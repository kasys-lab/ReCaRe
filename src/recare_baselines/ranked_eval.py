"""Evaluate an externally produced ranked-list JSONL against a qrels file.

This is the entry point for scoring a system that was *not* run through this
repository (e.g. a shared-task submission). The input is one JSON object per
query holding only the ranked document IDs, with no scores::

    {"qid": "q123", "ranked": ["d456", "d789", "d012"]}

Metrics are computed with the same code path as the paper baselines
(:func:`recare_baselines.eval.evaluate` / ``evaluate_per_query``), so a run
converted to this format reproduces the numbers in ``results/metrics/``.

Conversion to a scored run
--------------------------
``ir_measures`` / ``trec_eval`` sort by score (descending) and do not look at
input order. We therefore assign the strictly decreasing pseudo-score
``len(ranked) - rank`` so that the evaluated order is exactly the list order
and no ties are introduced.

Note that for a *scored* run, ``trec_eval`` breaks score ties by docid
(descending), not by input order. To reproduce a scored run's numbers from a
ranked list, the list must therefore be in ``(score, docid)`` descending order.

Per-query list processing
-------------------------
Applied in this order, matching the CAIL 2026 ReCaRe track rules and the paper:

1. A document repeated within one list is kept at its first position only.
2. The list is truncated to the first ``max_docs`` (default 1,000) documents.
3. With ``filter_self``, ``docid == qid`` is removed (Rev2Rev). Because this
   happens after truncation, a run that ranks the query article at rank 1 is
   scored on its remaining 999 documents, exactly like the paper baselines.

Query coverage
--------------
* Queries in the qrels but absent from the run are counted as 0 for every
  metric by default (``missing="zero"``; equivalent to ``trec_eval -c``).
  ``missing="skip"`` averages over the covered queries only, which is what
  ``cli._persist`` does for ``--limit-queries`` smoke runs. For a run that covers
  every qrels query the two settings give identical numbers.
* Queries in the run but absent from the qrels are ignored (with a warning).
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .eval import Qrels, Run, evaluate, evaluate_per_query, filter_self_match

log = logging.getLogger(__name__)

RankedRun = dict[str, list[str]]

MISSING_POLICIES = ("zero", "skip")
DEFAULT_MAX_DOCS = 1000  # CAIL 2026 ReCaRe track: at most 1,000 documents per query


@dataclass
class RankedEvalResult:
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    n_queries: int  # queries the averages are taken over
    n_qrels_queries: int
    n_run_queries: int
    missing_qids: list[str] = field(default_factory=list)  # in qrels, not in run
    extra_qids: list[str] = field(default_factory=list)  # in run, not in qrels
    n_duplicate_docs_dropped: int = 0
    n_truncated_queries: int = 0  # queries whose list exceeded max_docs
    n_self_matches_dropped: int = 0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _docid(item, *, path: Path, lineno: int) -> str:
    """Accept ``"doc"`` (spec) and, leniently, ``["doc", score]`` items."""
    if isinstance(item, str):
        return item
    if isinstance(item, list) and item and isinstance(item[0], str):
        return item[0]
    raise ValueError(
        f"{path}:{lineno}: 'ranked' items must be document-ID strings, got {item!r}"
    )


def _open_text(path: Path):
    """Open a text file, transparently decompressing ``*.gz``."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def read_ranked(path: Path) -> RankedRun:
    """Read ``{"qid", "ranked"}`` JSONL (optionally ``.gz``) as given.

    Lists are returned unmodified; de-duplication and truncation happen in
    :func:`evaluate_ranked`. A repeated ``qid`` line is an error, since there
    is no sound way to merge two rankings.
    """
    path = Path(path)
    run: RankedRun = {}
    with _open_text(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if not isinstance(rec, dict) or "qid" not in rec or "ranked" not in rec:
                raise ValueError(f"{path}:{lineno}: expected keys 'qid' and 'ranked'")
            qid = str(rec["qid"])
            if qid in run:
                raise ValueError(f"{path}:{lineno}: duplicate qid {qid!r}")
            if not isinstance(rec["ranked"], list):
                raise ValueError(f"{path}:{lineno}: 'ranked' must be a list")
            run[qid] = [_docid(item, path=path, lineno=lineno) for item in rec["ranked"]]
    return run


def read_qrels(path: Path) -> Qrels:
    """Read qrels in either supported format, auto-detected per line.

    * ReCaRe / BEIR JSONL: ``{"query-id": ..., "corpus-id": ..., "score": 1}``
      (the HF ``qrels-{task}-{lang}/{split}.jsonl`` files)
    * TREC: ``qid  iter  docid  rel`` (whitespace separated)
    """
    path = Path(path)
    qrels: Qrels = {}
    with _open_text(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                rec = json.loads(line)
                try:
                    qid, did = str(rec["query-id"]), str(rec["corpus-id"])
                except KeyError as e:
                    raise ValueError(
                        f"{path}:{lineno}: JSONL qrels need 'query-id' and 'corpus-id'"
                    ) from e
                score = int(rec.get("score", 0))
            else:
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"{path}:{lineno}: TREC qrels need 4 columns, got {len(parts)}"
                    )
                qid, _, did, rel = parts
                score = int(rel)
            qrels.setdefault(qid, {})[did] = score
    return qrels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def to_scored_run(ranked: RankedRun) -> Run:
    """Give each list strictly decreasing scores so evaluation order == list order."""
    return {
        qid: [(did, float(len(docs) - i)) for i, did in enumerate(docs)]
        for qid, docs in ranked.items()
    }


def prepare_lists(ranked: RankedRun, max_docs: int | None) -> tuple[RankedRun, int, int]:
    """De-duplicate (first occurrence wins), then truncate to ``max_docs``.

    Returns ``(lists, n_duplicates_dropped, n_truncated_queries)``.
    """
    out: RankedRun = {}
    n_dup = n_trunc = 0
    for qid, docs in ranked.items():
        seen: set[str] = set()
        uniq = []
        for did in docs:
            if did in seen:
                n_dup += 1
                continue
            seen.add(did)
            uniq.append(did)
        if max_docs is not None and len(uniq) > max_docs:
            n_trunc += 1
            uniq = uniq[:max_docs]
        out[qid] = uniq
    if n_dup:
        log.warning("dropped %d duplicate document(s) within ranked lists", n_dup)
    if n_trunc:
        log.warning("truncated %d ranked list(s) to %d documents", n_trunc, max_docs)
    return out, n_dup, n_trunc


def evaluate_ranked(
    ranked: RankedRun,
    qrels: Qrels,
    *,
    filter_self: bool = False,
    missing: str = "zero",
    max_docs: int | None = DEFAULT_MAX_DOCS,
) -> RankedEvalResult:
    """Score a ranked-list run with the paper metric set.

    Each list is de-duplicated and truncated to ``max_docs`` (``None`` = no
    limit), then ``filter_self`` drops ``docid == qid`` rows (required for
    Rev2Rev to match the paper; see
    :func:`recare_baselines.eval.filter_self_match`).
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}, got {missing!r}")
    if max_docs is not None and max_docs < 1:
        raise ValueError(f"max_docs must be positive or None, got {max_docs!r}")

    lists, n_dup, n_trunc = prepare_lists(ranked, max_docs)
    run = to_scored_run(lists)
    n_self = 0
    if filter_self:
        before = sum(len(v) for v in run.values())
        run = filter_self_match(run)
        n_self = before - sum(len(v) for v in run.values())

    # A query whose list is empty (possibly only after self-match filtering)
    # retrieves nothing; treat it exactly like a query absent from the run.
    missing_qids = sorted(q for q in qrels if not run.get(q))
    extra_qids = sorted(q for q in run if q not in qrels)
    if extra_qids:
        log.warning("%d run qid(s) not in qrels are ignored", len(extra_qids))
    if missing_qids:
        log.warning(
            "%d qrels qid(s) have no ranking in the run (policy: %s)",
            len(missing_qids),
            missing,
        )

    eval_run = {q: docs for q, docs in run.items() if q in qrels and docs}
    if not eval_run:
        raise ValueError("no query in the run appears in the qrels")
    if missing == "zero":
        # ``ir_measures`` averages over every qrels query and scores the ones
        # without a ranking as 0, i.e. ``trec_eval -c`` semantics.
        eval_qrels = qrels
    else:
        eval_qrels = {q: rels for q, rels in qrels.items() if q in eval_run}

    # Same calls as ``cli._persist`` -> identical numbers for full-coverage runs.
    metrics = evaluate(eval_run, eval_qrels)
    per_query = evaluate_per_query(eval_run, eval_qrels)
    if missing == "zero" and missing_qids:
        measure_keys = next(iter(per_query.values())).keys()
        for q in missing_qids:
            per_query.setdefault(q, {k: 0.0 for k in measure_keys})
    n_queries = len(eval_qrels)

    return RankedEvalResult(
        metrics=metrics,
        per_query=per_query,
        n_queries=n_queries,
        n_qrels_queries=len(qrels),
        n_run_queries=len(ranked),
        missing_qids=missing_qids,
        extra_qids=extra_qids,
        n_duplicate_docs_dropped=n_dup,
        n_truncated_queries=n_trunc,
        n_self_matches_dropped=n_self,
    )


def evaluate_ranked_files(
    run_path: Path,
    qrels_path: Path,
    *,
    filter_self: bool = False,
    missing: str = "zero",
    max_docs: int | None = DEFAULT_MAX_DOCS,
) -> RankedEvalResult:
    return evaluate_ranked(
        read_ranked(run_path),
        read_qrels(qrels_path),
        filter_self=filter_self,
        missing=missing,
        max_docs=max_docs,
    )
