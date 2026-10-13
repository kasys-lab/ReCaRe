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

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .eval import Qrels, Run, evaluate, evaluate_per_query, filter_self_match

log = logging.getLogger(__name__)

RankedRun = dict[str, list[str]]

MISSING_POLICIES = ("zero", "skip")


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


def read_ranked(path: Path) -> tuple[RankedRun, int]:
    """Read ``{"qid", "ranked"}`` JSONL.

    Returns ``(run, n_duplicates_dropped)``. A document repeated within one
    query's list is kept at its first (highest) position only; ``trec_eval``
    would otherwise count it more than once. A repeated ``qid`` line is an
    error, since there is no sound way to merge two rankings.
    """
    path = Path(path)
    run: RankedRun = {}
    n_dup = 0
    with open(path, encoding="utf-8") as f:
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
            seen: set[str] = set()
            ranked: list[str] = []
            for item in rec["ranked"]:
                did = _docid(item, path=path, lineno=lineno)
                if did in seen:
                    n_dup += 1
                    continue
                seen.add(did)
                ranked.append(did)
            run[qid] = ranked
    if n_dup:
        log.warning("dropped %d duplicate document(s) within ranked lists", n_dup)
    return run, n_dup


def read_qrels(path: Path) -> Qrels:
    """Read qrels in either supported format, auto-detected per line.

    * ReCaRe / BEIR JSONL: ``{"query-id": ..., "corpus-id": ..., "score": 1}``
      (the HF ``qrels-{task}-{lang}/{split}.jsonl`` files)
    * TREC: ``qid  iter  docid  rel`` (whitespace separated)
    """
    path = Path(path)
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as f:
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


def evaluate_ranked(
    ranked: RankedRun,
    qrels: Qrels,
    *,
    filter_self: bool = False,
    missing: str = "zero",
) -> RankedEvalResult:
    """Score a ranked-list run with the paper metric set.

    ``filter_self`` drops ``docid == qid`` rows first (required for Rev2Rev to
    match the paper; see :func:`recare_baselines.eval.filter_self_match`).
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}, got {missing!r}")

    run = to_scored_run(ranked)
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
        n_duplicate_docs_dropped=0,
        n_self_matches_dropped=n_self,
    )


def evaluate_ranked_files(
    run_path: Path,
    qrels_path: Path,
    *,
    filter_self: bool = False,
    missing: str = "zero",
) -> RankedEvalResult:
    ranked, n_dup = read_ranked(run_path)
    qrels = read_qrels(qrels_path)
    result = evaluate_ranked(ranked, qrels, filter_self=filter_self, missing=missing)
    result.n_duplicate_docs_dropped = n_dup
    return result
