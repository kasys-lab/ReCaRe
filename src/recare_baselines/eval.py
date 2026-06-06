"""Evaluation: Recall@10/100/1000, nDCG@10, MAP via ``ir_measures``.

We delegate metric arithmetic to the ``ir_measures`` package, which wraps
``pytrec_eval``. Callers pass plain dicts; we adapt them to ``Qrel`` /
``ScoredDoc`` namedtuples on the fly.

Conventions follow ``trec_eval``:

* qrels: ``score >= 1`` is positive (binary for R@k / MAP, graded for nDCG@10).
* runs: per-query lists pre-sorted by score descending; ``ir_measures`` does
  not require sorting but we keep the order stable for deterministic output.
"""

from __future__ import annotations

from collections.abc import Iterator

import ir_measures
from ir_measures import AP, Qrel, R, ScoredDoc, nDCG

# Public type aliases.
Qrels = dict[str, dict[str, int]]
Run = dict[str, list[tuple[str, float]]]


# Measures used everywhere baselines are reported.
_DEFAULT_MEASURES = (
    R @ 10,
    R @ 100,
    R @ 1000,
    nDCG @ 10,
    nDCG @ 100,
    nDCG @ 1000,
    AP,
)


def filter_self_match(run: Run) -> Run:
    """Drop ``qid == docid`` rows from each query's ranking.

    For ReCaRe Rev2Rev the query is itself a corpus article, so the query
    document trivially matches itself at retrieval time. The HF dataset card
    recommends post-filtering the run rather than re-indexing per query; we
    drop offending rows and let downstream ranks fill the gap.
    """
    out: Run = {}
    for qid, ranked in run.items():
        out[qid] = [(d, s) for d, s in ranked if d != qid]
    return out


def _qrel_iter(qrels: Qrels) -> Iterator[Qrel]:
    for qid, rels in qrels.items():
        for did, score in rels.items():
            yield Qrel(qid, did, int(score))


def _run_iter(run: Run) -> Iterator[ScoredDoc]:
    for qid, ranked in run.items():
        for did, score in ranked:
            yield ScoredDoc(qid, did, float(score))


def evaluate(run: Run, qrels: Qrels) -> dict[str, float]:
    """Compute the standard metric set and return a JSON-friendly dict."""
    results = ir_measures.calc_aggregate(
        list(_DEFAULT_MEASURES), _qrel_iter(qrels), _run_iter(run)
    )
    return {
        "recall_10": float(results[R @ 10]),
        "recall_100": float(results[R @ 100]),
        "recall_1000": float(results[R @ 1000]),
        "ndcg_10": float(results[nDCG @ 10]),
        "ndcg_100": float(results[nDCG @ 100]),
        "ndcg_1000": float(results[nDCG @ 1000]),
        "map": float(results[AP]),
    }


def evaluate_per_query(run: Run, qrels: Qrels) -> dict[str, dict[str, float]]:
    """Per-query metric breakdown (useful for failure analysis)."""
    by_qid: dict[str, dict[str, float]] = {}
    for measure in _DEFAULT_MEASURES:
        for r in ir_measures.iter_calc([measure], _qrel_iter(qrels), _run_iter(run)):
            by_qid.setdefault(r.query_id, {})[str(measure)] = float(r.value)
    return by_qid
