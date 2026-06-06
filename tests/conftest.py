"""Shared pytest fixtures for ReCaRe baseline tests.

The fixtures here let unit tests exercise the I/O contracts of the package
(loaders, runfile readers, metrics aggregation) without hitting HuggingFace
or instantiating real models. Real-model tests should be marked
``slow`` and gated on environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class SyntheticBundle:
    """Tiny ReCaRe-shaped bundle for unit tests.

    Paths are absolute under a ``tmp_path``. JSONL files use the exact schema
    the loaders expect (``_id``/``text`` for corpus and queries,
    ``query-id``/``corpus-id``/``score`` for qrels).
    """

    corpus_path: Path
    queries_path: Path
    qrels_path: Path
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


@pytest.fixture
def synthetic_bundle(tmp_path) -> SyntheticBundle:
    """Build a 5-query × 50-doc synthetic ReCaRe bundle.

    Document texts contain words such that:
      - ``q0``'s positives use the word "alpha"
      - ``q1``'s positives use "beta"
      - etc.
    BM25 should rank positives near the top for each query, giving callers a
    realistic-shaped ranking to exercise downstream code (metrics, runfiles)
    without paying for real model inference.
    """
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    n_queries = 5
    n_docs = 50

    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i in range(n_docs):
            tag = words[i % n_queries]
            text = (
                f"this is document {i} discussing {tag} matters in legal context. "
                f"the regulation applies broadly to {tag}-related entities."
            )
            f.write(json.dumps({"_id": f"d{i}", "text": text}) + "\n")

    queries: dict[str, str] = {}
    queries_path = tmp_path / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for i in range(n_queries):
            qid = f"q{i}"
            text = f"explain how the {words[i]} provisions are amended"
            queries[qid] = text
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")

    qrels: dict[str, dict[str, int]] = {}
    qrels_path = tmp_path / "qrels.jsonl"
    with open(qrels_path, "w", encoding="utf-8") as f:
        for i in range(n_queries):
            qid = f"q{i}"
            qrels[qid] = {}
            # Each query has 2 positives whose docs contain its keyword.
            for k in range(2):
                doc_idx = i + k * n_queries
                docid = f"d{doc_idx}"
                qrels[qid][docid] = 1
                f.write(
                    json.dumps({"query-id": qid, "corpus-id": docid, "score": 1}) + "\n"
                )

    return SyntheticBundle(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        queries=queries,
        qrels=qrels,
    )
