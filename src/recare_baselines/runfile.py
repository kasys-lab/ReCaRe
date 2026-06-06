"""Persistent JSONL run-file format.

A run-file stores per-query top-k retrieval results so we can:

* feed BM25 top-100 to rerankers (#7) and hard-negative miners (#3),
* re-evaluate with different metrics without rerunning the model,
* keep raw retriever outputs auditable.

Format (one JSON object per query)::

    {"qid": "32010L0066", "results": [["doc-id-1", 24.7], ["doc-id-2", 22.1], ...]}
"""

from __future__ import annotations

import json
from pathlib import Path

Run = dict[str, list[tuple[str, float]]]


def write_run(path: Path, run: Run, *, top_k: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in run.items():
            if top_k is not None:
                ranked = ranked[:top_k]
            f.write(
                json.dumps({"qid": qid, "results": [[d, s] for d, s in ranked]})
                + "\n"
            )


def read_run(path: Path) -> Run:
    out: Run = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["qid"]] = [(d, float(s)) for d, s in rec["results"]]
    return out
