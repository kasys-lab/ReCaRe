"""Hard-negative construction for dense retriever domain adaptation.

Phase 1 of the domain-adaptation pipeline needs two artifacts:

* split-aware dense top-100 files for train / validation queries
* one hard-negative training triple per ``(query, positive)`` pair

The mining rule mirrors the issue memo: start from the initial dense model's
post-filtered top-100, remove qrels positives, remove ``rev2rev`` self matches,
then pick one remaining document with a fixed RNG seed.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from . import data
from . import dense as dense_mod
from .eval import filter_self_match
from .runfile import Run, read_run, write_run

logger = logging.getLogger(__name__)

DENSE_TOP100_DIRNAME = "dense_top100"
TRAINING_DATA_DIRNAME = "training_data/dense"
TRAINING_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class DenseTrainingExample:
    qid: str
    positive_doc_id: str
    hard_negative_doc_id: str

    def to_json(self) -> dict[str, str]:
        return {
            "qid": self.qid,
            "positive_doc_id": self.positive_doc_id,
            "hard_negative_doc_id": self.hard_negative_doc_id,
        }


@dataclass(frozen=True)
class MiningResult:
    examples: list[DenseTrainingExample]
    seed: int
    n_queries: int
    n_queries_with_positive: int
    n_positive_pairs: int
    n_skipped_queries: int
    n_missing_top100_queries: int


@dataclass(frozen=True)
class DenseTop100BuildResult:
    path: Path
    created: bool
    n_queries: int
    split: str
    top_k: int


@dataclass(frozen=True)
class TrainingDataBuildResult:
    path: Path
    mining: MiningResult

    @property
    def n_examples(self) -> int:
        return len(self.mining.examples)


def _validate_training_split(split: str) -> None:
    if split not in TRAINING_SPLITS:
        raise ValueError(
            f"split must be one of {TRAINING_SPLITS} for domain-adaptation data, "
            f"got {split!r}"
        )


def dense_top100_path(
    task: str,
    lang: str,
    model_key: str,
    split: str,
    *,
    intermediate_root: Path = data.INTERMEDIATE_ROOT,
) -> Path:
    """Return the canonical dense top-100 path for a split.

    Train / validation use split-aware subdirectories. The test split keeps the
    historical flat path because rerankers and reports already consume it.
    """
    base = intermediate_root / DENSE_TOP100_DIRNAME
    name = f"{task}_{lang}_{model_key}.jsonl"
    if split in TRAINING_SPLITS:
        return base / split / name
    if split == "test":
        return base / name
    raise ValueError(f"split must be train/validation/test, got {split!r}")


def training_data_path(
    task: str,
    lang: str,
    model_key: str,
    split: str,
    *,
    intermediate_root: Path = data.INTERMEDIATE_ROOT,
) -> Path:
    _validate_training_split(split)
    return (
        intermediate_root
        / TRAINING_DATA_DIRNAME
        / split
        / f"{task}_{lang}_{model_key}.jsonl"
    )


def write_dense_top100(
    run: Run,
    *,
    task: str,
    lang: str,
    model_key: str,
    split: str,
    top_k: int = 100,
    intermediate_root: Path = data.INTERMEDIATE_ROOT,
) -> Path:
    """Write a dense top-100 file without touching the legacy test flat path."""
    _validate_training_split(split)
    path = dense_top100_path(
        task,
        lang,
        model_key,
        split,
        intermediate_root=intermediate_root,
    )
    write_run(path, run, top_k=top_k)
    return path


def load_dense_top100(
    *,
    task: str,
    lang: str,
    model_key: str,
    split: str,
    intermediate_root: Path = data.INTERMEDIATE_ROOT,
) -> Run:
    path = dense_top100_path(
        task,
        lang,
        model_key,
        split,
        intermediate_root=intermediate_root,
    )
    return read_run(path)


def _positive_doc_ids(qrels_for_query: dict[str, int]) -> list[str]:
    return sorted(did for did, rel in qrels_for_query.items() if int(rel) > 0)


def hard_negative_candidates(
    qid: str,
    ranked: list[tuple[str, float]],
    positives: set[str],
    *,
    task: str,
) -> list[str]:
    """Filter one query's dense ranking down to valid hard-negative doc ids."""
    out: list[str] = []
    seen: set[str] = set()
    for doc_id, _score in ranked:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in positives:
            continue
        if task == "rev2rev" and doc_id == qid:
            continue
        out.append(doc_id)
    return out


def mine_training_examples(
    qrels: dict[str, dict[str, int]],
    dense_top100: Run,
    *,
    task: str,
    seed: int = 13,
) -> MiningResult:
    """Mine dense contrastive triples from qrels and split-specific top-100."""
    rng = random.Random(seed)
    examples: list[DenseTrainingExample] = []
    n_positive_pairs = 0
    n_queries_with_positive = 0
    n_skipped_queries = 0
    n_missing_top100_queries = 0

    for qid in sorted(qrels):
        positives = _positive_doc_ids(qrels[qid])
        if not positives:
            continue
        n_queries_with_positive += 1
        n_positive_pairs += len(positives)

        ranked = dense_top100.get(qid)
        if ranked is None:
            n_missing_top100_queries += 1
            n_skipped_queries += 1
            continue

        candidates = hard_negative_candidates(
            qid,
            ranked,
            set(positives),
            task=task,
        )
        if not candidates:
            n_skipped_queries += 1
            continue

        for positive_doc_id in positives:
            examples.append(
                DenseTrainingExample(
                    qid=qid,
                    positive_doc_id=positive_doc_id,
                    hard_negative_doc_id=rng.choice(candidates),
                )
            )

    result = MiningResult(
        examples=examples,
        seed=seed,
        n_queries=len(qrels),
        n_queries_with_positive=n_queries_with_positive,
        n_positive_pairs=n_positive_pairs,
        n_skipped_queries=n_skipped_queries,
        n_missing_top100_queries=n_missing_top100_queries,
    )
    logger.info(
        "dense hard-negative mining: examples=%d positive_pairs=%d "
        "skipped_queries=%d missing_top100=%d seed=%d",
        len(examples),
        n_positive_pairs,
        n_skipped_queries,
        n_missing_top100_queries,
        seed,
    )
    return result


def write_training_examples(path: Path, examples: list[DenseTrainingExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")


def read_training_examples(path: Path) -> list[DenseTrainingExample]:
    examples: list[DenseTrainingExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            examples.append(
                DenseTrainingExample(
                    qid=rec["qid"],
                    positive_doc_id=rec["positive_doc_id"],
                    hard_negative_doc_id=rec["hard_negative_doc_id"],
                )
            )
    return examples


def build_dense_top100(
    model_key: str,
    task: str,
    lang: str,
    *,
    split: str,
    top_k: int = 100,
    search_top_k: int = 1000,
    batch_size: int = 64,
    force: bool = False,
    intermediate_root: Path = data.INTERMEDIATE_ROOT,
) -> DenseTop100BuildResult:
    """Run the initial dense retriever and persist split-aware top-100."""
    _validate_training_split(split)
    path = dense_top100_path(
        task,
        lang,
        model_key,
        split,
        intermediate_root=intermediate_root,
    )
    if path.exists() and not force:
        logger.info("dense top-100 already present at %s", path)
        existing = read_run(path)
        return DenseTop100BuildResult(
            path=path,
            created=False,
            n_queries=len(existing),
            split=split,
            top_k=top_k,
        )

    rd = data.load(task, lang, split=split)
    run = dense_mod.run_search(
        model_key,
        lang,
        rd.queries,
        top_k=max(top_k, search_top_k),
        batch_size=batch_size,
    )
    eval_run = filter_self_match(run) if task == "rev2rev" else run
    write_dense_top100(
        eval_run,
        task=task,
        lang=lang,
        model_key=model_key,
        split=split,
        top_k=top_k,
        intermediate_root=intermediate_root,
    )
    return DenseTop100BuildResult(
        path=path,
        created=True,
        n_queries=len(rd.queries),
        split=split,
        top_k=top_k,
    )


def build_training_examples(
    model_key: str,
    task: str,
    lang: str,
    *,
    split: str,
    seed: int = 13,
    top100_path: Path | None = None,
    out_path: Path | None = None,
    intermediate_root: Path = data.INTERMEDIATE_ROOT,
) -> TrainingDataBuildResult:
    """Build and write dense training triples for one split."""
    _validate_training_split(split)
    rd = data.load(task, lang, split=split)
    if top100_path is None:
        top100_path = dense_top100_path(
            task,
            lang,
            model_key,
            split,
            intermediate_root=intermediate_root,
        )
    dense_top100 = read_run(top100_path)
    mining = mine_training_examples(rd.qrels, dense_top100, task=task, seed=seed)

    if out_path is None:
        out_path = training_data_path(
            task,
            lang,
            model_key,
            split,
            intermediate_root=intermediate_root,
        )
    write_training_examples(out_path, mining.examples)
    logger.info(
        "wrote dense training examples: path=%s examples=%d skipped_queries=%d",
        out_path,
        len(mining.examples),
        mining.n_skipped_queries,
    )
    return TrainingDataBuildResult(path=out_path, mining=mining)
