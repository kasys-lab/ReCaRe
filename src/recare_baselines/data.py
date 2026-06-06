"""ReCaRe data access shared by all baselines.

Reuses the ReCaRe HuggingFace dataset (``kasys/ReCaRe``). Each task / language
has the same shared corpus (``corpus-{lang}/corpus.jsonl``), distinct queries
(``queries-{task}-{lang}/queries.jsonl``), and qrels split into train /
validation / test JSONL.

For baseline retrieval we use the **test** split (per project decision) and
search the **full** corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = REPO_ROOT / "indexes"
RESULTS_ROOT = REPO_ROOT / "results"
INTERMEDIATE_ROOT = RESULTS_ROOT / "intermediate"
PYSERINI_CORPUS_ROOT = INDEX_ROOT / "pyserini_corpus"

TASKS = ("rat2rev", "rev2rev")
LANGS = ("en", "ja")


@dataclass
class ReCaReData:
    task: str
    lang: str
    split: str
    corpus_path: Path  # original HF JSONL (read-only cache)
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]  # qid -> {doc_id -> score}

    @property
    def lucene_lang(self) -> str:
        """Lucene Analyzer language code (``en`` / ``ja``)."""
        return self.lang


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _hf_path(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="kasys/ReCaRe",
            filename=filename,
            repo_type="dataset",
        )
    )


def load(task: str, lang: str, split: str = "test") -> ReCaReData:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split must be train/validation/test, got {split!r}")

    corpus_path = _hf_path(f"corpus-{lang}/corpus.jsonl")
    queries_path = _hf_path(f"queries-{task}-{lang}/queries.jsonl")
    qrels_path = _hf_path(f"qrels-{task}-{lang}/{split}.jsonl")

    qrels: dict[str, dict[str, int]] = {}
    for rec in _read_jsonl(qrels_path):
        qrels.setdefault(rec["query-id"], {})[rec["corpus-id"]] = int(rec.get("score", 0))

    split_qids = set(qrels.keys())
    queries: dict[str, str] = {}
    for rec in _read_jsonl(queries_path):
        if rec["_id"] in split_qids:
            queries[rec["_id"]] = rec.get("text", "")

    return ReCaReData(
        task=task,
        lang=lang,
        split=split,
        corpus_path=corpus_path,
        queries=queries,
        qrels=qrels,
    )


def iter_corpus(data: ReCaReData):
    """Yield ``(doc_id, text)`` tuples for every doc in the corpus."""
    for rec in _read_jsonl(data.corpus_path):
        yield rec["_id"], rec.get("text", "")


def write_pyserini_corpus(lang: str) -> Path:
    """Materialize the language-specific corpus as a Pyserini-friendly JSONL.

    Output: ``indexes/pyserini_corpus/{lang}/docs.jsonl`` with one JSON object
    per line containing ``id`` and ``contents`` fields (Pyserini's expected
    schema). Returns the path to that file.
    """
    out_dir = PYSERINI_CORPUS_ROOT / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "docs.jsonl"
    if out_path.exists():
        return out_path
    corpus_path = _hf_path(f"corpus-{lang}/corpus.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in _read_jsonl(corpus_path):
            out.write(
                json.dumps(
                    {"id": rec["_id"], "contents": rec.get("text", "")},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


def write_pyserini_corpus_augmented(
    lang: str, aug_jsonl: Path, *, suffix: str
) -> Path:
    """Materialize an augmented corpus: ``contents = original + " " + augmentation``.

    The augmentation JSONL is expected to use the schema
    ``{"id": <docid>, "contents": <augmentation text>}`` (matches the layout
    of ``data/recare_d2q/`` and ``data/recare_d2e/``). Augmentation text is
    appended to the original corpus text — this is the canonical Doc2Query /
    LLM doc-expansion concatenation in the literature (e.g., Nogueira+ 2019,
    2F-01 §3.3.4). Docs in the original corpus that are missing from the
    augmentation file pass through unchanged.

    Output path: ``indexes/pyserini_corpus/{lang}+{suffix}/docs.jsonl``.
    Caller chooses ``suffix`` (e.g. ``"d2q"`` or ``"d2e"``); the suffix
    propagates to the Lucene index path and downstream model_key.

    Returns the output file path. Reuses an existing file if present.
    """
    out_dir = PYSERINI_CORPUS_ROOT / f"{lang}+{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "docs.jsonl"
    if out_path.exists():
        return out_path

    aug_text: dict[str, str] = {}
    with open(aug_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aug_text[rec["id"]] = rec.get("contents", "") or ""

    corpus_path = _hf_path(f"corpus-{lang}/corpus.jsonl")
    n_aug = n_passthrough = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in _read_jsonl(corpus_path):
            doc_id = rec["_id"]
            base = rec.get("text", "")
            ext = aug_text.get(doc_id, "")
            if ext:
                contents = f"{base} {ext}"
                n_aug += 1
            else:
                contents = base
                n_passthrough += 1
            out.write(
                json.dumps({"id": doc_id, "contents": contents}, ensure_ascii=False)
                + "\n"
            )
    return out_path
