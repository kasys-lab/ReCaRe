"""BM25 retrieval over ReCaRe via Pyserini Lucene.

Pipeline:

1. ``build_index(lang)`` constructs a Lucene index per language under
   ``indexes/lucene/{lang}`` from the Pyserini-format JSONL prepared by
   :func:`recare_baselines.data.write_pyserini_corpus`.
2. ``run_search(...)`` issues queries against that index. Lucene has no input
   length cap, so long Rat2Rev EU queries are passed through as-is.

BM25 hyperparameters: ``k1=0.9``, ``b=0.4``. Per-language Lucene Analyzers
(English / Japanese) are selected by the ``--language`` flag passed to
``pyserini.index.lucene``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from . import data  # noqa: E402 — package __init__ installs the OpenAI env stub

logger = logging.getLogger(__name__)

INDEX_ROOT = data.INDEX_ROOT / "lucene"

_LANG_PYSERINI_FLAG = {
    "en": "en",  # EnglishAnalyzer
    "ja": "ja",  # JapaneseAnalyzer (Kuromoji)
}

_BM25_K1 = 0.9
_BM25_B = 0.4


def index_path(lang: str, doc_aug_suffix: str | None = None) -> Path:
    """Lucene index path. Plain corpus → ``indexes/lucene/{lang}``.
    Doc-augmented corpus → ``indexes/lucene/{lang}+{suffix}`` (e.g. ``en+d2q``).
    """
    if doc_aug_suffix:
        return INDEX_ROOT / f"{lang}+{doc_aug_suffix}"
    return INDEX_ROOT / lang


def build_index(
    lang: str,
    *,
    threads: int = 8,
    force: bool = False,
    doc_augmentation: Path | None = None,
    doc_aug_suffix: str | None = None,
) -> Path:
    """Build a Lucene index for the language-specific ReCaRe corpus.

    If ``doc_augmentation`` is given, build an index over the augmented
    corpus (original doc + ``doc_augmentation`` JSONL contents). The caller
    must also pass ``doc_aug_suffix`` (e.g. ``"d2q"``, ``"d2e"``) — this
    becomes part of the index path so the searcher can pick the right one.
    """
    if lang not in _LANG_PYSERINI_FLAG:
        raise ValueError(f"unsupported lang {lang!r}")
    if doc_augmentation is not None and not doc_aug_suffix:
        raise ValueError("doc_augmentation requires doc_aug_suffix")
    out_dir = index_path(lang, doc_aug_suffix)
    if out_dir.exists() and not force:
        logger.info("Lucene index already present at %s; reuse (force=True to rebuild)", out_dir)
        return out_dir
    if force and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if doc_augmentation is not None:
        corpus_jsonl = data.write_pyserini_corpus_augmented(
            lang, doc_augmentation, suffix=doc_aug_suffix
        )
    else:
        corpus_jsonl = data.write_pyserini_corpus(lang)
    logger.info("Indexing %s -> %s", corpus_jsonl, out_dir)

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(corpus_jsonl.parent),
        "--index",
        str(out_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
        "--language",
        _LANG_PYSERINI_FLAG[lang],
    ]
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    subprocess.run(cmd, check=True, env=env)
    return out_dir


def _searcher(lang: str, doc_aug_suffix: str | None = None):
    from pyserini.search.lucene import LuceneSearcher

    s = LuceneSearcher(str(index_path(lang, doc_aug_suffix)))
    s.set_bm25(_BM25_K1, _BM25_B)
    if lang in _LANG_PYSERINI_FLAG:
        s.set_language(_LANG_PYSERINI_FLAG[lang])
    return s


def _hits_to_list(hits) -> list[tuple[str, float]]:
    return [(h.docid, float(h.score)) for h in hits]


def run_search(
    queries: dict[str, str],
    *,
    lang: str,
    top_k: int = 1000,
    doc_aug_suffix: str | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Run BM25 over a dict of queries against the per-language Lucene index.

    If ``doc_aug_suffix`` is set, the searcher loads the augmented index
    at ``indexes/lucene/{lang}+{suffix}``. The caller is responsible for
    having built that index via :func:`build_index` first.
    """
    searcher = _searcher(lang, doc_aug_suffix)
    out: dict[str, list[tuple[str, float]]] = {}
    for qid, qtext in queries.items():
        if not qtext.strip():
            out[qid] = []
            continue
        hits = searcher.search(qtext, k=top_k)
        out[qid] = _hits_to_list(hits)
    return out
