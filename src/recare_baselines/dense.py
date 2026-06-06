"""Multilingual dense retrievers (short- and long-context).

Models registered here cover both Issue #5 (short-context, 512 tok cap:
mDPR / mContriever / mE5-base) and Issue #6 (long-context, 8192 tok cap:
BGE-M3 dense mode / jina-embeddings-v3).

Pipeline:

1. ``encode_corpus(model_key, lang)`` runs the encoder over every document in
   the language-specific ReCaRe corpus and writes the resulting matrix to
   ``indexes/dense/{model_key}/{lang}/{embeddings.npy, ids.txt}``.
2. ``run_search(model_key, lang, queries, top_k)`` encodes each query and
   returns the top-``k`` matches by inner product (cosine when the model's
   ``normalize`` is set).

Inputs longer than ``max_length`` are **head-truncated** via the HuggingFace
tokenizer's default ``truncation=True, max_length=...``. We do not chunk +
RRF: long queries that exceed a model's cap drop the tail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import data

logger = logging.getLogger(__name__)

INDEX_DIR = data.INDEX_ROOT / "dense"


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short identifier used in paths and CLI ("me5-base", "bge-m3", ...)
    hf_id: str
    pooling: str  # "cls" or "mean" or "jina_encode"
    normalize: bool  # whether to L2-normalize embeddings
    query_prefix: str = ""
    passage_prefix: str = ""
    max_length: int = 512
    trust_remote_code: bool = False
    # Default per-call encoder kwargs (e.g. {"task": "retrieval.passage"} for jina-v3).
    # The encoder picks the right entry from ``query_kwargs`` / ``passage_kwargs``.
    query_kwargs: dict = field(default_factory=dict)
    passage_kwargs: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    # ---- short-context (issue #5) ---------------------------------------
    "mdpr": ModelSpec(
        key="mdpr",
        hf_id="castorini/mdpr-tied-pft-msmarco",
        pooling="cls",
        normalize=False,
    ),
    "mcontriever": ModelSpec(
        key="mcontriever",
        hf_id="facebook/mcontriever",
        pooling="mean",
        normalize=False,
    ),
    "me5-base": ModelSpec(
        key="me5-base",
        hf_id="intfloat/multilingual-e5-base",
        pooling="mean",
        normalize=True,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    # ---- smoke-test tier ------------------------------------------------
    # multilingual-e5-small (~118M params, 384-dim, EN/JA supported).
    # Same prefix convention as me5-base. Used by scripts/smoke_test.sh for
    # CPU-only laptop sanity checks; not part of the paper's reported baselines.
    "me5-small": ModelSpec(
        key="me5-small",
        hf_id="intfloat/multilingual-e5-small",
        pooling="mean",
        normalize=True,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    # ---- long-context (issue #6) ----------------------------------------
    "bge-m3": ModelSpec(
        key="bge-m3",
        hf_id="BAAI/bge-m3",
        pooling="cls",
        normalize=True,
        max_length=8192,
    ),
    "jina-v3": ModelSpec(
        key="jina-v3",
        hf_id="jinaai/jina-embeddings-v3",
        pooling="jina_encode",  # uses the model's own .encode() method
        normalize=True,
        max_length=8192,
        trust_remote_code=True,
        query_kwargs={"task": "retrieval.query"},
        passage_kwargs={"task": "retrieval.passage"},
    ),
}


SHORT_CONTEXT_KEYS = ("mdpr", "mcontriever", "me5-base")
LONG_CONTEXT_KEYS = ("bge-m3", "jina-v3")


# ---------------------------------------------------------------------------
# Document-side chunking for max-P scoring
# ---------------------------------------------------------------------------


def _maxp_chunk_budget(spec: ModelSpec) -> int:
    """How many *content* BPE tokens each max-P passage may carry.

    Subtracts the encoder's prefix (``passage: ``) and a small safety margin
    for special tokens (``[CLS]``, ``[SEP]``) from the model's ``max_length``.
    """
    return max(spec.max_length - 8, 64)


def _split_text_for_maxp(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split ``text`` into non-overlapping BPE chunks each ≤ ``max_tokens`` tokens.

    Used symmetrically for queries and documents in max-P scoring. Inputs that
    already fit return a single-element list (so callers can treat the chunk
    list uniformly). Inputs that exceed the cap are sliced on raw BPE token
    boundaries — no sentence respect — so the encoder sees exactly
    ``max_tokens`` content tokens per chunk.
    """
    if not text or not text.strip():
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    return [
        tokenizer.decode(ids[i : i + max_tokens], skip_special_tokens=True)
        for i in range(0, len(ids), max_tokens)
    ]


# Back-compat alias.
_split_doc_for_maxp = _split_text_for_maxp


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(spec: ModelSpec):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code
    )
    model = AutoModel.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code
    )
    model.eval()
    model.to(_device())
    if _device() == "cuda":
        model = model.half()
    return tokenizer, model


def _pool(last_hidden_state, attention_mask, pooling: str):
    if pooling == "cls":
        return last_hidden_state[:, 0]
    # Mean pooling over non-pad tokens.
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def _encode_batch(spec: ModelSpec, tokenizer, model, texts: list[str], is_query: bool):
    import torch

    if spec.pooling == "jina_encode":
        # jina-embeddings-v3 has a custom forward that takes a ``task`` kwarg
        # to switch task-specific LoRA adapters. We delegate to model.encode()
        # which handles tokenization, pooling, and normalization internally.
        kwargs = spec.query_kwargs if is_query else spec.passage_kwargs
        with torch.no_grad():
            out = model.encode(
                texts,
                max_length=spec.max_length,
                truncate_dim=None,
                **kwargs,
            )
        # ``out`` is a numpy.ndarray (jina default) or torch.Tensor.
        if hasattr(out, "cpu"):
            out = out.cpu().numpy()
        return np.asarray(out, dtype=np.float32)

    prefix = spec.query_prefix if is_query else spec.passage_prefix
    inputs = tokenizer(
        [prefix + t for t in texts],
        padding=True,
        truncation=True,
        max_length=spec.max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device()) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    emb = _pool(out.last_hidden_state, inputs["attention_mask"], spec.pooling)
    if spec.normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.float().cpu().numpy()


def _index_paths(
    model_key: str, lang: str, doc_aug_suffix: str | None = None
) -> tuple[Path, Path]:
    """Embedding paths. Augmented variants go to ``{model_key}/{lang}+{suffix}``."""
    sub = f"{lang}+{doc_aug_suffix}" if doc_aug_suffix else lang
    base = INDEX_DIR / model_key / sub
    base.mkdir(parents=True, exist_ok=True)
    return base / "embeddings.npy", base / "ids.txt"


def encode_corpus(
    model_key: str,
    lang: str,
    *,
    batch_size: int = 64,
    force: bool = False,
    doc_augmentation: Path | None = None,
    doc_aug_suffix: str | None = None,
) -> Path:
    """Encode every document in the language-specific corpus (single passage per doc).

    If ``doc_augmentation`` is given, encode the augmented corpus
    (``original_text + " " + augmentation_contents``) and place the output
    under ``indexes/dense/{model_key}/{lang}+{suffix}/``. ``doc_aug_suffix``
    is required when ``doc_augmentation`` is set.
    """
    if doc_augmentation is not None and not doc_aug_suffix:
        raise ValueError("doc_augmentation requires doc_aug_suffix")
    spec = MODELS[model_key]
    emb_path, ids_path = _index_paths(model_key, lang, doc_aug_suffix)
    if emb_path.exists() and ids_path.exists() and not force:
        logger.info("Dense index already present at %s", emb_path)
        return emb_path

    # Use loader's iter_corpus on the cached HF JSONL.
    corpus_path = data._hf_path(f"corpus-{lang}/corpus.jsonl")
    ids: list[str] = []
    texts: list[str] = []
    import json as _json

    aug_text: dict[str, str] = {}
    if doc_augmentation is not None:
        with open(doc_augmentation, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                aug_text[rec["id"]] = rec.get("contents", "") or ""

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = _json.loads(line)
            doc_id = rec["_id"]
            base_text = rec.get("text", "")
            if doc_augmentation is not None and aug_text.get(doc_id):
                texts.append(f"{base_text} {aug_text[doc_id]}")
            else:
                texts.append(base_text)
            ids.append(doc_id)

    tokenizer, model = _load_model(spec)

    # Probe one sample to determine output dim — for jina-v3 the .encode()
    # output dim can differ from model.config.hidden_size after projection.
    out_dim = _encode_batch(spec, tokenizer, model, [texts[0]], is_query=False).shape[1]
    embeddings = np.empty((len(texts), out_dim), dtype=np.float32)

    desc = f"enc-doc:{model_key}:{lang}"
    if doc_aug_suffix:
        desc += f"+{doc_aug_suffix}"
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[start : start + batch_size]
        embeddings[start : start + len(batch)] = _encode_batch(
            spec, tokenizer, model, batch, is_query=False
        )

    np.save(emb_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        for did in ids:
            f.write(did + "\n")
    return emb_path


def encode_corpus_maxp(
    model_key: str,
    lang: str,
    *,
    batch_size: int = 64,
    force: bool = False,
) -> Path:
    """Encode the corpus with **passage-level chunking** for max-P scoring.

    Each document is BPE-tokenized once; if the token count exceeds
    :func:`_maxp_chunk_budget`, the doc is split into non-overlapping passages
    of that budget. Every passage is encoded independently. The output index
    consists of three files under ``indexes/dense/{model_key}-maxp/{lang}/``:

    - ``embeddings.npy``: shape ``(n_passages, dim)``
    - ``doc_ids.txt``: one doc id per passage (the same doc id repeats once
      per passage of that doc)
    - ``boundaries.npy``: shape ``(n_docs + 1,)`` int32; ``boundaries[i]`` is
      the first passage index of doc ``i``, ``boundaries[-1] = n_passages``.
      Documents are stored in the original corpus order so that the unique
      doc-id list at index ``i`` is recoverable as ``doc_ids[boundaries[i]]``.

    The boundary array lets :func:`run_search_maxp` reduce passage scores to
    per-doc max scores in a single ``np.maximum.reduceat`` call.
    """
    spec = MODELS[model_key]
    base = INDEX_DIR / f"{model_key}-maxp" / lang
    base.mkdir(parents=True, exist_ok=True)
    emb_path = base / "embeddings.npy"
    ids_path = base / "doc_ids.txt"
    bounds_path = base / "boundaries.npy"
    if (
        emb_path.exists()
        and ids_path.exists()
        and bounds_path.exists()
        and not force
    ):
        logger.info("max-P dense index already present at %s", emb_path)
        return emb_path

    corpus_path = data._hf_path(f"corpus-{lang}/corpus.jsonl")
    import json as _json

    tokenizer, model = _load_model(spec)
    chunk_budget = _maxp_chunk_budget(spec)

    # Build the chunk list and per-doc boundary array.
    passage_texts: list[str] = []
    doc_ids: list[str] = []
    boundaries: list[int] = []  # length n_docs; boundaries[i] = first passage idx of doc i
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = _json.loads(line)
            doc_id = rec["_id"]
            text = rec.get("text", "")
            chunks = _split_doc_for_maxp(text, tokenizer, chunk_budget) or [""]
            boundaries.append(len(passage_texts))
            for c in chunks:
                passage_texts.append(c)
                doc_ids.append(doc_id)

    boundaries.append(len(passage_texts))  # sentinel
    n_split_docs = sum(1 for i in range(len(boundaries) - 1) if boundaries[i + 1] - boundaries[i] > 1)
    logger.info(
        "%s/%s max-P: %d docs -> %d passages (%d split, max-tokens-per-passage=%d)",
        model_key,
        lang,
        len(boundaries) - 1,
        len(passage_texts),
        n_split_docs,
        chunk_budget,
    )

    # Probe + encode.
    out_dim = _encode_batch(spec, tokenizer, model, [passage_texts[0]], is_query=False).shape[1]
    embeddings = np.empty((len(passage_texts), out_dim), dtype=np.float32)
    for start in tqdm(
        range(0, len(passage_texts), batch_size),
        desc=f"enc-maxp:{model_key}:{lang}",
    ):
        batch = passage_texts[start : start + batch_size]
        embeddings[start : start + len(batch)] = _encode_batch(
            spec, tokenizer, model, batch, is_query=False
        )

    np.save(emb_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        for did in doc_ids:
            f.write(did + "\n")
    np.save(bounds_path, np.asarray(boundaries, dtype=np.int64))
    return emb_path


def _load_index(
    model_key: str, lang: str, doc_aug_suffix: str | None = None
) -> tuple[np.ndarray, list[str]]:
    emb_path, ids_path = _index_paths(model_key, lang, doc_aug_suffix)
    embeddings = np.load(emb_path).astype(np.float32)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return embeddings, ids


def _rank_inner_product(
    qids: list[str],
    q_embs: np.ndarray,
    embeddings: np.ndarray,
    ids: list[str],
    *,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Rank documents for each query by inner product using the baseline path.

    Fine-tuned retrieval reuses this helper so the score computation and top-k
    tie handling stay aligned with ``run_search``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if embeddings.shape[0] != len(ids):
        raise ValueError(
            f"index row/doc-id mismatch: {embeddings.shape[0]} embeddings vs {len(ids)} ids"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("cannot rank against an empty embedding index")

    out: dict[str, list[tuple[str, float]]] = {}
    scores = q_embs @ embeddings.T  # (n_q, n_d)
    eff_k = min(top_k, scores.shape[1])
    for i, qid in enumerate(qids):
        s = scores[i]
        idx = np.argpartition(-s, eff_k - 1)[:eff_k]
        idx = idx[np.argsort(-s[idx])]
        out[qid] = [(ids[j], float(s[j])) for j in idx]
    return out


def run_search(
    model_key: str,
    lang: str,
    queries: dict[str, str],
    *,
    top_k: int = 1000,
    batch_size: int = 64,
    doc_aug_suffix: str | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Encode the queries and return top-``k`` matches by inner product.

    Long queries are head-truncated by the tokenizer (``truncation=True,
    max_length=spec.max_length``). Use :func:`run_search_maxp` for the max-P
    variant that chunks long queries.

    If ``doc_aug_suffix`` is set, the search loads the augmented embedding
    index at ``indexes/dense/{model_key}/{lang}+{suffix}/``.
    """
    spec = MODELS[model_key]
    embeddings, ids = _load_index(model_key, lang, doc_aug_suffix)
    tokenizer, model = _load_model(spec)

    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    q_embs = np.empty((len(qids), embeddings.shape[1]), dtype=np.float32)
    for start in tqdm(range(0, len(qids), batch_size), desc=f"enc-q:{model_key}:{lang}"):
        batch = q_texts[start : start + batch_size]
        q_embs[start : start + len(batch)] = _encode_batch(
            spec, tokenizer, model, batch, is_query=True
        )

    # Inner product on full matrix; for mE5 with normalized vectors this is cosine.
    # For ~91K docs × 768 dims at fp32, the matmul is small enough to do in one shot.
    return _rank_inner_product(qids, q_embs, embeddings, ids, top_k=top_k)


def _encode_query_chunks(
    spec: ModelSpec,
    tokenizer,
    model,
    queries: dict[str, str],
    chunk_budget: int,
    batch_size: int,
):
    """Split each query into BPE chunks and encode every chunk.

    Returns ``(qids, chunk_embs, boundaries)`` where ``boundaries[i]`` is the
    first chunk index of query ``qids[i]`` and ``boundaries[-1] = total``.
    """
    qids = list(queries.keys())
    all_chunks: list[str] = []
    boundaries: list[int] = []
    for q in qids:
        boundaries.append(len(all_chunks))
        chunks = _split_text_for_maxp(queries[q], tokenizer, chunk_budget) or [""]
        all_chunks.extend(chunks)
    boundaries.append(len(all_chunks))
    if not all_chunks:
        return qids, np.empty((0, 0), dtype=np.float32), np.asarray(boundaries, dtype=np.int64)
    out_dim = _encode_batch(spec, tokenizer, model, [all_chunks[0]], is_query=True).shape[1]
    chunk_embs = np.empty((len(all_chunks), out_dim), dtype=np.float32)
    for start in tqdm(
        range(0, len(all_chunks), batch_size), desc=f"enc-q-maxp:{spec.key}"
    ):
        batch = all_chunks[start : start + batch_size]
        chunk_embs[start : start + len(batch)] = _encode_batch(
            spec, tokenizer, model, batch, is_query=True
        )
    return qids, chunk_embs, np.asarray(boundaries, dtype=np.int64)


def run_search_maxp(
    model_key: str,
    lang: str,
    queries: dict[str, str],
    *,
    top_k: int = 1000,
    batch_size: int = 64,
) -> dict[str, list[tuple[str, float]]]:
    """Doc-side max-P: ``score_d = max_{c in d} sim(truncate(q), c)``.

    The query is encoded normally (with head truncation at ``spec.max_length``).
    Each corpus document was pre-chunked at index time; we score the query
    against every passage and reduce to a per-doc score by taking the maximum
    across that doc's passages. Documents that fit in one chunk yield the
    same score as :func:`run_search`.

    Requires :func:`encode_corpus_maxp` to have been run for ``(model_key, lang)``.
    """
    spec = MODELS[model_key]
    base = INDEX_DIR / f"{model_key}-maxp" / lang
    emb_path = base / "embeddings.npy"
    ids_path = base / "doc_ids.txt"
    bounds_path = base / "boundaries.npy"
    if not (emb_path.exists() and ids_path.exists() and bounds_path.exists()):
        raise FileNotFoundError(
            f"max-P index missing at {base}; run encode_corpus_maxp({model_key!r}, {lang!r}) first."
        )

    chunk_embs = np.load(emb_path).astype(np.float32)
    with open(ids_path, "r", encoding="utf-8") as f:
        chunk_to_doc = [line.strip() for line in f if line.strip()]
    boundaries = np.load(bounds_path)
    n_docs = boundaries.shape[0] - 1
    # The first chunk index of each doc (length n_docs); doc_ids ordered by corpus order.
    unique_doc_ids = [chunk_to_doc[boundaries[i]] for i in range(n_docs)]

    tokenizer, model = _load_model(spec)
    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    q_embs = np.empty((len(qids), chunk_embs.shape[1]), dtype=np.float32)
    for start in tqdm(
        range(0, len(qids), batch_size), desc=f"maxp-q:{model_key}:{lang}"
    ):
        batch = q_texts[start : start + batch_size]
        q_embs[start : start + len(batch)] = _encode_batch(
            spec, tokenizer, model, batch, is_query=True
        )

    # Score every query against every passage; reduce by per-doc max via reduceat.
    scores_passages = q_embs @ chunk_embs.T  # (n_q, n_passages)
    eff_k = min(top_k, n_docs)
    out: dict[str, list[tuple[str, float]]] = {}
    seg_starts = boundaries[:-1]
    for i, qid in enumerate(qids):
        per_doc = np.maximum.reduceat(scores_passages[i], seg_starts)
        idx = np.argpartition(-per_doc, eff_k - 1)[:eff_k]
        idx = idx[np.argsort(-per_doc[idx])]
        out[qid] = [(unique_doc_ids[j], float(per_doc[j])) for j in idx]
    return out


def run_search_maxp_q(
    model_key: str,
    lang: str,
    queries: dict[str, str],
    *,
    top_k: int = 1000,
    batch_size: int = 64,
) -> dict[str, list[tuple[str, float]]]:
    """Query-side max-P: ``score_d = max_{c in q} sim(c, truncate(d))``.

    Documents are encoded with head-truncation (single embedding per doc, the
    same index :func:`encode_corpus` produces). Each query is split into
    non-overlapping BPE chunks; we score every chunk against every doc and
    reduce by taking the max across the query's chunks. Single-chunk queries
    degenerate to :func:`run_search`.
    """
    spec = MODELS[model_key]
    embeddings, ids = _load_index(model_key, lang)  # head-truncated docs
    tokenizer, model = _load_model(spec)
    chunk_budget = _maxp_chunk_budget(spec)

    qids, chunk_embs, q_boundaries = _encode_query_chunks(
        spec, tokenizer, model, queries, chunk_budget, batch_size
    )

    # (n_q_chunks, n_docs) inner products.
    scores = chunk_embs @ embeddings.T
    eff_k = min(top_k, embeddings.shape[0])
    out: dict[str, list[tuple[str, float]]] = {}
    for i, qid in enumerate(qids):
        start, end = int(q_boundaries[i]), int(q_boundaries[i + 1])
        if end - start == 1:
            per_doc = scores[start]
        else:
            per_doc = scores[start:end].max(axis=0)
        idx = np.argpartition(-per_doc, eff_k - 1)[:eff_k]
        idx = idx[np.argsort(-per_doc[idx])]
        out[qid] = [(ids[j], float(per_doc[j])) for j in idx]
    return out


def run_search_maxp_both(
    model_key: str,
    lang: str,
    queries: dict[str, str],
    *,
    top_k: int = 1000,
    batch_size: int = 64,
) -> dict[str, list[tuple[str, float]]]:
    """Both-sides max-P: ``score_d = max_{qc in q, dc in d} sim(qc, dc)``.

    Both query and document are split into non-overlapping BPE chunks. We
    score every (query-chunk, doc-chunk) pair and reduce twice — first to a
    per-doc max along the doc-chunk axis, then to a single per-doc score by
    taking the max across the query-chunk axis. Equivalent to taking the max
    over the full chunk × chunk grid.

    Requires :func:`encode_corpus_maxp` for the doc index.
    """
    spec = MODELS[model_key]
    base = INDEX_DIR / f"{model_key}-maxp" / lang
    emb_path = base / "embeddings.npy"
    ids_path = base / "doc_ids.txt"
    bounds_path = base / "boundaries.npy"
    if not (emb_path.exists() and ids_path.exists() and bounds_path.exists()):
        raise FileNotFoundError(
            f"max-P doc index missing at {base}; run encode_corpus_maxp({model_key!r}, {lang!r}) first."
        )

    d_chunk_embs = np.load(emb_path).astype(np.float32)
    with open(ids_path, "r", encoding="utf-8") as f:
        chunk_to_doc = [line.strip() for line in f if line.strip()]
    d_boundaries = np.load(bounds_path)
    n_docs = d_boundaries.shape[0] - 1
    unique_doc_ids = [chunk_to_doc[d_boundaries[i]] for i in range(n_docs)]

    tokenizer, model = _load_model(spec)
    chunk_budget = _maxp_chunk_budget(spec)
    qids, q_chunk_embs, q_boundaries = _encode_query_chunks(
        spec, tokenizer, model, queries, chunk_budget, batch_size
    )

    scores = q_chunk_embs @ d_chunk_embs.T  # (n_q_chunks, n_d_chunks)
    eff_k = min(top_k, n_docs)
    seg_starts = d_boundaries[:-1]
    out: dict[str, list[tuple[str, float]]] = {}
    for i, qid in enumerate(qids):
        qs, qe = int(q_boundaries[i]), int(q_boundaries[i + 1])
        # Per-doc max along d-chunk axis for each q-chunk: shape (n_q_chunks_for_this_q, n_docs).
        per_doc_per_qc = np.maximum.reduceat(scores[qs:qe], seg_starts, axis=1)
        if qe - qs == 1:
            per_doc = per_doc_per_qc[0]
        else:
            per_doc = per_doc_per_qc.max(axis=0)
        idx = np.argpartition(-per_doc, eff_k - 1)[:eff_k]
        idx = idx[np.argsort(-per_doc[idx])]
        out[qid] = [(unique_doc_ids[j], float(per_doc[j])) for j in idx]
    return out
