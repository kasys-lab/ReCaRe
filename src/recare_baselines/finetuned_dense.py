"""Fine-tuned dense retriever encoding and search.

Phase 3 of the domain-adaptation pipeline evaluates checkpoints created by
``train_dense``. This module mirrors ``dense.py`` where possible: it loads a
checkpoint, encodes the corpus, and delegates ranking to the same inner-product
helper used by baseline dense search.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from . import data
from . import dense as dense_mod
from . import train_dense
from .runfile import Run

logger = logging.getLogger(__name__)

INDEX_DIR = data.INDEX_ROOT / "dense_finetuned"
FINETUNE_ROOT = train_dense.FINETUNE_ROOT


@dataclass(frozen=True)
class FineTunedIndexPaths:
    embeddings: Path
    ids: Path


@dataclass(frozen=True)
class FineTunedEncodeResult:
    alias: str
    checkpoint_dir: Path
    embeddings_path: Path
    ids_path: Path
    n_docs: int
    created: bool


@dataclass(frozen=True)
class FineTunedSearchResult:
    alias: str
    checkpoint_dir: Path
    run: Run


def validate_alias(alias: str) -> str:
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("alias must be a non-empty string")
    alias = alias.strip()
    if alias in {".", ".."} or Path(alias).name != alias or "/" in alias or "\\" in alias:
        raise ValueError(f"alias must be a path-safe name, got {alias!r}")
    return alias


def index_paths(
    alias: str,
    lang: str,
    *,
    index_root: Path = data.INDEX_ROOT,
    create: bool = True,
) -> FineTunedIndexPaths:
    """Return canonical fine-tuned corpus embedding paths for an alias."""
    alias = validate_alias(alias)
    if lang not in data.LANGS:
        raise ValueError(f"lang must be one of {data.LANGS}, got {lang!r}")
    base = index_root / "dense_finetuned" / alias / lang
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return FineTunedIndexPaths(
        embeddings=base / "embeddings.npy",
        ids=base / "ids.txt",
    )


def default_checkpoint_dir(
    model_key: str,
    task: str,
    lang: str,
    *,
    checkpoint: str = "best",
    finetune_root: Path = FINETUNE_ROOT,
) -> Path:
    if checkpoint not in {"best", "last"}:
        raise ValueError("checkpoint must be 'best' or 'last'")
    return finetune_root / model_key / f"{task}_{lang}" / checkpoint


def resolve_checkpoint_dir(
    checkpoint: str | Path,
    *,
    model_key: str | None = None,
    task: str | None = None,
    lang: str | None = None,
    finetune_root: Path = FINETUNE_ROOT,
) -> Path:
    checkpoint_str = str(checkpoint)
    if checkpoint_str in {"best", "last"}:
        if not (model_key and task and lang):
            raise ValueError(
                "model_key, task, and lang are required when checkpoint is 'best' or 'last'"
            )
        return default_checkpoint_dir(
            model_key,
            task,
            lang,
            checkpoint=checkpoint_str,
            finetune_root=finetune_root,
        )
    return Path(checkpoint).expanduser()


def read_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    """Read metadata saved by ``train_dense`` if present."""
    data_objs: list[dict[str, Any]] = []
    for path in (
        checkpoint_dir / "checkpoint_meta.json",
        checkpoint_dir.parent / "train_config.json",
    ):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            data_objs.append(obj)

    metadata: dict[str, Any] = {}
    for obj in reversed(data_objs):
        metadata.update(obj)
    return metadata


def resolve_model_key(checkpoint_dir: Path, model_key: str | None = None) -> str:
    if model_key:
        if model_key not in dense_mod.MODELS:
            raise ValueError(f"unknown model_key: {model_key!r}")
        return model_key
    metadata = read_checkpoint_metadata(checkpoint_dir)
    meta_model = metadata.get("model")
    if isinstance(meta_model, str) and meta_model in dense_mod.MODELS:
        return meta_model
    raise ValueError(
        "model_key is required when the checkpoint metadata does not contain a known model"
    )


def _resolve_tuning_method(
    metadata: dict[str, Any],
    explicit_tuning_method: str | None,
) -> str:
    tuning_method = explicit_tuning_method or str(metadata.get("tuning_method", "full"))
    if tuning_method not in {"full", "lora"}:
        raise ValueError(f"unsupported tuning_method: {tuning_method!r}")
    return tuning_method


def resolve_alias(
    checkpoint_dir: Path,
    *,
    alias: str | None = None,
    model_key: str | None = None,
    task: str | None = None,
    lang: str | None = None,
    tuning_method: str | None = None,
) -> str:
    """Resolve the output alias for a fine-tuned checkpoint."""
    if alias:
        return validate_alias(alias)

    metadata = read_checkpoint_metadata(checkpoint_dir)
    meta_alias = metadata.get("output_alias")
    if isinstance(meta_alias, str) and meta_alias.strip():
        return validate_alias(meta_alias)

    model_key = model_key or metadata.get("model")
    task = task or metadata.get("task")
    lang = lang or metadata.get("lang")
    tuning_method = _resolve_tuning_method(metadata, tuning_method)
    if not (isinstance(model_key, str) and isinstance(task, str) and isinstance(lang, str)):
        raise ValueError(
            "alias could not be inferred; pass --alias or use a checkpoint with metadata"
        )
    return validate_alias(
        train_dense.default_output_alias(
            model_key,
            task,
            lang,
            tuning_method=tuning_method,
        )
    )


def _prime_local_remote_code_cache(checkpoint_dir: Path) -> None:
    """Copy all local remote-code modules into Transformers' dynamic cache.

    For local checkpoints, Transformers keys the dynamic module cache by the
    checkpoint directory basename (``best`` / ``last`` here) and may copy only
    direct imports. Jina's saved modeling code has transitive relative imports,
    so pre-populate the cache with every saved Python module before loading.
    """
    if not checkpoint_dir.is_dir():
        return
    py_files = list(checkpoint_dir.glob("*.py"))
    if not py_files:
        return

    from transformers.dynamic_module_utils import (
        HF_MODULES_CACHE,
        TRANSFORMERS_DYNAMIC_MODULE_NAME,
        _sanitize_module_name,
    )

    submodule = _sanitize_module_name(checkpoint_dir.name)
    cache_dir = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / submodule
    cache_dir.mkdir(parents=True, exist_ok=True)
    init_path = cache_dir / "__init__.py"
    if not init_path.exists():
        init_path.touch()
    for source in py_files:
        shutil.copy2(source, cache_dir / source.name)


class FineTunedDenseEncoder:
    """Inference encoder loaded from a fine-tuned checkpoint."""

    def __init__(
        self,
        spec: dense_mod.ModelSpec,
        checkpoint_dir: Path,
        *,
        max_length: int | None = None,
        tuning_method: str = "full",
        device: str | None = None,
        fp16: bool | None = None,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if tuning_method not in {"full", "lora"}:
            raise ValueError(f"unsupported tuning_method: {tuning_method!r}")

        PeftModel = None
        if tuning_method == "lora" and not train_dense._uses_native_jina_lora(spec):
            try:
                from peft import PeftModel as _PeftModel
            except ImportError as e:  # pragma: no cover - optional Phase 4 path
                raise RuntimeError(
                    "LoRA checkpoint loading requires the optional 'peft' package. "
                    "Install it when enabling Phase 4 LoRA support."
                ) from e
            PeftModel = _PeftModel

        self.spec = spec
        self.max_length = max_length or spec.max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = (self.device.type == "cuda") if fp16 is None else bool(fp16)
        if self.device.type != "cuda":
            self.use_fp16 = False
        if train_dense._uses_native_jina_lora(spec):
            self.use_fp16 = False

        tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else spec.hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=spec.trust_remote_code,
        )
        if tuning_method == "lora" and train_dense._uses_native_jina_lora(spec):
            _prime_local_remote_code_cache(checkpoint_dir)
            self.model = AutoModel.from_pretrained(
                checkpoint_dir,
                trust_remote_code=spec.trust_remote_code,
                **train_dense.inference_model_load_kwargs(spec),
            )
        elif tuning_method == "lora":
            base_model = AutoModel.from_pretrained(
                spec.hf_id,
                trust_remote_code=spec.trust_remote_code,
                **train_dense.inference_model_load_kwargs(spec),
            )
            assert PeftModel is not None
            self.model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        else:
            if spec.trust_remote_code:
                _prime_local_remote_code_cache(checkpoint_dir)
            self.model = AutoModel.from_pretrained(
                checkpoint_dir,
                trust_remote_code=spec.trust_remote_code,
                **train_dense.inference_model_load_kwargs(spec),
            )
        if train_dense._uses_native_jina_lora(spec):
            self.model.float()
        self.model.eval()
        self.model.to(self.device)
        if self.use_fp16 and not train_dense._uses_native_jina_lora(spec):
            self.model = self.model.half()

    def encode_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        task = train_dense._jina_task(self.spec, is_query=is_query)
        task_instruction = train_dense._jina_task_instruction(self.model, task)
        inputs = self.tokenizer(
            [prefix + task_instruction + text for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = train_dense._forward_model(self.model, inputs, task=task)
        emb = train_dense._pool_model_output(
            out,
            inputs["attention_mask"],
            self.spec.pooling,
        )
        if self.spec.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb.float().cpu().numpy()


def _load_encoder(
    checkpoint_dir: Path,
    *,
    model_key: str | None = None,
    max_length: int | None = None,
    tuning_method: str | None = None,
    device: str | None = None,
    fp16: bool | None = None,
) -> tuple[str, str, FineTunedDenseEncoder]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    model_key = resolve_model_key(checkpoint_dir, model_key)
    metadata = read_checkpoint_metadata(checkpoint_dir)
    tuning_method = _resolve_tuning_method(metadata, tuning_method)
    encoder = FineTunedDenseEncoder(
        dense_mod.MODELS[model_key],
        checkpoint_dir,
        max_length=max_length,
        tuning_method=tuning_method,
        device=device,
        fp16=fp16,
    )
    return model_key, tuning_method, encoder


def encode_corpus(
    checkpoint_dir: Path,
    lang: str,
    *,
    model_key: str | None = None,
    task: str | None = None,
    alias: str | None = None,
    batch_size: int = 64,
    force: bool = False,
    max_length: int | None = None,
    tuning_method: str | None = None,
    index_root: Path = data.INDEX_ROOT,
    device: str | None = None,
    fp16: bool | None = None,
) -> FineTunedEncodeResult:
    """Encode the full corpus with a fine-tuned checkpoint."""
    if lang not in data.LANGS:
        raise ValueError(f"lang must be one of {data.LANGS}, got {lang!r}")
    if task is not None and task not in data.TASKS:
        raise ValueError(f"task must be one of {data.TASKS}, got {task!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    checkpoint_dir = Path(checkpoint_dir)
    alias = resolve_alias(
        checkpoint_dir,
        alias=alias,
        model_key=model_key,
        task=task,
        lang=lang,
        tuning_method=tuning_method,
    )
    paths = index_paths(alias, lang, index_root=index_root)
    if paths.embeddings.exists() and paths.ids.exists() and not force:
        with open(paths.ids, "r", encoding="utf-8") as f:
            n_docs = sum(1 for line in f if line.strip())
        logger.info("fine-tuned dense index already present at %s", paths.embeddings)
        return FineTunedEncodeResult(
            alias=alias,
            checkpoint_dir=checkpoint_dir,
            embeddings_path=paths.embeddings,
            ids_path=paths.ids,
            n_docs=n_docs,
            created=False,
        )

    _model_key, _tuning_method, encoder = _load_encoder(
        checkpoint_dir,
        model_key=model_key,
        max_length=max_length,
        tuning_method=tuning_method,
        device=device,
        fp16=fp16,
    )

    corpus_path = data._hf_path(f"corpus-{lang}/corpus.jsonl")
    doc_ids: list[str] = []
    texts: list[str] = []
    for rec in data._read_jsonl(corpus_path):
        doc_ids.append(rec["_id"])
        texts.append(rec.get("text", ""))
    if not texts:
        raise ValueError(f"corpus is empty: {corpus_path}")

    out_dim = encoder.encode_batch([texts[0]], is_query=False).shape[1]
    embeddings = np.empty((len(texts), out_dim), dtype=np.float32)
    for start in tqdm(range(0, len(texts), batch_size), desc=f"enc-ft:{alias}:{lang}"):
        batch = texts[start : start + batch_size]
        embeddings[start : start + len(batch)] = encoder.encode_batch(
            batch,
            is_query=False,
        )

    np.save(paths.embeddings, embeddings)
    with open(paths.ids, "w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            f.write(doc_id + "\n")
    return FineTunedEncodeResult(
        alias=alias,
        checkpoint_dir=checkpoint_dir,
        embeddings_path=paths.embeddings,
        ids_path=paths.ids,
        n_docs=len(doc_ids),
        created=True,
    )


def _load_index(
    alias: str,
    lang: str,
    *,
    index_root: Path = data.INDEX_ROOT,
) -> tuple[np.ndarray, list[str]]:
    paths = index_paths(alias, lang, index_root=index_root, create=False)
    if not (paths.embeddings.exists() and paths.ids.exists()):
        raise FileNotFoundError(
            f"fine-tuned dense index missing at {paths.embeddings.parent}; "
            "run encode-finetuned-dense first"
        )
    embeddings = np.load(paths.embeddings).astype(np.float32)
    with open(paths.ids, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if embeddings.shape[0] != len(ids):
        raise ValueError(
            f"index row/doc-id mismatch for {alias}/{lang}: "
            f"{embeddings.shape[0]} embeddings vs {len(ids)} ids"
        )
    return embeddings, ids


def run_search(
    checkpoint_dir: Path,
    lang: str,
    queries: dict[str, str],
    *,
    model_key: str | None = None,
    task: str | None = None,
    alias: str | None = None,
    top_k: int = 1000,
    batch_size: int = 64,
    max_length: int | None = None,
    tuning_method: str | None = None,
    index_root: Path = data.INDEX_ROOT,
    device: str | None = None,
    fp16: bool | None = None,
) -> Run:
    """Run fine-tuned dense search over a pre-encoded corpus index."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    checkpoint_dir = Path(checkpoint_dir)
    alias = resolve_alias(
        checkpoint_dir,
        alias=alias,
        model_key=model_key,
        task=task,
        lang=lang,
        tuning_method=tuning_method,
    )
    embeddings, ids = _load_index(alias, lang, index_root=index_root)
    _model_key, _tuning_method, encoder = _load_encoder(
        checkpoint_dir,
        model_key=model_key,
        max_length=max_length,
        tuning_method=tuning_method,
        device=device,
        fp16=fp16,
    )

    qids = list(queries.keys())
    q_texts = [queries[qid] for qid in qids]
    q_embs = np.empty((len(qids), embeddings.shape[1]), dtype=np.float32)
    for start in tqdm(range(0, len(qids), batch_size), desc=f"enc-q-ft:{alias}:{lang}"):
        batch = q_texts[start : start + batch_size]
        q_embs[start : start + len(batch)] = encoder.encode_batch(
            batch,
            is_query=True,
        )

    return dense_mod._rank_inner_product(qids, q_embs, embeddings, ids, top_k=top_k)
