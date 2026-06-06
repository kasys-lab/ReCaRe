"""Supervised contrastive fine-tuning for dense retrievers.

Phase 2 of the domain-adaptation pipeline trains the short-context dense
encoders with qrels positives plus one mined hard negative per positive pair.
The trainable forward path mirrors :mod:`recare_baselines.dense`, but keeps
gradients attached and returns torch tensors for the InfoNCE loss.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import data
from . import dense as dense_mod
from . import hard_negative

logger = logging.getLogger(__name__)

FINETUNE_ROOT = data.RESULTS_ROOT / "dense_finetune"


@dataclass(frozen=True)
class DenseTextTrainingExample:
    qid: str
    positive_doc_id: str
    hard_negative_doc_id: str
    qrel_positive_doc_ids: tuple[str, ...]
    query: str
    positive: str
    hard_negative: str


@dataclass(frozen=True)
class DenseTrainConfig:
    model: str
    task: str
    lang: str
    base_hf_model_id: str
    tuning_method: str
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    learning_rate: float
    warmup_ratio: float
    epochs: int
    patience: int
    seed: int
    temperature: float
    device: str
    use_amp: bool
    precision: str
    requested_fp16: bool | None
    gradient_checkpointing: bool
    train_dense_top100_path: str
    validation_dense_top100_path: str
    train_training_data_path: str
    validation_training_data_path: str
    train_examples: int
    validation_examples: int
    output_alias: str


@dataclass(frozen=True)
class DenseTrainResult:
    output_dir: Path
    best_dir: Path
    last_dir: Path
    train_config_path: Path
    train_steps_path: Path
    metrics_path: Path
    best_val_loss: float
    epochs_trained: int
    global_step: int
    stopped_early: bool


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float = 0.0
    best_val_loss: float = math.inf
    bad_epochs: int = 0

    def update(self, val_loss: float) -> tuple[bool, bool]:
        """Record a validation loss and return ``(improved, should_stop)``."""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


class DenseTrainingDataset(Dataset):
    def __init__(self, examples: list[DenseTextTrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DenseTextTrainingExample:
        return self.examples[idx]


def collate_dense_examples(batch: list[DenseTextTrainingExample]) -> dict[str, list[str]]:
    return {
        "qid": [ex.qid for ex in batch],
        "positive_doc_id": [ex.positive_doc_id for ex in batch],
        "hard_negative_doc_id": [ex.hard_negative_doc_id for ex in batch],
        "qrel_positive_doc_ids": [ex.qrel_positive_doc_ids for ex in batch],
        "query": [ex.query for ex in batch],
        "positive": [ex.positive for ex in batch],
        "hard_negative": [ex.hard_negative for ex in batch],
    }


class TrainableDenseEncoder(torch.nn.Module):
    """A trainable dense encoder using the existing ``dense.ModelSpec`` contract."""

    def __init__(
        self,
        spec: dense_mod.ModelSpec,
        *,
        max_length: int | None = None,
        tuning_method: str = "full",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if tuning_method not in {"full", "lora"}:
            raise ValueError(f"unsupported tuning_method: {tuning_method!r}")

        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        self.max_length = max_length or spec.max_length
        self.tuning_method = tuning_method
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            spec.hf_id,
            trust_remote_code=spec.trust_remote_code,
            **trainable_model_load_kwargs(spec, tuning_method=tuning_method),
        )
        if tuning_method == "lora" and not gradient_checkpointing:
            _disable_gradient_checkpointing(self.model)
        elif tuning_method == "lora" and not _uses_native_jina_lora(spec):
            if _supports_peft_gradient_checkpointing(self.model):
                _enable_gradient_checkpointing(self.model)
        if tuning_method == "lora" and _uses_native_jina_lora(spec):
            return
        if tuning_method == "lora":
            self.model = _apply_lora(self.model)

    def forward(self, texts: list[str], *, is_query: bool) -> torch.Tensor:
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        task = _jina_task(self.spec, is_query=is_query)
        task_instruction = _jina_task_instruction(self.model, task)
        inputs = self.tokenizer(
            [prefix + task_instruction + text for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = _forward_model(self.model, inputs, task=task)
        emb = _pool_model_output(out, inputs["attention_mask"], self.spec.pooling)
        if self.spec.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def _jina_task(spec: dense_mod.ModelSpec, *, is_query: bool) -> str | None:
    """Return jina-v3's task adapter name for trainable forward, if needed."""
    if spec.pooling != "jina_encode":
        return None
    kwargs = spec.query_kwargs if is_query else spec.passage_kwargs
    task = kwargs.get("task")
    return task if isinstance(task, str) else None


def _uses_native_jina_lora(spec: dense_mod.ModelSpec) -> bool:
    """Whether a model has built-in Jina task LoRA adapters."""
    return spec.pooling == "jina_encode"


def trainable_model_load_kwargs(
    spec: dense_mod.ModelSpec,
    *,
    tuning_method: str = "full",
) -> dict[str, Any]:
    """Extra AutoModel load kwargs needed for differentiable training."""
    kwargs: dict[str, Any] = {}
    if spec.key == "jina-v3":
        kwargs["use_flash_attn"] = True
        if tuning_method == "full":
            kwargs["lora_main_params_trainable"] = True
    return kwargs


def _jina_train_amp_dtype() -> torch.dtype:
    """AMP dtype for Jina FlashAttention training."""
    value = os.getenv("RECARE_JINA_AMP_DTYPE", "bf16").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(
        "RECARE_JINA_AMP_DTYPE must be one of bf16/fp16, "
        f"got {value!r}"
    )


def inference_model_load_kwargs(spec: dense_mod.ModelSpec) -> dict[str, Any]:
    """Extra AutoModel load kwargs needed for fine-tuned inference."""
    if spec.key == "jina-v3":
        return {"use_flash_attn": False}
    return {}


def _jina_task_instruction(model: torch.nn.Module, task: str | None) -> str:
    """Return Jina's task instruction prefix used by its encode() helper."""
    if task is None:
        return ""
    task_instructions = getattr(model, "_task_instructions", None)
    if isinstance(task_instructions, dict):
        instruction = task_instructions.get(task)
        if isinstance(instruction, str):
            return instruction
    return ""


def _forward_model(model: torch.nn.Module, inputs: dict[str, torch.Tensor], *, task: str | None):
    """Forward with jina's task adapter when the model supports it."""
    if task is None:
        return model(**inputs)
    adaptation_map = getattr(model, "_adaptation_map", None)
    if isinstance(adaptation_map, dict) and task in adaptation_map:
        task_id = adaptation_map[task]
        adapter_mask = torch.full(
            (inputs["input_ids"].shape[0],),
            task_id,
            dtype=torch.int32,
            device=inputs["input_ids"].device,
        )
        try:
            return model(**inputs, adapter_mask=adapter_mask)
        except TypeError as e:
            raise RuntimeError(
                f"model forward did not accept adapter_mask for task={task!r}; "
                "this would collapse jina-v3's query/passage adapter distinction"
            ) from e
    raise RuntimeError(
        f"model does not expose a native adapter id for task={task!r}; this would "
        "collapse jina-v3's query/passage adapter distinction"
    )


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable activation checkpointing when supported by the loaded model."""
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def _disable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Disable activation checkpointing for Jina's variable-length FlashAttention path."""
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            try:
                module.gradient_checkpointing = False
            except Exception:  # pragma: no cover - defensive for remote-code properties
                pass


def _supports_peft_gradient_checkpointing(model: torch.nn.Module) -> bool:
    """Whether PEFT can attach input-gradient hooks for checkpointing."""
    if not hasattr(model, "gradient_checkpointing_enable"):
        return False
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if get_input_embeddings is None:
        return False
    try:
        return get_input_embeddings() is not None
    except NotImplementedError:
        return False


def _resolve_gradient_checkpointing(
    spec: dense_mod.ModelSpec,
    *,
    tuning_method: str,
    requested: bool | None,
) -> bool:
    if requested is not None:
        return requested
    return not (tuning_method == "lora" and _uses_native_jina_lora(spec))


def _pool_model_output(out: Any, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool transformer outputs for the trainable path.

    Baseline jina-v3 inference delegates to ``model.encode()``. During training
    we need a differentiable path, so we call the model forward and mean-pool
    its token states while still passing jina's task adapter when available.
    """
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        hidden = out.last_hidden_state
    elif isinstance(out, dict) and out.get("last_hidden_state") is not None:
        hidden = out["last_hidden_state"]
    elif isinstance(out, (tuple, list)) and len(out) > 0:
        hidden = out[0]
    else:
        raise ValueError("model output does not contain token hidden states")

    if pooling == "jina_encode":
        pooling = "mean"
    return dense_mod._pool(hidden, attention_mask, pooling)


def _default_lora_target_modules(model: torch.nn.Module) -> list[str]:
    preferred = {
        "query",
        "key",
        "value",
        "dense",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "fc1",
        "fc2",
    }
    excluded = {"classifier", "score", "lm_head", "pooler"}
    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        leaf = parts[-1]
        if leaf not in preferred:
            continue
        if any(part in excluded for part in parts):
            continue
        found.add(leaf)
    if not found:
        raise ValueError("could not infer LoRA target linear modules for this model")
    return sorted(found)


def _apply_lora(model: torch.nn.Module) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "LoRA fine-tuning requires the optional 'peft' package. "
            "Install project dependencies after updating pyproject.toml."
        ) from e

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=_default_lora_target_modules(model),
    )
    return get_peft_model(model, config)


def contrastive_logits(
    query_embeddings: torch.Tensor,
    positive_doc_embeddings: torch.Tensor,
    hard_negative_embeddings: torch.Tensor,
    *,
    temperature: float,
    false_negative_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build InfoNCE logits with positives first and hard negatives second.

    Given a batch size ``B``, candidates are ``concat(P, N)`` with shape
    ``(2B, dim)``. Labels are therefore ``0..B-1``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    batch_size = query_embeddings.shape[0]
    if positive_doc_embeddings.shape[0] != batch_size:
        raise ValueError("positive_doc_embeddings must have the same batch size")
    if hard_negative_embeddings.shape[0] != batch_size:
        raise ValueError("hard_negative_embeddings must have the same batch size")

    docs = torch.cat([positive_doc_embeddings, hard_negative_embeddings], dim=0)
    logits = query_embeddings @ docs.T / temperature
    labels = torch.arange(batch_size, device=query_embeddings.device)
    if false_negative_mask is not None:
        expected_shape = logits.shape
        if false_negative_mask.shape != expected_shape:
            raise ValueError(
                "false_negative_mask must have shape "
                f"{expected_shape}, got {false_negative_mask.shape}"
            )
        false_negative_mask = false_negative_mask.to(
            device=logits.device, dtype=torch.bool
        ).clone()
        false_negative_mask[torch.arange(batch_size, device=logits.device), labels] = (
            False
        )
        logits = logits.masked_fill(false_negative_mask, torch.finfo(logits.dtype).min)
    return logits, labels


def contrastive_loss(
    query_embeddings: torch.Tensor,
    positive_doc_embeddings: torch.Tensor,
    hard_negative_embeddings: torch.Tensor,
    *,
    temperature: float,
    false_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits, labels = contrastive_logits(
        query_embeddings,
        positive_doc_embeddings,
        hard_negative_embeddings,
        temperature=temperature,
        false_negative_mask=false_negative_mask,
    )
    return F.cross_entropy(logits, labels)


def qrels_false_negative_mask(
    qrel_positive_doc_ids: list[tuple[str, ...]],
    positive_doc_ids: list[str],
    hard_negative_doc_ids: list[str],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mask batch candidates that are known qrels positives for each query."""
    batch_size = len(positive_doc_ids)
    if len(qrel_positive_doc_ids) != batch_size:
        raise ValueError("qrel_positive_doc_ids must match positive_doc_ids length")
    if len(hard_negative_doc_ids) != batch_size:
        raise ValueError("hard_negative_doc_ids must match positive_doc_ids length")

    candidate_doc_ids = positive_doc_ids + hard_negative_doc_ids
    mask = torch.zeros(
        (batch_size, len(candidate_doc_ids)), dtype=torch.bool, device=device
    )
    for row, positives in enumerate(qrel_positive_doc_ids):
        positive_set = set(positives)
        for col, doc_id in enumerate(candidate_doc_ids):
            if doc_id in positive_set:
                mask[row, col] = True
        mask[row, row] = False
    return mask


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_output_dir(model_key: str, task: str, lang: str) -> Path:
    return FINETUNE_ROOT / model_key / f"{task}_{lang}"


def default_output_alias(
    model_key: str, task: str, lang: str, *, tuning_method: str = "full"
) -> str:
    if tuning_method == "lora":
        return f"{model_key}-lora-ft-{task}-{lang}"
    return f"{model_key}-ft-{task}-{lang}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_corpus_subset(corpus_path: Path, doc_ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in data._read_jsonl(corpus_path):
        doc_id = rec["_id"]
        if doc_id in doc_ids:
            out[doc_id] = rec.get("text", "")
    missing = sorted(doc_ids - set(out))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} document ids missing from corpus: {preview}")
    return out


def load_text_training_examples(
    path: Path,
    *,
    task: str,
    lang: str,
    split: str,
) -> list[DenseTextTrainingExample]:
    """Load id triples and attach query / document text for training."""
    id_examples = hard_negative.read_training_examples(path)
    rd = data.load(task, lang, split=split)
    doc_ids = {
        did
        for ex in id_examples
        for did in (ex.positive_doc_id, ex.hard_negative_doc_id)
    }
    doc_texts = _read_corpus_subset(rd.corpus_path, doc_ids)

    examples: list[DenseTextTrainingExample] = []
    missing_queries: list[str] = []
    for ex in id_examples:
        query = rd.queries.get(ex.qid)
        if query is None:
            missing_queries.append(ex.qid)
            continue
        qrel_positive_doc_ids = tuple(hard_negative._positive_doc_ids(rd.qrels[ex.qid]))
        examples.append(
            DenseTextTrainingExample(
                qid=ex.qid,
                positive_doc_id=ex.positive_doc_id,
                hard_negative_doc_id=ex.hard_negative_doc_id,
                qrel_positive_doc_ids=qrel_positive_doc_ids,
                query=query,
                positive=doc_texts[ex.positive_doc_id],
                hard_negative=doc_texts[ex.hard_negative_doc_id],
            )
        )

    if missing_queries:
        preview = ", ".join(sorted(set(missing_queries))[:5])
        raise KeyError(
            f"{len(set(missing_queries))} query ids missing from {split} queries: {preview}"
        )
    return examples


def _ensure_training_data(
    model_key: str,
    task: str,
    lang: str,
    *,
    split: str,
    seed: int,
    auto_build_data: bool,
    force_data: bool,
    top_k: int,
    search_top_k: int,
    build_batch_size: int,
) -> Path:
    examples_path = hard_negative.training_data_path(task, lang, model_key, split)
    if examples_path.exists() and not force_data:
        return examples_path
    if not auto_build_data:
        raise FileNotFoundError(
            f"dense training data not found at {examples_path}; run "
            "`build-dense-training-data` first or omit --no-auto-build-data"
        )

    top100_path = hard_negative.dense_top100_path(task, lang, model_key, split)
    if force_data or not top100_path.exists():
        emb_path, ids_path = dense_mod._index_paths(model_key, lang)
        if not (emb_path.exists() and ids_path.exists()):
            logger.info("encoding dense corpus before top100 mining: %s", emb_path)
            dense_mod.encode_corpus(
                model_key,
                lang,
                batch_size=build_batch_size,
                force=False,
            )
        logger.info("building dense top100 before training: %s", top100_path)
        hard_negative.build_dense_top100(
            model_key,
            task,
            lang,
            split=split,
            top_k=top_k,
            search_top_k=search_top_k,
            batch_size=build_batch_size,
            force=force_data,
        )

    logger.info("building dense training examples before training: %s", examples_path)
    hard_negative.build_training_examples(
        model_key,
        task,
        lang,
        split=split,
        seed=seed,
        out_path=examples_path,
    )
    return examples_path


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining = total_steps - current_step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _encode_batch_loss(
    encoder: TrainableDenseEncoder,
    batch: dict[str, list[str]],
    *,
    temperature: float,
) -> torch.Tensor:
    q_emb = encoder(batch["query"], is_query=True)
    p_emb = encoder(batch["positive"], is_query=False)
    n_emb = encoder(batch["hard_negative"], is_query=False)
    false_negative_mask = qrels_false_negative_mask(
        batch["qrel_positive_doc_ids"],
        batch["positive_doc_id"],
        batch["hard_negative_doc_id"],
        device=q_emb.device,
    )
    return contrastive_loss(
        q_emb,
        p_emb,
        n_emb,
        temperature=temperature,
        false_negative_mask=false_negative_mask,
    )


@torch.no_grad()
def evaluate_loss(
    encoder: TrainableDenseEncoder,
    loader: DataLoader,
    *,
    temperature: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device_type: str,
) -> float:
    encoder.eval()
    loss_sum = 0.0
    n_examples = 0
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
        if device_type == "cuda"
        else nullcontext()
    )
    for batch in loader:
        with amp_ctx:
            loss = _encode_batch_loss(encoder, batch, temperature=temperature)
        batch_size = len(batch["query"])
        loss_sum += float(loss.item()) * batch_size
        n_examples += batch_size
    if n_examples == 0:
        raise ValueError("validation loader is empty")
    return loss_sum / n_examples


def _save_last_state(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    early_stopping: EarlyStoppingState,
) -> None:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": early_stopping.best_val_loss,
        "bad_epochs": early_stopping.bad_epochs,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def train_dense(
    model_key: str,
    task: str,
    lang: str,
    *,
    batch_size: int = 64,
    grad_accum_steps: int = 1,
    lr: float = 1e-5,
    warmup_ratio: float = 0.1,
    epochs: int = 100,
    patience: int = 3,
    seed: int = 13,
    temperature: float = 0.05,
    tuning_method: str = "full",
    max_length: int | None = None,
    output_dir: Path | None = None,
    auto_build_data: bool = True,
    force_data: bool = False,
    top_k: int = 100,
    search_top_k: int = 1000,
    build_batch_size: int = 64,
    log_every: int = 1,
    device: str | None = None,
    fp16: bool | None = None,
    gradient_checkpointing: bool | None = None,
) -> DenseTrainResult:
    """Run full supervised contrastive fine-tuning for one model/task/lang."""
    if model_key not in dense_mod.MODELS:
        raise ValueError(f"unknown model_key: {model_key!r}")
    if task not in data.TASKS:
        raise ValueError(f"task must be one of {data.TASKS}, got {task!r}")
    if lang not in data.LANGS:
        raise ValueError(f"lang must be one of {data.LANGS}, got {lang!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if patience <= 0:
        raise ValueError("patience must be > 0")
    if log_every <= 0:
        raise ValueError("log_every must be > 0")

    spec = dense_mod.MODELS[model_key]
    gradient_checkpointing = _resolve_gradient_checkpointing(
        spec,
        tuning_method=tuning_method,
        requested=gradient_checkpointing,
    )
    output_dir = output_dir or default_output_dir(model_key, task, lang)
    best_dir = output_dir / "best"
    last_dir = output_dir / "last"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = _ensure_training_data(
        model_key,
        task,
        lang,
        split="train",
        seed=seed,
        auto_build_data=auto_build_data,
        force_data=force_data,
        top_k=top_k,
        search_top_k=search_top_k,
        build_batch_size=build_batch_size,
    )
    val_path = _ensure_training_data(
        model_key,
        task,
        lang,
        split="validation",
        seed=seed,
        auto_build_data=auto_build_data,
        force_data=force_data,
        top_k=top_k,
        search_top_k=search_top_k,
        build_batch_size=build_batch_size,
    )

    train_examples = load_text_training_examples(
        train_path, task=task, lang=lang, split="train"
    )
    val_examples = load_text_training_examples(
        val_path, task=task, lang=lang, split="validation"
    )
    if not train_examples:
        raise ValueError(f"no training examples in {train_path}")
    if not val_examples:
        raise ValueError(f"no validation examples in {val_path}")

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (torch_device.type == "cuda") if fp16 is None else bool(fp16)
    amp_dtype = torch.float16
    if torch_device.type != "cuda":
        use_amp = False
    elif _uses_native_jina_lora(spec) and use_amp:
        amp_dtype = _jina_train_amp_dtype()
    precision = (
        "bf16_amp"
        if use_amp and amp_dtype == torch.bfloat16
        else "fp16_amp"
        if use_amp
        else "fp32"
    )

    cfg = DenseTrainConfig(
        model=model_key,
        task=task,
        lang=lang,
        base_hf_model_id=spec.hf_id,
        tuning_method=tuning_method,
        max_length=max_length or spec.max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        effective_batch_size=batch_size * grad_accum_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        epochs=epochs,
        patience=patience,
        seed=seed,
        temperature=temperature,
        device=str(torch_device),
        use_amp=use_amp,
        precision=precision,
        requested_fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        train_dense_top100_path=str(
            hard_negative.dense_top100_path(task, lang, model_key, "train")
        ),
        validation_dense_top100_path=str(
            hard_negative.dense_top100_path(task, lang, model_key, "validation")
        ),
        train_training_data_path=str(train_path),
        validation_training_data_path=str(val_path),
        train_examples=len(train_examples),
        validation_examples=len(val_examples),
        output_alias=default_output_alias(
            model_key, task, lang, tuning_method=tuning_method
        ),
    )
    train_config_path = output_dir / "train_config.json"
    train_config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_steps_path = output_dir / "train_steps.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    train_steps_path.write_text("", encoding="utf-8")
    metrics_path.write_text("", encoding="utf-8")

    _set_seed(seed)
    encoder = TrainableDenseEncoder(
        spec,
        max_length=max_length,
        tuning_method=tuning_method,
        gradient_checkpointing=gradient_checkpointing,
    ).to(torch_device)
    train_dataset = DenseTrainingDataset(train_examples)
    val_dataset = DenseTrainingDataset(val_examples)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_dense_examples,
        pin_memory=torch_device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dense_examples,
        pin_memory=torch_device.type == "cuda",
    )

    trainable_params = [p for p in encoder.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("no trainable parameters found")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_updates = max(1, updates_per_epoch * epochs)
    scheduler = _make_scheduler(
        optimizer, total_steps=total_updates, warmup_ratio=warmup_ratio
    )
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    early_stopping = EarlyStoppingState(patience=patience)

    global_step = 0
    epochs_trained = 0
    stopped_early = False
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
        if torch_device.type == "cuda"
        else nullcontext()
    )

    logger.info(
        "training dense model=%s task=%s lang=%s train=%d validation=%d "
        "batch=%d grad_accum=%d epochs=%d device=%s fp16=%s",
        model_key,
        task,
        lang,
        len(train_examples),
        len(val_examples),
        batch_size,
        grad_accum_steps,
        epochs,
        torch_device,
        use_amp,
    )

    for epoch in range(1, epochs + 1):
        encoder.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_seen = 0
        pending_accum = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            with amp_ctx:
                loss = _encode_batch_loss(encoder, batch, temperature=temperature)
            batch_n = len(batch["query"])
            train_loss_sum += float(loss.item()) * batch_n
            train_seen += batch_n

            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
            pending_accum += 1
            is_update_step = (
                pending_accum == grad_accum_steps or batch_idx == len(train_loader)
            )
            if not is_update_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending_accum = 0
            global_step += 1

            if global_step % log_every == 0:
                append_jsonl(
                    train_steps_path,
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": float(loss.item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    },
                )

        epoch_train_loss = train_loss_sum / max(1, train_seen)
        val_loss = evaluate_loss(
            encoder,
            val_loader,
            temperature=temperature,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            device_type=torch_device.type,
        )
        improved, should_stop = early_stopping.update(val_loss)
        epochs_trained = epoch

        if improved:
            encoder.save_checkpoint(best_dir)
            (best_dir / "checkpoint_meta.json").write_text(
                json.dumps(asdict(cfg) | {"checkpoint": "best"}, indent=2),
                encoding="utf-8",
            )

        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "step": global_step,
                "train_loss": epoch_train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "best": improved,
                "bad_epochs": early_stopping.bad_epochs,
            },
        )
        logger.info(
            "epoch=%d step=%d train_loss=%.6f val_loss=%.6f best=%s bad_epochs=%d",
            epoch,
            global_step,
            epoch_train_loss,
            val_loss,
            improved,
            early_stopping.bad_epochs,
        )
        if should_stop:
            stopped_early = True
            break

    encoder.save_checkpoint(last_dir)
    (last_dir / "checkpoint_meta.json").write_text(
        json.dumps(asdict(cfg) | {"checkpoint": "last"}, indent=2),
        encoding="utf-8",
    )
    _save_last_state(
        last_dir / "trainer_state.pt",
        epoch=epochs_trained,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
    )

    return DenseTrainResult(
        output_dir=output_dir,
        best_dir=best_dir,
        last_dir=last_dir,
        train_config_path=train_config_path,
        train_steps_path=train_steps_path,
        metrics_path=metrics_path,
        best_val_loss=early_stopping.best_val_loss,
        epochs_trained=epochs_trained,
        global_step=global_step,
        stopped_early=stopped_early,
    )


def read_epoch_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_training_curve(metrics_path: Path, out_path: Path | None = None) -> Path:
    """Plot train / validation loss from an epoch-level metrics JSONL file."""
    records = read_epoch_metrics(metrics_path)
    if not records:
        raise ValueError(f"no records found in {metrics_path}")
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config = Path(tempfile.gettempdir()) / "recare_matplotlib_cache"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in records]
    train_loss = [float(r["train_loss"]) for r in records]
    val_loss = [float(r["val_loss"]) for r in records]
    out_path = out_path or metrics_path.with_name("loss_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, marker="o", label="train")
    ax.plot(epochs, val_loss, marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(metrics_path.parent.name)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
