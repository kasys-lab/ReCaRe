"""Domain-adaptation evaluation artifacts and aggregation.

Retrieval modules such as ``dense.py`` and ``finetuned_dense.py`` return runs.
This module owns the fine-tuned retrieval artifact paths and the aggregate
JSON artifact built from metrics files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import data
from . import finetuned_dense
from . import train_dense

logger = logging.getLogger(__name__)

METRICS_DIR = data.RESULTS_ROOT / "metrics"
DOMAIN_ADAPTATION_JSON = data.RESULTS_ROOT / "domain_adaptation.json"
FINETUNE_ROOT = train_dense.FINETUNE_ROOT

METRIC_KEYS = (
    "recall_10",
    "recall_100",
    "recall_1000",
    "ndcg_10",
    "ndcg_100",
    "ndcg_1000",
    "map",
)


@dataclass(frozen=True)
class FineTunedArtifactPaths:
    runfile: Path
    top100: Path
    metrics: Path


@dataclass(frozen=True)
class DomainAdaptationAggregateResult:
    json_path: Path
    n_records: int


def artifact_paths(
    alias: str,
    task: str,
    lang: str,
    *,
    results_root: Path = data.RESULTS_ROOT,
) -> FineTunedArtifactPaths:
    """Return canonical fine-tuned run/top100/metrics paths."""
    alias = finetuned_dense.validate_alias(alias)
    if task not in data.TASKS:
        raise ValueError(f"task must be one of {data.TASKS}, got {task!r}")
    if lang not in data.LANGS:
        raise ValueError(f"lang must be one of {data.LANGS}, got {lang!r}")
    return FineTunedArtifactPaths(
        runfile=results_root / "intermediate" / f"{alias}_runs" / f"{task}_{lang}.jsonl",
        top100=results_root
        / "intermediate"
        / "dense_top100"
        / f"{task}_{lang}_{alias}.jsonl",
        metrics=results_root / "metrics" / f"{alias}_{task}_{lang}.json",
    )


def infer_alias_parts(alias: str) -> dict[str, str]:
    """Best-effort parser for aliases like ``me5-base-ft-rat2rev-en``."""
    alias = finetuned_dense.validate_alias(alias)
    if "-lora-ft-" in alias:
        base, rest = alias.split("-lora-ft-", 1)
        method = "lora"
    elif "-ft-" in alias:
        base, rest = alias.split("-ft-", 1)
        method = "full"
    else:
        return {}

    pieces = rest.rsplit("-", 1)
    if len(pieces) != 2:
        return {"base_model": base, "tuning_method": method}
    train_task, train_lang = pieces
    return {
        "base_model": base,
        "train_task": train_task,
        "train_lang": train_lang,
        "tuning_method": method,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def _discover_train_configs(
    finetune_root: Path = FINETUNE_ROOT,
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(finetune_root.glob("*/*/train_config.json")):
        try:
            cfg = _read_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("skipping unreadable train config: %s", e)
            continue
        alias = cfg.get("output_alias")
        if isinstance(alias, str) and alias.strip():
            alias = finetuned_dense.validate_alias(alias)
            cfg_with_path = cfg | {"train_config_path": str(path)}
            existing = configs.get(alias)
            if existing is None:
                configs[alias] = cfg_with_path
                continue
            existing_path = Path(str(existing.get("train_config_path", "")))
            existing_model_dir = existing_path.parent.parent.name
            new_model_dir = path.parent.parent.name
            model_key = str(cfg.get("model") or "")
            if existing_model_dir != model_key and new_model_dir == model_key:
                configs[alias] = cfg_with_path
    return configs


def _config_for_training_spec(
    configs: dict[str, dict[str, Any]],
    *,
    base_model: str,
    train_task: str,
    train_lang: str,
) -> dict[str, Any]:
    """Return a unique train config matching parsed alias parts, if one exists."""
    matches = [
        cfg
        for cfg in configs.values()
        if cfg.get("model") == base_model
        and cfg.get("task") == train_task
        and cfg.get("lang") == train_lang
    ]
    if len(matches) == 1:
        return matches[0]
    return {}


def _looks_like_finetuned_alias(value: str) -> bool:
    return "-ft-" in value or "-lora-ft-" in value


def _baseline_metrics(
    metrics_dir: Path,
    *,
    base_model: str,
    task: str,
    lang: str,
) -> dict[str, float] | None:
    path = metrics_dir / f"{base_model}_{task}_{lang}.json"
    if not path.exists():
        return None
    try:
        cell = _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    metrics = cell.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {k: float(metrics[k]) for k in METRIC_KEYS if k in metrics}


def _metric_delta(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, float] | None,
) -> dict[str, float] | None:
    if baseline_metrics is None:
        return None
    delta: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key in metrics and key in baseline_metrics:
            delta[key] = float(metrics[key]) - float(baseline_metrics[key])
    return delta


def _training_progress(cfg: dict[str, Any]) -> dict[str, int | float]:
    """Read actual training progress from the epoch-level metrics JSONL."""
    train_config_path = cfg.get("train_config_path")
    if not isinstance(train_config_path, str) or not train_config_path:
        return {}
    metrics_path = Path(train_config_path).with_name("metrics.jsonl")
    if not metrics_path.exists():
        return {}

    last: dict[str, Any] | None = None
    best: dict[str, Any] | None = None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue
                last = rec
                if rec.get("best"):
                    best = rec
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("skipping unreadable training metrics: %s", e)
        return {}

    if last is None:
        return {}
    out: dict[str, int | float] = {}
    if "epoch" in last:
        out["epochs_trained"] = int(last["epoch"])
    if "step" in last:
        out["global_step"] = int(last["step"])
    if best is not None:
        if "epoch" in best:
            out["best_epoch"] = int(best["epoch"])
        if "val_loss" in best:
            out["best_val_loss"] = float(best["val_loss"])
    return out


def aggregate_domain_adaptation(
    *,
    out_json: Path = DOMAIN_ADAPTATION_JSON,
    metrics_dir: Path = METRICS_DIR,
    finetune_root: Path = FINETUNE_ROOT,
    models: set[str] | None = None,
) -> DomainAdaptationAggregateResult:
    """Aggregate fine-tuned metrics into machine-readable JSON."""
    configs = _discover_train_configs(finetune_root)
    records: list[dict[str, Any]] = []

    for path in sorted(metrics_dir.glob("*.json")):
        if path.name.endswith(".api_ledger.json"):
            continue
        try:
            cell = _read_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("skipping unreadable metrics file: %s", e)
            continue
        if "metrics" not in cell:
            continue

        model = str(cell.get("model", ""))
        task = str(cell.get("task", ""))
        lang = str(cell.get("lang", ""))
        metrics = cell.get("metrics")
        if not isinstance(metrics, dict):
            continue

        if model in configs or _looks_like_finetuned_alias(model):
            if models is not None and model not in models:
                continue
            inferred = infer_alias_parts(model)
            cfg = configs.get(model, {})
            if not cfg:
                cfg = _config_for_training_spec(
                    configs,
                    base_model=str(inferred.get("base_model") or ""),
                    train_task=str(inferred.get("train_task") or ""),
                    train_lang=str(inferred.get("train_lang") or ""),
                )
            base_model = str(cfg.get("model") or inferred.get("base_model") or "")
            train_task = str(cfg.get("task") or inferred.get("train_task") or "")
            train_lang = str(cfg.get("lang") or inferred.get("train_lang") or "")
            tuning_method = str(
                cfg.get("tuning_method") or inferred.get("tuning_method") or ""
            )
            baseline = (
                _baseline_metrics(metrics_dir, base_model=base_model, task=task, lang=lang)
                if base_model
                else None
            )
            after_metrics = {k: float(metrics[k]) for k in METRIC_KEYS if k in metrics}
            training = _training_progress(cfg)
            records.append(
                {
                    "alias": model,
                    "base_model": base_model,
                    "train_task": train_task,
                    "train_lang": train_lang,
                    "eval_task": task,
                    "eval_lang": lang,
                    "tuning_method": tuning_method,
                    "before": {
                        "model": base_model,
                        "metrics": baseline,
                    },
                    "after": {
                        "model": model,
                        "metrics": after_metrics,
                    },
                    "delta": _metric_delta(metrics, baseline),
                    "training": training,
                }
            )

    records.sort(key=lambda r: (r["base_model"], r["train_task"], r["train_lang"], r["alias"]))
    payload = {"records": records}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return DomainAdaptationAggregateResult(
        json_path=out_json,
        n_records=len(records),
    )
