"""Tests for fine-tuned dense retrieval artifact handling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recare_baselines import dense as dense_mod
from recare_baselines import domain_adaptation
from recare_baselines import finetuned_dense


def test_finetuned_alias_paths_do_not_overlap_baseline_paths(tmp_path):
    alias = "me5-base-ft-rat2rev-en"
    index_root = tmp_path / "indexes"
    results_root = tmp_path / "results"

    paths = finetuned_dense.index_paths(
        alias,
        "en",
        index_root=index_root,
        create=False,
    )
    artifacts = domain_adaptation.artifact_paths(
        alias,
        "rat2rev",
        "en",
        results_root=results_root,
    )

    assert paths.embeddings == (
        index_root / "dense_finetuned" / alias / "en" / "embeddings.npy"
    )
    assert paths.ids == index_root / "dense_finetuned" / alias / "en" / "ids.txt"
    assert artifacts.runfile == (
        results_root / "intermediate" / f"{alias}_runs" / "rat2rev_en.jsonl"
    )
    assert artifacts.top100 == (
        results_root
        / "intermediate"
        / "dense_top100"
        / f"rat2rev_en_{alias}.jsonl"
    )
    assert artifacts.metrics == results_root / "metrics" / f"{alias}_rat2rev_en.json"

    baseline_index = index_root / "dense" / "me5-base" / "en" / "embeddings.npy"
    baseline_top100 = (
        results_root / "intermediate" / "dense_top100" / "rat2rev_en_me5-base.jsonl"
    )
    baseline_metrics = results_root / "metrics" / "me5-base_rat2rev_en.json"
    assert paths.embeddings != baseline_index
    assert artifacts.top100 != baseline_top100
    assert artifacts.metrics != baseline_metrics


def test_alias_rejects_path_separators():
    with pytest.raises(ValueError):
        finetuned_dense.validate_alias("../bad")


def test_resolve_alias_prefers_checkpoint_metadata(tmp_path):
    checkpoint_dir = tmp_path / "dense_finetune" / "me5-base" / "rat2rev_en" / "best"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "output_alias": "custom-ft-alias",
                "model": "me5-base",
                "task": "rat2rev",
                "lang": "en",
                "tuning_method": "full",
            }
        ),
        encoding="utf-8",
    )

    assert finetuned_dense.resolve_alias(checkpoint_dir) == "custom-ft-alias"


def test_dense_inner_product_ranking_helper_is_reused_for_search():
    qids = ["q1", "q2"]
    q_embs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.8],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    ids = ["doc-x", "doc-y", "doc-z"]

    runs = dense_mod._rank_inner_product(qids, q_embs, embeddings, ids, top_k=2)

    assert [doc_id for doc_id, _score in runs["q1"]] == ["doc-x", "doc-z"]
    assert [doc_id for doc_id, _score in runs["q2"]] == ["doc-y", "doc-z"]
    assert runs["q1"][0][1] == pytest.approx(1.0)
    assert runs["q2"][0][1] == pytest.approx(0.8)


def test_aggregate_domain_adaptation_writes_json_only(tmp_path):
    results_root = tmp_path / "results"
    metrics_dir = results_root / "metrics"
    finetune_root = results_root / "dense_finetune"
    metrics_dir.mkdir(parents=True)
    train_dir = finetune_root / "me5-base" / "rat2rev_en"
    train_dir.mkdir(parents=True)

    alias = "me5-base-ft-rat2rev-en"
    (train_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model": "me5-base",
                "task": "rat2rev",
                "lang": "en",
                "output_alias": alias,
                "tuning_method": "full",
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "me5-base_rat2rev_en.json").write_text(
        json.dumps(
            {
                "model": "me5-base",
                "task": "rat2rev",
                "lang": "en",
                "split": "test",
                "metrics": {
                    "recall_100": 0.4,
                    "ndcg_10": 0.1,
                    "map": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / f"{alias}_rat2rev_en.json").write_text(
        json.dumps(
            {
                "model": alias,
                "task": "rat2rev",
                "lang": "en",
                "split": "test",
                "metrics": {
                    "recall_100": 0.6,
                    "ndcg_10": 0.2,
                    "map": 0.08,
                },
            }
        ),
        encoding="utf-8",
    )

    out_json = results_root / "domain_adaptation.json"
    out_md = results_root / "domain_adaptation.md"
    result = domain_adaptation.aggregate_domain_adaptation(
        out_json=out_json,
        metrics_dir=metrics_dir,
        finetune_root=finetune_root,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    records = payload["records"]
    assert result.n_records == 1
    assert len(records) == 1
    assert records[0]["alias"] == alias
    assert records[0]["base_model"] == "me5-base"
    assert records[0]["train_task"] == "rat2rev"
    assert records[0]["eval_task"] == "rat2rev"
    assert records[0]["before"]["model"] == "me5-base"
    assert records[0]["before"]["metrics"]["recall_100"] == 0.4
    assert records[0]["after"]["model"] == alias
    assert records[0]["after"]["metrics"]["recall_100"] == 0.6
    assert records[0]["delta"]["recall_100"] == pytest.approx(0.2)
    assert "baselines" not in payload
    assert "metrics_path" not in records[0]
    assert "train_config_path" not in records[0]
    assert result.json_path == out_json
    assert not out_md.exists()


def test_aggregate_uses_lora_train_config_for_legacy_full_alias(tmp_path):
    results_root = tmp_path / "results"
    metrics_dir = results_root / "metrics"
    finetune_root = results_root / "dense_finetune"
    metrics_dir.mkdir(parents=True)
    train_dir = finetune_root / "bge-m3" / "rat2rev_en"
    train_dir.mkdir(parents=True)

    (train_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model": "bge-m3",
                "task": "rat2rev",
                "lang": "en",
                "output_alias": "bge-m3-lora-ft-rat2rev-en",
                "tuning_method": "lora",
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "bge-m3-ft-rat2rev-en_rat2rev_en.json").write_text(
        json.dumps(
            {
                "model": "bge-m3-ft-rat2rev-en",
                "task": "rat2rev",
                "lang": "en",
                "split": "test",
                "metrics": {
                    "recall_100": 0.6,
                    "ndcg_10": 0.2,
                    "map": 0.08,
                },
            }
        ),
        encoding="utf-8",
    )

    out_json = results_root / "domain_adaptation.json"
    domain_adaptation.aggregate_domain_adaptation(
        out_json=out_json,
        metrics_dir=metrics_dir,
        finetune_root=finetune_root,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["records"][0]["tuning_method"] == "lora"
    assert payload["records"][0]["alias"] == "bge-m3-ft-rat2rev-en"
    assert payload["records"][0]["base_model"] == "bge-m3"
