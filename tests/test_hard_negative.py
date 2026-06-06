"""Tests for dense hard-negative mining used by domain adaptation."""

from __future__ import annotations

import json
import logging
import random
from types import SimpleNamespace

from recare_baselines import hard_negative as hn
from recare_baselines.runfile import read_run, write_run


def test_hard_negative_excludes_positive_and_rev2rev_self_match():
    qrels = {"q1": {"positive": 1}}
    dense_top100 = {
        "q1": [
            ("q1", 0.99),
            ("positive", 0.95),
            ("hard-negative", 0.80),
        ]
    }

    result = hn.mine_training_examples(qrels, dense_top100, task="rev2rev", seed=7)

    assert result.n_skipped_queries == 0
    assert [example.to_json() for example in result.examples] == [
        {
            "qid": "q1",
            "positive_doc_id": "positive",
            "hard_negative_doc_id": "hard-negative",
        }
    ]


def test_empty_hard_negative_candidates_are_skipped_and_logged(caplog):
    qrels = {
        "q1": {"p1": 1},
        "q2": {"p2": 1},
    }
    dense_top100 = {
        "q1": [("p1", 1.0), ("n1", 0.8)],
        "q2": [("p2", 1.0)],
    }

    with caplog.at_level(logging.INFO, logger=hn.__name__):
        result = hn.mine_training_examples(qrels, dense_top100, task="rat2rev", seed=13)

    assert result.n_skipped_queries == 1
    assert result.n_missing_top100_queries == 0
    assert len(result.examples) == 1
    assert result.examples[0].hard_negative_doc_id == "n1"
    assert "skipped_queries=1" in caplog.text


def test_seed_reproducibility_and_choice_sequence():
    qrels = {"q1": {"p1": 1, "p2": 1, "p3": 1}}
    candidates = ["n0", "n1", "n2", "n3"]
    dense_top100 = {"q1": [(did, 1.0 / (i + 1)) for i, did in enumerate(candidates)]}

    result_a = hn.mine_training_examples(qrels, dense_top100, task="rat2rev", seed=42)
    result_b = hn.mine_training_examples(qrels, dense_top100, task="rat2rev", seed=42)
    expected_rng = random.Random(42)
    expected = [expected_rng.choice(candidates) for _ in range(3)]

    assert result_a.examples == result_b.examples
    assert [example.hard_negative_doc_id for example in result_a.examples] == expected


def test_split_aware_dense_top100_path_does_not_overwrite_flat_test_path(tmp_path):
    intermediate_root = tmp_path / "results" / "intermediate"
    flat_path = intermediate_root / "dense_top100" / "rat2rev_en_me5-base.jsonl"
    flat_path.parent.mkdir(parents=True)
    flat_path.write_text("sentinel\n", encoding="utf-8")

    train_path = hn.write_dense_top100(
        {"q1": [("d1", 1.0), ("d2", 0.5)]},
        task="rat2rev",
        lang="en",
        model_key="me5-base",
        split="train",
        top_k=1,
        intermediate_root=intermediate_root,
    )

    assert train_path == (
        intermediate_root / "dense_top100" / "train" / "rat2rev_en_me5-base.jsonl"
    )
    assert flat_path.read_text(encoding="utf-8") == "sentinel\n"
    assert read_run(train_path) == {"q1": [("d1", 1.0)]}


def test_build_training_examples_writes_split_aware_jsonl(tmp_path, monkeypatch):
    intermediate_root = tmp_path / "results" / "intermediate"
    top100_path = hn.dense_top100_path(
        "rat2rev",
        "en",
        "me5-base",
        "validation",
        intermediate_root=intermediate_root,
    )
    write_run(top100_path, {"q1": [("p1", 1.0), ("n1", 0.8)]})

    def fake_load(task, lang, split):
        assert (task, lang, split) == ("rat2rev", "en", "validation")
        return SimpleNamespace(qrels={"q1": {"p1": 1}})

    monkeypatch.setattr(hn.data, "load", fake_load)

    result = hn.build_training_examples(
        "me5-base",
        "rat2rev",
        "en",
        split="validation",
        seed=3,
        intermediate_root=intermediate_root,
    )

    assert result.path == (
        intermediate_root
        / "training_data"
        / "dense"
        / "validation"
        / "rat2rev_en_me5-base.jsonl"
    )
    records = [
        json.loads(line)
        for line in result.path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert records == [
        {
            "qid": "q1",
            "positive_doc_id": "p1",
            "hard_negative_doc_id": "n1",
        }
    ]
