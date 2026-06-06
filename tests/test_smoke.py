"""End-to-end smoke test that exercises the public package surface.

These tests verify the install is healthy without hitting HuggingFace, the
network, or any heavy models. They are intentionally cheap so they can run
in CI on every PR. For a real-model smoke test, see
``scripts/smoke_test.sh``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_top_level_imports_resolve():
    """All public modules can be imported without side effects on disk."""
    from recare_baselines import (  # noqa: F401
        bm25,
        cli,
        data,
        dense,
        domain_adaptation,
        eval as eval_mod,
        expansion,
        finetuned_dense,
        hard_negative,
        rankgpt,
        reranker,
        runfile,
        stats,
        train_dense,
    )


def test_cli_help_runs():
    """``recare-baselines --help`` exits 0 and lists all subcommands."""
    from click.testing import CliRunner

    from recare_baselines.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    expected_subcommands = {
        "run-bm25",
        "run-dense",
        "rerank",
        "rankgpt",
        "expand-queries",
        "encode-dense",
        "train-dense",
    }
    for sub in expected_subcommands:
        assert sub in result.output, f"missing subcommand {sub} in CLI help"


def test_dense_registry_contains_paper_models():
    """All dense models cited in Table 2 are registered, plus me5-small."""
    from recare_baselines import dense

    paper_models = {"mdpr", "mcontriever", "me5-base", "bge-m3", "jina-v3"}
    smoke_models = {"me5-small"}
    for key in paper_models | smoke_models:
        assert key in dense.MODELS, f"dense model {key} not registered"
        spec = dense.MODELS[key]
        assert spec.hf_id, f"empty hf_id for {key}"


def test_reranker_registry_contains_paper_models():
    """All cross-encoder rerankers cited in Table 3 are registered."""
    from recare_baselines import reranker

    paper_models = {
        "bge-reranker-v2-m3",
        "jina-reranker-v2",
        "qwen3-reranker-4b",
        "qwen3-reranker-8b",
    }
    for key in paper_models:
        assert key in reranker.MODELS, f"reranker {key} not registered"
        spec = reranker.MODELS[key]
        assert spec.hf_id, f"empty hf_id for {key}"


def test_runfile_roundtrip(tmp_path: Path):
    """Write a tiny run and read it back, verifying schema and ordering."""
    from recare_baselines.runfile import read_run, write_run

    run = {
        "q0": [("d3", 4.2), ("d1", 3.1), ("d2", 0.5)],
        "q1": [("d0", 9.0)],
    }
    out_path = tmp_path / "run.jsonl"
    write_run(out_path, run)
    loaded = read_run(out_path)
    assert set(loaded.keys()) == {"q0", "q1"}
    assert [doc for doc, _ in loaded["q0"]] == ["d3", "d1", "d2"]
    assert loaded["q0"][0][1] == pytest.approx(4.2)


def test_evaluate_basic_metrics(synthetic_bundle):
    """Evaluate a hand-built ranking; verify Recall and nDCG behave sanely.

    Two positives per query at ranks 1 and 6 of a 10-doc top-K → Recall@10
    must be 1.0 and nDCG@10 must be > 0 (the rank-1 positive dominates DCG).
    """
    from recare_baselines.eval import evaluate

    qrels = synthetic_bundle.qrels
    run = {}
    for qid, rels in qrels.items():
        positives = list(rels.keys())
        non_pos = [f"d{i + 1000}" for i in range(8)]
        run[qid] = [
            (positives[0], 10.0),
            *[(d, 9.0 - i) for i, d in enumerate(non_pos[:4])],
            (positives[1], 5.0),
            *[(d, 4.0 - i) for i, d in enumerate(non_pos[4:7])],
        ]

    metrics = evaluate(run, qrels)
    assert metrics["recall_10"] == pytest.approx(1.0)
    assert metrics["ndcg_10"] > 0.5  # positive in top-1 dominates DCG
