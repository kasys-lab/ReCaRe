"""Tests for ``evaluate-ranked`` (ranked-list JSONL + qrels -> metrics)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from recare_baselines import ranked_eval as re_mod
from recare_baselines.cli import cli
from recare_baselines.eval import evaluate, evaluate_per_query


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_matches_scored_pipeline(synthetic_bundle):
    """Same ordering as a scored run -> exactly the numbers ``cli._persist`` gets."""
    scored = {
        f"q{i}": [(f"d{(i + 7 * k) % 50}", 100.0 - k) for k in range(50)] for i in range(5)
    }
    ranked = {q: [d for d, _ in docs] for q, docs in scored.items()}
    result = re_mod.evaluate_ranked(ranked, synthetic_bundle.qrels)
    assert result.metrics == evaluate(scored, synthetic_bundle.qrels)
    assert result.per_query == evaluate_per_query(scored, synthetic_bundle.qrels)
    assert result.n_queries == 5 and not result.missing_qids


def test_list_order_is_evaluated_order():
    qrels = {"q": {"b": 1}}
    # "b" second -> nDCG@10 < 1; first -> 1. Pseudo-scores must not reorder.
    assert re_mod.evaluate_ranked({"q": ["a", "b"]}, qrels).metrics["ndcg_10"] < 1.0
    assert re_mod.evaluate_ranked({"q": ["b", "a"]}, qrels).metrics["ndcg_10"] == 1.0


def test_self_match_filter():
    qrels = {"q": {"p": 1}}
    ranked = {"q": ["q", "p"]}
    unfiltered = re_mod.evaluate_ranked(ranked, qrels)
    filtered = re_mod.evaluate_ranked(ranked, qrels, filter_self=True)
    assert filtered.n_self_matches_dropped == 1
    assert filtered.metrics["map"] == 1.0
    assert unfiltered.metrics["map"] == 0.5


def test_missing_query_policies():
    qrels = {"a": {"x": 1}, "b": {"y": 1}}
    ranked = {"a": ["x"], "c": ["z"]}  # b missing, c not in qrels
    zero = re_mod.evaluate_ranked(ranked, qrels, missing="zero")
    skip = re_mod.evaluate_ranked(ranked, qrels, missing="skip")
    assert zero.missing_qids == ["b"] and zero.extra_qids == ["c"]
    assert zero.n_queries == 2 and zero.metrics["map"] == 0.5
    assert zero.per_query["b"]["AP"] == 0.0
    assert skip.n_queries == 1 and skip.metrics["map"] == 1.0
    assert "b" not in skip.per_query


def test_empty_list_after_filter_counts_as_missing():
    qrels = {"a": {"x": 1}, "b": {"y": 1}}
    ranked = {"a": ["x"], "b": ["b"]}  # b only retrieves itself
    res = re_mod.evaluate_ranked(ranked, qrels, filter_self=True)
    assert res.missing_qids == ["b"]
    assert res.metrics["map"] == 0.5


def test_duplicates_keep_first_occurrence():
    qrels = {"q": {"b": 1}}
    res = re_mod.evaluate_ranked({"q": ["a", "a", "b", "a"]}, qrels)
    assert res.n_duplicate_docs_dropped == 2
    assert res.metrics["map"] == 0.5  # b at rank 2, not 3


def test_truncation_to_max_docs():
    qrels = {"q": {"p": 1}}
    ranked = {"q": [f"n{i}" for i in range(1000)] + ["p"]}  # p at rank 1001
    res = re_mod.evaluate_ranked(ranked, qrels)
    assert res.n_truncated_queries == 1
    assert res.metrics["map"] == 0.0 and res.metrics["recall_1000"] == 0.0
    unlimited = re_mod.evaluate_ranked(ranked, qrels, max_docs=None)
    assert unlimited.n_truncated_queries == 0 and unlimited.metrics["map"] > 0.0


def test_truncation_counts_unique_docs_and_precedes_self_filter():
    qrels = {"q": {"p": 1}}
    # 2 duplicates do not use up the budget; p is the 3rd unique doc.
    res = re_mod.evaluate_ranked({"q": ["a", "a", "b", "b", "p"]}, qrels, max_docs=3)
    assert res.n_truncated_queries == 0 and res.metrics["recall_10"] == 1.0
    # Self match occupies a slot before filtering (paper: top-1000 then filter).
    res = re_mod.evaluate_ranked({"q": ["q", "a", "p"]}, qrels, max_docs=2, filter_self=True)
    assert res.n_truncated_queries == 1 and res.metrics["recall_10"] == 0.0


def test_read_ranked_gzip(tmp_path):
    import gzip

    p = tmp_path / "run.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"qid": "q", "ranked": ["a", "b"]}) + "\n")
    assert re_mod.read_ranked(p) == {"q": ["a", "b"]}


def test_read_ranked_errors(tmp_path):
    p = _write_jsonl(tmp_path / "dup.jsonl", [{"qid": "q", "ranked": []}] * 2)
    with pytest.raises(ValueError, match="duplicate qid"):
        re_mod.read_ranked(p)

    p = _write_jsonl(tmp_path / "bad.jsonl", [{"qid": "q", "ranked": [1]}])
    with pytest.raises(ValueError, match="document-ID strings"):
        re_mod.read_ranked(p)


def test_read_qrels_jsonl_and_trec(tmp_path, synthetic_bundle):
    assert re_mod.read_qrels(synthetic_bundle.qrels_path) == synthetic_bundle.qrels
    trec = tmp_path / "qrels.txt"
    trec.write_text("q1 0 d1 1\nq1 0 d2 0\nq2 Q0 d3 2\n")
    assert re_mod.read_qrels(trec) == {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 2}}


def test_cli_end_to_end(tmp_path, synthetic_bundle):
    run = _write_jsonl(
        tmp_path / "run.jsonl",
        [{"qid": q, "ranked": list(rels) + ["d49"]} for q, rels in synthetic_bundle.qrels.items()],
    )
    out = tmp_path / "metrics.json"
    res = CliRunner().invoke(
        cli,
        ["evaluate-ranked", str(run), str(synthetic_bundle.qrels_path), "--output", str(out)],
    )
    assert res.exit_code == 0, res.output
    cell = json.loads(out.read_text())
    assert cell["metrics"]["recall_10"] == 1.0
    assert cell["n_queries"] == 5 and set(cell["per_query"]) == set(synthetic_bundle.qrels)


def test_cli_rev2rev_defaults_to_self_match_filter(tmp_path):
    run = _write_jsonl(tmp_path / "run.jsonl", [{"qid": "q", "ranked": ["q", "p"]}])
    qrels = _write_jsonl(tmp_path / "qrels.jsonl", [{"query-id": "q", "corpus-id": "p", "score": 1}])
    runner = CliRunner()

    res = runner.invoke(cli, ["evaluate-ranked", str(run), str(qrels), "--task", "rev2rev"])
    assert res.exit_code == 0, res.output
    assert '"self_match_filtered": true' in res.output

    res = runner.invoke(cli, ["evaluate-ranked", str(run), str(qrels)])
    assert res.exit_code == 0, res.output
    assert "--filter-self-match" in res.output  # warning emitted
