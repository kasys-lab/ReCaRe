"""Significance testing for baselines vs BM25.

For each ``(task, lang, metric)`` cell we compare every non-BM25 method to
BM25 using a **paired two-sided t-test** on per-query metric values, then
apply **Holm correction** across the ``k`` methods (k = number of non-BM25
methods in that cell). This replaces the prior Tukey HSD setup, which over-
corrected by treating all 105 model pairs as a single family per cell.

Output columns
--------------
``task, lang, metric, baseline, method, n_queries, meandiff, t_stat,
p_raw, p_adj_holm, reject`` — one row per (cell, method).

Conventions
-----------
- Per-query values come from ``results/metrics/*.json`` (the ``per_query``
  field, populated by :func:`recare_baselines.eval.evaluate_per_query`).
- Queries are aligned by qid; queries missing in either side are dropped.
- ``alpha = 0.05``; ``meandiff = mean(method - baseline)``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "results" / "metrics"
TEST_DIR = REPO_ROOT / "results" / "ttest_holm"

DEFAULT_BASELINE = "bm25"
_METRIC_KEYS = ("R@10", "R@100", "R@1000", "nDCG@10", "nDCG@100", "nDCG@1000", "AP")

# Metrics on which Table 4 reports adapted-vs-base significance.
DA_METRIC_KEYS = ("R@100", "nDCG@10")


def _load_cell_files() -> dict[tuple[str, str], list[Path]]:
    cells: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for f in METRICS_DIR.glob("*.json"):
        cell = json.loads(f.read_text())
        if "per_query" not in cell:
            continue
        cells[(cell["task"], cell["lang"])].append(f)
    return cells


def _per_query_long(files: Iterable[Path], metric_key: str) -> pd.DataFrame:
    """Long-form ``(qid, model, value)`` for one cell + metric."""
    rows = []
    for f in files:
        cell = json.loads(f.read_text())
        for qid, q_metrics in cell["per_query"].items():
            if metric_key in q_metrics:
                rows.append(
                    {"qid": qid, "model": cell["model"], "value": q_metrics[metric_key]}
                )
    return pd.DataFrame(rows)


def paired_ttest_vs_baseline(
    task: str,
    lang: str,
    metric_key: str,
    *,
    baseline: str = DEFAULT_BASELINE,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired t-test of every non-baseline method vs ``baseline`` for one cell.

    Holm-corrects across the ``k`` non-baseline methods present in the cell.
    Returns one row per method (or an empty DataFrame if the cell has no
    metrics, or the baseline is missing).
    """
    cells = _load_cell_files()
    files = cells.get((task, lang))
    if not files:
        return pd.DataFrame()
    df = _per_query_long(files, metric_key)
    if df.empty:
        return pd.DataFrame()

    # Wide form: rows=qid, cols=model, values=metric value
    wide = df.pivot_table(index="qid", columns="model", values="value", aggfunc="first")
    if baseline not in wide.columns:
        return pd.DataFrame()

    methods = [m for m in wide.columns if m != baseline]
    rows: list[dict] = []
    pvals: list[float] = []
    for m in methods:
        sub = wide[[baseline, m]].dropna()
        if len(sub) < 2:
            continue
        diff = sub[m] - sub[baseline]
        if diff.std() == 0:
            # Identical scoring on every aligned query → t is undefined.
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat, p_val = scipy_stats.ttest_rel(sub[m], sub[baseline])
            t_stat, p_val = float(t_stat), float(p_val)
        rows.append(
            {
                "task": task,
                "lang": lang,
                "metric": metric_key,
                "baseline": baseline,
                "method": m,
                "n_queries": int(len(sub)),
                "meandiff": float(diff.mean()),
                "t_stat": t_stat,
                "p_raw": p_val,
            }
        )
        pvals.append(p_val)

    if not rows:
        return pd.DataFrame()

    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    for r, padj, rej in zip(rows, p_adj, reject):
        r["p_adj_holm"] = float(padj)
        r["reject"] = bool(rej)

    return pd.DataFrame(rows)


def run_all(
    metric_keys: Iterable[str] = _METRIC_KEYS,
    *,
    baseline: str = DEFAULT_BASELINE,
    alpha: float = 0.05,
) -> pd.DataFrame:
    cells = _load_cell_files()
    out: list[pd.DataFrame] = []
    for (task, lang), _ in cells.items():
        for mk in metric_keys:
            t = paired_ttest_vs_baseline(
                task, lang, mk, baseline=baseline, alpha=alpha
            )
            if not t.empty:
                out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def write_results(
    metric_keys: Iterable[str] = _METRIC_KEYS,
    *,
    baseline: str = DEFAULT_BASELINE,
    alpha: float = 0.05,
) -> Path:
    table = run_all(metric_keys, baseline=baseline, alpha=alpha)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEST_DIR / "ttest_holm_all.csv"
    table.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Domain adaptation (Table 4): adapted model vs its own base model
# ---------------------------------------------------------------------------


def _per_query_map(metrics_dir: Path, model: str, task: str, lang: str) -> dict | None:
    """Return the ``{qid: {metric: value}}`` per-query map for one metrics cell."""
    f = metrics_dir / f"{model}_{task}_{lang}.json"
    if not f.exists():
        return None
    cell = json.loads(f.read_text())
    return cell.get("per_query")


def domain_adaptation_holm(
    records: Iterable[dict],
    *,
    metrics_dir: Path = METRICS_DIR,
    metric_keys: Iterable[str] = DA_METRIC_KEYS,
    alpha: float = 0.05,
) -> dict[str, dict[str, dict]]:
    """Per-query paired two-sided t-test of each adapted model vs **its own base
    model**, Holm-corrected within each ``(eval_task, eval_lang, metric)`` family.

    Unlike :func:`paired_ttest_vs_baseline` (which fixes the baseline to BM25 and
    pools every method in a cell), this compares ``alias`` to ``base_model`` only
    and forms one Holm family per (task, lang, metric) over the ``k`` adapted
    models — the test reported for Table 4's ``†``.

    ``records`` are domain-adaptation records, each with ``alias``,
    ``base_model``, ``eval_task``, ``eval_lang`` (see
    :func:`recare_baselines.domain_adaptation.aggregate_domain_adaptation`).

    Returns ``{alias: {metric_key: {n_queries, meandiff, t_stat, p_raw,
    p_adj_holm, reject}}}``.
    """
    raw: dict[tuple[str, str], dict] = {}
    groups: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)

    for r in records:
        alias = r.get("alias")
        base = r.get("base_model")
        task = r.get("eval_task")
        lang = r.get("eval_lang")
        if not (alias and base and task and lang):
            continue
        adapted_pq = _per_query_map(metrics_dir, alias, task, lang)
        base_pq = _per_query_map(metrics_dir, base, task, lang)
        if not adapted_pq or not base_pq:
            continue
        for mk in metric_keys:
            qids = [
                q
                for q in adapted_pq
                if q in base_pq and mk in adapted_pq[q] and mk in base_pq[q]
            ]
            if len(qids) < 2:
                continue
            after = [float(adapted_pq[q][mk]) for q in qids]
            before = [float(base_pq[q][mk]) for q in qids]
            diff = [a - b for a, b in zip(after, before)]
            if len(set(diff)) <= 1:
                # No variance in the paired differences → t undefined; treat as
                # non-significant (matches paired_ttest_vs_baseline).
                t_stat, p_val = 0.0, 1.0
            else:
                t_stat, p_val = scipy_stats.ttest_rel(after, before)
                t_stat, p_val = float(t_stat), float(p_val)
            key = (alias, mk)
            raw[key] = {
                "n_queries": int(len(qids)),
                "meandiff": float(sum(diff) / len(diff)),
                "t_stat": t_stat,
                "p_raw": float(p_val),
            }
            groups[(task, lang, mk)].append(key)

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for _family, members in groups.items():
        pvals = [raw[k]["p_raw"] for k in members]
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
        for k, rej, padj in zip(members, reject, p_adj):
            info = dict(raw[k])
            info["p_adj_holm"] = float(padj)
            info["reject"] = bool(rej)
            out[k[0]][k[1]] = info
    return dict(out)
