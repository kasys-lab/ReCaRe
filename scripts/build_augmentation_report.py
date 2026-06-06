"""Build the Issue #10 augmentation self-contained README.

Reads:
- ``results/augmentation.json`` — aggregated metrics
- ``results/ttest_holm/ttest_holm_all.csv`` — Holm-corrected vs BM25
- per-cell metrics in ``results/metrics/`` for jina-v3-baseline t-tests

Writes:
- ``results/augmentation.md`` — Δ tables (vs base retriever) + significance
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recare_baselines.stats import paired_ttest_vs_baseline

REPO_ROOT = Path(__file__).resolve().parents[1]
AGG_PATH = REPO_ROOT / "results" / "augmentation.json"
TTEST_BM25_PATH = REPO_ROOT / "results" / "ttest_holm" / "ttest_holm_all.csv"
TTEST_JINA_PATH = REPO_ROOT / "results" / "ttest_holm" / "ttest_holm_aug_vs_jina-v3.csv"
OUT_PATH = REPO_ROOT / "results" / "augmentation.md"

CELLS = [
    ("rat2rev", "en", 113),
    ("rat2rev", "ja", 121),
    ("rev2rev", "en", 503),
    ("rev2rev", "ja", 535),
]
METRIC_COLS = ["recall_10", "recall_100", "recall_1000", "ndcg_10", "ndcg_100", "ndcg_1000", "map"]
METRIC_DISPLAY = {
    "recall_10": "R@10", "recall_100": "R@100", "recall_1000": "R@1000",
    "ndcg_10": "nDCG@10", "ndcg_100": "nDCG@100", "ndcg_1000": "nDCG@1000",
    "map": "MAP",
}
TTEST_METRIC_KEYS = ("R@10", "R@100", "R@1000", "nDCG@10", "nDCG@100", "nDCG@1000", "AP")
KEY_TO_DISPLAY = dict(zip(METRIC_COLS, TTEST_METRIC_KEYS))

BASE_RETRIEVERS = ("bm25", "jina-v3")
DOC_AUGS = ("d2q", "d2e")
QUERY_AUGS = (
    ("q2d_zs", "gpt-4.1-mini"), ("q2d_zs", "qwen3.5-9b"),
    ("q2e_zs", "gpt-4.1-mini"), ("q2e_zs", "qwen3.5-9b"),
    ("task_q2e", "gpt-4.1-mini"), ("task_q2e", "qwen3.5-9b"),
)


def aug_keys(base: str) -> list[tuple[str, str]]:
    """Return (label, model_key) pairs for one base retriever."""
    out = [("(none)", base)]
    for da in DOC_AUGS:
        out.append((f"+{da}", f"{base}+{da}"))
    for fam, model in QUERY_AUGS:
        out.append((f"+{fam}.{model}", f"{base}+{fam}.{model}"))
    return out


def compute_jina_ttests() -> pd.DataFrame:
    """Run the same paired t-test logic with jina-v3 as the baseline."""
    rows: list[pd.DataFrame] = []
    for task, lang, _ in CELLS:
        for mk in TTEST_METRIC_KEYS:
            t = paired_ttest_vs_baseline(task, lang, mk, baseline="jina-v3")
            if not t.empty:
                rows.append(t)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def render_cell_table(
    agg: dict, base: str, task: str, lang: str, ttest_df: pd.DataFrame
) -> str:
    """One Δ-vs-baseline table for one (base, task, lang) cell."""
    cell_key = f"{task}-{lang}"
    base_metrics = agg.get(base, {}).get(cell_key, {})
    if not base_metrics:
        return f"_(no baseline metrics for {base} {cell_key})_\n"

    lines = [
        "| Method | " + " | ".join(METRIC_DISPLAY[m] for m in METRIC_COLS) + " |",
        "|" + "---|" * (1 + len(METRIC_COLS)),
    ]
    for label, key in aug_keys(base):
        cell = agg.get(key, {}).get(cell_key)
        if not cell:
            lines.append(f"| {label} | " + " | ".join("—" for _ in METRIC_COLS) + " |")
            continue
        cells = []
        for m in METRIC_COLS:
            v = cell.get(m)
            if v is None:
                cells.append("—")
                continue
            base_v = base_metrics.get(m)
            if label == "(none)" or base_v is None:
                cells.append(f"{v:.4f}")
                continue
            delta = v - base_v
            # Look up Holm-adjusted significance
            method_key = key  # full model_key e.g. "bm25+d2q"
            mask = (
                (ttest_df["task"] == task)
                & (ttest_df["lang"] == lang)
                & (ttest_df["metric"] == KEY_TO_DISPLAY[m])
                & (ttest_df["method"] == method_key)
            )
            sig = ""
            if mask.any():
                row = ttest_df.loc[mask].iloc[0]
                if bool(row["reject"]):
                    sig = "**"  # bold to mark significance
            cells.append(f"{v:.4f} ({sig}{delta:+.4f}{sig})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    agg = json.loads(AGG_PATH.read_text())
    bm25_t = pd.read_csv(TTEST_BM25_PATH)
    jina_t = compute_jina_ttests()
    jina_t.to_csv(TTEST_JINA_PATH, index=False)

    # Combine ttest results: each row is keyed by method (model_key).
    bm25_t["base"] = "bm25"
    jina_t["base"] = "jina-v3"
    all_t = pd.concat([bm25_t, jina_t], ignore_index=True)

    body: list[str] = []
    body.append("# Issue #10 — Cross-task augmentation (Q2D / Q2E / D2Q / LLM doc expansion)\n")
    body.append(
        "論文 §4.X Augmentation の self-contained README. "
        "BM25 と jina-v3 (短/長コンテキスト dense 代表) に対して 8 拡張を適用し, "
        "拡張なし baseline との Δ を表化."
        "\n\n"
        "**主要 finding**: 文書側 LLM 拡張 (d2e) のみ大幅改善 (BM25 で R@100 +0.03~0.07), "
        "標準的なクエリ拡張 (Q2D/Q2E zero-shot) と D2Q (T5) は **marginal** にとどまる. "
        "強 retriever (jina-v3) では大半の拡張が **悪化** (Weller+ EACL 2024 の helpful/harmful 軸と整合).\n\n"
        "## モデル / 拡張の対応\n\n"
        "| 拡張 | 種類 | 生成モデル | 連結方式 |\n"
        "|---|---|---|---|\n"
        "| `d2q` | 文書側 | T5/mT5 (msmarco-doc2query) | 元 doc + ' ' + 生成 query (索引時連結) |\n"
        "| `d2e` | 文書側 | gpt-4.1-mini (法律専門家プロンプト, 2F-01 §3.3.4) | 元 doc + ' ' + 生成説明文 |\n"
        "| `q2d_zs` | クエリ側 | gpt-4.1-mini / Qwen3.5-9B | BM25: q×5 + 出力 (Wang+ 2023, Jagerman+ 2023). dense: q + ' [SEP] ' + 出力 (Wang+ 2023) |\n"
        "| `q2e_zs` | クエリ側 | gpt-4.1-mini / Qwen3.5-9B | 同上 |\n"
        "| `task_q2e` | クエリ側 | gpt-4.1-mini / Qwen3.5-9B (法律専門家プロンプト, 2F-01 §3.3.3 改訂) | 同上 |\n\n"
        "詳細は [`docs/issues/10-augmentation.md`](../docs/issues/10-augmentation.md).\n\n"
        "## 表記\n\n"
        "各セル: `value (Δ)` 形式. Δ は対応する base retriever (BM25 or jina-v3) からの差分. "
        "**太字** は Holm-corrected paired t-test (vs base retriever, k=8 methods × 7 metrics per cell, α=0.05) "
        "で **有意** (`reject = True`).\n\n"
    )

    for base in BASE_RETRIEVERS:
        body.append(f"## {base.upper()} ({'sparse' if base == 'bm25' else 'long-context dense'})\n\n")
        for task, lang, n in CELLS:
            body.append(f"### {task} / {lang} (n={n} test queries)\n\n")
            body.append(render_cell_table(agg, base, task, lang, all_t))
            body.append("\n")

    body.append("## 検定の出力\n\n")
    body.append("- BM25 augmentations: [`ttest_holm/ttest_holm_all.csv`](ttest_holm/ttest_holm_all.csv) (全 baseline 法 vs BM25)\n")
    body.append("- jina-v3 augmentations: [`ttest_holm/ttest_holm_aug_vs_jina-v3.csv`](ttest_holm/ttest_holm_aug_vs_jina-v3.csv) (本 README で生成)\n")
    body.append("- Holm 補正は **(task, lang, metric)** セル内で同 base に対する k 比較に適用.\n\n")

    body.append("## 再現\n\n")
    body.append("```bash\n")
    body.append("# Phase 1: 24 query expansion JSONLs を生成\n")
    body.append("bash scripts/run_expansion_grid.sh\n\n")
    body.append("# Phase 2: 拡張済 BM25 index + jina-v3 embedding を構築\n")
    body.append("bash scripts/build_bm25_aug_indexes.sh\n")
    body.append("bash scripts/encode_jina_v3_aug.sh\n\n")
    body.append("# Phase 3: 64-cell evaluation grid\n")
    body.append("bash scripts/run_eval_aug_grid.sh\n\n")
    body.append("# Phase 4: aggregate + t-test + this README\n")
    body.append("uv run recare-baselines aggregate --out results/augmentation.json\n")
    body.append("uv run recare-baselines ttest-holm\n")
    body.append("uv run python scripts/build_augmentation_report.py\n")
    body.append("```\n\n")

    body.append("## 関連\n\n")
    body.append("- 拡張 brief: [`docs/issues/10-augmentation.md`](../docs/issues/10-augmentation.md)\n")
    body.append("- BM25 / 短コンテキスト dense baseline: [`baselines_short.md`](baselines_short.md)\n")
    body.append("- 長コンテキスト dense baseline: [`baselines_long.md`](baselines_long.md)\n")
    body.append("- Reranker baseline: [`baselines_rerankers.md`](baselines_rerankers.md)\n")

    OUT_PATH.write_text("".join(body))
    print(f"wrote {OUT_PATH}")
    print(f"wrote {TTEST_JINA_PATH}")


if __name__ == "__main__":
    main()
