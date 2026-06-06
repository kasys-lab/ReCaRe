# Results

This directory contains all numerical results from the §4.2 retrieval
experiments in the paper.

## Index

| File | Paper reference | What it covers |
|---|---|---|
| [`baselines.md`](baselines.md) | Table 2 (rows 1-6) | BM25 + 5 dense retrievers, Recall@100 + full metric set |
| [`rerankers.md`](rerankers.md) | Table 3 (rows 1-4) | 4 cross-encoder rerankers on BM25 top-100 |
| [`rankgpt.md`](rankgpt.md) | Table 3 (rows 5-7) | 3 zero-shot listwise LLM rerankers + Azure cost ledger |
| [`augmentation.md`](augmentation.md) | Table 2 (augmentation block) | Q2E / Q2D / d2q / d2e on BM25 and jina-v3 |
| [`domain_adaptation.md`](domain_adaptation.md) | Table 4 | Dense retriever fine-tuning, before→after with Holm-corrected t-tests |
| [`ttest_holm/`](ttest_holm/) | All significance markers (†) | Per-cell Holm-corrected paired t-tests vs BM25 |

## Data layout

- **Aggregates**: `baselines_short.json`, `baselines_long.json`,
  `augmentation.json`, `domain_adaptation.json` — one JSON object per model
  with a `{task}-{lang}` map of metrics.
- **Per-cell metrics**: `metrics/<model_key>_<task>_<lang>.json` (188 files).
  Schema:
  ```json
  {
    "model": "...",
    "task": "rat2rev",
    "lang": "en",
    "n_queries": 113,
    "metrics": {
      "recall_10": ..., "recall_100": ..., "recall_1000": ...,
      "ndcg_10": ...,   "ndcg_100": ...,   "ndcg_1000": ...,
      "map": ...
    },
    "per_query": {"q123": {"R@10": ..., "nDCG@10": ..., "AP": ...}, ...}
  }
  ```
- **RankGPT cost ledgers**: `metrics/rankgpt-<model>+bm25_<task>_<lang>.api_ledger.json`
  with `n_calls`, `input_tokens`, `output_tokens`, `cost_usd`.

## Paper-table → result-file map

| Paper | Cells | Source files |
|---|---|---|
| Table 2 BM25 row | 4 | `metrics/bm25_*.json` |
| Table 2 short-context dense (mDPR/mContriever/mE5) | 12 | `metrics/{mdpr,mcontriever,me5-base}_*.json` |
| Table 2 long-context dense (BGE-M3/Jina) | 8 | `metrics/{bge-m3,jina-v3}_*.json` |
| Table 2 augmentation × {BM25, Jina} × {d2e, d2q, Q2E, Q2D} | 32 (paper) / 64 (full grid) | `metrics/{bm25,jina-v3}+*_*.json` |
| Table 3 cross-encoder rerankers | 16 (paper) / 20 (full grid) | `metrics/*+bm25_*.json` |
| Table 3 RankGPT | 12 | `metrics/rankgpt-*+bm25_*.json` |
| Table 4 domain adaptation | 40 cells × 2 metrics | `domain_adaptation.json` |

## Beyond-the-paper experiments

The repository also retains supplementary experiments not reported in the
paper for space reasons but useful as ablations or comparison points:

- `qwen3-reranker-0.6b` (smallest Qwen3 reranker): see `metrics/qwen3-reranker-0.6b+bm25_*.json`
- Full 7-metric set (R@{10,100,1000}, nDCG@{10,100,1000}, MAP) for every cell — see per-cell JSON
- Both `gpt-4.1-mini` AND `qwen3.5-9b` query expansions for every family — see `metrics/{bm25,jina-v3}+{q2d_zs,q2e_zs,task_q2e}.{model}_*.json`
- Holm-corrected pairwise t-tests of every method vs BM25 in
  `ttest_holm/ttest_holm_all.csv` (paper only marks BM25 baselines vs new methods)
- Holm-corrected pairwise t-tests of every augmentation vs jina-v3 baseline
  in `ttest_holm/ttest_holm_aug_vs_jina-v3.csv` (paper marks vs jina-v3 in
  the augmentation block of Table 2)
