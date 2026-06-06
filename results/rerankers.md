# Cross-encoder rerankers on BM25 top-100 (Paper Table 3)

## Setup

- **First-stage**: BM25 top-100 from
  [`results/intermediate/bm25_top100/{task}_{lang}.jsonl`](intermediate/).
  Rerankers re-order this top-100 list; ranks 101-1000 are preserved.
- **Metrics**: nDCG@10, nDCG@100, MAP. (Recall@1000 cannot improve because
  the candidate pool is fixed at 100.)
- **Significance**: paired two-sided t-test vs BM25 within each (task, lang,
  metric), Holm correction across all models in this section.

## Models

| Key | HF ID | Family | max_length | Batch size |
|---|---|---|---|---|
| `bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` | seqcls (`AutoModelForSequenceClassification`) | 8192 | 16 |
| `jina-reranker-v2` | `jinaai/jina-reranker-v2-base-multilingual` | seqcls (trust_remote_code) | 1024 | 16 |
| `qwen3-reranker-4b` | `Qwen/Qwen3-Reranker-4B` | qwen3-lm (yes/no probability) | 8192 | 4 |
| `qwen3-reranker-8b` | `Qwen/Qwen3-Reranker-8B` | qwen3-lm (yes/no probability) | 8192 | 2 |

The `qwen3-lm` family scores each (query, doc) pair by prompting Qwen3 with
a yes/no instruction and reading the conditional probability of the "yes"
token — see [`src/recare_baselines/reranker.py`](../src/recare_baselines/reranker.py)
for the prompt template.

A 0.6B Qwen3 reranker is also implemented (`qwen3-reranker-0.6b`) as a
supplementary lightweight comparison; results in `metrics/`.

## Table 3 (paper, partial: rerankers only) — nDCG@10

| Model | Rat2Rev EU | Rat2Rev JA | Rev2Rev EU | Rev2Rev JA |
|---|---|---|---|---|
| **BM25** | 0.116 | 0.271 | 0.201 | 0.273 |
| BGE-Reranker-v2-M3 | 0.104 | **0.301** | 0.125 | 0.232 |
| Jina Reranker v2 | 0.180† | 0.238 | 0.196 | 0.241 |
| Qwen3-Reranker-4B | 0.191† | 0.295 | 0.192 | 0.265 |
| Qwen3-Reranker-8B | **0.204**† | **0.302** | **0.209** | **0.278** |

(`†` denotes Holm-corrected p<0.05 vs BM25.)

## Findings (paper §4.2)

1. **Rerankers mainly help Rat2Rev EU.** All three larger rerankers
   significantly improve nDCG@10 on Rat2Rev EU; Qwen3-8B is the strongest.
   Rev2Rev gains are smaller and not statistically significant.
2. **BGE-Reranker degrades nDCG@10 in 3 of 4 cells.** Its 8192-token
   context is well-suited to long ReCaRe articles, but its training
   distribution emphasizes short-passage relevance and rev2rev's
   article-to-article matching is out-of-distribution.
3. **Implicit co-revision is hard.** Rev2Rev requires recognizing that two
   articles will be co-revised, an implicit relationship that lexical and
   cross-encoder scoring cannot easily capture.

## Per-cell data

Aggregate: [`baselines_long.json`](baselines_long.json) (includes both
rerankers and long-context retrievers; see model keys with `+bm25` suffix).

Per-cell: `metrics/<reranker>+bm25_<task>_<lang>.json`.
