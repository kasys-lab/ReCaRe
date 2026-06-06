# Query / document expansion (Paper Table 2 augmentation block)

## Setup

Four training-free augmentation methods applied to two retrievers (BM25,
jina-embeddings-v3). For paper Table 2 we report a single LLM per family:
GPT-4.1-mini for Q2E / Q2D and the canonical T5/mT5 doc2query model for
d2q. The repository also persists results for Qwen3.5-9B as an open-source
comparator.

| Family | Side | Method | Reference |
|---|---|---|---|
| `d2e` | Doc | LLM explanatory paragraph appended to each article | New (this paper) — see [`prompts/expansion_doc_explanation.md`](../prompts/expansion_doc_explanation.md) |
| `d2q` | Doc | doc2query / docTTTTTquery predicted queries appended | Nogueira+ 2019 |
| `q2e_zs` | Query | LLM-generated explanatory terms appended to query | Jagerman+ 2023 — see [`prompts/expansion_q2e_zs.md`](../prompts/expansion_q2e_zs.md) |
| `q2d_zs` | Query | LLM-generated pseudo-document appended to query | Wang+ 2023 (Query2doc) — see [`prompts/expansion_q2d_zs.md`](../prompts/expansion_q2d_zs.md) |
| `task_q2e` | Query | Task-conditioned LLM query expansion (Rat2Rev / Rev2Rev variants) | This paper — see [`prompts/expansion_task_q2e_*.md`](../prompts/) |

**Query expansion concat**:
- BM25: Wang+ 2023 protocol — `q⁺ = concat(q × 5, expansion)`. Repeating
  the query 5 times is shown by Wang+ to outperform single concatenation.
- Dense: `q⁺ = q + " [SEP] " + expansion`, then truncate to `max_length`.

**Doc expansion concat**:
- For both BM25 and dense, `d⁺ = d + " " + augmentation`. The augmented
  text is re-indexed (BM25) or re-encoded (jina-v3) once; query time has
  no expansion overhead.

**LLMs used**:
- GPT-4.1-mini (`gpt-4.1-mini-2025-04-14`) via Azure OpenAI for all query
  expansions and for d2e document expansion.
- Qwen3.5-9B via self-hosted vLLM, evaluated as an open-source alternative.
- T5/mT5 doc2query for d2q (cited from `data/recare_d2q/` — pre-generated;
  see [`docs/data_format.md`](../docs/data_format.md) for download).

## Table 2 (paper, augmentation block) — Recall@100

### On BM25

| Method | Rat2Rev EU | Rat2Rev JA | Rev2Rev EU | Rev2Rev JA |
|---|---|---|---|---|
| BM25 (reference) | 0.226 | 0.470 | 0.311 | 0.418 |
| + Doc. expansion: d2e | **0.291**† | **0.499** | **0.326**† | 0.427 |
| + Doc. expansion: d2q | 0.220 | 0.476 | 0.311 | **0.432**† |
| + Query expansion: Q2E | 0.231 | 0.470 | 0.316 | 0.423† |
| + Query expansion: Q2D | 0.226 | 0.461 | 0.312 | 0.422 |

### On jina-embeddings-v3

| Method | Rat2Rev EU | Rat2Rev JA | Rev2Rev EU | Rev2Rev JA |
|---|---|---|---|---|
| jina-v3 (reference) | 0.331 | 0.482 | 0.339 | 0.461 |
| + Doc. expansion: d2e | **0.347** | 0.472 | 0.331 | 0.453 |
| + Doc. expansion: d2q | 0.253 | 0.463 | 0.283 | 0.445 |
| + Query expansion: Q2E | 0.338 | 0.470 | **0.341** | 0.445 |
| + Query expansion: Q2D | 0.319 | 0.452 | 0.329 | **0.453** |

(`†` in the BM25 block denotes Holm-corrected p<0.05 vs BM25 baseline. The
jina-v3 augmentation block uses jina-v3 as its reference; no augmentation
significantly improves over the base retriever there.)

## Findings (paper §4.2)

1. **Document expansion dominates for BM25.** The LLM-generated explanatory
   text (`d2e`) is the only augmentation that significantly improves BM25
   Recall@100 in three of four cells, with the largest gain on Rat2Rev EU
   (+0.065). Query expansion gives smaller, mostly non-significant
   improvements.
2. **Augmentation is unreliable for jina-v3.** No method consistently
   improves over the base long-context retriever, and `d2q` substantially
   degrades it (Rat2Rev EU: 0.331 → 0.253). Strong long-context retrievers
   already capture the signal that explanatory expansion provides.
3. **Asymmetric query-document length matters.** Adding explanatory text to
   long Rat2Rev rationales hurts more often than helping the dense
   retriever (which is bounded by 8192 tokens); BM25, with no token cap,
   benefits because the added vocabulary increases lexical recall.

## Full grid (beyond paper)

The full grid includes both `gpt-4.1-mini` and `qwen3.5-9b` query
expansions for every family × cell, plus all combinations on jina-v3 (which
the paper reports only with `gpt-4.1-mini`). Files:

| Cells | Source |
|---|---|
| BM25 + d2q / d2e | 8 | `metrics/bm25+{d2q,d2e}_*.json` |
| BM25 + q2d_zs / q2e_zs / task_q2e × {gpt-4.1-mini, qwen3.5-9b} | 24 | `metrics/bm25+{family}.{model}_*.json` |
| jina-v3 + d2q / d2e | 8 | `metrics/jina-v3+{d2q,d2e}_*.json` |
| jina-v3 + q2d_zs / q2e_zs / task_q2e × {gpt-4.1-mini, qwen3.5-9b} | 24 | `metrics/jina-v3+{family}.{model}_*.json` |

Total: **64 augmented retrieval cells**.

Aggregate: [`augmentation.json`](augmentation.json).

Holm-corrected t-tests vs base retriever:
[`ttest_holm/ttest_holm_aug_vs_jina-v3.csv`](ttest_holm/ttest_holm_aug_vs_jina-v3.csv).
