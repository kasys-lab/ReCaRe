# RankGPT zero-shot LLM reranking on BM25 top-100 (Paper Table 3)

## Setup

- **Protocol**: Sun et al., 2023 sliding-window listwise reranking via
  `rank_llm.SafeOpenai.rerank_batch`. Window size 20, stride 10, 9 windows
  per query covering the full top-100.
- **Backend**: Azure OpenAI Chat Completions (`temperature=0.0`).
- **Per-passage cap**: 300 words (rank_llm default).
- **Prompt**: see [`prompts/rankgpt_listwise.md`](../prompts/rankgpt_listwise.md).
- **First-stage**: BM25 top-100 (same input as cross-encoder rerankers).

## Models evaluated

| CLI `--model` | Azure deployment | Released |
|---|---|---|
| `gpt-4.1-nano-2025-04-14` | gpt-4.1-nano | 2025-04-14 |
| `gpt-4.1-mini-2025-04-14` | gpt-4.1-mini | 2025-04-14 |
| `gpt-5.4-mini-2026-03-17` | gpt-5.4-mini | 2026-03-17 |

## Table 3 (paper, RankGPT rows) — nDCG@10

| Model | Rat2Rev EU | Rat2Rev JA | Rev2Rev EU | Rev2Rev JA |
|---|---|---|---|---|
| BM25 (reference) | 0.116 | 0.271 | 0.201 | 0.273 |
| RankGPT gpt-4.1-nano | 0.130 | 0.295 | 0.185 | 0.266 |
| RankGPT gpt-4.1-mini | 0.205† | **0.373**† | 0.198 | **0.295** |
| RankGPT gpt-5.4-mini | **0.225**† | 0.356† | **0.218** | 0.283 |

(`†` denotes Holm-corrected p<0.05 vs BM25.)

## Findings (paper §4.2)

1. **RankGPT is the strongest Rat2Rev reranker.** gpt-5.4-mini achieves the
   best nDCG@10 in Rat2Rev EU and gpt-4.1-mini in Rat2Rev JA. Both
   significantly outperform BM25.
2. **Rev2Rev remains hard for LLM rerankers too.** RankGPT improvements on
   Rev2Rev are smaller in magnitude and not statistically significant under
   Holm correction. This mirrors the cross-encoder result: identifying
   co-revised articles is the dominant difficulty in Rev2Rev.
3. **gpt-4.1-nano underperforms.** Despite being the cheapest model, it
   shows no significant improvement over BM25 in any cell.

## Cost ledger

Total Azure spend for the 12-cell grid at the rates that applied during
paper writing was approximately **$156 USD**, broken down as:

| Model | Approx. spend (USD) |
|---|---|
| gpt-4.1-nano | $5.10 |
| gpt-4.1-mini | $77.00 |
| gpt-5.4-mini | $74.00 |
| **Total** | **~$156** |

Per-cell ledgers (`metrics/rankgpt-<model>+bm25_<task>_<lang>.api_ledger.json`)
record the exact `n_calls`, `input_tokens`, `output_tokens`, and `cost_usd`
that were charged. Run

```bash
uv run recare-baselines rankgpt-cost --model gpt-4.1-mini-2025-04-14
```

for an estimate at current Azure rates before re-running.

## Reproducibility caveats

- LLM outputs are not perfectly deterministic even at `temperature=0.0`;
  rerank orderings may differ by ≤1% across re-runs.
- `gpt-5.4-mini` requires a separate Azure deployment with the appropriate
  API version (e.g. `2026-03-17-preview`); other models can share a single
  deployment.
- We do not modify ranks 101-1000 of the input top-K; nDCG@100 / nDCG@1000
  are therefore largely inherited from BM25.

## Per-cell data

Per-cell metrics: `metrics/rankgpt-<model>+bm25_<task>_<lang>.json`.

Per-cell cost ledgers: `metrics/rankgpt-<model>+bm25_<task>_<lang>.api_ledger.json`.
