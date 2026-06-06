# Prompt: RankGPT Listwise Sliding-Window Reranking

| | |
|---|---|
| **Reranking strategy** | Zero-shot listwise; sliding-window over a first-stage top-K run |
| **Used for** | Table 3 RankGPT row (3 Azure OpenAI models × 4 cells) |
| **Reference** | Sun et al., 2023. *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.* EMNLP 2023. [arXiv:2304.09542](https://arxiv.org/abs/2304.09542) |
| **Implementation** | [`src/recare_baselines/rankgpt.py`](../src/recare_baselines/rankgpt.py) — wraps `rank_llm.SafeOpenai.rerank_batch` |
| **CLI** | `uv run recare-baselines rankgpt <task> <lang> --model <id> --first-stage bm25` |
| **First-stage input** | BM25 top-100 (`results/intermediate/bm25_top100/{task}_{lang}.jsonl`) |
| **Window / stride** | `window_size=20`, `stride=10` → 9 sliding windows per query (covers top-100) |
| **Per-passage cap** | 300 words (rank_llm default; configurable via `--passage-word-cap`) |
| **Models evaluated** | `gpt-4.1-nano-2025-04-14`, `gpt-4.1-mini-2025-04-14`, `gpt-5.4-mini-2026-03-17` |
| **API backend** | Azure OpenAI Chat Completions (see `AZURE_OPENAI_*` env vars) |

## Sliding-window protocol (Sun+ 2023)

Given a first-stage ranked list of `top_k=100` passages:

1. Initialize the list as the BM25 ordering of the top 100.
2. Define windows of size 20 with stride 10, starting from the END of the list:
   `[80, 100), [70, 90), [60, 80), …, [0, 20)`. Each window's head 10 items
   overlap the previous (i.e. later-scored) window's tail, so refined
   orderings propagate from the bottom up.
3. For each window, send the LLM a single multi-turn chat with the prompt
   template below. The model returns a re-ordered list of identifiers; we
   overwrite that slice in place.
4. After 9 windows, return the resulting list, scored by
   `score = top_k - rank_position` so higher rank = higher score.

We do not modify positions `[top_k, |D|)` — RankGPT only reranks the input
top-100; the tail of the run is preserved verbatim.

## Prompt template (Sun+ 2023)

We follow `rank_llm`'s implementation of the original Sun+ 2023 prompt
verbatim. A single chat exchange per window has three turns:

**System (turn 1)**:
```
You are RankGPT, an intelligent assistant that can rank passages based on
their relevancy to the query.
```

**User (turn 1)**:
```
I will provide you with {window_size} passages, each indicated by a
numerical identifier []. Rank the passages based on their relevance to the
search query: {query}.
```

**Assistant (turn 1)** (canned acknowledgement, not generated):
```
Okay, please provide the passages.
```

**User (turns 2 through window_size+1)** — one user/assistant exchange per
passage, where the assistant acknowledgement is canned:

```
[i] {passage_text}
```
```
Received passage [i].
```

**User (final)**:
```
Search Query: {query}.
Rank the {window_size} passages above based on their relevance to the
search query. All the passages should be included and listed using
identifiers, in descending order of relevance. The output format should be
[] > [] > ... e.g., [1] > [2] > ... Only respond with the ranking results,
do not say any word or explain.
```

The expected response is a single line of the form `[3] > [11] > [4] > …`
listing all `window_size` identifiers (1-indexed within the window). We
parse this with `rank_llm`'s `safe_extract_results` and apply it to the
slice.

`{passage_text}` is truncated to 300 words (default; configurable). Long
articles in ReCaRe (especially Rat2Rev EU rationales and full revised
articles) are head-truncated to fit; we have observed that the tail of long
articles is rarely the determinative content for relevance.

## Cost ledger

Per-call cost depends on the model rate and the article text density. The
paper reports a total Azure spend of ~$156 USD for the 12-cell grid at the
rates that applied during writing. Run

```bash
uv run recare-baselines rankgpt-cost --model gpt-4.1-mini-2025-04-14
```

to estimate cost for a fresh grid with current rates.

Per-cell ledgers are written to
`results/metrics/rankgpt-<model>+bm25_<task>_<lang>.api_ledger.json` and
contain `n_calls`, `input_tokens`, `output_tokens`, and `cost_usd`.

## Reproducibility notes

- `temperature=0.0` is hard-coded in the wrapped `rank_llm.SafeOpenai` call;
  outputs are mostly deterministic but the OpenAI / Azure backends may show
  ≤1% rank disagreement across re-runs.
- `gpt-5.4-mini` requires a separate Azure deployment under
  `AZURE_OPENAI_API_VERSION=2026-03-17-preview`. Override at the env level
  per model if your deployment uses a different version.
- We use the model's chat endpoint with a 9-step sliding window; first-stage
  BM25 top-100 is reused verbatim from
  [`results/intermediate/bm25_top100/`](../results/intermediate/) so RankGPT
  numbers can be reproduced independently of BM25 indexing.
