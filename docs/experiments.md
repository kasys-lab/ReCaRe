# Experiment details

A reference manual for the retrieval methods evaluated on ReCaRe. For
paper-table reproduction, see [`reproduction.md`](reproduction.md). For data
formats, see [`data_format.md`](data_format.md).

## Method matrix

| Method | Paper table | Code | Script | Result MD |
|---|---|---|---|---|
| BM25 | Table 2 row 1 | [`src/recare_baselines/bm25.py`](../src/recare_baselines/bm25.py) | [`scripts/run_bm25.sh`](../scripts/run_bm25.sh) | [`results/baselines.md`](../results/baselines.md) |
| Short-context dense (mDPR, mContriever, mE5) | Table 2 | [`src/recare_baselines/dense.py`](../src/recare_baselines/dense.py) | [`scripts/run_short_dense.sh`](../scripts/run_short_dense.sh) | [`results/baselines.md`](../results/baselines.md) |
| Long-context dense (BGE-M3, jina-v3) | Table 2 | [`src/recare_baselines/dense.py`](../src/recare_baselines/dense.py) | [`scripts/run_long_dense.sh`](../scripts/run_long_dense.sh) | [`results/baselines.md`](../results/baselines.md) |
| Cross-encoder rerankers | Table 3 | [`src/recare_baselines/reranker.py`](../src/recare_baselines/reranker.py) | [`scripts/run_rerankers.sh`](../scripts/run_rerankers.sh) | [`results/rerankers.md`](../results/rerankers.md) |
| RankGPT (zero-shot LLM) | Table 3 | [`src/recare_baselines/rankgpt.py`](../src/recare_baselines/rankgpt.py) | [`scripts/run_rankgpt.sh`](../scripts/run_rankgpt.sh) | [`results/rankgpt.md`](../results/rankgpt.md) |
| Query / doc expansion | Table 2 | [`src/recare_baselines/expansion.py`](../src/recare_baselines/expansion.py) | [`scripts/run_expansion.sh`](../scripts/run_expansion.sh) | [`results/augmentation.md`](../results/augmentation.md) |
| Domain adaptation | Table 4 | [`src/recare_baselines/{hard_negative,train_dense,finetuned_dense}.py`](../src/recare_baselines/) | [`scripts/run_domain_adaptation.sh`](../scripts/run_domain_adaptation.sh) | [`results/domain_adaptation.md`](../results/domain_adaptation.md) |

## BM25

- Backend: Pyserini Lucene 9.x (`LuceneSearcher.set_bm25(0.9, 0.4)`).
- Analyzers:
  - English: `EnglishAnalyzer` (Porter stemmer + default Lucene stopwords).
  - Japanese: `JapaneseAnalyzer` (Kuromoji morphological analysis + default
    stopwords).
- Index location: `indexes/lucene/{lang}/` (or `{lang}+{aug-suffix}` for
  augmented).
- Top-K: 1000 by default; the top-100 slice is saved separately to
  `results/intermediate/bm25_top100/` for downstream rerankers.

## Dense retrievers

All dense models are run with `truncate` strategy for paper numbers. The
default similarity is cosine when the model L2-normalizes (`me5-*`,
`bge-m3`, `jina-v3`) and inner product otherwise (`mdpr`, `mcontriever`).

| Model | Params | max_length | Pooling | Trust remote code | Output dim |
|---|---|---|---|---|---|
| `mdpr` | 178M | 512 | CLS | No | 768 |
| `mcontriever` | 178M | 512 | mean | No | 768 |
| `me5-base` | 278M | 512 | mean | No | 768 |
| `me5-small` (smoke) | 118M | 512 | mean | No | 384 |
| `bge-m3` | 568M | 8192 | CLS | No | 1024 |
| `jina-v3` | 572M | 8192 | task-LoRA `.encode()` | Yes | 1024 |

For each model, indices live at `indexes/dense/{key}/{lang}/embeddings.npy`
and `ids.txt`. Augmented variants go to `{key}/{lang}+{aug-suffix}/`.

`me5-small` is registered for smoke tests only; it is NOT in the paper.

### Long-input strategies (supplementary)

Beyond `truncate`, the codebase implements three max-P variants:
`maxp-doc`, `maxp-q`, `maxp-both`. They split queries/documents into
non-overlapping BPE chunks of size `max_length - 8` and aggregate via max
similarity. These are not reported in the paper but are persisted as
`metrics/{model}-maxp-{doc,q,both}_*.json` for ablation studies.

## Cross-encoder rerankers

Two scoring families:
- `seqcls`: `AutoModelForSequenceClassification` with one logit per (q, d)
  pair. Used for `bge-reranker-v2-m3` and `jina-reranker-v2`.
- `qwen3-lm`: Qwen3 causal LM prompted with a yes/no instruction; score =
  log-probability of "yes". Used for `qwen3-reranker-{0.6b,4b,8b}`.

Input pairs are truncated to `max_length` after concatenation; for the
`seqcls` family the tokenizer's special-token convention applies, for the
`qwen3-lm` family the prompt template in `reranker.py` is wrapped around.

Top-100 candidates from BM25 are loaded from
`results/intermediate/bm25_top100/{task}_{lang}.jsonl`; ranks 101-1000 are
preserved from the first stage so Recall@1000 is unchanged.

## RankGPT (zero-shot LLM)

See [`prompts/rankgpt_listwise.md`](../prompts/rankgpt_listwise.md) for the
exact Sun et al. 2023 sliding-window template. Highlights:

- Window size 20, stride 10, 9 windows per query.
- Passage cap 300 words (rank_llm default).
- Temperature 0.0.
- Azure OpenAI Chat Completions; deployment-specific API version per model.

We wrap `rank_llm.SafeOpenai.rerank_batch`, which handles retries and
chunked execution. Per-cell API ledgers are persisted to
`metrics/rankgpt-<model>+bm25_<task>_<lang>.api_ledger.json`.

## Query / document expansion

| Family | Implementation | Prompt | Concat (BM25) | Concat (dense) |
|---|---|---|---|---|
| `d2e` | LLM generates a one-paragraph explanation per article | [`prompts/expansion_doc_explanation.md`](../prompts/expansion_doc_explanation.md) | `d⁺ = d + " " + aug` | same |
| `d2q` | T5/mT5 docTTTTTquery predicted queries per article | (external; pre-generated) | `d⁺ = d + " " + aug` | same |
| `q2e_zs` | LLM generates expansion keywords for the query | [`prompts/expansion_q2e_zs.md`](../prompts/expansion_q2e_zs.md) | Wang+ 2023 `q × 5 + aug` | `q + " [SEP] " + aug` |
| `q2d_zs` | LLM generates a pseudo-document for the query | [`prompts/expansion_q2d_zs.md`](../prompts/expansion_q2d_zs.md) | Wang+ 2023 `q × 5 + aug` | `q + " [SEP] " + aug` |
| `task_q2e` | Task-conditioned LLM query expansion (Rat2Rev / Rev2Rev) | [`prompts/expansion_task_q2e_*.md`](../prompts/) | Wang+ 2023 `q × 5 + aug` | `q + " [SEP] " + aug` |

Generated outputs are persisted to `data/expansion/<family>/<model>_<task>_<lang>_test.jsonl`
with schema:
```json
{"qid": "q123", "expansion": "..."}
```

For doc-side expansion, the augmented corpus is materialized into a fresh
Pyserini JSONL and re-indexed (BM25) or re-encoded (jina-v3). Each
augmented index lives at `indexes/lucene/{lang}+{aug}/` or
`indexes/dense/{model}/{lang}+{aug}/`.

## Domain adaptation

Pipeline:
1. **Hard-negative mining** ([`hard_negative.py`](../src/recare_baselines/hard_negative.py)):
   for each training query, sample one hard negative uniformly from the
   base model's top-100 candidates after excluding all positives. Triplets
   are persisted to `results/intermediate/training_data/dense/{split}/`.
2. **Training** ([`train_dense.py`](../src/recare_baselines/train_dense.py)):
   InfoNCE loss with hard + in-batch negatives. AdamW, lr=1e-5, warmup
   ratio 0.1, temperature 0.05, seed 13. Short-context models use full FT;
   long-context use LoRA via `peft`. Early stopping on validation loss
   with patience 3.
3. **Evaluation** ([`finetuned_dense.py`](../src/recare_baselines/finetuned_dense.py)):
   the adapted corpus is re-encoded, queries are re-encoded with the
   adapted weights, and the same `run_search` path computes metrics.
4. **Aggregation** ([`domain_adaptation.py`](../src/recare_baselines/domain_adaptation.py)):
   collects `{model}-{lora-,}ft-{task}-{lang}` aliases and emits
   `results/domain_adaptation.json` with before/after, delta, and
   per-query t-test inputs.

## Per-cell metrics file format

```json
{
  "model": "bm25",
  "task": "rat2rev",
  "lang": "en",
  "n_queries": 113,
  "metrics": {
    "recall_10": 0.0740, "recall_100": 0.2256, "recall_1000": 0.4063,
    "ndcg_10": 0.1163,   "ndcg_100": 0.1552,   "ndcg_1000": 0.2058,
    "map": 0.0705
  },
  "per_query": {
    "q123": {"R@10": 1.0, "nDCG@10": 0.85, "AP": 0.62, ...},
    "q456": {...}
  }
}
```

The `per_query` block uses `ir_measures` string keys (`R@10`, `nDCG@10`,
`AP`, …). It is consumed by `stats.py` for the paired t-tests.

## Runtime expectations

Approximate wall-clock on a single A100 (rough guide):

| Operation | Time |
|---|---|
| BM25 index build (per lang) | 1-2 min |
| BM25 search (per cell) | 30 sec |
| mDPR / mContriever / mE5 encode (per lang) | 5-15 min |
| BGE-M3 encode (per lang) | 30-60 min |
| jina-v3 encode (per lang) | 30-90 min |
| Cross-encoder rerank one cell, top-100, 113 queries | 5-30 min |
| Qwen3-8B rerank one cell | 30-120 min |
| RankGPT one cell (full Azure round-trip) | 30-90 min, ~$15 |
| Domain adaptation: one (model, task, lang) train | 30 min - 4 h |
