# Data formats

The ReCaRe dataset itself is hosted on HuggingFace Datasets at
[`kasys/ReCaRe`](https://huggingface.co/datasets/kasys/ReCaRe) (DOI
[10.57967/hf/8642](https://doi.org/10.57967/hf/8642)). This document
describes the on-disk formats used by the retrieval code in this
repository.

## ReCaRe dataset (HF)

Loaded via [`src/recare_baselines/data.py`](../src/recare_baselines/data.py),
which downloads the JSONL shards lazily via `huggingface_hub.hf_hub_download`
and caches them under `~/.cache/huggingface/`. Override the cache location
with `HF_HOME`.

| File on HF | Schema | Purpose |
|---|---|---|
| `corpus-{lang}/corpus.jsonl` | `{"_id": "doc_id", "text": "article body"}` | Full article-level corpus (91,361 EU / 90,170 JA) |
| `queries-{task}-{lang}/queries.jsonl` | `{"_id": "query_id", "text": "..."}` | Per-task query set |
| `qrels-{task}-{lang}/{train,validation,test}.jsonl` | `{"query-id": "qid", "corpus-id": "doc_id", "score": 1}` | Relevance judgments |

Split sizes (test):
- Rat2Rev: 113 EU / 121 JA
- Rev2Rev: 503 EU / 535 JA

## Runfile format

`results/intermediate/<key>_runs/<task>_<lang>.jsonl` and
`results/intermediate/<source>_top100/<task>_<lang>.jsonl` use a one-line-per-query JSONL:

```json
{"qid": "q123", "ranked": [["d456", 12.345], ["d789", 11.222], ...]}
```

`ranked` is sorted by score descending. Doc IDs are strings; scores are
floats. The `runfile` module ([`src/recare_baselines/runfile.py`](../src/recare_baselines/runfile.py))
provides `read_run(path) → dict[str, list[tuple[str, float]]]` and
`write_run(run, path)`.

## Per-cell metrics JSON

`results/metrics/<model_key>_<task>_<lang>.json`:

```json
{
  "model": "bm25",
  "task": "rat2rev",
  "lang": "en",
  "n_queries": 113,
  "metrics": {
    "recall_10": ..., "recall_100": ..., "recall_1000": ...,
    "ndcg_10": ...,   "ndcg_100": ...,   "ndcg_1000": ...,
    "map": ...
  },
  "per_query": {
    "q123": {"R@10": ..., "nDCG@10": ..., "AP": ...}
  }
}
```

`per_query` uses `ir_measures` string keys (`R@10`, `nDCG@10`, `AP`, ...).
The paired t-tests in `src/recare_baselines/stats.py` consume `per_query`.

## RankGPT API ledger JSON

`results/metrics/rankgpt-<model>+<first_stage>_<task>_<lang>.api_ledger.json`:

```json
{
  "model": "gpt-4.1-mini-2025-04-14",
  "task": "rat2rev",
  "lang": "en",
  "first_stage": "bm25",
  "n_calls": 1017,
  "input_tokens": 8932412,
  "output_tokens": 84321,
  "cost_usd": 14.89,
  "per_query_calls": {"q123": 9, "q456": 9, ...},
  "config": {
    "window_size": 20,
    "stride": 10,
    "top_k": 100,
    "passage_word_cap": 300,
    "context_size": 32768,
    "azure_api_version": "2025-04-14-preview"
  }
}
```

## Domain adaptation aggregate JSON

`results/domain_adaptation.json` has a top-level `records` list with one
entry per (base_model, train_task, train_lang, eval_task, eval_lang)
combination:

```json
{
  "records": [
    {
      "alias": "bge-m3-lora-ft-rat2rev-en",
      "base_model": "bge-m3",
      "train_task": "rat2rev", "train_lang": "en",
      "eval_task": "rat2rev",  "eval_lang": "en",
      "tuning_method": "lora",
      "before": {"model": "bge-m3", "metrics": {...}},
      "after":  {"model": "bge-m3-lora-ft-rat2rev-en", "metrics": {...}},
      "delta": {"recall_100": +0.065, ...},
      "training": {"epochs_trained": 9, "best_epoch": 6, "best_val_loss": 3.78, "global_step": 1161}
    },
    ...
  ]
}
```

## Holm-corrected t-test CSV

`results/ttest_holm/ttest_holm_all.csv` columns:

| Column | Meaning |
|---|---|
| `model` | candidate model key |
| `task`, `lang` | cell identifier |
| `metric` | one of `recall_10/100/1000`, `ndcg_10/100/1000`, `map` |
| `baseline` | "bm25" |
| `family_size` | k, the number of methods in the Holm family for this (task, lang, metric) |
| `t_stat` | paired-t statistic |
| `p_raw` | uncorrected two-sided p-value |
| `p_holm` | Holm-corrected p-value |
| `significant` | `True` if `p_holm < 0.05` |
| `delta` | `mean(per-query candidate metric) - mean(per-query baseline metric)` |

## LLM expansion JSONL

`data/expansion/<family>/<model>_<task>_<lang>_test.jsonl`:

```json
{"qid": "q123", "expansion": "..."}
```

One line per query (the test-split subset). For `task_q2e` on Rev2Rev, the
prompt also receives the article-before/article-after pair from a side
metadata file loaded via `expansion_mod.load_rev2rev_article_pairs`; this
is encoded in the prompt-generation step and not stored separately.

## Pre-generated full-corpus expansions (HF sidecar)

The d2q (T5 docTTTTTquery) and d2e (LLM explanation) full-corpus
expansions are not committed to this repo because of their size
(~305 MB total). They are released as a separate HuggingFace dataset at
[`kasys/ReCaRe-expansions`](https://huggingface.co/datasets/kasys/ReCaRe-expansions)
(public, tag `v0.1.0`). Fetch them with the bundled CLI, which places the
files at the exact paths the augmentation pipeline expects:

```bash
uv run recare-baselines fetch-expansions          # all (d2e + d2q, en + ja)
# or selectively: --method d2e --lang en
```

(`scripts/run_expansion.sh` Phase 2 calls this automatically.) Equivalently,
download with the HF CLI:

```bash
huggingface-cli download kasys/ReCaRe-expansions \
    --repo-type dataset \
    --local-dir data/
```

This populates `data/recare_d2q/recare_<lang>_d2q_documents.jsonl` and
`data/recare_d2e/recare_<lang>_d2e_documents.jsonl`, each with schema:

```json
{"id": "doc_id", "contents": "expansion text"}
```

Once these files exist, `scripts/run_expansion.sh PHASES=2 ...` will build
the augmented BM25 indexes and jina-v3 embeddings.

## Domain adaptation checkpoints (HF Models)

The 20 fine-tuned checkpoints (5 base models × 2 tasks × 2 languages) are too
large to ship in the source tree (~15 GB total). They are released as a single
HuggingFace model repo
[`kasys/ReCaRe-domain-adaptation`](https://huggingface.co/kasys/ReCaRe-domain-adaptation),
one subfolder per checkpoint named `<model>_<task>_<lang>`. Each subfolder
ships:

- LoRA weights (`adapter_model.safetensors`, for `bge-m3`) or the full
  fine-tuned model (`model.safetensors`, for `jina-v3`, `mdpr`, `mcontriever`,
  `me5-base`) plus its tokenizer
- `checkpoint_meta.json` (training hyperparameters + `output_alias`)

Fetch them with the bundled CLI, which places each checkpoint at the path the
evaluation expects (`results/dense_finetune/<model>/<task>_<lang>/best`):

```bash
uv run recare-baselines fetch-finetuned                 # all 20
# or selectively: --model bge-m3 --task rat2rev --lang en
```

`scripts/run_domain_adaptation.sh` Phase 3 calls this automatically for any
cell whose checkpoint is missing locally, so `PHASES=3 bash
scripts/run_domain_adaptation.sh` reproduces the Table 4 evaluation without
re-training. Equivalently, download one checkpoint directly:

```python
from huggingface_hub import snapshot_download
ckpt = snapshot_download("kasys/ReCaRe-domain-adaptation",
                         allow_patterns="bge-m3_rat2rev_en/*")
```
