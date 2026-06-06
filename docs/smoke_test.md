# Smoke test (CPU-only, no API keys)

The smoke test verifies that the install is healthy and the BM25 / dense
retrieval pipelines end-to-end work on your machine, using a lightweight
model (`multilingual-e5-small`, 118M params) on a deliberately small subset
of queries.

This is **not** a paper-faithful evaluation — Recall@100 and nDCG@10 from
the smoke test will differ from Tables 2-3 because they use 5 queries
instead of 113-535 and a smoke-tier model instead of the paper's
baselines.

## What it runs

| Step | Models | Cells | Queries per cell |
|---|---|---|---|
| 1. BM25 build + search | Pyserini Lucene | 4 (rat2rev × en/ja, rev2rev × en/ja) | 5 |
| 2. Dense encode + search | `me5-small` | 4 | 5 |
| 3. Cross-encoder rerank (optional) | `bge-reranker-v2-m3` | 4 | 5 |

Step 3 is skipped by default because `bge-reranker-v2-m3` (568M params)
runs slowly on CPU. To force it: `RUN_RERANK=1 bash scripts/smoke_test.sh`.

## Prerequisites

- Python 3.11 + `uv sync` completed
- JDK 21 with `JAVA_HOME` set (Pyserini)
- Internet access for HuggingFace download (~3 GB for the dataset cache +
  ~500 MB for me5-small + ~2 GB for bge-reranker-v2-m3 if `RUN_RERANK=1`)
- ~16 GB RAM, no GPU required

No API keys are needed.

## Run

```bash
bash scripts/smoke_test.sh
```

Expected wall-clock on a 2024-era 8-core x86_64 laptop with 16 GB RAM:
- First run (cold cache): 10-15 min
- Repeat runs (warm cache): ~1 min

To increase coverage during development, raise the query limit:
```bash
LIMIT_QUERIES=20 bash scripts/smoke_test.sh
```

## Outputs

After completion:
- `results/metrics/bm25_{task}_{lang}.json` — 4 cells of BM25 metrics
- `results/metrics/me5-small_{task}_{lang}.json` — 4 cells of dense metrics
- `results/smoke_test.log` — timestamped log of every step

The smoke test prints the Rat2Rev EU BM25 metrics on the last line so you
can eyeball success:

```
Sample (Rat2Rev EU BM25):
{
  "model": "bm25",
  "task": "rat2rev",
  "lang": "en",
  "n_queries": 5,
  "metrics": {
    "recall_10": ...,
    "recall_100": ...,
    ...
  }
}
```

## Unit tests

A lightweight pytest suite covers the public package surface without
hitting the network or any model:

```bash
uv run pytest tests/
```

`tests/test_smoke.py` verifies:
- All public modules import without side effects
- `recare-baselines --help` lists the expected subcommands
- The dense and reranker model registries contain every model cited in the
  paper
- Run-file round-trip preserves order and scores
- Metrics evaluation produces sane values on a synthetic ranking

`tests/test_{train_dense,finetuned_dense,hard_negative}.py` cover the
domain-adaptation helpers (contrastive loss shape, early stopping,
checkpoint paths, hard-negative exclusion logic).

These tests run in ~5 seconds on CPU.

## Troubleshooting

- **`Pyserini` ImportError**: install JDK 21 and set `JAVA_HOME`. On
  Ubuntu: `sudo apt install openjdk-21-jdk-headless && export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`.
- **`OSError: HF download failed`**: ensure outbound HTTPS to
  `huggingface.co`. Set `HF_HOME` to control the cache location.
- **`CUDA out of memory` during rerank**: the smoke reranker step requires
  ~3 GB GPU memory. Skip with `RUN_RERANK=0` (the default) or reduce
  `--batch-size` in the script.
- **Smoke takes much longer than expected**: the first run downloads the
  ReCaRe corpus (~1.5 GB compressed) plus the models. Subsequent runs
  reuse the cache.
