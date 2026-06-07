# Reproducing the paper's retrieval experiments

This document maps each table in §4.2 of the paper to the exact shell
commands and expected output files.

The CLI is `recare-baselines`; installed by `uv sync` from the repository
root.

## Prerequisites

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- JDK 21 (for BM25 via Pyserini). Set `JAVA_HOME` accordingly.
- A CUDA-capable GPU is strongly recommended for dense retrievers; required
  for Qwen3 rerankers and the 8192-token jina-v3 / BGE-M3 corpus encoding.
- For RankGPT (Table 3 RankGPT rows): Azure OpenAI access — set
  `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`.
- For LLM query/doc expansion **generation** (Table 2 augmentation, Phase
  1): same Azure variables plus optional `QWEN_API_BASE` for the
  self-hosted Qwen3.5-9B alternative. The repository ships GPT-4.1-mini
  test-split outputs in `data/expansion/`; if you skip Phase 1 generation
  you can complete Table 2 augmentation evaluation without any LLM keys.

**BM25, dense retrievers, cross-encoder rerankers, and domain adaptation
do not require any OpenAI / Azure credentials.** The package's `__init__`
installs a placeholder `OPENAI_API_KEY` to satisfy unrelated upstream
imports (pyserini, rank_llm); the real Azure variables only matter at
actual LLM-call time.

The dataset is fetched automatically from the HuggingFace dataset
`kasys/ReCaRe`; no manual download is needed for the test split.

## Setup

```bash
git clone https://github.com/kasys-lab/ReCaRe.git
cd ReCaRe
uv sync
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64   # adjust to your install
```

## Table 2 — First-stage retrieval

### BM25 row

```bash
bash scripts/run_bm25.sh
```

This builds the EU/JA Lucene indexes (idempotent) and runs all 4 cells.
Outputs:
- Indexes: `indexes/lucene/{en,ja}/` (not committed)
- Runfiles: `results/intermediate/bm25_runs/{task}_{lang}.jsonl`
- Top-100 for downstream rerankers: `results/intermediate/bm25_top100/{task}_{lang}.jsonl`
- Metrics: `results/metrics/bm25_{task}_{lang}.json`

Expected Recall@100: 0.226 / 0.470 / 0.311 / 0.418 (matches paper Table 2).

### Short-context dense (mDPR, mContriever, mE5-base)

```bash
bash scripts/run_short_dense.sh
```

GPU recommended. Each model takes ~5-15 min/lang to encode the corpus and
~1 min to search. Outputs follow the same pattern with `model_key` in place
of `bm25`.

### Long-context dense (BGE-M3, Jina v3)

```bash
bash scripts/run_long_dense.sh
```

GPU required for tractable runtime. jina-v3 needs `trust_remote_code=True`
(handled automatically). Corpus encoding can take 30-60 min/lang on a
single A100; longer on consumer cards.

### Augmentation block (Q2E, Q2D, d2e, d2q on BM25 and jina-v3)

Three-phase pipeline. See [`scripts/run_expansion.sh`](../scripts/run_expansion.sh)
for the full grid; high-level:

```bash
# Phase 1: generate query expansions (needs Azure API key)
PHASES=1 bash scripts/run_expansion.sh

# Phase 2: build augmented indexes (d2q / d2e corpus expansions are
#          auto-fetched from HF kasys/ReCaRe-expansions; see docs/data_format.md)
PHASES=2 bash scripts/run_expansion.sh

# Phase 3: evaluate the 64 augmented cells
PHASES=3 bash scripts/run_expansion.sh
```

Phase 1 can be **skipped** if you reuse the GPT-4.1-mini test-split
expansions already committed in `data/expansion/`. Qwen3.5-9B variants
require re-generation against a self-hosted endpoint.

## Table 3 — Reranking on BM25 top-100

### Cross-encoder rerankers

```bash
bash scripts/run_rerankers.sh
```

Requires the BM25 top-100 (from `scripts/run_bm25.sh`). The order is
small→large so a later OOM still leaves smaller models' results on disk.
Qwen3-8B requires ≥40 GB of GPU memory; reduce batch size in the script
or use `MODELS="bge-reranker-v2-m3 jina-reranker-v2 qwen3-reranker-4b"` to
skip it.

### RankGPT (zero-shot listwise LLM)

```bash
bash scripts/run_rankgpt.sh
```

Cost: approximately $156 USD total at the rates that applied during paper
writing (gpt-4.1-nano $5, gpt-4.1-mini $77, gpt-5.4-mini $74). Run

```bash
uv run recare-baselines rankgpt-cost --model gpt-4.1-mini-2025-04-14
```

first to estimate at current rates. Resume-safe via `--skip-existing`:
already-completed cells are detected by their JSON and skipped.

## Table 4 — Domain adaptation

```bash
bash scripts/run_domain_adaptation.sh
```

Three phases in one script (toggle via `PHASES`):
1. Build dense top-100 on train/validation splits (for the hard-negative
   pool).
2. Train 5 models × 4 cells = 20 adapters. Short-context models use full
   FT; long-context use LoRA. Each adapter trains for up to 100 epochs with
   patience=3 on validation loss.
3. Encode the adapted corpus, evaluate on test, and aggregate into
   `results/domain_adaptation.json`.

The 20 trained checkpoints are too large to ship in the repo; they will be
uploaded to HuggingFace Hub at `kasys-lab/recare-<model>-<task>-<lang>`.
See [`docs/data_format.md`](data_format.md) for the download command.

## Verifying paper numbers

After running the four scripts above, compare against the paper:

```bash
# Table 2 BM25 Rat2Rev EU R@100
uv run python -c "import json; print(json.load(open('results/metrics/bm25_rat2rev_en.json'))['metrics']['recall_100'])"
# → 0.2256... (paper rounds to 0.226)

# Table 3 RankGPT gpt-5.4-mini Rat2Rev EU nDCG@10
uv run python -c "import json; print(json.load(open('results/metrics/rankgpt-gpt-5.4-mini-2026-03-17+bm25_rat2rev_en.json'))['metrics']['ndcg_10'])"
# → 0.2253... (paper rounds to 0.225)
```

Aggregate JSONs (`results/baselines_short.json`, etc.) provide all
4-cell × 7-metric blocks in a single file for easier programmatic checking.

## Significance markers

The `†` markers in paper Tables 2-4 correspond to Holm-corrected paired
t-tests. Pre-computed t-test tables:
- vs BM25 baseline: [`results/ttest_holm/ttest_holm_all.csv`](../results/ttest_holm/ttest_holm_all.csv)
- vs jina-v3 baseline (augmentation block): [`results/ttest_holm/ttest_holm_aug_vs_jina-v3.csv`](../results/ttest_holm/ttest_holm_aug_vs_jina-v3.csv)
- Domain adaptation (before vs after): inside `results/domain_adaptation.json` records

To re-compute:
```bash
uv run recare-baselines ttest-holm --help
```
