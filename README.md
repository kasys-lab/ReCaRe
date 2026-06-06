# ReCaRe — Retrieval Baselines

Retrieval baselines and full experimental code for the paper

> **ReCaRe: A Bilingual Legal Benchmark for Revision Candidate Retrieval.**
> Takumi Ito, Yuma Kurokawa, Makoto P. Kato, Sumio Fujita.

The benchmark dataset itself is released at
[`kasys/ReCaRe`](https://huggingface.co/datasets/kasys/ReCaRe)
([DOI 10.57967/hf/8642](https://doi.org/10.57967/hf/8642)). This
repository provides the code, run scripts, prompts, results, and
documentation for **§4.2 Retrieval Experiments** of the paper — BM25,
dense retrievers, cross-encoder rerankers, RankGPT zero-shot listwise
LLM reranking, query/document expansion, and dense-retriever domain
adaptation.

The dataset construction pipeline (§3.1), validation pipeline (§3.3),
and characterization metrics (§4.1) are intentionally outside the scope of
this public release.

## Contents

```
ReCaRe/
├── src/recare_baselines/   # Python package (CLI: recare-baselines)
├── scripts/                # End-to-end run scripts for each paper table
├── prompts/                # LLM prompt cards (expansion, RankGPT)
├── docs/                   # Reproduction guide, experiment details
├── tests/                  # pytest unit + smoke tests
├── results/                # Per-cell metrics + result-summary markdown
└── data/expansion/         # LLM query expansions (test split, ~7 MB)
```

## Quickstart

```bash
# 1. Install (Python 3.11; uv handles the venv)
git clone https://github.com/kasys-lab/ReCaRe.git
cd ReCaRe
uv sync

# 2. Smoke test: BM25 + me5-small dense on 5 queries per cell
#    (~10 min on a CPU laptop; no API keys required)
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
bash scripts/smoke_test.sh

# 3. Run paper Table 2 BM25 row (all 4 cells, full test split)
bash scripts/run_bm25.sh
```

The HuggingFace dataset is fetched on demand; no manual download is
needed. See [`docs/smoke_test.md`](docs/smoke_test.md) for details.

## Reproduction

Each script reproduces one block of the paper:

| Paper | Script | What it does |
|---|---|---|
| Table 2 BM25 row | [`scripts/run_bm25.sh`](scripts/run_bm25.sh) | Index EN/JA, run 4 cells |
| Table 2 short-context dense | [`scripts/run_short_dense.sh`](scripts/run_short_dense.sh) | mDPR / mContriever / mE5 × 4 cells |
| Table 2 long-context dense | [`scripts/run_long_dense.sh`](scripts/run_long_dense.sh) | BGE-M3 / jina-v3 × 4 cells (GPU) |
| Table 2 augmentation | [`scripts/run_expansion.sh`](scripts/run_expansion.sh) | Q2E/Q2D/d2q/d2e × {BM25, jina-v3} |
| Table 3 cross-encoder rerankers | [`scripts/run_rerankers.sh`](scripts/run_rerankers.sh) | BGE / Jina / Qwen3-4B/8B × 4 cells |
| Table 3 RankGPT | [`scripts/run_rankgpt.sh`](scripts/run_rankgpt.sh) | 3 Azure LLMs × 4 cells (≈ $156) |
| Table 4 domain adaptation | [`scripts/run_domain_adaptation.sh`](scripts/run_domain_adaptation.sh) | 5 models × 4 cells, hard-neg → train → eval |

See [`docs/reproduction.md`](docs/reproduction.md) for the full step-by-step
guide, including expected runtimes, GPU requirements, and how to verify
the reproduced numbers against the paper.

## Results

All numerical results are committed in `results/`:

- [`results/baselines.md`](results/baselines.md) — Table 2 (BM25 + 5 dense
  retrievers)
- [`results/rerankers.md`](results/rerankers.md) — Table 3 cross-encoder
  rerankers
- [`results/rankgpt.md`](results/rankgpt.md) — Table 3 RankGPT + Azure
  cost ledger
- [`results/augmentation.md`](results/augmentation.md) — Table 2
  augmentation block
- [`results/domain_adaptation.md`](results/domain_adaptation.md) — Table 4
  with Holm-corrected paired t-tests

Per-cell metrics (188 JSON files) live under `results/metrics/`. The full
results index is at [`results/README.md`](results/README.md), which also
maps every paper-table cell to its source JSON.

## Beyond-the-paper experiments

The repository also retains supplementary experiments that were performed
but not included in the paper for space reasons:

- The 0.6B Qwen3 reranker (smallest variant)
- max-P chunking strategies for dense retrievers (`maxp-doc`, `maxp-q`,
  `maxp-both`) as supplementary long-input ablations
- Full 7-metric set (R@{10,100,1000}, nDCG@{10,100,1000}, MAP) for every
  cell — the paper reports only Recall@100 (Table 2) and nDCG@10 (Tables
  3-4)
- Both `gpt-4.1-mini` AND `qwen3.5-9b` query expansions for every family
  — paper reports the `gpt-4.1-mini` cells only
- Holm-corrected paired t-tests of every method vs BM25, and every
  augmentation vs jina-v3 (see [`results/ttest_holm/`](results/ttest_holm/))

See [`results/README.md`](results/README.md) for the complete map.

## Pre-generated artifacts on HuggingFace

Two large generated artifacts are hosted on HuggingFace rather than
committed to this repo:

- **d2e / d2q full-corpus expansions** (~150 MB per language) →
  `kasys/ReCaRe-expansions` (dataset)
- **Domain-adaptation checkpoints** (5 base models × 4 cells = 20
  adapters) → `kasys-lab/recare-<base>-<task>-<lang>` (models)

See [`docs/data_format.md`](docs/data_format.md) for download commands.

## Prerequisites

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- JDK 21 (for BM25 via Pyserini)
- CUDA-capable GPU strongly recommended for dense retrievers; required for
  Qwen3 rerankers and the 8192-token jina-v3 / BGE-M3 corpus encoding
- Azure OpenAI access **only** for RankGPT (Table 3 RankGPT rows) and LLM
  query/doc expansion generation (Table 2 augmentation block, Phase 1):
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_VERSION`

BM25, dense retrievers (Table 2), cross-encoder rerankers (Table 3 rows
1-4), and domain adaptation (Table 4) all run without any OpenAI / Azure
credentials. The smoke test (`scripts/smoke_test.sh`) works on CPU and
requires no API keys.

## Citation

```bibtex
@misc{ito2026recare,
  title  = {ReCaRe: A Bilingual Legal Benchmark for Revision Candidate
            Retrieval},
  author = {Ito, Takumi and Kurokawa, Yuma and Kato, Makoto P. and
            Fujita, Sumio},
  year   = {2026},
  note   = {Manuscript under review}
}
```

The benchmark dataset has its own citation; see the
[ReCaRe HuggingFace dataset card](https://huggingface.co/datasets/kasys/ReCaRe).

## License

- **Code**: MIT (see [LICENSE](LICENSE))
- **Results, prompts, documentation**: CC BY 4.0
- **Dataset (`kasys/ReCaRe`)**: CC BY 4.0 (EU statutory articles, e-Gov
  Japanese statutes) and CC0 1.0 (metadata); see the dataset card for
  per-component terms.
