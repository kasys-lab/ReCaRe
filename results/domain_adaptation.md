# Domain adaptation: dense retriever fine-tuning (Paper Table 4)

## Setup

- **Training data**: Triplets `(query, positive, hard_negative)` from the
  train split. Hard negative is sampled uniformly at random from the base
  model's top-100 candidates after excluding all positive articles for that
  query. One hard negative per (qid, positive) pair.
- **Loss**: InfoNCE with the hard negative + in-batch negatives.
- **Tuning method**:
  - Short-context (`mdpr`, `mcontriever`, `me5-base`): full fine-tuning,
    batch size 64.
  - Long-context (`bge-m3`, `jina-v3`): LoRA fine-tuning, batch size 8.
- **Schedule**: AdamW, lr=1e-5, warmup ratio 0.1, up to 100 epochs with
  early stopping (`patience=3` on validation loss), seed 13.
- **Evaluation**: each adapted checkpoint is evaluated on the same test
  cell it was trained for (e.g. `me5-base-ft-rat2rev-en` evaluates on
  Rat2Rev EU). 5 models × 4 cells = 20 fine-tuned adapters.
- **Significance**: paired two-sided t-test on per-query metrics
  (Recall@100, nDCG@10) vs the base model, Holm-corrected within each
  (task, lang, metric) family of `k=5` models.

## Table 4 (paper) — Recall@100 / nDCG@10, Base → Adapted

### Rat2Rev EU

| Model | Recall@100 | nDCG@10 |
|---|---|---|
| mDPR | 0.084 → 0.137 | 0.055 → 0.059 |
| mContriever | 0.163 → 0.256† | 0.058 → 0.126† |
| mE5 | 0.221 → 0.290† | 0.110 → 0.139 |
| BGE-M3 | 0.220 → 0.285† | 0.125 → 0.143† |
| Jina v3 | 0.331 → 0.339 | 0.163 → 0.148 |

### Rat2Rev JA

| Model | Recall@100 | nDCG@10 |
|---|---|---|
| mDPR | 0.349 → 0.323 | 0.226 → 0.131 |
| mContriever | 0.416 → 0.443 | 0.205 → 0.226 |
| mE5 | 0.426 → 0.424 | 0.246 → 0.190 |
| BGE-M3 | 0.426 → 0.499† | 0.237 → 0.266 |
| Jina v3 | 0.482 → 0.499 | 0.268 → **0.287**† |

### Rev2Rev EU

| Model | Recall@100 | nDCG@10 |
|---|---|---|
| mDPR | 0.179 → 0.214† | 0.120 → 0.079 |
| mContriever | 0.261 → 0.320† | 0.163 → 0.150 |
| mE5 | 0.284 → 0.351† | 0.164 → 0.183† |
| BGE-M3 | 0.306 → 0.317† | 0.172 → 0.182† |
| Jina v3 | 0.339 → 0.365† | 0.203 → 0.187 |

### Rev2Rev JA

| Model | Recall@100 | nDCG@10 |
|---|---|---|
| mDPR | 0.394 → 0.380 | 0.244 → 0.166 |
| mContriever | 0.424 → 0.436 | 0.261 → 0.189 |
| mE5 | 0.393 → **0.495**† | 0.247 → 0.261 |
| BGE-M3 | 0.398 → 0.402† | 0.246 → 0.247 |
| Jina v3 | 0.461 → 0.474† | 0.261 → 0.267† |

(`†` denotes Holm-corrected p<0.05 of Adapted vs Base, family size k=5
per (task, lang, metric).)

## Findings (paper §4.2)

1. **Domain adaptation reliably improves Recall@100.** Significant gains
   appear for the majority of cells, especially in EU and on Rev2Rev. This
   indicates that in-domain supervision helps dense models surface
   additional relevant articles into the candidate pool.
2. **nDCG@10 gains are selective.** Significant top-rank improvements
   appear only for some models (mContriever and BGE on Rat2Rev EU; mE5 and
   BGE on Rev2Rev EU; Jina on both Rat2Rev JA nDCG@10 and Rev2Rev JA
   nDCG@10), and several smaller models actually lose top-rank
   effectiveness after adaptation (e.g. mDPR nDCG@10 drops in three of
   four cells).
3. **Long-context retrievers remain competitive after adaptation.** Jina
   and BGE-M3 are still among the strongest models post-adaptation, which
   matches the long-text structure of ReCaRe.

## Beyond-the-paper diagnostics

- Per-cell training curves (`loss_curve.png`), config (`train_config.json`),
  and step-level metrics (`train_steps.jsonl`, `metrics.jsonl`) are saved
  per adapter under `results/dense_finetune/{model}/{task}_{lang}/`
  (gitignored — they are reproducible from `scripts/run_domain_adaptation.sh`).
- Hard-negative pool: `results/intermediate/dense_top100/{train,validation}/`
  (gitignored).
- Triplet training data: `results/intermediate/training_data/dense/{split}/`
  (gitignored).

## Per-cell data

Aggregate: [`domain_adaptation.json`](domain_adaptation.json), with one
record per (base_model, train_task, train_lang, eval_task, eval_lang)
combination, including before/after metrics, delta, and training summary.

Per-cell after-adaptation metrics: `metrics/{model}-{lora-,}ft-{task}-{lang}_{task}_{lang}.json`.
