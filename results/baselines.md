# Baselines: First-stage retrieval (Paper Table 2)

## Setup

- **Dataset**: ReCaRe test split — Rat2Rev (113/121 queries for EU/JA) and
  Rev2Rev (503/535 queries for EU/JA), full article-level corpus (91,361 EU /
  90,170 JA).
- **Metrics**: Recall@{10,100,1000}, nDCG@{10,100,1000}, MAP via
  `ir_measures` (binary qrels for Recall/MAP, graded `2^rel - 1` gain for
  nDCG). For Rev2Rev, the qid == docid row is filtered post-hoc per the
  dataset card recommendation.
- **Significance**: Per-query paired two-sided t-test vs BM25 within each
  (task, lang, metric); Holm correction over the family of methods reported
  in this section. `†` marks Holm-corrected p<0.05 in the paper.

## Models

| Key | HF ID | Architecture | Pooling | Similarity | Prefix | max_length |
|---|---|---|---|---|---|---|
| `bm25` | Pyserini Lucene | sparse | — | BM25 (k1=0.9, b=0.4) | — | unlimited |
| `mdpr` | `castorini/mdpr-tied-pft-msmarco` | mBERT | CLS pooler | inner product | none | 512 |
| `mcontriever` | `facebook/mcontriever` | mBERT | mean | inner product | none | 512 |
| `me5-base` | `intfloat/multilingual-e5-base` | mBERT | mean | cosine (L2-norm) | `query: ` / `passage: ` | 512 |
| `bge-m3` | `BAAI/bge-m3` | XLM-R-Large | CLS | cosine | none | 8192 |
| `jina-v3` | `jinaai/jina-embeddings-v3` | XLM-R-Base + task LoRA | model `.encode()` | cosine | task-LoRA selected internally | 8192 |

**BM25 analyzers**: language-specific Lucene Analyzers — Porter stemmer +
default stopwords for English, Kuromoji morphological analysis + default
stopwords for Japanese. Configured via `LuceneSearcher.set_bm25(0.9, 0.4)`.

**Long-input handling** (all dense models use `truncate` for paper numbers):
| Strategy | Definition |
|---|---|
| `truncate` (default) | `score(q,d) = sim(truncate(q), truncate(d))` |
| `maxp-doc` | `score(q,d) = max_{c ∈ d} sim(truncate(q), c)` |
| `maxp-q` | `score(q,d) = max_{c ∈ q} sim(c, truncate(d))` |
| `maxp-both` | `score(q,d) = max_{qc ∈ q, dc ∈ d} sim(qc, dc)` |

Paper Table 2 uses `truncate` for all dense retrievers. The other strategies
are evaluated as supplementary ablations and persisted as
`metrics/{model}-maxp-{doc,q,both}_{task}_{lang}.json`.

## Results

### Table 2 (paper) — Recall@100 on ReCaRe test split

| Model | Rat2Rev EU | Rat2Rev JA | Rev2Rev EU | Rev2Rev JA |
|---|---|---|---|---|
| **BM25** | **0.226** | **0.470** | 0.311 | 0.418 |
| *Short-context dense (512 tok, truncate)* | | | | |
| mDPR | 0.084 | 0.349 | 0.179 | 0.394 |
| mContriever | 0.163 | 0.416 | 0.261 | 0.424 |
| mE5 | 0.221 | 0.426 | 0.284 | 0.393 |
| *Long-context dense (8192 tok)* | | | | |
| BGE-M3 | 0.220 | 0.426 | 0.306 | 0.398 |
| Jina v3 | **0.331**† | **0.482** | **0.339**† | **0.461**† |

(`†` denotes Holm-corrected p<0.05 vs BM25.)

### Detailed metrics (all measures)

Aggregates: [`baselines_short.json`](baselines_short.json) (BM25 + 3 short),
[`baselines_long.json`](baselines_long.json) (BGE-M3 + Jina). Each model key
maps to four `{task}-{lang}` cells with the full 7-metric set.

Per-cell JSON: `metrics/{model}_{task}_{lang}.json`.

## Findings (matches paper §4.2 narrative)

1. **BM25 is a strong first-stage baseline.** It outperforms all
   short-context dense retrievers in 3 of 4 cells on Recall@100 — even
   though Table 1 documents low query-document lexical overlap. This is
   consistent with the long, content-rich queries in Rat2Rev.
2. **Long-context dense (Jina v3) is the only consistent winner.**
   Statistically significant Recall@100 improvements over BM25 in Rat2Rev EU
   and both Rev2Rev settings. BGE-M3 is competitive but does not
   significantly improve over BM25 in any cell.
3. **Short-context dense underperforms** when queries or documents exceed
   512 tokens — most notably on Rat2Rev EU where queries average 558.6
   tokens.

## Related cells beyond Table 2

| Variant | Cells | Files |
|---|---|---|
| max-P doc/query/both strategies | 4 models × 4 cells × 3 strategies = 48 | `metrics/{model}-maxp-*_*.json` |
| Paired t-tests vs BM25 | All methods × all metrics | `ttest_holm/ttest_holm_all.csv` |
