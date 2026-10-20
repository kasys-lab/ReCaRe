# Evaluating a ReCaRe Run

This guide explains how to score a ReCaRe run file on your own machine with
the **same script the organizers use for the CAIL 2026 ReCaRe track**. Because
the qrels are public, you can compute your official scores before you submit.

The script is the `evaluate-ranked` command of this repository. It takes a run
file (your ranked results) and a qrels file (the relevance judgments) and
reports nDCG, Recall, and MAP.

---

## Quick start

```bash
# 1. Install (Python 3.11 and uv required)
git clone https://github.com/kasys-lab/ReCaRe.git
cd ReCaRe
uv sync

# 2. Download the test qrels (a few MB)
uv run huggingface-cli download kasys/ReCaRe --repo-type dataset \
    --include "qrels-*/test.jsonl" --local-dir recare_data

# 3. Score a run
uv run recare-baselines evaluate-ranked \
    tsukuba_rat2rev-en_run1.jsonl \
    recare_data/qrels-rat2rev-en/test.jsonl \
    --task rat2rev
```

The command prints a JSON summary. The number that determines the official
ranking is `metrics.ndcg_10`.

To try the script before you have a run of your own, score one of the
[sample runs](#sample-runs) instead:

```bash
uv run recare-baselines evaluate-ranked \
    data/sample_runs/bm25_rat2rev-en_run1.jsonl.gz \
    recare_data/qrels-rat2rev-en/test.jsonl \
    --task rat2rev
# -> "ndcg_10": 0.1163..., "recall_100": 0.2255...
```

> `uv sync` installs the full baseline stack (Pyserini, PyTorch, and so on),
> not only the evaluation code, so the first install can take a while.

---

## 1. Prepare your run file

A run file has **one JSON object per line, one line per query**. Each line
holds the query ID and the document IDs in rank order, best first. Scores are
not used.

```json
{"qid": "32010L0084", "ranked": ["32001L0083_20091005_0000000000_1", "32001L0083_20091005_0000000000_8", "..."]}
```

| Field | Content |
|---|---|
| `qid` | The `_id` of a query in `queries-<task>-<lang>/queries.jsonl` |
| `ranked` | The `_id`s of corpus articles (`corpus-<lang>/corpus.jsonl`), highest-ranked first |

Track rules for the file:

- Every query in the subtask's test split must appear **exactly once**.
- Submit **at most 1,000 documents per query**. If a list is longer, the
  script scores only its first 1,000 documents.
- Name the file `<team_id>_<subtask_id>_<run_id>.jsonl`, for example
  `tsukuba_rat2rev-en_run1.jsonl`. `run1` is your primary run.

If you produced a scored ranking, sort it by score in descending order before
you write the `ranked` list. The script evaluates documents **in the order
they appear in the list**.

Submit plain `.jsonl` files. The script also reads gzip-compressed files
(`.jsonl.gz`), which is how the sample runs are stored.

### Sample runs

`data/sample_runs/` contains BM25 runs for the two English subtasks in the
official format. They show what a valid submission looks like.

| File | Queries | Documents per query | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| `bm25_rat2rev-en_run1.jsonl.gz` | 113 | 1,000 | 0.1163 | 0.2256 |
| `bm25_rev2rev-en_run1.jsonl.gz` | 503 | 1,000 | 0.2013 | 0.3106 |

They are the BM25 (Pyserini) top-1,000 lists behind the baseline numbers in
[`results/metrics/`](results/metrics/), and scoring them reproduces those
numbers exactly. Documents with equal BM25 scores are listed in descending
docid order, which is how `trec_eval` orders ties. To see a file in submission
form, decompress it with `gunzip -k`.

In the Rev2Rev sample, every query ranks its own article first, as BM25
naturally does. With `--task rev2rev`, the script removes that entry and scores
the remaining 999 documents (`n_self_matches_dropped` = 503).

---

## 2. Get the qrels

The qrels are in the Hugging Face dataset
[`kasys/ReCaRe`](https://huggingface.co/datasets/kasys/ReCaRe), one file per
subtask and split:

```
qrels-<task>-<lang>/<split>.jsonl      # e.g. qrels-rev2rev-en/test.jsonl
{"query-id": "<query_id>", "corpus-id": "<doc_id>", "score": 1}
```

The command in Quick start step 2 downloads all four test files into
`recare_data/`. TREC-format qrels (`qid 0 docid rel`) also work, and the script
detects the format automatically.

The **test-split qrels are for scoring only**. Track rules do not allow you to
use them for training, tuning, prompt construction, or model selection.

---

## 3. Score each subtask

Pass `--task` so that the script applies the correct protocol for that task.

| Subtask | Command |
|---|---|
| `rat2rev-en` | `uv run recare-baselines evaluate-ranked RUN recare_data/qrels-rat2rev-en/test.jsonl --task rat2rev` |
| `rev2rev-en` | `uv run recare-baselines evaluate-ranked RUN recare_data/qrels-rev2rev-en/test.jsonl --task rev2rev` |
| `rat2rev-ja` | `uv run recare-baselines evaluate-ranked RUN recare_data/qrels-rat2rev-ja/test.jsonl --task rat2rev` |
| `rev2rev-ja` | `uv run recare-baselines evaluate-ranked RUN recare_data/qrels-rev2rev-ja/test.jsonl --task rev2rev` |

**Why `--task rev2rev` matters.** In Rev2Rev, the query is itself an article in
the corpus, and its query ID is that article's document ID. A system therefore
often retrieves the query article at rank 1. With `--task rev2rev`, the script
removes the query's own ID from its ranked list before scoring, as the ReCaRe
paper does. Without this option, the query article counts as a non-relevant
document. If you forget the option, the script prints a warning when it finds
a query that ranks itself.

---

## 4. Read the output

Example output (abridged):

```json
{
  "task": "rat2rev",
  "n_queries": 113,
  "n_missing_queries": 0,
  "n_extra_queries": 0,
  "self_match_filtered": false,
  "n_duplicate_docs_dropped": 0,
  "metrics": {
    "recall_10": 0.0740, "recall_100": 0.2256, "recall_1000": 0.4063,
    "ndcg_10": 0.1163,   "ndcg_100": 0.1552,   "ndcg_1000": 0.2058,
    "map": 0.0705
  }
}
```

| Metric | Role in the track |
|---|---|
| `ndcg_10` | **Primary metric.** Determines the subtask ranking. |
| `recall_100` | Secondary metric. Used as the tie-breaker. |
| `recall_10`, `recall_1000`, `ndcg_100`, `ndcg_1000`, `map` | Reported in the results table. They do not affect the ranking. |

Before you submit, check these fields:

- `n_missing_queries` should be `0`. A missing query scores 0 on every metric
  and lowers your averages.
- `n_extra_queries` should be `0`. A nonzero value usually means that you used
  the wrong query set or language.
- `n_duplicate_docs_dropped` should be `0`. When a document appears more than
  once in a list, only its first occurrence counts.
- `n_truncated_queries` should be `0`. A nonzero value means that some lists
  had more than 1,000 documents and were cut.

To save per-query scores, for example for your own significance tests, add
`--output my_scores.json`. The file has the same layout as the files in
`results/metrics/` and includes a `per_query` field.

---

## 5. How scores are computed

- The metrics come from [`ir_measures`](https://ir-measur.es/), which wraps
  `trec_eval`. This is the same code that produced the baseline numbers in
  [`results/`](results/), so a baseline run converted to this format
  reproduces those numbers.
- The script gives each document a score that decreases with its rank, so
  that your list order is exactly the order that is evaluated. Ties never
  occur.
- nDCG uses `trec_eval`'s linear gain with a log2 discount. ReCaRe qrels are
  binary (`score` = 1), so the gain choice does not change any result.
- Each list is processed in this order: (1) duplicate documents are removed,
  keeping the first occurrence; (2) the list is cut to its first 1,000
  documents; (3) for Rev2Rev, the query's own ID is removed. Because step 3
  comes after step 2, a list that contains the query article is scored on 999
  documents. The ReCaRe paper baselines are scored the same way.
- A query that is in the qrels but not in your run, or whose list is empty,
  scores 0. A query that is in your run but not in the qrels is ignored.

---

## 6. Options

| Option | Default | Effect |
|---|---|---|
| `--task {rat2rev,rev2rev}` | none | Records the task. `rev2rev` also turns on the self-match filter. |
| `--filter-self-match` / `--no-filter-self-match` | on only with `--task rev2rev` | Overrides the self-match filter. |
| `--max-docs N` | `1000` | Scores only the first N documents of each list, counted after duplicates are removed. `0` means no limit. |
| `--missing {zero,skip}` | `zero` | `zero` scores missing queries 0, as the official scoring does. `skip` averages over the queries in your run only, which is useful while you debug a partial run. |
| `--output PATH` | none | Writes the full result, including per-query scores, to a JSON file. |
| `--model`, `--lang`, `--split` | none | Labels copied into the output. They do not affect scoring. |

---

## 7. Use from Python

```python
from recare_baselines.ranked_eval import evaluate_ranked_files

result = evaluate_ranked_files(
    "tsukuba_rev2rev-en_run1.jsonl",
    "recare_data/qrels-rev2rev-en/test.jsonl",
    filter_self=True,          # same as --task rev2rev
)
print(result.metrics["ndcg_10"], result.missing_qids)
```

---

## Contact

Questions about the track or the evaluation: Makoto P. Kato
(<mpkato@acm.org>). For bugs in the script, please open an issue on
[GitHub](https://github.com/kasys-lab/ReCaRe/issues).
