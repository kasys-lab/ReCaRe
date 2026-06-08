"""CLI for running BM25 / dense baselines on ReCaRe.

Examples
--------
::

    # Build per-language Lucene indexes (one-time).
    uv run recare-baselines build-bm25-index en
    uv run recare-baselines build-bm25-index ja

    # Run BM25 on one (task, lang) pair, write run-file + metrics.
    uv run recare-baselines run-bm25 rat2rev en
    uv run recare-baselines run-bm25 rev2rev ja

    # Run all 4 BM25 settings.
    uv run recare-baselines run-bm25-all

    # Encode the per-language corpus with one dense model (one-time per model).
    uv run recare-baselines encode-dense me5-base en
    uv run recare-baselines encode-dense me5-base ja

    # Run a dense model over one (task, lang) pair.
    uv run recare-baselines run-dense me5-base rat2rev en

    # Run the full 16-cell grid (4 models × 2 tasks × 2 langs).
    uv run recare-baselines run-all

    # Aggregate everything into results/baselines_short.json.
    uv run recare-baselines aggregate
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from . import bm25 as bm25_mod
from . import (
    data,
    dense as dense_mod,
    domain_adaptation as domain_adaptation_mod,
    expansion as expansion_mod,
    finetuned_dense as finetuned_dense_mod,
    hard_negative as hard_negative_mod,
    rankgpt as rankgpt_mod,
    reranker as rerank_mod,
    train_dense as train_dense_mod,
)
from .eval import evaluate, evaluate_per_query, filter_self_match
from .runfile import write_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
BM25_TOP100_DIR = INTERMEDIATE_DIR / "bm25_top100"
DENSE_TOP100_DIR = INTERMEDIATE_DIR / "dense_top100"
DENSE_RUNS_DIR = INTERMEDIATE_DIR / "dense_runs"
METRICS_DIR = RESULTS_DIR / "metrics"
SUMMARY_PATH = RESULTS_DIR / "baselines_short.json"

REQUIRED_METRIC_KEYS = (
    "recall_10",
    "recall_100",
    "recall_1000",
    "ndcg_10",
    "ndcg_100",
    "ndcg_1000",
    "map",
)


def _metrics_cell_complete(persist_key: str, task: str, lang: str) -> bool:
    """Return True iff a *valid* metrics cell exists at the canonical path.

    "Valid" means: file exists, parses as JSON, has a ``metrics`` dict
    containing every key in :data:`REQUIRED_METRIC_KEYS` with a finite numeric
    value. A bare ``Path.exists()`` would mistake a partially-written or
    corrupted file for a completed run; this guard is what justifies the
    ``--skip-existing`` flag treating presence as completion.
    """
    path = METRICS_DIR / f"{persist_key}_{task}_{lang}.json"
    if not path.exists():
        return False
    try:
        cell = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    metrics = cell.get("metrics")
    if not isinstance(metrics, dict):
        return False
    for k in REQUIRED_METRIC_KEYS:
        v = metrics.get(k)
        if not isinstance(v, (int, float)):
            return False
    return True


@click.group()
def cli() -> None:
    """ReCaRe baselines: BM25 + multilingual short-context dense retrievers."""


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------


@cli.command("fetch-expansions")
@click.option(
    "--method",
    type=click.Choice(["d2e", "d2q", "all"]),
    default="all",
    show_default=True,
    help="Which document-side expansion to fetch.",
)
@click.option(
    "--lang",
    type=click.Choice(["en", "ja", "all"]),
    default="all",
    show_default=True,
)
def fetch_expansions(method, lang):
    """Download document-side expansions (d2e/d2q) from HF into ``data/``.

    Pulls from kasys/ReCaRe-expansions and places files at the canonical
    ``data/recare_{method}/recare_{lang}_{method}_documents.jsonl`` layout the
    augmentation scripts expect. Idempotent; existing files are reused.
    """
    methods = data.DOC_EXPANSION_METHODS if method == "all" else (method,)
    langs = data.LANGS if lang == "all" else (lang,)
    paths = data.fetch_doc_expansions(methods=methods, langs=langs)
    for p in paths:
        click.echo(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    click.echo(f"fetched {len(paths)} file(s) from {data.DOC_EXPANSION_REPO}")


@cli.command("fetch-finetuned")
@click.option(
    "--model",
    type=click.Choice([*finetuned_dense_mod.DA_MODEL_KEYS, "all"]),
    default="all",
    show_default=True,
    help="Which base model's domain-adapted checkpoints to fetch.",
)
@click.option("--task", type=click.Choice(["rat2rev", "rev2rev", "all"]), default="all", show_default=True)
@click.option("--lang", type=click.Choice(["en", "ja", "all"]), default="all", show_default=True)
def fetch_finetuned(model, task, lang):
    """Download domain-adapted checkpoints (Table 4) from HF.

    Pulls from kasys/ReCaRe-domain-adaptation and places each checkpoint at
    ``results/dense_finetune/<model>/<task>_<lang>/best`` — the layout Phase 3
    of run_domain_adaptation.sh evaluates. Lets you reproduce the evaluation
    without re-running training. Idempotent; existing checkpoints are reused.
    """
    models = finetuned_dense_mod.DA_MODEL_KEYS if model == "all" else (model,)
    tasks = data.TASKS if task == "all" else (task,)
    langs = data.LANGS if lang == "all" else (lang,)
    paths = finetuned_dense_mod.fetch_finetuned_checkpoints(
        model_keys=models, tasks=tasks, langs=langs
    )
    for p in paths:
        click.echo(f"  {p}")
    click.echo(
        f"fetched {len(paths)} checkpoint(s) from "
        f"{finetuned_dense_mod.DA_MODELS_REPO}"
    )


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


@cli.command("build-bm25-index")
@click.argument("lang", type=click.Choice(["en", "ja"]))
@click.option("--threads", type=int, default=8, show_default=True)
@click.option("--force", is_flag=True, help="Rebuild from scratch even if index exists.")
@click.option(
    "--doc-augmentation",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a doc-augmentation JSONL ({\"id\": ..., \"contents\": <aug>}). "
    "Builds an index over (original doc) + ' ' + (augmentation), placed at "
    "indexes/lucene/{lang}+{aug-suffix}.",
)
@click.option(
    "--aug-suffix",
    default=None,
    help="Index/key suffix for the augmented variant (e.g. 'd2q', 'd2e'). "
    "Required with --doc-augmentation.",
)
def build_bm25_index(lang, threads, force, doc_augmentation, aug_suffix):
    """Build a Lucene BM25 index for the given language.

    Without ``--doc-augmentation`` this builds the canonical per-language
    index. With ``--doc-augmentation FILE --aug-suffix d2q`` it builds the
    Doc2Query / LLM-doc-expanded variant at ``indexes/lucene/{lang}+d2q``.
    """
    if doc_augmentation is not None and not aug_suffix:
        raise click.UsageError("--doc-augmentation requires --aug-suffix.")
    out = bm25_mod.build_index(
        lang,
        threads=threads,
        force=force,
        doc_augmentation=doc_augmentation,
        doc_aug_suffix=aug_suffix,
    )
    click.echo(f"index ready: {out}")


@cli.command("run-bm25")
@click.argument("task", type=click.Choice(["rat2rev", "rev2rev"]))
@click.argument("lang", type=click.Choice(["en", "ja"]))
@click.option("--top-k", type=int, default=1000, show_default=True)
@click.option("--split", default="test", type=click.Choice(["train", "validation", "test"]))
@click.option(
    "--query-augmentation",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a query-expansion JSONL produced by `expand-queries`. "
    "Applies Wang+ Query2doc / Jagerman+ 2023's BM25 protocol: "
    "q⁺ = concat(q×5, expansion).",
)
@click.option(
    "--doc-aug-suffix",
    default=None,
    help="Suffix of the doc-augmented Lucene index to use (e.g. 'd2q', 'd2e'). "
    "Index must have been built via `build-bm25-index --doc-augmentation ... "
    "--aug-suffix <suffix>` first. Combinable with --query-augmentation.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="If a *valid* metrics cell already exists for this (model, task, lang), "
    "skip re-running. Validity is verified by re-parsing the JSON.",
)
@click.option(
    "--limit-queries",
    type=int,
    default=None,
    help="Smoke / debug: only evaluate the first N queries (deterministic order). "
    "Persisted metrics are still written; for paper-faithful numbers omit this flag.",
)
def run_bm25(
    task, lang, split, top_k, query_augmentation, doc_aug_suffix, skip_existing,
    limit_queries,
):
    """Run BM25 search and write run-file + metrics."""
    rd = data.load(task, lang, split=split)
    queries = rd.queries
    if limit_queries is not None and limit_queries > 0:
        queries = dict(list(queries.items())[:limit_queries])
    model_key_parts = ["bm25"]
    if doc_aug_suffix:
        model_key_parts.append(doc_aug_suffix)
    if query_augmentation is not None:
        queries, stats = expansion_mod.apply_query_expansion(
            queries, query_augmentation, mode="bm25_n5"
        )
        suffix = expansion_mod.derive_persist_suffix(query_augmentation)
        model_key_parts.append(suffix)
        click.echo(
            f"  query_augmentation={query_augmentation.name} stats={stats}",
            err=True,
        )
    model_key = "+".join(model_key_parts)
    if skip_existing and _metrics_cell_complete(model_key, task, lang):
        click.echo(
            f"skip {model_key} {task}/{lang}: metrics cell already complete",
            err=True,
        )
        return
    click.echo(
        f"BM25 {task}/{lang} | n_queries={len(queries)} | top_k={top_k} | "
        f"model_key={model_key} | doc_aug_suffix={doc_aug_suffix or '(none)'}",
        err=True,
    )
    run = bm25_mod.run_search(
        queries, lang=lang, top_k=top_k, doc_aug_suffix=doc_aug_suffix
    )
    _persist(run, rd, model_key=model_key, split=split)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


@cli.command("encode-dense")
@click.argument("model_key", type=click.Choice(sorted(dense_mod.MODELS.keys())))
@click.argument("lang", type=click.Choice(["en", "ja"]))
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--force", is_flag=True)
@click.option(
    "--strategy",
    type=click.Choice(["truncate", "maxp"]),
    default="truncate",
    show_default=True,
    help="'truncate' encodes one passage per doc (head-truncate to max_length). "
    "'maxp' chunks long docs into multiple passages so search can take the "
    "per-doc max similarity (score_d = max_{c in d} sim(truncate(q), c)).",
)
@click.option(
    "--doc-augmentation",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a doc-augmentation JSONL ({\"id\": ..., \"contents\": <aug>}). "
    "Encodes (original doc) + ' ' + (augmentation), placed at "
    "indexes/dense/{model_key}/{lang}+{aug-suffix}.",
)
@click.option(
    "--aug-suffix",
    default=None,
    help="Embedding/key suffix for the augmented variant (e.g. 'd2q', 'd2e'). "
    "Required with --doc-augmentation. Truncate strategy only.",
)
def encode_dense(model_key, lang, batch_size, force, strategy, doc_augmentation, aug_suffix):
    """Encode the language-specific ReCaRe corpus with a dense model."""
    if doc_augmentation is not None:
        if not aug_suffix:
            raise click.UsageError("--doc-augmentation requires --aug-suffix.")
        if strategy != "truncate":
            raise click.UsageError(
                "--doc-augmentation is only supported with strategy=truncate."
            )
    if strategy == "maxp":
        out = dense_mod.encode_corpus_maxp(
            model_key, lang, batch_size=batch_size, force=force
        )
    else:
        out = dense_mod.encode_corpus(
            model_key,
            lang,
            batch_size=batch_size,
            force=force,
            doc_augmentation=doc_augmentation,
            doc_aug_suffix=aug_suffix,
        )
    click.echo(f"corpus index ({strategy}): {out}")


@cli.command("run-dense")
@click.argument("model_key", type=click.Choice(sorted(dense_mod.MODELS.keys())))
@click.argument("task", type=click.Choice(["rat2rev", "rev2rev"]))
@click.argument("lang", type=click.Choice(["en", "ja"]))
@click.option("--top-k", type=int, default=1000, show_default=True)
@click.option("--split", default="test", type=click.Choice(["train", "validation", "test"]))
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option(
    "--strategy",
    type=click.Choice(["truncate", "maxp-doc", "maxp-q", "maxp-both", "maxp"]),
    default="truncate",
    show_default=True,
    help=(
        "Long-input handling. 'truncate': head-truncate q and d to max_length. "
        "'maxp-doc': score_d = max_{c in d} sim(truncate(q), c). "
        "'maxp-q': score_d = max_{c in q} sim(c, truncate(d)). "
        "'maxp-both': score_d = max_{qc in q, dc in d} sim(qc, dc). "
        "'maxp' is an alias for 'maxp-doc'."
    ),
)
@click.option(
    "--query-augmentation",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a query-expansion JSONL produced by `expand-queries`. "
    "Applies Wang+ Query2doc dense protocol: q⁺ = q + ' [SEP] ' + expansion.",
)
@click.option(
    "--doc-aug-suffix",
    default=None,
    help="Suffix of the doc-augmented dense index to use (e.g. 'd2q', 'd2e'). "
    "Embeddings must have been built via `encode-dense ... --doc-augmentation "
    "FILE --aug-suffix <suffix>` first. Truncate strategy only.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="If a *valid* metrics cell already exists for this (model, task, lang), "
    "skip re-running. Validity is verified by re-parsing the JSON.",
)
@click.option(
    "--limit-queries",
    type=int,
    default=None,
    help="Smoke / debug: only evaluate the first N queries (deterministic order).",
)
def run_dense(
    model_key, task, lang, split, top_k, batch_size, strategy,
    query_augmentation, doc_aug_suffix, skip_existing, limit_queries,
):
    """Run a dense model over a (task, lang) pair using the pre-encoded index."""
    if strategy == "maxp":
        strategy = "maxp-doc"
    if doc_aug_suffix and strategy != "truncate":
        raise click.UsageError(
            "--doc-aug-suffix only supported with strategy=truncate."
        )
    rd = data.load(task, lang, split=split)
    queries = rd.queries
    if limit_queries is not None and limit_queries > 0:
        queries = dict(list(queries.items())[:limit_queries])
    aug_suffix = ""
    if doc_aug_suffix:
        aug_suffix = f"+{doc_aug_suffix}"
    if query_augmentation is not None:
        if strategy != "truncate":
            raise click.UsageError(
                "--query-augmentation is currently supported only with strategy=truncate "
                "(max-P variants chunk the query, which is incompatible with the "
                "Wang+ '[SEP]'-style concat we apply here)."
            )
        queries, stats = expansion_mod.apply_query_expansion(
            queries, query_augmentation, mode="dense_sep"
        )
        aug_suffix += "+" + expansion_mod.derive_persist_suffix(query_augmentation)
        click.echo(
            f"  query_augmentation={query_augmentation.name} stats={stats}",
            err=True,
        )
    # Compute persist_key up front so --skip-existing can short-circuit
    # before the (potentially expensive) query encoding.
    if strategy == "maxp-doc":
        persist_key = f"{model_key}-maxp-doc{aug_suffix}"
    elif strategy == "maxp-q":
        persist_key = f"{model_key}-maxp-q{aug_suffix}"
    elif strategy == "maxp-both":
        persist_key = f"{model_key}-maxp-both{aug_suffix}"
    else:
        persist_key = f"{model_key}{aug_suffix}"
    if skip_existing and _metrics_cell_complete(persist_key, task, lang):
        click.echo(
            f"skip {persist_key} {task}/{lang}: metrics cell already complete",
            err=True,
        )
        return
    click.echo(
        f"{model_key} {task}/{lang} strategy={strategy} | n_queries={len(queries)} | "
        f"top_k={top_k} | doc_aug_suffix={doc_aug_suffix or '(none)'}",
        err=True,
    )
    if strategy == "maxp-doc":
        run = dense_mod.run_search_maxp(
            model_key, lang, queries, top_k=top_k, batch_size=batch_size
        )
    elif strategy == "maxp-q":
        run = dense_mod.run_search_maxp_q(
            model_key, lang, queries, top_k=top_k, batch_size=batch_size
        )
    elif strategy == "maxp-both":
        run = dense_mod.run_search_maxp_both(
            model_key, lang, queries, top_k=top_k, batch_size=batch_size
        )
    else:
        run = dense_mod.run_search(
            model_key, lang, queries, top_k=top_k, batch_size=batch_size,
            doc_aug_suffix=doc_aug_suffix,
        )
    _persist(run, rd, model_key=persist_key, split=split)


@cli.command("build-dense-top100")
@click.argument("model_key", type=click.Choice(sorted(dense_mod.MODELS.keys())))
@click.argument("task", type=click.Choice(data.TASKS))
@click.argument("lang", type=click.Choice(data.LANGS))
@click.option("--split", default="train", type=click.Choice(["train", "validation"]))
@click.option("--top-k", type=int, default=100, show_default=True)
@click.option(
    "--search-top-k",
    type=int,
    default=1000,
    show_default=True,
    help="Internal retrieval depth before writing the split-aware top-k file.",
)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--force", is_flag=True, help="Rebuild even if the top100 file exists.")
def build_dense_top100_cmd(
    model_key,
    task,
    lang,
    split,
    top_k,
    search_top_k,
    batch_size,
    force,
):
    """Build split-aware dense top-100 for train / validation queries."""
    result = hard_negative_mod.build_dense_top100(
        model_key,
        task,
        lang,
        split=split,
        top_k=top_k,
        search_top_k=search_top_k,
        batch_size=batch_size,
        force=force,
    )
    verb = "wrote" if result.created else "exists"
    click.echo(
        f"{verb} {result.path} | split={result.split} "
        f"n_queries={result.n_queries} top_k={result.top_k}"
    )


@cli.command("build-dense-training-data")
@click.argument("model_key", type=click.Choice(sorted(dense_mod.MODELS.keys())))
@click.argument("task", type=click.Choice(data.TASKS))
@click.argument("lang", type=click.Choice(data.LANGS))
@click.option("--split", default="train", type=click.Choice(["train", "validation"]))
@click.option("--seed", type=int, default=13, show_default=True)
@click.option(
    "--top100-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override dense top100 input path. Defaults to the split-aware location.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Override output JSONL path. Defaults to results/intermediate/training_data/dense/.",
)
def build_dense_training_data_cmd(model_key, task, lang, split, seed, top100_path, out_path):
    """Mine dense hard negatives and write training triples."""
    result = hard_negative_mod.build_training_examples(
        model_key,
        task,
        lang,
        split=split,
        seed=seed,
        top100_path=top100_path,
        out_path=out_path,
    )
    mining = result.mining
    click.echo(
        f"wrote {result.path} | examples={result.n_examples} "
        f"positive_pairs={mining.n_positive_pairs} "
        f"skipped_queries={mining.n_skipped_queries} "
        f"missing_top100={mining.n_missing_top100_queries} seed={mining.seed}"
    )


@cli.command("train-dense")
@click.argument("model_key", type=click.Choice(sorted(dense_mod.MODELS.keys())))
@click.argument("task", type=click.Choice(data.TASKS))
@click.argument("lang", type=click.Choice(data.LANGS))
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--grad-accum-steps", type=int, default=1, show_default=True)
@click.option("--lr", type=float, default=1e-5, show_default=True)
@click.option("--warmup-ratio", type=float, default=0.1, show_default=True)
@click.option("--epochs", type=int, default=100, show_default=True)
@click.option("--patience", type=int, default=3, show_default=True)
@click.option("--seed", type=int, default=13, show_default=True)
@click.option("--temperature", type=float, default=0.05, show_default=True)
@click.option(
    "--tuning-method",
    type=click.Choice(["full", "lora"]),
    default="full",
    show_default=True,
)
@click.option(
    "--max-length",
    type=int,
    default=None,
    help="Override the model spec max_length.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override output directory. Defaults to results/dense_finetune/{model}/{task}_{lang}.",
)
@click.option(
    "--no-auto-build-data",
    is_flag=True,
    help="Require existing training JSONL instead of creating top100/training data.",
)
@click.option(
    "--force-data",
    is_flag=True,
    help="Rebuild split-aware top100 and training JSONL before training.",
)
@click.option(
    "--top-k",
    type=int,
    default=100,
    show_default=True,
    help="Top-k depth saved for auto-built split-aware dense top100.",
)
@click.option(
    "--search-top-k",
    type=int,
    default=1000,
    show_default=True,
    help="Internal retrieval depth for auto-built split-aware dense top100.",
)
@click.option(
    "--build-batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Batch size used when auto-building dense top100.",
)
@click.option("--log-every", type=int, default=1, show_default=True)
@click.option(
    "--device",
    default=None,
    help="Training device override, e.g. 'cuda', 'cuda:0', or 'cpu'.",
)
@click.option(
    "--fp16/--no-fp16",
    default=None,
    help="Override mixed precision. Defaults to enabled on CUDA only.",
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=None,
    help=(
        "Override activation checkpointing. Defaults to disabled for native "
        "Jina LoRA and enabled otherwise."
    ),
)
def train_dense_cmd(
    model_key,
    task,
    lang,
    batch_size,
    grad_accum_steps,
    lr,
    warmup_ratio,
    epochs,
    patience,
    seed,
    temperature,
    tuning_method,
    max_length,
    output_dir,
    no_auto_build_data,
    force_data,
    top_k,
    search_top_k,
    build_batch_size,
    log_every,
    device,
    fp16,
    gradient_checkpointing,
):
    """Fine-tune a dense retriever with supervised contrastive loss."""
    result = train_dense_mod.train_dense(
        model_key,
        task,
        lang,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        warmup_ratio=warmup_ratio,
        epochs=epochs,
        patience=patience,
        seed=seed,
        temperature=temperature,
        tuning_method=tuning_method,
        max_length=max_length,
        output_dir=output_dir,
        auto_build_data=not no_auto_build_data,
        force_data=force_data,
        top_k=top_k,
        search_top_k=search_top_k,
        build_batch_size=build_batch_size,
        log_every=log_every,
        device=device,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
    )
    click.echo(
        f"trained {model_key} {task}/{lang} | out={result.output_dir} "
        f"best_val_loss={result.best_val_loss:.6f} "
        f"epochs={result.epochs_trained} steps={result.global_step} "
        f"stopped_early={result.stopped_early}"
    )
    click.echo(f"best checkpoint: {result.best_dir}")
    click.echo(f"last checkpoint: {result.last_dir}")
    click.echo(f"metrics: {result.metrics_path}")


@cli.command("plot-training-curve")
@click.argument(
    "metrics_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output PNG path. Defaults to loss_curve.png next to metrics.jsonl.",
)
def plot_training_curve_cmd(metrics_path, out_path):
    """Plot train / validation loss from dense fine-tuning metrics JSONL."""
    out = train_dense_mod.plot_training_curve(metrics_path, out_path)
    click.echo(f"wrote {out}")


# ---------------------------------------------------------------------------
# Fine-tuned dense retrieval (domain adaptation Phase 3)
# ---------------------------------------------------------------------------


@cli.command("encode-finetuned-dense")
@click.argument(
    "checkpoint_dir",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--lang", required=True, type=click.Choice(data.LANGS))
@click.option(
    "--model-key",
    type=click.Choice(sorted(dense_mod.MODELS.keys())),
    default=None,
    help="Base dense model key. Inferred from checkpoint metadata when omitted.",
)
@click.option(
    "--task",
    type=click.Choice(data.TASKS),
    default=None,
    help="Training task, only needed when alias cannot be inferred from metadata.",
)
@click.option(
    "--alias",
    default=None,
    help="Fine-tuned model alias. Defaults to checkpoint metadata output_alias.",
)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--force", is_flag=True, help="Re-encode even if the index exists.")
@click.option("--max-length", type=int, default=None, help="Override max_length.")
@click.option(
    "--tuning-method",
    type=click.Choice(["full", "lora"]),
    default=None,
    help="Override checkpoint tuning method. Defaults to metadata or full.",
)
@click.option("--device", default=None, help="Inference device override.")
@click.option(
    "--fp16/--no-fp16",
    default=None,
    help="Override mixed precision. Defaults to enabled on CUDA only.",
)
def encode_finetuned_dense_cmd(
    checkpoint_dir,
    lang,
    model_key,
    task,
    alias,
    batch_size,
    force,
    max_length,
    tuning_method,
    device,
    fp16,
):
    """Encode the corpus with a fine-tuned dense checkpoint."""
    checkpoint_dir = finetuned_dense_mod.resolve_checkpoint_dir(
        checkpoint_dir,
        model_key=model_key,
        task=task,
        lang=lang,
    )
    result = finetuned_dense_mod.encode_corpus(
        checkpoint_dir,
        lang,
        model_key=model_key,
        task=task,
        alias=alias,
        batch_size=batch_size,
        force=force,
        max_length=max_length,
        tuning_method=tuning_method,
        device=device,
        fp16=fp16,
    )
    verb = "wrote" if result.created else "exists"
    click.echo(
        f"{verb} fine-tuned corpus index alias={result.alias} lang={lang} "
        f"n_docs={result.n_docs} path={result.embeddings_path}"
    )


@cli.command("run-finetuned-dense")
@click.argument(
    "checkpoint_dir",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.argument("task", type=click.Choice(data.TASKS))
@click.argument("lang", type=click.Choice(data.LANGS))
@click.option(
    "--model-key",
    type=click.Choice(sorted(dense_mod.MODELS.keys())),
    default=None,
    help="Base dense model key. Inferred from checkpoint metadata when omitted.",
)
@click.option(
    "--alias",
    default=None,
    help="Fine-tuned model alias. Defaults to checkpoint metadata output_alias.",
)
@click.option("--top-k", type=int, default=1000, show_default=True)
@click.option("--split", default="test", type=click.Choice(["train", "validation", "test"]))
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--max-length", type=int, default=None, help="Override max_length.")
@click.option(
    "--tuning-method",
    type=click.Choice(["full", "lora"]),
    default=None,
    help="Override checkpoint tuning method. Defaults to metadata or full.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="If a valid metrics cell already exists for this alias/task/lang, skip.",
)
@click.option("--device", default=None, help="Inference device override.")
@click.option(
    "--fp16/--no-fp16",
    default=None,
    help="Override mixed precision. Defaults to enabled on CUDA only.",
)
def run_finetuned_dense_cmd(
    checkpoint_dir,
    task,
    lang,
    model_key,
    alias,
    top_k,
    split,
    batch_size,
    max_length,
    tuning_method,
    skip_existing,
    device,
    fp16,
):
    """Run a fine-tuned dense checkpoint and save baseline-compatible outputs."""
    checkpoint_dir = finetuned_dense_mod.resolve_checkpoint_dir(
        checkpoint_dir,
        model_key=model_key,
        task=task,
        lang=lang,
    )
    persist_key = finetuned_dense_mod.resolve_alias(
        checkpoint_dir,
        alias=alias,
        model_key=model_key,
        task=task,
        lang=lang,
        tuning_method=tuning_method,
    )
    if skip_existing and _metrics_cell_complete(persist_key, task, lang):
        click.echo(
            f"skip {persist_key} {task}/{lang}: metrics cell already complete",
            err=True,
        )
        return

    rd = data.load(task, lang, split=split)
    click.echo(
        f"{persist_key} {task}/{lang} | n_queries={len(rd.queries)} | top_k={top_k}",
        err=True,
    )
    run = finetuned_dense_mod.run_search(
        checkpoint_dir,
        lang,
        rd.queries,
        model_key=model_key,
        task=task,
        alias=persist_key,
        top_k=top_k,
        batch_size=batch_size,
        max_length=max_length,
        tuning_method=tuning_method,
        device=device,
        fp16=fp16,
    )
    _persist(run, rd, model_key=persist_key, split=split)


# ---------------------------------------------------------------------------
# Bulk runners
# ---------------------------------------------------------------------------


@cli.command("run-bm25-all")
@click.option("--top-k", type=int, default=1000, show_default=True)
@click.pass_context
def run_bm25_all(ctx, top_k):
    """Run BM25 over all 4 (task, lang) pairs."""
    for task in ("rat2rev", "rev2rev"):
        for lang in ("en", "ja"):
            ctx.invoke(run_bm25, task=task, lang=lang, split="test", top_k=top_k)


@cli.command("run-all")
@click.option("--top-k", type=int, default=1000, show_default=True)
@click.option(
    "--models",
    default="bm25,mdpr,mcontriever,me5-base",
    show_default=True,
    help="Comma-separated model keys to run.",
)
@click.pass_context
def run_all(ctx, top_k, models):
    """Run every requested model on the 4 ReCaRe settings."""
    keys = [m.strip() for m in models.split(",") if m.strip()]
    for k in keys:
        for lang in ("en", "ja"):
            if k != "bm25":
                ctx.invoke(encode_dense, model_key=k, lang=lang, batch_size=64, force=False)
        for task in ("rat2rev", "rev2rev"):
            for lang in ("en", "ja"):
                if k == "bm25":
                    ctx.invoke(run_bm25, task=task, lang=lang, split="test", top_k=top_k)
                else:
                    ctx.invoke(
                        run_dense,
                        model_key=k,
                        task=task,
                        lang=lang,
                        split="test",
                        top_k=top_k,
                        batch_size=64,
                    )


# ---------------------------------------------------------------------------
# Cross-encoder rerankers (Issue #7)
# ---------------------------------------------------------------------------


@cli.command("rerank")
@click.argument("model_key", type=click.Choice(sorted(rerank_mod.MODELS.keys())))
@click.argument("task", type=click.Choice(["rat2rev", "rev2rev"]))
@click.argument("lang", type=click.Choice(["en", "ja"]))
@click.option(
    "--first-stage",
    default="bm25",
    show_default=True,
    help="Which top-K run-file to rerank: 'bm25' (results/intermediate/bm25_top100/) or "
    "a dense model key like 'bge-m3' (results/intermediate/dense_top100/).",
)
@click.option("--split", default="test", type=click.Choice(["train", "validation", "test"]))
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="If a *valid* metrics cell already exists for this (model, task, lang), "
    "skip re-running. Validity is verified by re-parsing the JSON and checking "
    "all required metric keys are present (so a partial/corrupt file does not "
    "trigger a false skip).",
)
@click.option(
    "--limit-queries",
    type=int,
    default=None,
    help="Smoke / debug: only rerank the first N queries (deterministic order).",
)
def rerank(model_key, task, lang, first_stage, split, skip_existing, limit_queries):
    """Rerank a first-stage top-100 run with a cross-encoder and write metrics."""
    persist_key = f"{model_key}+{first_stage}"
    if skip_existing and _metrics_cell_complete(persist_key, task, lang):
        click.echo(
            f"skip {persist_key} {task}/{lang}: metrics cell already complete",
            err=True,
        )
        return
    rd = data.load(task, lang, split=split)
    queries = rd.queries
    candidates = rerank_mod.load_first_stage(first_stage, task, lang)
    if limit_queries is not None and limit_queries > 0:
        queries = dict(list(queries.items())[:limit_queries])
        candidates = {qid: cands for qid, cands in candidates.items() if qid in queries}
    corpus_texts = rerank_mod.load_corpus_texts(lang)
    click.echo(
        f"{model_key} rerank {task}/{lang} | first_stage={first_stage} | "
        f"n_queries={len(queries)}",
        err=True,
    )
    run = rerank_mod.rerank_run(model_key, queries, candidates, corpus_texts)
    _persist(run, rd, model_key=persist_key, split=split)


# ---------------------------------------------------------------------------
# Zero-shot LLM ranking — RankGPT sliding window (Issue #8)
# ---------------------------------------------------------------------------


@cli.command("expand-queries")
@click.argument(
    "prompt_family",
    type=click.Choice(["q2d_zs", "q2e_zs", "task_q2e"]),
)
@click.argument(
    "model_slug",
    type=click.Choice(sorted(expansion_mod.MODELS.keys())),
)
@click.argument("task", type=click.Choice(["rat2rev", "rev2rev"]))
@click.argument("lang", type=click.Choice(["en", "ja"]))
@click.option("--max-tokens", default=512, show_default=True, type=int)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--concurrency", default=16, show_default=True, type=int)
@click.option(
    "--no-skip-existing",
    is_flag=True,
    default=False,
    help="Re-generate even if expansion JSONL already has the qid.",
)
def expand_queries_cmd(
    prompt_family, model_slug, task, lang, max_tokens, temperature, concurrency,
    no_skip_existing,
):
    """Generate query-side expansions and persist to data/expansion/.

    The output JSONL is the *only* contract between this command and the
    downstream retrieval steps. After this completes, retrieval CLIs read
    expansion text from disk (no further LLM calls).
    """
    rd = data.load(task, lang, split="test")
    article_pairs = None
    if prompt_family == "task_q2e" and task == "rev2rev":
        click.echo(f"loading metadata-{lang} for article_before/after lookup", err=True)
        article_pairs = expansion_mod.load_rev2rev_article_pairs(lang)
    expansion_mod.run_cell(
        prompt_family=prompt_family,
        model_slug=model_slug,
        task=task,
        lang=lang,
        queries=rd.queries,
        article_pairs=article_pairs,
        max_tokens=max_tokens,
        temperature=temperature,
        concurrency=concurrency,
        skip_existing=not no_skip_existing,
    )


@cli.command("rankgpt-cost")
@click.option("--model", default="gpt-4.1-mini-2025-04-14", show_default=True)
@click.option("--window", default=20, show_default=True, type=int)
@click.option("--stride", default=10, show_default=True, type=int)
@click.option("--top-k", default=100, show_default=True, type=int)
def rankgpt_cost(model, window, stride, top_k):
    """Estimate USD cost to rerank all 4 ReCaRe cells with RankGPT (BM25 top-100 input)."""
    cfg = rankgpt_mod.RankGPTConfig(
        model=model, window_size=window, stride=stride, top_k=top_k
    )
    n_queries = {}
    for task in ("rat2rev", "rev2rev"):
        for lang in ("en", "ja"):
            rd = data.load(task, lang, split="test")
            n_queries[f"{task}-{lang}"] = len(rd.queries)
    est = rankgpt_mod.estimate_cost(n_queries, cfg)
    click.echo(f"Model: {cfg.model}  | window={cfg.window_size} stride={cfg.stride}")
    click.echo(
        f"  calls/query = {est['__calls_per_query__']:.0f}  "
        f"(assuming ~{est['__assumed_input_tokens_per_call__']:.0f} input + "
        f"{est['__assumed_output_tokens_per_call__']:.0f} output tokens / call)"
    )
    click.echo("")
    for cell, n in n_queries.items():
        cost = est[cell]
        click.echo(f"  {cell:<14} | n_queries={n:>4d}  | est cost ≈ ${cost:>6.2f}")
    click.echo(f"  {'TOTAL':<14} |               | est cost ≈ ${est['__total__']:>6.2f}")


@cli.command("rankgpt")
@click.argument("task", type=click.Choice(["rat2rev", "rev2rev"]))
@click.argument("lang", type=click.Choice(["en", "ja"]))
@click.option(
    "--first-stage",
    default="bm25",
    show_default=True,
    help="Which top-100 run-file to rerank ('bm25' = bm25_top100/, "
    "or a dense-model key from dense_top100/).",
)
@click.option("--model", default="gpt-4.1-mini-2025-04-14", show_default=True)
@click.option("--window", default=20, show_default=True, type=int)
@click.option("--stride", default=10, show_default=True, type=int)
@click.option("--top-k", default=100, show_default=True, type=int)
@click.option("--split", default="test", type=click.Choice(["train", "validation", "test"]))
@click.option("--passage-word-cap", default=300, show_default=True, type=int,
              help="Per-passage word-count cap inside the prompt (rank_llm convention).")
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="If a *valid* metrics cell already exists for this (model, task, lang), "
    "skip re-running. Validity is verified by re-parsing the JSON and checking "
    "all required metric keys are present.",
)
@click.option(
    "--limit-queries",
    type=int,
    default=None,
    help="Smoke / debug: only rerank the first N queries (deterministic order). "
    "Useful for capping API spend during sanity checks.",
)
def rankgpt_cmd(
    task, lang, first_stage, model, window, stride, top_k, split, passage_word_cap,
    skip_existing, limit_queries,
):
    """Run RankGPT sliding-window over a first-stage top-K run via rank_llm."""
    persist_key = f"rankgpt-{model}+{first_stage}"
    if skip_existing and _metrics_cell_complete(persist_key, task, lang):
        click.echo(
            f"skip {persist_key} {task}/{lang}: metrics cell already complete",
            err=True,
        )
        return
    cfg = rankgpt_mod.RankGPTConfig(
        model=model,
        window_size=window,
        stride=stride,
        top_k=top_k,
        passage_word_cap=passage_word_cap,
    )
    rd = data.load(task, lang, split=split)
    queries = rd.queries
    candidates = rankgpt_mod.load_first_stage(first_stage, task, lang)
    if limit_queries is not None and limit_queries > 0:
        queries = dict(list(queries.items())[:limit_queries])
        candidates = {qid: cands for qid, cands in candidates.items() if qid in queries}
    corpus_texts = rankgpt_mod.load_corpus_texts(lang)
    click.echo(
        f"RankGPT {task}/{lang} | model={cfg.model} | window={cfg.window_size}/"
        f"stride={cfg.stride} | top_k={cfg.top_k} | first_stage={first_stage} | "
        f"n_queries={len(queries)}",
        err=True,
    )
    run, ledger = rankgpt_mod.run_cell(
        task, lang, queries, candidates, corpus_texts, cfg
    )
    persist_key = f"rankgpt-{cfg.model}+{first_stage}"
    _persist(run, rd, model_key=persist_key, split=split)
    _save_api_ledger(ledger, persist_key, cfg, task, lang, first_stage)


def _save_api_ledger(ledger, persist_key, cfg, task, lang, first_stage):
    ledger_path = METRICS_DIR / f"{persist_key}_{task}_{lang}.api_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "model": cfg.model,
                "task": task,
                "lang": lang,
                "first_stage": first_stage,
                "n_calls": ledger.n_calls,
                "input_tokens": ledger.input_tokens,
                "output_tokens": ledger.output_tokens,
                "cost_usd": round(ledger.cost_usd, 4),
                "per_query_calls": ledger.per_query_calls,
                "config": {
                    "window_size": cfg.window_size,
                    "stride": cfg.stride,
                    "top_k": cfg.top_k,
                    "passage_word_cap": cfg.passage_word_cap,
                    "context_size": cfg.context_size,
                    "azure_api_version": cfg.azure_api_version,
                },
            },
            indent=2,
        )
    )
    click.echo(f"wrote {ledger_path}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@cli.command("aggregate")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=SUMMARY_PATH,
    show_default=True,
)
@click.option(
    "--models",
    default=None,
    help="Comma-separated subset of model keys to include (default: all).",
)
def aggregate(out_path, models):
    """Aggregate per-cell metrics into one JSON keyed by model → cell."""
    keep = None
    if models:
        keep = {m.strip() for m in models.split(",") if m.strip()}
    table: dict[str, dict[str, dict[str, float]]] = {}
    for f in sorted(METRICS_DIR.glob("*.json")):
        # Skip RankGPT API-ledger sidecars — they share the metrics dir but
        # use a different schema (n_calls / cost_usd, not metrics / per_query).
        if f.name.endswith(".api_ledger.json"):
            continue
        cell = json.loads(f.read_text())
        if "metrics" not in cell:
            continue
        model = cell["model"]
        if keep is not None and model not in keep:
            continue
        key = f"{cell['task']}-{cell['lang']}"
        table.setdefault(model, {})[key] = cell["metrics"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2, sort_keys=True))
    click.echo(f"wrote {out_path}")


@cli.command("aggregate-domain-adaptation")
@click.option(
    "--out-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=domain_adaptation_mod.DOMAIN_ADAPTATION_JSON,
    show_default=True,
)
@click.option(
    "--models",
    default=None,
    help="Comma-separated fine-tuned aliases to include (default: all discovered).",
)
def aggregate_domain_adaptation_cmd(out_json, models):
    """Aggregate fine-tuned dense metrics into a JSON report."""
    keep = None
    if models:
        keep = {m.strip() for m in models.split(",") if m.strip()}
    result = domain_adaptation_mod.aggregate_domain_adaptation(
        out_json=out_json,
        models=keep,
    )
    click.echo(f"wrote {result.json_path} | records={result.n_records}")


@cli.command("ttest-holm")
@click.option(
    "--baseline",
    default="bm25",
    show_default=True,
    help="Baseline model to compare every other method against.",
)
@click.option("--alpha", default=0.05, show_default=True, type=float)
def ttest_holm_cmd(baseline, alpha):
    """Paired t-test of every non-baseline method vs ``baseline`` per (task, lang, metric) cell, Holm-corrected across the k methods in that cell.

    Writes ``results/ttest_holm/ttest_holm_all.csv``. Per-query metrics come
    from ``results/metrics/`` (the ``per_query`` field, populated by
    :func:`recare_baselines.eval.evaluate_per_query`).
    """
    from .stats import write_results

    out = write_results(baseline=baseline, alpha=alpha)
    click.echo(f"wrote {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persist(run, rd, *, model_key: str, split: str) -> None:
    """Write run-file (top-1000, post-filter), top-100 (BM25 only), and metrics file.

    For Rev2Rev we apply the ``qid == docid`` post-filter recommended by the
    ReCaRe dataset card — the query article itself appears in the corpus and
    would otherwise be a trivial match.

    Per-query metrics are saved alongside the aggregate to support paired
    significance testing (Tukey HSD) across models.
    """
    runs_dir = INTERMEDIATE_DIR / f"{model_key}_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runfile_path = runs_dir / f"{rd.task}_{rd.lang}.jsonl"
    write_run(runfile_path, run)  # raw, pre-filter run-file (auditable)
    click.echo(f"wrote {runfile_path}", err=True)

    eval_run = filter_self_match(run) if rd.task == "rev2rev" else run

    if model_key == "bm25":
        BM25_TOP100_DIR.mkdir(parents=True, exist_ok=True)
        # Top-100 used by rerankers (#7) and hard-negative miners (#3): persist
        # the post-filtered version so downstream consumers don't re-trip the
        # self-match issue.
        write_run(
            BM25_TOP100_DIR / f"{rd.task}_{rd.lang}.jsonl",
            eval_run,
            top_k=100,
        )
    else:
        # Per issue #6: rerankers (#7, #8) also consume long-context dense top-100.
        DENSE_TOP100_DIR.mkdir(parents=True, exist_ok=True)
        write_run(
            DENSE_TOP100_DIR / f"{rd.task}_{rd.lang}_{model_key}.jsonl",
            eval_run,
            top_k=100,
        )

    # Evaluate only over queries that the run actually covered. For paper-faithful
    # runs this is equivalent to ``rd.qrels`` because every test query is searched;
    # under ``--limit-queries N`` the qrels for the skipped queries are excluded
    # so smoke-test metrics are not diluted by missing queries.
    eval_qrels = {qid: rels for qid, rels in rd.qrels.items() if qid in eval_run}
    metrics = evaluate(eval_run, eval_qrels)
    per_query = evaluate_per_query(eval_run, eval_qrels)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cell = {
        "model": model_key,
        "task": rd.task,
        "lang": rd.lang,
        "split": split,
        "n_queries": len(eval_qrels),
        "n_queries_total": len(rd.qrels),
        "self_match_filtered": rd.task == "rev2rev",
        "metrics": metrics,
        "per_query": per_query,
    }
    metrics_path = METRICS_DIR / f"{model_key}_{rd.task}_{rd.lang}.json"
    metrics_path.write_text(json.dumps(cell, indent=2))
    # Echo summary (drop verbose per-query map for the stdout view).
    click.echo(json.dumps({k: v for k, v in cell.items() if k != "per_query"}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    cli()
