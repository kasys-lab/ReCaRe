"""Zero-shot LLM ranking via RankGPT sliding-window (Issue #8).

Wraps :class:`rank_llm.rerank.listwise.rank_gpt.SafeOpenai` from
``castorini/rank_llm`` so we follow the standard reference implementation
(Sun+ 2023, multi-turn listwise prompt template
``rank_llm/rerank/prompt_templates/rank_gpt_template.yaml``). The protocol:

1. Initialise the candidate list as the first-stage ranking (top-100,
   best→worst).
2. Slide a window of size 20 from end to beginning with stride 10. For each
   window, query the LLM with the multi-turn template and overwrite the
   slice in place. The previous window's tail (10 items) overlaps the next
   window's head, propagating refined orderings up the list.
3. After 9 windows, return the final list.

Backend: Azure OpenAI by default (``OPENAI_ENDPOINT`` from ``.env``). Falls
back to public OpenAI when no endpoint is set. ``rank_llm`` configures the
legacy module-level ``openai.api_type/base/version/key`` knobs internally;
we just pass the right values.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens; update if OpenAI changes)
# ---------------------------------------------------------------------------

PRICE_PER_M = {
    # GPT-4.1 family (April 2025)
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    # GPT-5 family (placeholders — update when official prices land)
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-mini-2026-03-17": {"input": 0.25, "output": 2.00},
    # Older / cheaper
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class RankGPTConfig:
    model: str = "gpt-4.1-mini-2025-04-14"
    context_size: int = 128_000  # gpt-4.1 family supports 128K
    window_size: int = 20
    stride: int = 10
    top_k: int = 100
    # Per-passage word-count cap inside ``_convert_doc_to_prompt_content``.
    # rank_llm uses ``300 * (window_size // window_len)`` by default; setting
    # it explicitly here lets us match the published Sun+ 2023 setup.
    passage_word_cap: int = 300
    # Azure API version. Required for our deployments.
    azure_api_version: str = "2024-10-21"


@dataclass
class TokenLedger:
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    per_query_calls: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, in_tok: int, out_tok: int, qid: str | None = None):
        self.n_calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        prices = PRICE_PER_M.get(model) or {"input": 0.0, "output": 0.0}
        self.cost_usd += in_tok * prices["input"] / 1e6 + out_tok * prices["output"] / 1e6
        if qid:
            self.per_query_calls[qid] = self.per_query_calls.get(qid, 0) + 1


# ---------------------------------------------------------------------------
# .env loading + Azure config detection
# ---------------------------------------------------------------------------


def _load_azure_config() -> tuple[str, str | None]:
    """Return ``(api_key, azure_endpoint or None)`` after loading ``.env``.

    Also normalises env vars for ``openai>=2.x``: that SDK ignores the legacy
    module-level ``openai.api_type/api_base`` knobs and only consults the new
    env vars (``AZURE_OPENAI_ENDPOINT``, ``OPENAI_API_TYPE`` etc.) when a
    client is auto-instantiated. We populate them here so rank_llm's
    ``openai.chat.completions.create()`` calls hit the right endpoint.
    """
    try:
        from dotenv import load_dotenv

        repo_root = Path(__file__).resolve().parents[2]
        load_dotenv(repo_root / ".env", override=False)
    except ImportError:
        pass
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Put it in $REPO_ROOT/.env (gitignored) "
            "or export it directly in the shell."
        )
    endpoint = os.environ.get("OPENAI_ENDPOINT") or os.environ.get(
        "AZURE_OPENAI_ENDPOINT"
    )
    if endpoint:
        # openai>=2.x reads these on lazy client construction.
        os.environ.setdefault("AZURE_OPENAI_ENDPOINT", endpoint)
        os.environ.setdefault("AZURE_OPENAI_API_KEY", api_key)
        os.environ.setdefault("OPENAI_API_TYPE", "azure")
        os.environ.setdefault(
            "OPENAI_API_VERSION", os.environ.get("OPENAI_API_VERSION", "2024-10-21")
        )
    return api_key, endpoint


# ---------------------------------------------------------------------------
# rank_llm wrapper
# ---------------------------------------------------------------------------


def _stub_vllm_dependent_modules():
    """Pre-stub ``rank_listwise_os_llm`` (and friends) so that importing
    ``rank_llm.rerank.listwise.__init__`` does not pull in ``vllm._C``.

    ``rank_llm`` ships with a vLLM-backed local-LLM reranker
    (``RankListwiseOSLLM``) that imports ``vllm`` at module top level.
    Our installed ``vllm`` wheel is built against CUDA 13 while the rest
    of the venv targets CUDA 12.6, so ``import vllm._C`` raises
    ``ImportError: libcudart.so.13``. We do not use vLLM (we use the
    Azure / OpenAI ``SafeOpenai`` path), so we register tiny stubs in
    ``sys.modules`` *before* the eager ``__init__.py`` import chain runs.
    """
    import sys
    import types

    targets = {
        # vLLM (cu13 wheel mismatch)
        "rank_llm.rerank.vllm_handler": ("VllmHandler",),
        "rank_llm.rerank.listwise.rank_listwise_os_llm": ("RankListwiseOSLLM",),
        "rank_llm.rerank.listwise.vicuna_reranker": ("VicunaReranker",),
        "rank_llm.rerank.listwise.zephyr_reranker": ("ZephyrReranker",),
        "rank_llm.rerank.listwise.lit5_reranker": ("Lit5Reranker",),
        # transformers-T5 (rank_llm vendors a copy that is incompatible with
        # transformers 4.57+; we don't need T5/FiD/MonoT5/DuoT5).
        "rank_llm.rerank.listwise.rank_fid": ("RankFiDDistill", "RankFiDScore"),
        "rank_llm.rerank.listwise.lit5": (),
        "rank_llm.rerank.listwise.lit5.model": ("FiD", "FiDCrossAttentionScore"),
        "rank_llm.rerank.listwise.lit5.modeling_t5": (),
        "rank_llm.rerank.pointwise.monot5": ("MonoT5",),
        "rank_llm.rerank.pairwise.duot5": ("DuoT5",),
    }
    for name, syms in targets.items():
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        for sym in syms:
            setattr(m, sym, type(sym, (), {}))
        sys.modules[name] = m


def _make_reranker(cfg: RankGPTConfig):
    """Instantiate rank_llm's ``SafeOpenai`` with our Azure / OpenAI config."""
    api_key, endpoint = _load_azure_config()
    _stub_vllm_dependent_modules()

    # Lazy import — rank_llm pulls in vLLM / SGLang adapters at import time.
    from rank_llm.rerank.listwise.rank_gpt import SafeOpenai
    from rank_llm.rerank.rankllm import PromptMode

    kwargs = dict(
        model=cfg.model,
        context_size=cfg.context_size,
        prompt_mode=PromptMode.RANK_GPT,
        window_size=cfg.window_size,
        stride=cfg.stride,
        keys=[api_key],
    )
    if endpoint:
        kwargs.update(
            api_type="azure",
            api_base=endpoint,
            api_version=cfg.azure_api_version,
        )
    return SafeOpenai(**kwargs)


def _build_requests(
    queries: dict[str, str],
    candidates: dict[str, list[tuple[str, float]]],
    corpus_texts: dict[str, str],
    top_k: int,
):
    """Convert our (queries, candidates, corpus) into a ``list[rank_llm.data.Request]``."""
    from rank_llm.data import Candidate, Query, Request

    requests = []
    for qid, qtext in queries.items():
        cand = candidates.get(qid) or []
        cand = cand[:top_k]
        rl_candidates = [
            Candidate(
                docid=did,
                score=float(score),
                doc={"text": corpus_texts.get(did, "")},
            )
            for did, score in cand
        ]
        requests.append(
            Request(query=Query(text=qtext, qid=qid), candidates=rl_candidates)
        )
    return requests


def _ledger_from_results(model: str, results) -> TokenLedger:
    """Sum invocation token counts in a ``list[rank_llm.data.Result]``."""
    ledger = TokenLedger()
    for r in results:
        qid = r.query.qid
        history = r.invocations_history or []
        for inv in history:
            in_tok = int(getattr(inv, "input_token_count", 0) or 0)
            out_tok = int(getattr(inv, "output_token_count", 0) or 0)
            ledger.add(model, in_tok, out_tok, qid=str(qid))
    return ledger


def run_cell(
    task: str,
    lang: str,
    queries: dict[str, str],
    candidates: dict[str, list[tuple[str, float]]],
    corpus_texts: dict[str, str],
    cfg: RankGPTConfig,
) -> tuple[dict[str, list[tuple[str, float]]], TokenLedger]:
    """Sliding-window rerank one cell via rank_llm's ``SafeOpenai.rerank_batch``.

    Returns ``({qid: [(doc_id, synthetic_score)]}, ledger)`` matching the rest
    of the baseline pipeline (synthetic score = ``-rank`` so descending sort
    preserves order).
    """
    reranker = _make_reranker(cfg)
    requests = _build_requests(queries, candidates, corpus_texts, cfg.top_k)
    logger.info(
        "Running rank_llm SafeOpenai: %d requests, model=%s, window=%d, stride=%d, top_k=%d",
        len(requests),
        cfg.model,
        cfg.window_size,
        cfg.stride,
        cfg.top_k,
    )
    results = reranker.rerank_batch(
        requests,
        rank_start=0,
        rank_end=cfg.top_k,
        populate_invocations_history=True,
    )
    ledger = _ledger_from_results(cfg.model, results)

    out: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        qid = str(r.query.qid)
        out[qid] = [
            (str(c.docid), -float(rank))
            for rank, c in enumerate(r.candidates[: cfg.top_k])
        ]
    return out, ledger


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


def _sliding_windows(top_k: int, window: int, stride: int) -> list[int]:
    if window > top_k:
        return [0]
    starts = list(range(0, top_k - window + 1, stride))
    if starts[-1] != top_k - window:
        starts.append(top_k - window)
    return list(reversed(starts))


def estimate_cost(
    n_queries_per_cell: dict[str, int],
    cfg: RankGPTConfig,
    *,
    avg_input_tokens_per_call: int = 10_000,
    avg_output_tokens_per_call: int = 100,
) -> dict[str, float]:
    """Pre-run cost estimate (USD) per cell + total.

    Defaults assume 9 sliding-window calls per query at ~10K input tokens each
    (multi-turn rank_llm prompt, 20 passages × ~300 tokens body + overhead)
    plus ~100 output tokens (the rank list).
    """
    prices = PRICE_PER_M.get(cfg.model) or {"input": 0.40, "output": 1.60}
    calls_per_q = len(_sliding_windows(cfg.top_k, cfg.window_size, cfg.stride))
    out: dict[str, float] = {}
    total = 0.0
    for cell, n_q in n_queries_per_cell.items():
        in_t = n_q * calls_per_q * avg_input_tokens_per_call
        out_t = n_q * calls_per_q * avg_output_tokens_per_call
        cost = in_t * prices["input"] / 1e6 + out_t * prices["output"] / 1e6
        out[cell] = cost
        total += cost
    out["__total__"] = total
    out["__calls_per_query__"] = float(calls_per_q)
    out["__assumed_input_tokens_per_call__"] = float(avg_input_tokens_per_call)
    out["__assumed_output_tokens_per_call__"] = float(avg_output_tokens_per_call)
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_first_stage(
    first_stage: str, task: str, lang: str
) -> dict[str, list[tuple[str, float]]]:
    repo_root = Path(__file__).resolve().parents[2]
    if first_stage == "bm25":
        path = (
            repo_root
            / "results"
            / "intermediate"
            / "bm25_top100"
            / f"{task}_{lang}.jsonl"
        )
    else:
        path = (
            repo_root
            / "results"
            / "intermediate"
            / "dense_top100"
            / f"{task}_{lang}_{first_stage}.jsonl"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, list[tuple[str, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["qid"]] = [(d, float(s)) for d, s in rec["results"]]
    return out


def load_corpus_texts(lang: str) -> dict[str, str]:
    from . import data

    path = data._hf_path(f"corpus-{lang}/corpus.jsonl")
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["_id"]] = rec.get("text", "")
    return out
