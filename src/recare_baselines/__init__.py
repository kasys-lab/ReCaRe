"""Baselines for ReCaRe (BM25, dense, rerankers, RankGPT, expansion, domain adaptation)."""

import os

__version__ = "0.1.0"

# Several upstream packages (pyserini ≥0.43, rank_llm) import an OpenAI
# client at top-level even when the caller only needs BM25 / dense retrieval
# or cross-encoder reranking. Without ``OPENAI_API_KEY`` in the env, the
# OpenAI SDK raises ``OpenAIError: Missing credentials`` during import and
# blocks unrelated subcommands. We install a placeholder here so that
# non-LLM paths (BM25, dense retrievers, cross-encoder rerankers, training,
# data loading, metrics aggregation) work on a fresh checkout with no keys
# configured. Real Azure OpenAI / OpenAI usage (recare-baselines rankgpt,
# expand-queries) overrides this via .env or explicit env vars.
os.environ.setdefault("OPENAI_API_KEY", "sk-recare-no-llm-stub")
