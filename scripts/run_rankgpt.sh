#!/usr/bin/env bash
# Table 3 RankGPT zero-shot listwise sliding-window reranking on BM25 top-100.
# 3 Azure OpenAI models × 4 cells = 12 runs.
#
# Total Azure cost ≈ $156 USD at the rates that applied during paper writing
# (gpt-4.1-nano $5, gpt-4.1-mini $77, gpt-5.4-mini $74). Run
#   uv run recare-baselines rankgpt-cost --model <id>
# for a fresh estimate.
#
# Required env vars (loaded from .env via python-dotenv, or exported). Either
# the AZURE_OPENAI_* or the OPENAI_* names work:
#   AZURE_OPENAI_API_KEY      (or OPENAI_API_KEY)
#   AZURE_OPENAI_ENDPOINT     (or OPENAI_ENDPOINT)
#   AZURE_OPENAI_API_VERSION  (or OPENAI_API_VERSION; default 2024-10-21)
#
# Prereqs:
#   - scripts/run_bm25.sh has produced bm25 top-100 runs.
#   - rank-llm is installed. It is optional (hard-depends on vLLM, which RankGPT
#     does not use), so install only the minimal set:
#       uv pip install --no-deps rank-llm==0.25.7 dacite ftfy wcwidth msgspec

set -uo pipefail
cd "$(dirname "$0")/.."

# Fail fast with guidance if rank-llm is missing, rather than mid-run.
if ! uv run python -c "import rank_llm" 2>/dev/null; then
    echo "ERROR: rank-llm is not installed. Install the minimal set:" >&2
    echo "  uv pip install --no-deps rank-llm==0.25.7 dacite ftfy wcwidth msgspec" >&2
    exit 1
fi

LOG=results/rankgpt_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" > "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

# Default models match the paper. Override via env:
#   MODELS="gpt-4.1-mini" ./scripts/run_rankgpt.sh
MODELS=${MODELS:-"gpt-4.1-nano gpt-4.1-mini gpt-5.4-mini"}

for model in $MODELS; do
    for task in rat2rev rev2rev; do
        for lang in en ja; do
            stamp ">>> $model $task $lang"
            uv run recare-baselines rankgpt "$task" "$lang" \
                --model "$model" --first-stage bm25 --skip-existing >> "$LOG" 2>&1
            rc=$?
            stamp "<<< $model $task $lang (rc=$rc)"
        done
    done
done

stamp "DONE"
