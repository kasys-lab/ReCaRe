#!/usr/bin/env bash
# Table 3 RankGPT zero-shot listwise sliding-window reranking on BM25 top-100.
# 3 Azure OpenAI models × 4 cells = 12 runs.
#
# Total Azure cost ≈ $156 USD at the rates that applied during paper writing
# (gpt-4.1-nano $5, gpt-4.1-mini $77, gpt-5.4-mini $74). Run
#   uv run recare-baselines rankgpt-cost --model <id>
# for a fresh estimate.
#
# Required env vars (typically loaded from .env via python-dotenv):
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_VERSION
#
# Prereq: scripts/run_bm25.sh has produced bm25 top-100 runs.

set -uo pipefail
cd "$(dirname "$0")/.."

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
