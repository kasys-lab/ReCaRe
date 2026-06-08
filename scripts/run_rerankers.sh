#!/usr/bin/env bash
# Table 3 cross-encoder rerankers on BM25 top-100.
#
# Paper reports 4 rerankers: bge-reranker-v2-m3, jina-reranker-v2,
# qwen3-reranker-4b, qwen3-reranker-8b. The 0.6b model is included here for
# completeness (supplementary, not in paper).
#
# Each cell is recomputed and its metrics file overwritten in place. Order is
# small→large so a later OOM still leaves smaller models' results on disk.
#
# Prereq: results/intermediate/bm25_top100/{task}_{lang}.jsonl must exist.
# Run scripts/run_bm25.sh first.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=results/rerank_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" >> "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

MODELS=${MODELS:-"bge-reranker-v2-m3 jina-reranker-v2 \
                  qwen3-reranker-0.6b qwen3-reranker-4b qwen3-reranker-8b"}

for model in $MODELS; do
    for task in rat2rev rev2rev; do
        for lang in en ja; do
            stamp ">>> $model $task $lang"
            uv run recare-baselines rerank "$model" "$task" "$lang" \
                --first-stage bm25 >> "$LOG" 2>&1
            rc=$?
            stamp "<<< $model $task $lang (rc=$rc)"
        done
    done
done

stamp "DONE"
