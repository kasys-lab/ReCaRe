#!/usr/bin/env bash
# Table 3 cross-encoder rerankers on BM25 top-100.
#
# Paper reports 4 rerankers: bge-reranker-v2-m3, jina-reranker-v2,
# qwen3-reranker-4b, qwen3-reranker-8b. The 0.6b model is included here for
# completeness (supplementary, not in paper).
#
# Resumable via --skip-existing (default): completed cells in results/metrics/
# are detected and skipped. Order is small→large so a later OOM still leaves
# smaller models' results on disk.
#
# NOTE: a fresh clone already ships the committed reference metrics, so the
# default run skips every cell ("metrics cell already complete", rc=0) without
# recomputing anything. To actually re-run the rerankers and verify they match
# the committed values, force recomputation with SKIP_EXISTING=0:
#     SKIP_EXISTING=0 bash scripts/run_rerankers.sh
# (cells are recomputed and their metrics files overwritten in place.)
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

# Default: skip cells already on disk (resume-safe). SKIP_EXISTING=0 forces
# recomputation — needed to verify reproduction against the committed metrics.
SKIP_EXISTING=${SKIP_EXISTING:-1}
skip_flag=()
[ "$SKIP_EXISTING" = "1" ] && skip_flag=(--skip-existing)

for model in $MODELS; do
    for task in rat2rev rev2rev; do
        for lang in en ja; do
            stamp ">>> $model $task $lang"
            uv run recare-baselines rerank "$model" "$task" "$lang" \
                --first-stage bm25 "${skip_flag[@]}" >> "$LOG" 2>&1
            rc=$?
            stamp "<<< $model $task $lang (rc=$rc)"
        done
    done
done

stamp "DONE"
