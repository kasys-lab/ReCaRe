#!/usr/bin/env bash
# Table 2 long-context dense block: BGE-M3, jina-v3 (8192 token cap)
# × {en, ja} × {rat2rev, rev2rev}. GPU strongly recommended; jina-v3 requires
# trust_remote_code=True.

set -euo pipefail
cd "$(dirname "$0")/.."

# jina-v3's 8192-token attention is O(seq^2); large batches OOM (a single
# attention matrix can need >30 GB). expandable_segments reduces allocator
# fragmentation. (Same setting run_expansion.sh uses for jina-v3 encoding.)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG=results/long_dense_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" > "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

MODELS=${MODELS:-"bge-m3 jina-v3"}

# Per-model batch size. jina-v3 OOMs at 16 on a 40 GB GPU, so default it to 4;
# bge-m3 tolerates 16. Override all models with BATCH_SIZE=N (e.g. on a larger
# GPU), which takes precedence over these per-model defaults.
batch_for() {
    if [ -n "${BATCH_SIZE:-}" ]; then echo "$BATCH_SIZE"; return; fi
    case "$1" in
        jina-v3) echo 4 ;;
        *)       echo 16 ;;
    esac
}

# Encode AND search each model before moving to the next. With the previous
# "encode all models, then search all models" layout, an OOM while encoding a
# later model (e.g. jina-v3) tripped `set -e` before the search phase ran, so
# an already-encoded earlier model (bge-m3) was never searched — its committed
# metrics then looked "reproduced" while actually untouched. Per-model ordering
# guarantees each finished model is fully evaluated on disk.
for model in $MODELS; do
    bs=$(batch_for "$model")
    for lang in en ja; do
        stamp ">>> encode $model $lang (batch=$bs)"
        uv run recare-baselines encode-dense "$model" "$lang" \
            --batch-size "$bs" >> "$LOG" 2>&1
        stamp "<<< encode $model $lang"
    done
    for task in rat2rev rev2rev; do
        for lang in en ja; do
            stamp ">>> search $model $task $lang (batch=$bs)"
            uv run recare-baselines run-dense "$model" "$task" "$lang" \
                --batch-size "$bs" >> "$LOG" 2>&1
            stamp "<<< search $model $task $lang"
        done
    done
done

stamp "DONE"
