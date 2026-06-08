#!/usr/bin/env bash
# Table 2 long-context dense block: BGE-M3, jina-v3 (8192 token cap)
# × {en, ja} × {rat2rev, rev2rev}. GPU strongly recommended; jina-v3 requires
# trust_remote_code=True.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG=results/long_dense_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" > "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

MODELS=${MODELS:-"bge-m3 jina-v3"}

# Per-model batch size: jina-v3's 8192-token O(seq^2) attention OOMs at 16 on a
# 40 GB GPU, so default it to 4. BATCH_SIZE=N overrides all models.
batch_for() {
    if [ -n "${BATCH_SIZE:-}" ]; then echo "$BATCH_SIZE"; return; fi
    case "$1" in
        jina-v3) echo 4 ;;
        *)       echo 16 ;;
    esac
}

# Encode and search each model before the next, so a later model's OOM does not
# leave an earlier (already-encoded) model unsearched.
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
