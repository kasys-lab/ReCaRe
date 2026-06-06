#!/usr/bin/env bash
# Table 2 long-context dense block: BGE-M3, jina-v3 (8192 token cap)
# × {en, ja} × {rat2rev, rev2rev}. GPU strongly recommended; jina-v3 requires
# trust_remote_code=True.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=results/long_dense_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" > "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

MODELS=${MODELS:-"bge-m3 jina-v3"}
BATCH_SIZE=${BATCH_SIZE:-16}

for model in $MODELS; do
    for lang in en ja; do
        stamp ">>> encode $model $lang"
        uv run recare-baselines encode-dense "$model" "$lang" \
            --batch-size "$BATCH_SIZE" >> "$LOG" 2>&1
        stamp "<<< encode $model $lang"
    done
done

for model in $MODELS; do
    for task in rat2rev rev2rev; do
        for lang in en ja; do
            stamp ">>> search $model $task $lang"
            uv run recare-baselines run-dense "$model" "$task" "$lang" \
                --batch-size "$BATCH_SIZE" >> "$LOG" 2>&1
            stamp "<<< search $model $task $lang"
        done
    done
done

stamp "DONE"
