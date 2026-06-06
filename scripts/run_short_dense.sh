#!/usr/bin/env bash
# Table 2 short-context dense block: mDPR, mContriever, mE5-base
# × {en, ja} × {rat2rev, rev2rev}. Encodes corpora once per (model, lang),
# then runs the 12 searches.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=results/short_dense_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" > "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

MODELS=${MODELS:-"mdpr mcontriever me5-base"}
BATCH_SIZE=${BATCH_SIZE:-128}

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
