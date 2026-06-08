#!/usr/bin/env bash
# Table 4 domain adaptation pipeline: hard-neg mining → train → evaluate.
#
# Three phases (toggle with PHASES env var, default all three):
#   Phase 1: build dense top-100 on train/validation splits (for hard-neg pool)
#   Phase 2: train (5 models × 4 cells). Short-context models (mdpr, mcontriever,
#            me5-base) use full FT; long-context (bge-m3, jina-v3) use LoRA.
#   Phase 3: encode the fine-tuned corpus and evaluate on test, then aggregate.
#
# Env overrides:
#   MODELS    (default: "mdpr mcontriever me5-base bge-m3 jina-v3")
#   TASKS     (default: "rat2rev rev2rev")
#   LANGS     (default: "en ja")
#   PHASES    (default: "1 2 3")
#   BATCH_SIZE, LR, EPOCHS, PATIENCE, SEED — passed to train-dense

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=${LOG:-results/domain_adaptation_runs.log}
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" >> "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

MODELS=${MODELS:-"mdpr mcontriever me5-base bge-m3 jina-v3"}
TASKS=${TASKS:-"rat2rev rev2rev"}
LANGS=${LANGS:-"en ja"}
PHASES=${PHASES:-"1 2 3"}

SHORT_BATCH_SIZE=${SHORT_BATCH_SIZE:-64}
LONG_BATCH_SIZE=${LONG_BATCH_SIZE:-8}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-3}
SEED=${SEED:-13}
TEMPERATURE=${TEMPERATURE:-0.05}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
TOP_K=${TOP_K:-1000}

is_long_context() {
    case "$1" in
        bge-m3|jina-v3) return 0 ;;
        *) return 1 ;;
    esac
}

# -------------------------------------------------------------- Phase 1
if [[ " $PHASES " == *" 1 "* ]]; then
    for model in $MODELS; do
        for task in $TASKS; do
            for lang in $LANGS; do
                for split in train validation; do
                    stamp ">>> build-dense-top100 $model $task $lang ($split)"
                    uv run recare-baselines build-dense-top100 \
                        "$model" "$task" "$lang" --split "$split" \
                        --batch-size "$EVAL_BATCH_SIZE" >> "$LOG" 2>&1
                    stamp "<<< build-dense-top100 $model $task $lang ($split)"
                done
            done
        done
    done
fi

# -------------------------------------------------------------- Phase 2 (train)
if [[ " $PHASES " == *" 2 "* ]]; then
    for model in $MODELS; do
        for task in $TASKS; do
            for lang in $LANGS; do
                if is_long_context "$model"; then
                    tuning=lora
                    bs="$LONG_BATCH_SIZE"
                else
                    tuning=full
                    bs="$SHORT_BATCH_SIZE"
                fi
                stamp ">>> train $model $task $lang ($tuning, bs=$bs)"
                uv run recare-baselines train-dense "$model" "$task" "$lang" \
                    --batch-size "$bs" \
                    --lr "$LR" --warmup-ratio 0.1 \
                    --epochs "$EPOCHS" --patience "$PATIENCE" \
                    --seed "$SEED" --temperature "$TEMPERATURE" \
                    --tuning-method "$tuning" >> "$LOG" 2>&1
                stamp "<<< train $model $task $lang ($tuning)"
            done
        done
    done
fi

# -------------------------------------------------------------- Phase 3 (eval)
if [[ " $PHASES " == *" 3 "* ]]; then
    for model in $MODELS; do
        for task in $TASKS; do
            for lang in $LANGS; do
                ckpt="results/dense_finetune/${model}/${task}_${lang}/best"
                if [[ ! -d "$ckpt" ]]; then
                    # No locally-trained checkpoint — fetch the released one from
                    # HF (kasys/ReCaRe-domain-adaptation) so eval can run without
                    # re-training. Set FETCH=0 to disable.
                    if [[ "${FETCH:-1}" == "1" ]]; then
                        stamp ">>> fetch-finetuned $model $task $lang"
                        uv run recare-baselines fetch-finetuned \
                            --model "$model" --task "$task" --lang "$lang" >> "$LOG" 2>&1
                    fi
                fi
                if [[ ! -d "$ckpt" ]]; then
                    stamp "skip $model $task $lang (no checkpoint at $ckpt)"
                    continue
                fi
                if is_long_context "$model"; then
                    tuning=lora
                    alias="${model}-lora-ft-${task}-${lang}"
                else
                    tuning=full
                    alias="${model}-ft-${task}-${lang}"
                fi
                stamp ">>> encode-finetuned $alias"
                uv run recare-baselines encode-finetuned-dense "$ckpt" \
                    --lang "$lang" --model-key "$model" --task "$task" \
                    --alias "$alias" --tuning-method "$tuning" \
                    --batch-size "$EVAL_BATCH_SIZE" >> "$LOG" 2>&1

                stamp ">>> run-finetuned $alias"
                uv run recare-baselines run-finetuned-dense "$ckpt" "$task" "$lang" \
                    --model-key "$model" --top-k "$TOP_K" --split test \
                    --alias "$alias" --tuning-method "$tuning" \
                    --batch-size "$EVAL_BATCH_SIZE" >> "$LOG" 2>&1
            done
        done
    done

    stamp ">>> aggregate-domain-adaptation"
    uv run recare-baselines aggregate-domain-adaptation \
        --out-json results/domain_adaptation.json >> "$LOG" 2>&1
    stamp "<<< aggregate-domain-adaptation"
fi

stamp "DONE"
