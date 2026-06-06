#!/usr/bin/env bash
# Table 2 augmentation block: query/document expansion on BM25 and jina-v3.
#
# Three phases:
#   Phase 1 (query side): Generate Q2D / Q2E / task-Q2E expansions via Azure
#                         OpenAI + Qwen3.5-9B (vLLM). Outputs to
#                         data/expansion/<family>/<model>_<task>_<lang>_test.jsonl
#                         Resume-safe; already-generated qids are skipped.
#   Phase 2 (doc side): Build augmented BM25 indexes and jina-v3 embeddings
#                       from the pre-released d2q / d2e doc-level outputs
#                       (hosted on HF Datasets at kasys/ReCaRe-expansions —
#                       see docs/data_format.md for download).
#   Phase 3 (evaluate): Run 64 augmented retrieval cells (32 BM25 + 32 jina-v3).
#
# Required for Phase 1 (LLM generation):
#   AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
#   (for gpt-4.1-mini); QWEN_API_BASE (for self-hosted Qwen3.5-9B).
#
# Phase 1 can be SKIPPED if you reuse the test-split JSONLs shipped in
# data/expansion/ (only Azure-generated; Qwen requires re-generation).

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=results/expansion_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" >> "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

PHASES=${PHASES:-"1 2 3"}
TASKS=(rat2rev rev2rev)
LANGS=(en ja)
DOC_AUGS=(d2q d2e)
QUERY_FAMILIES=(q2d_zs q2e_zs task_q2e)
QUERY_MODELS=(gpt-4.1-mini qwen3.5-9b)
JINA_QUERY_BS=${JINA_QUERY_BS:-8}

# -------------------------------------------------------------- Phase 1
if [[ " $PHASES " == *" 1 "* ]]; then
    for family in "${QUERY_FAMILIES[@]}"; do
        for model in "${QUERY_MODELS[@]}"; do
            for task in "${TASKS[@]}"; do
                for lang in "${LANGS[@]}"; do
                    stamp ">>> expand-queries $family $model $task $lang"
                    uv run recare-baselines expand-queries \
                        "$family" "$model" "$task" "$lang" \
                        --concurrency 8 >> "$LOG" 2>&1
                    stamp "<<< expand-queries $family $model $task $lang (rc=$?)"
                done
            done
        done
    done
fi

# -------------------------------------------------------------- Phase 2 (BM25 doc-aug indexes)
# Requires data/recare_d2q/ and data/recare_d2e/ to exist (download from HF —
# see docs/data_format.md).
if [[ " $PHASES " == *" 2 "* ]]; then
    declare -a JOBS=(
        "en d2q data/recare_d2q/recare_en_d2q_documents.jsonl"
        "ja d2q data/recare_d2q/recare_ja_d2q_documents.jsonl"
        "en d2e data/recare_d2e/recare_en_d2e_documents.jsonl"
        "ja d2e data/recare_d2e/recare_ja_d2e_documents.jsonl"
    )
    for spec in "${JOBS[@]}"; do
        read -r lang suffix path <<< "$spec"
        if [[ ! -f "$path" ]]; then
            stamp "skip $lang+$suffix: $path not found (download from HF first)"
            continue
        fi
        stamp ">>> bm25 index $lang+$suffix"
        uv run recare-baselines build-bm25-index "$lang" \
            --doc-augmentation "$path" --aug-suffix "$suffix" \
            --threads 8 >> "$LOG" 2>&1
        stamp "<<< bm25 index $lang+$suffix (rc=$?)"

        stamp ">>> jina-v3 encode $lang+$suffix"
        uv run recare-baselines encode-dense jina-v3 "$lang" \
            --doc-augmentation "$path" --aug-suffix "$suffix" \
            --batch-size 4 >> "$LOG" 2>&1
        stamp "<<< jina-v3 encode $lang+$suffix (rc=$?)"
    done
fi

# -------------------------------------------------------------- Phase 3 (evaluate)
if [[ " $PHASES " == *" 3 "* ]]; then
    # BM25 doc-aug
    for aug in "${DOC_AUGS[@]}"; do
        for task in "${TASKS[@]}"; do
            for lang in "${LANGS[@]}"; do
                stamp ">>> bm25+$aug $task/$lang"
                uv run recare-baselines run-bm25 "$task" "$lang" \
                    --doc-aug-suffix "$aug" --skip-existing >> "$LOG" 2>&1
                stamp "<<< bm25+$aug $task/$lang (rc=$?)"
            done
        done
    done
    # BM25 query-aug
    for fam in "${QUERY_FAMILIES[@]}"; do
        for model in "${QUERY_MODELS[@]}"; do
            for task in "${TASKS[@]}"; do
                for lang in "${LANGS[@]}"; do
                    path="data/expansion/$fam/${model}_${task}_${lang}_test.jsonl"
                    stamp ">>> bm25+$fam.$model $task/$lang"
                    uv run recare-baselines run-bm25 "$task" "$lang" \
                        --query-augmentation "$path" --skip-existing >> "$LOG" 2>&1
                    stamp "<<< bm25+$fam.$model $task/$lang (rc=$?)"
                done
            done
        done
    done
    # jina-v3 doc-aug
    for aug in "${DOC_AUGS[@]}"; do
        for task in "${TASKS[@]}"; do
            for lang in "${LANGS[@]}"; do
                stamp ">>> jina-v3+$aug $task/$lang"
                uv run recare-baselines run-dense jina-v3 "$task" "$lang" \
                    --doc-aug-suffix "$aug" --batch-size "$JINA_QUERY_BS" \
                    --skip-existing >> "$LOG" 2>&1
                stamp "<<< jina-v3+$aug $task/$lang (rc=$?)"
            done
        done
    done
    # jina-v3 query-aug
    for fam in "${QUERY_FAMILIES[@]}"; do
        for model in "${QUERY_MODELS[@]}"; do
            for task in "${TASKS[@]}"; do
                for lang in "${LANGS[@]}"; do
                    path="data/expansion/$fam/${model}_${task}_${lang}_test.jsonl"
                    stamp ">>> jina-v3+$fam.$model $task/$lang"
                    uv run recare-baselines run-dense jina-v3 "$task" "$lang" \
                        --query-augmentation "$path" --batch-size "$JINA_QUERY_BS" \
                        --skip-existing >> "$LOG" 2>&1
                    stamp "<<< jina-v3+$fam.$model $task/$lang (rc=$?)"
                done
            done
        done
    done
fi

stamp "DONE"
