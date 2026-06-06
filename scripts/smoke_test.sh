#!/usr/bin/env bash
# CPU-friendly smoke test (no GPU, no API keys required).
#
# What it runs:
#   1. BM25 on the full corpora, 5 queries per cell × 4 cells = 4 runs.
#   2. me5-small dense (118M params, 384-dim, multilingual) on the full
#      corpora, 5 queries per cell × 4 cells = 4 runs.
#   3. bge-reranker-v2-m3 on BM25 top-100, 5 queries per cell × 4 cells = 4
#      runs. (Step 3 is SKIPPED by default on machines without GPU; pass
#      RUN_RERANK=1 to force it.)
#
# Expected runtime on a 2024-era 8-core x86_64 laptop with 16 GB RAM:
#   - BM25 indexing (one-time): ~1 min/lang
#   - me5-small corpus encoding (one-time): ~3 min/lang
#   - All searches: ~30 sec total
#   Grand total: ~10-15 min the first run, ~1 min on a warm cache.
#
# This is NOT a paper-faithful evaluation: it uses an unregistered smoke-tier
# model (me5-small) and only 5 queries. Recall@100 / nDCG@10 numbers will
# differ from Tables 2/3. The goal is verifying the install end-to-end.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${JAVA_HOME:-}" ]]; then
    echo "WARNING: JAVA_HOME is not set. BM25 (Pyserini) requires JDK 21." >&2
fi

LIMIT_QUERIES=${LIMIT_QUERIES:-5}
RUN_RERANK=${RUN_RERANK:-0}
SMOKE_LOG=results/smoke_test.log
mkdir -p "$(dirname "$SMOKE_LOG")"
echo "$(date -Is) START (limit-queries=$LIMIT_QUERIES, rerank=$RUN_RERANK)" > "$SMOKE_LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$SMOKE_LOG"
}

# Step 1: BM25 indexes + 4 searches
for lang in en ja; do
    stamp ">>> build-bm25-index $lang"
    uv run recare-baselines build-bm25-index "$lang" --threads 4 >> "$SMOKE_LOG" 2>&1
    stamp "<<< build-bm25-index $lang"
done

for task in rat2rev rev2rev; do
    for lang in en ja; do
        stamp ">>> bm25 $task $lang"
        uv run recare-baselines run-bm25 "$task" "$lang" \
            --limit-queries "$LIMIT_QUERIES" >> "$SMOKE_LOG" 2>&1
        stamp "<<< bm25 $task $lang"
    done
done

# Step 2: me5-small dense (smoke-tier model)
for lang in en ja; do
    stamp ">>> encode me5-small $lang"
    uv run recare-baselines encode-dense me5-small "$lang" \
        --batch-size 32 >> "$SMOKE_LOG" 2>&1
    stamp "<<< encode me5-small $lang"
done

for task in rat2rev rev2rev; do
    for lang in en ja; do
        stamp ">>> dense me5-small $task $lang"
        uv run recare-baselines run-dense me5-small "$task" "$lang" \
            --batch-size 32 --limit-queries "$LIMIT_QUERIES" >> "$SMOKE_LOG" 2>&1
        stamp "<<< dense me5-small $task $lang"
    done
done

# Step 3: reranker (skipped by default; heavy for CPU-only)
if [[ "$RUN_RERANK" == "1" ]]; then
    for task in rat2rev rev2rev; do
        for lang in en ja; do
            stamp ">>> rerank bge-reranker-v2-m3 $task $lang"
            uv run recare-baselines rerank bge-reranker-v2-m3 "$task" "$lang" \
                --first-stage bm25 --limit-queries "$LIMIT_QUERIES" >> "$SMOKE_LOG" 2>&1
            stamp "<<< rerank bge-reranker-v2-m3 $task $lang"
        done
    done
fi

stamp "DONE"
echo ""
echo "Smoke test complete. Metrics written to results/metrics/."
echo "Sample (Rat2Rev EU BM25):"
cat results/metrics/bm25_rat2rev_en.json 2>/dev/null || echo "  (no file)"
