#!/usr/bin/env bash
# Table 2 BM25 row: indexes both languages and runs all 4 cells
# (rat2rev/rev2rev × en/ja). Requires JAVA_HOME pointing at JDK 21.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${JAVA_HOME:-}" ]]; then
    echo "WARNING: JAVA_HOME is not set. Pyserini requires JDK 21." >&2
    echo "         Set e.g.: export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64" >&2
fi

LOG=results/bm25_runs.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) START" > "$LOG"

stamp() {
    echo "$(date -Is) $*" | tee -a "$LOG"
}

# Build language-specific Lucene indexes (idempotent: skip if present).
for lang in en ja; do
    stamp ">>> build-bm25-index $lang"
    uv run recare-baselines build-bm25-index "$lang" --threads 8 >> "$LOG" 2>&1
    stamp "<<< build-bm25-index $lang"
done

# Search all 4 cells.
for task in rat2rev rev2rev; do
    for lang in en ja; do
        stamp ">>> run-bm25 $task $lang"
        uv run recare-baselines run-bm25 "$task" "$lang" >> "$LOG" 2>&1
        stamp "<<< run-bm25 $task $lang"
    done
done

stamp "DONE"
