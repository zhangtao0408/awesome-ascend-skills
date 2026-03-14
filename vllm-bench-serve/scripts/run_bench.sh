#!/usr/bin/env bash
# run_bench.sh — Execute a single vllm bench serve command with error handling
# Usage: ./run_bench.sh "full vllm bench serve command"
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"vllm bench serve --base-url ... --model ... [options]\""
  exit 1
fi

BENCH_CMD="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Extract result-dir from command if present
RESULT_DIR=$(echo "$BENCH_CMD" | grep -oP '(?<=--result-dir\s)\S+' || echo "./bench_results")

# Create result directory
mkdir -p "$RESULT_DIR"

# Create log directory
LOG_DIR="${RESULT_DIR%/}/../logs"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp/bench_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bench_${TIMESTAMP}.log"

echo "=== Benchmark Execution ==="
echo "  Time:    $(date)"
echo "  Log:     $LOG_FILE"
echo "  Results: $RESULT_DIR"
echo ""
echo "  Command:"
echo "    $BENCH_CMD"
echo ""

# Execute benchmark
EXIT_CODE=0
eval "$BENCH_CMD" 2>&1 | tee "$LOG_FILE" || EXIT_CODE=$?

echo ""
echo "=== Execution Complete ==="
echo "  Exit code: $EXIT_CODE"
echo "  Log: $LOG_FILE"

# Check for result files
if [[ $EXIT_CODE -eq 0 ]]; then
  RESULT_FILES=$(find "$RESULT_DIR" -name "*.json" -newer "$LOG_FILE" 2>/dev/null | head -5)
  if [[ -n "$RESULT_FILES" ]]; then
    echo "  Result files:"
    echo "$RESULT_FILES" | while read -r f; do echo "    $f"; done
  else
    echo "  Warning: No new result JSON files found in $RESULT_DIR"
  fi
else
  echo "  ERROR: Benchmark failed with exit code $EXIT_CODE"
  echo "  Check log for details: $LOG_FILE"
fi

exit $EXIT_CODE
