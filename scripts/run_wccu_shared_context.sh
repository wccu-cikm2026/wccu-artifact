#!/usr/bin/env bash
# Shared-context workload. Use WCCU_USE_LLM=0 for deterministic/offline mode.
# Usage: EXP_TAG=wccu_shared WCCU_SHARED_REPETITIONS=10 ./scripts/run_wccu_shared_context.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_shared_$(date +%Y%m%d_%H%M%S)}"
USE_LLM="${WCCU_USE_LLM:-1}"
REPETITIONS="${WCCU_SHARED_REPETITIONS:-10}"
CONDITIONS="${WCCU_SHORT_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only}"
MAX_OUTPUT_TOKENS="${WCCU_MAX_OUTPUT_TOKENS:-1200}"
TIMEOUT_SECONDS="${WCCU_TIMEOUT_SECONDS:-90}"
OUT="results/${EXP_TAG}/shared_context_workload.json"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_shared_context.sh
use_llm=${USE_LLM}
repetitions=${REPETITIONS}
conditions=${CONDITIONS}
max_output_tokens=${MAX_OUTPUT_TOKENS}
timeout_seconds=${TIMEOUT_SECONDS}
NOTE

LLM_FLAG=()
NAME="shared_context_workload"
if [[ "$USE_LLM" != "0" ]]; then
  LLM_FLAG=(--use-llm)
  NAME="shared_context_workload_llm"
fi

LOG="logs/${EXP_TAG}_${NAME}.log"
echo "[wccu] running shared-context workload"
python -m wccu_eval.eval.run_shared_context_workload \
  --repetitions "$REPETITIONS" \
  --condition "$CONDITIONS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --out "$OUT" \
  "${LLM_FLAG[@]}" \
  2>&1 | tee "$LOG"

python - "$OUT" "analysis/${EXP_TAG}/shared_context_summary.csv" <<'PY'
import csv, json, sys
from pathlib import Path
src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
payload = json.loads(src.read_text(encoding='utf-8'))
rows = payload.get('aggregated', [])
keys = sorted({k for r in rows for k in r})
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"[wccu] wrote {out}")
PY

echo "[wccu] shared-context workload complete: ${OUT}"
