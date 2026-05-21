#!/usr/bin/env bash
# Main LLM obligation benchmark for paper tables.
# Usage: EXP_TAG=wccu_full WCCU_LIMIT_PER_FAMILY=30 ./scripts/run_wccu_main_llm_obligation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_main_$(date +%Y%m%d_%H%M%S)}"
FAMILIES="${WCCU_FAMILIES:-freshness,commitment,authority,operation,derived_view,witness_gap,safe}"
LIMIT_PER_FAMILY="${WCCU_LIMIT_PER_FAMILY:-30}"
REPETITIONS="${WCCU_REPETITIONS:-1}"
CONDITIONS="${WCCU_MAIN_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_wccu_no_read_validation,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only}"
CERTIFICATE_GUIDANCE="${WCCU_CERTIFICATE_GUIDANCE:-unguided}"
MAX_OUTPUT_TOKENS="${WCCU_MAX_OUTPUT_TOKENS:-1200}"
TIMEOUT_SECONDS="${WCCU_TIMEOUT_SECONDS:-90}"
MOCK_LLM="${MOCK_LLM:-0}"
OUT="${WCCU_MAIN_OUT:-results/${EXP_TAG}/llm_obligation_${CERTIFICATE_GUIDANCE}.json}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_main_llm_obligation.sh
families=${FAMILIES}
limit_per_family=${LIMIT_PER_FAMILY}
repetitions=${REPETITIONS}
conditions=${CONDITIONS}
certificate_guidance=${CERTIFICATE_GUIDANCE}
max_output_tokens=${MAX_OUTPUT_TOKENS}
timeout_seconds=${TIMEOUT_SECONDS}
mock_llm=${MOCK_LLM}
NOTE

MOCK_FLAG=()
if [[ "$MOCK_LLM" == "1" ]]; then
  MOCK_FLAG=(--mock-llm)
fi

LOG="logs/${EXP_TAG}_llm_obligation_${CERTIFICATE_GUIDANCE}.log"
echo "[wccu] running main LLM obligation benchmark"
echo "[wccu] output: ${OUT}"

python -m wccu_eval.eval.run_llm_obligation_benchmark \
  --families "$FAMILIES" \
  --limit-per-family "$LIMIT_PER_FAMILY" \
  --repetitions "$REPETITIONS" \
  --condition "$CONDITIONS" \
  --certificate-guidance "$CERTIFICATE_GUIDANCE" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --out "$OUT" \
  "${MOCK_FLAG[@]}" \
  2>&1 | tee "$LOG"

python - "$OUT" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.load(open(path, encoding='utf-8'))
generations = payload.get('generations') or []
results = payload.get('results') or []
failed = sum(1 for r in results if r.get('failed'))
print(f"WCCU LLM obligation summary: generations={len(generations)} results={len(results)} failed_results={failed}")
if not generations:
    raise SystemExit('No successful LLM generations were recorded; inspect logs/errors before using tables.')
PY

python -m wccu_eval.scripts.make_llm_obligation_tables \
  "$OUT" \
  --out-prefix "analysis/${EXP_TAG}/llm_obligation_${CERTIFICATE_GUIDANCE}"

python -m wccu_eval.scripts.make_coordination_metrics_table \
  "$OUT" \
  --out-csv "analysis/${EXP_TAG}/coordination_metrics_llm_obligation_${CERTIFICATE_GUIDANCE}.csv" \
  --out-tex "analysis/${EXP_TAG}/coordination_metrics_llm_obligation_${CERTIFICATE_GUIDANCE}.tex" \
  --full-tex

echo "[wccu] main LLM obligation benchmark complete: ${OUT}"
