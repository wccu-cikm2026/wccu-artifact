#!/usr/bin/env bash
# Small live LLM smoke tests. Reads provider/model/API keys from .env.
# Usage: EXP_TAG=wccu_smoke ./scripts/run_wccu_llm_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_smoke_$(date +%Y%m%d_%H%M%S)}"
SMOKE_LIMIT="${WCCU_SMOKE_LIMIT_PER_FAMILY:-2}"
SHARED_SMOKE_REPS="${WCCU_SHARED_SMOKE_REPETITIONS:-1}"
MAX_OUTPUT_TOKENS="${WCCU_MAX_OUTPUT_TOKENS:-1200}"
TIMEOUT_SECONDS="${WCCU_TIMEOUT_SECONDS:-90}"
SHORT_CONDITIONS="${WCCU_SHORT_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_llm_smoke.sh
smoke_limit_per_family=${SMOKE_LIMIT}
shared_smoke_repetitions=${SHARED_SMOKE_REPS}
short_conditions=${SHORT_CONDITIONS}
max_output_tokens=${MAX_OUTPUT_TOKENS}
timeout_seconds=${TIMEOUT_SECONDS}
NOTE

run_logged() {
  local name="$1"
  shift
  local log="logs/${EXP_TAG}_${name}.log"
  echo "[wccu] running ${name}"
  echo "[wccu] command: $*" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
}

make_llm_tables() {
  local json_path="$1"
  local prefix="$2"
  python -m wccu_eval.scripts.make_llm_obligation_tables "$json_path" --out-prefix "$prefix"
}

echo "[wccu] repo: ${REPO_ROOT}"
echo "[wccu] EXP_TAG: ${EXP_TAG}"

run_logged check_llm_provider \
  python -m wccu_eval.scripts.check_llm_provider \
    --scenario high_risk_rule_change \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --error-log "results/${EXP_TAG}/check_llm_provider.errors.jsonl"

run_logged llm_obligation_smoke \
  python -m wccu_eval.eval.run_llm_obligation_benchmark \
    --families freshness,authority,operation,derived_view,witness_gap,safe \
    --limit-per-family "$SMOKE_LIMIT" \
    --repetitions 1 \
    --certificate-guidance unguided \
    --condition "$SHORT_CONDITIONS" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --out "results/${EXP_TAG}/llm_obligation_smoke.json"

make_llm_tables "results/${EXP_TAG}/llm_obligation_smoke.json" "analysis/${EXP_TAG}/llm_obligation_smoke"

run_logged shared_context_smoke \
  python -m wccu_eval.eval.run_shared_context_workload \
    --use-llm \
    --repetitions "$SHARED_SMOKE_REPS" \
    --condition "$SHORT_CONDITIONS" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --out "results/${EXP_TAG}/shared_context_smoke.json"

echo "[wccu] LLM smoke complete for EXP_TAG=${EXP_TAG}"
