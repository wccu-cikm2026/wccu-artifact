#!/usr/bin/env bash
# Offline, no-LLM sanity checks for the WCCU code path.
# Usage:
#   EXP_TAG=wccu_sanity ./scripts/run_wccu_offline_sanity.sh
# Optional:
#   WCCU_SKIP_UNITTEST=1 ./scripts/run_wccu_offline_sanity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_sanity_$(date +%Y%m%d_%H%M%S)}"
SHORT_CONDITIONS="${WCCU_SHORT_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only}"
SKIP_UNITTEST="${WCCU_SKIP_UNITTEST:-0}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

write_note() {
  cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_offline_sanity.sh
short_conditions=${SHORT_CONDITIONS}
skip_unittest=${SKIP_UNITTEST}
NOTE
}

run_logged() {
  local name="$1"
  shift
  local log="logs/${EXP_TAG}_${name}.log"
  echo "[wccu] running ${name}"
  echo "[wccu] command: $*" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
}

write_note

echo "[wccu] repo: ${REPO_ROOT}"
echo "[wccu] EXP_TAG: ${EXP_TAG}"

run_logged compileall python -m compileall wccu_eval tests scripts

if [[ "$SKIP_UNITTEST" != "1" ]]; then
  run_logged unittest python -m unittest discover -s tests -v
else
  echo "[wccu] skipping unittest because WCCU_SKIP_UNITTEST=1"
fi

run_logged mock_multi_model \
  python -m wccu_eval.eval.run_multi_model_obligation_benchmark \
    --mock-llm \
    --families freshness,authority,operation,derived_view,witness_gap,safe \
    --limit-per-family 1 \
    --repetitions 1 \
    --condition "$SHORT_CONDITIONS" \
    --out "results/${EXP_TAG}/mock_multi_model.json"

run_logged mock_shared_context \
  python -m wccu_eval.eval.run_shared_context_workload \
    --repetitions 1 \
    --condition "$SHORT_CONDITIONS" \
    --out "results/${EXP_TAG}/mock_shared_context.json"

run_logged mock_wccu_stress \
  python -m wccu_eval.eval.run_wccu_stress \
    --cases 2 \
    --writers 2 \
    --atom-count 6 \
    --invalidation-prob 0.5 \
    --seed 7 \
    --condition adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only \
    --repetitions 1 \
    --out "results/${EXP_TAG}/mock_wccu_stress.json"

echo "[wccu] offline sanity complete for EXP_TAG=${EXP_TAG}"
