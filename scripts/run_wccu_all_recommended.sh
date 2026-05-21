#!/usr/bin/env bash
# Runs the recommended WCCU experiment sequence.
#
# Default for paper-strength evidence: CooperBench-derived live-LLM experiments.
# Controlled obligation, shared-context, stress, and mock/offline diagnostics are
# optional and should be treated as appendix/artifact checks unless explicitly
# reported as such.
#
# Usage:
#   EXP_TAG=wccu_full COOPERBENCH_SNAPSHOT=/path/to/cooperbench ./scripts/run_wccu_all_recommended.sh
#   EXP_TAG=wccu_full COOPERBENCH_INPUT=data/converted.jsonl ./scripts/run_wccu_all_recommended.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export EXP_TAG="${EXP_TAG:-wccu_full_$(date +%Y%m%d_%H%M%S)}"
RUN_PRIMARY="${WCCU_RUN_PRIMARY_REAL_LLM:-1}"
RUN_OFFLINE="${WCCU_RUN_OFFLINE_SANITY:-0}"
RUN_LLM_SMOKE="${WCCU_RUN_LLM_SMOKE:-0}"
RUN_CONTROLLED="${WCCU_RUN_CONTROLLED_OBLIGATION:-0}"
RUN_SHARED="${WCCU_RUN_SHARED_CONTEXT:-0}"
RUN_STRESS="${WCCU_RUN_STRESS_DIAGNOSTICS:-0}"
RUN_GUIDANCE="${WCCU_RUN_GUIDANCE:-0}"
RUN_MULTI_MODEL="${WCCU_RUN_MULTI_MODEL:-0}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_all_recommended.sh
primary_real_llm=${RUN_PRIMARY}
cooperbench_input=${COOPERBENCH_INPUT:-}
cooperbench_snapshot=${COOPERBENCH_SNAPSHOT:-${COOPERBENCH_RAW_INPUT:-}}
run_offline_sanity=${RUN_OFFLINE}
run_llm_smoke=${RUN_LLM_SMOKE}
run_controlled_obligation=${RUN_CONTROLLED}
run_shared_context=${RUN_SHARED}
run_stress_diagnostics=${RUN_STRESS}
run_guidance=${RUN_GUIDANCE}
run_multi_model=${RUN_MULTI_MODEL}
NOTE

echo "[wccu] repo: ${REPO_ROOT}"
echo "[wccu] EXP_TAG: ${EXP_TAG}"
echo "[wccu] results: results/${EXP_TAG}"
echo "[wccu] analysis: analysis/${EXP_TAG}"

if [[ "$RUN_PRIMARY" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_primary_real_llm_experiments.sh"
fi

if [[ "$RUN_OFFLINE" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_offline_sanity.sh"
fi
if [[ "$RUN_LLM_SMOKE" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_llm_smoke.sh"
fi
if [[ "$RUN_CONTROLLED" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_main_llm_obligation.sh"
fi
if [[ "$RUN_SHARED" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_shared_context.sh"
fi
if [[ "$RUN_STRESS" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_stress_and_target_ablation.sh"
fi
if [[ "$RUN_GUIDANCE" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_certificate_guidance_ablation.sh"
fi
if [[ "$RUN_MULTI_MODEL" == "1" ]]; then
  bash "$SCRIPT_DIR/run_wccu_multi_model.sh"
fi

# The primary script already packages outputs. If it was disabled, package here.
if [[ "$RUN_PRIMARY" != "1" ]]; then
  bash "$SCRIPT_DIR/package_wccu_experiment_outputs.sh" --tag "$EXP_TAG"
fi

echo "[wccu] recommended experiment suite complete for EXP_TAG=${EXP_TAG}"
