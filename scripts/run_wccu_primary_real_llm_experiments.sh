#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/load_wccu_env.sh"
# Primary paper-strength WCCU experiment pipeline.
#
# This script makes the CooperBench-derived, live-LLM experiments the default
# path.  Synthetic/deterministic/mock diagnostics are not run unless explicitly
# requested via flags below.
#
# Required input, one of:
#   COOPERBENCH_SNAPSHOT=/path/to/local/CooperBench/snapshot_or_hf_download
#   COOPERBENCH_INPUT=/path/to/already_converted_cooperbench.jsonl
#
# Required live LLM configuration:
#   LLM_PROVIDER=openai LLM_MODEL=<model> ...
#   or values in .env
#
# Main outputs:
#   data/$EXP_TAG/cooperbench_converted.jsonl
#   data/$EXP_TAG/cooperbench_subset*.jsonl
#   data/$EXP_TAG/cooperbench_commitment_stale*.jsonl
#   results/$EXP_TAG/cooperbench_workspace_wccu.json
#   results/$EXP_TAG/cooperbench_commitment_stale_wccu.json
#   analysis/$EXP_TAG/cooperbench_dataset_report.*
#   analysis/$EXP_TAG/live_llm_wccu_ablation.*
#   ${EXP_TAG}_outputs.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export EXP_TAG="${EXP_TAG:-wccu_real_llm_$(date +%Y%m%d_%H%M%S)}"
export WCCU_REQUIRE_COOPERBENCH="${WCCU_REQUIRE_COOPERBENCH:-1}"
export WCCU_COOPER_MOCK_LLM="${WCCU_COOPER_MOCK_LLM:-0}"

RUN_UNIT_TESTS="${WCCU_RUN_UNIT_TESTS:-1}"
RUN_CONTROLLED_OBLIGATION="${WCCU_RUN_CONTROLLED_OBLIGATION:-0}"
RUN_BOUNDARY_DIAGNOSTICS="${WCCU_RUN_BOUNDARY_DIAGNOSTICS:-0}"
INCLUDE_SOURCE="${WCCU_PACKAGE_INCLUDE_SOURCE:-0}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs" "data/${EXP_TAG}"

cat > "results/${EXP_TAG}/PRIMARY_RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_primary_real_llm_experiments.sh
primary_workload=CooperBench-derived collaborative-coding metadata + live LLM agents
cooperbench_input=${COOPERBENCH_INPUT:-}
cooperbench_snapshot=${COOPERBENCH_SNAPSHOT:-${COOPERBENCH_RAW_INPUT:-}}
mock_llm=${WCCU_COOPER_MOCK_LLM}
run_controlled_obligation=${RUN_CONTROLLED_OBLIGATION}
run_boundary_diagnostics=${RUN_BOUNDARY_DIAGNOSTICS}
NOTE

if [[ "${WCCU_COOPER_MOCK_LLM}" == "1" && "${WCCU_ALLOW_MOCK_AS_PRIMARY:-0}" != "1" ]]; then
  echo "[wccu-primary] Refusing to run mock LLM as primary evidence. Set WCCU_ALLOW_MOCK_AS_PRIMARY=1 only for local smoke tests." >&2
  exit 2
fi

# Keep provider failures attached to this experiment tag unless the caller
# explicitly routes them elsewhere.
export LLM_ERROR_LOG_PATH="${LLM_ERROR_LOG_PATH:-results/${EXP_TAG}/provider_errors.jsonl}"

python -m wccu_eval.scripts.check_real_llm_config \
  2>&1 | tee "logs/${EXP_TAG}_real_llm_config.log"

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
  python -m compileall wccu_eval tests \
    2>&1 | tee "logs/${EXP_TAG}_compile.log"
  python -m unittest discover -s tests -v \
    2>&1 | tee "logs/${EXP_TAG}_unit_tests.log"
fi

python -m wccu_eval.scripts.check_llm_provider \
  --scenario high_risk_rule_change \
  2>&1 | tee "logs/${EXP_TAG}_check_llm_provider.log"

bash "$SCRIPT_DIR/run_wccu_cooperbench_derived.sh" \
  2>&1 | tee "logs/${EXP_TAG}_primary_cooperbench_derived.log"

CONVERTED="${COOPERBENCH_CONVERTED:-${WCCU_COOPER_DATA_DIR:-data/${EXP_TAG}}/cooperbench_converted.jsonl}"
SUBSET_SIZE="${WCCU_COOPER_SUBSET_SIZE:-30}"
SEED="${WCCU_COOPER_SEED:-7}"
SUBSET="${COOPERBENCH_SUBSET:-${WCCU_COOPER_DATA_DIR:-data/${EXP_TAG}}/cooperbench_subset${SUBSET_SIZE}_seed${SEED}.jsonl}"
COMMITMENT_LIMIT="${WCCU_COOPER_COMMITMENT_LIMIT:-${SUBSET_SIZE}}"
COMMITMENT_DIAG="${COOPERBENCH_COMMITMENT_DIAG:-${WCCU_COOPER_DATA_DIR:-data/${EXP_TAG}}/cooperbench_commitment_stale_subset${COMMITMENT_LIMIT}_seed${SEED}.jsonl}"

python -m wccu_eval.scripts.make_cooperbench_dataset_report \
  --converted "$CONVERTED" \
  --subset "$SUBSET" \
  --commitment-diag "$COMMITMENT_DIAG" \
  --out-json "analysis/${EXP_TAG}/cooperbench_dataset_report.json" \
  --out-csv "analysis/${EXP_TAG}/cooperbench_dataset_report.csv" \
  --out-md "analysis/${EXP_TAG}/cooperbench_dataset_report.md" \
  2>&1 | tee "logs/${EXP_TAG}_cooperbench_dataset_report.log"

ABLATION_INPUTS=()
[[ -f "results/${EXP_TAG}/cooperbench_commitment_stale_wccu.json" ]] && ABLATION_INPUTS+=("results/${EXP_TAG}/cooperbench_commitment_stale_wccu.json")
[[ -f "results/${EXP_TAG}/cooperbench_workspace_wccu.json" ]] && ABLATION_INPUTS+=("results/${EXP_TAG}/cooperbench_workspace_wccu.json")
if [[ ${#ABLATION_INPUTS[@]} -gt 0 ]]; then
  python -m wccu_eval.scripts.make_wccu_real_llm_ablation_table \
    "${ABLATION_INPUTS[@]}" \
    --group-by-family \
    --out-csv "analysis/${EXP_TAG}/live_llm_wccu_ablation.csv" \
    --out-json "analysis/${EXP_TAG}/live_llm_wccu_ablation.json" \
    --out-tex "analysis/${EXP_TAG}/live_llm_wccu_ablation.tex" \
    2>&1 | tee "logs/${EXP_TAG}_live_llm_wccu_ablation.log"
fi

if [[ "$RUN_CONTROLLED_OBLIGATION" == "1" ]]; then
  echo "[wccu-primary] Running controlled live-LLM obligation benchmark as supplemental evidence."
  bash "$SCRIPT_DIR/run_wccu_main_llm_obligation.sh" \
    2>&1 | tee "logs/${EXP_TAG}_supplemental_llm_obligation.log"
fi

if [[ "$RUN_BOUNDARY_DIAGNOSTICS" == "1" ]]; then
  echo "[wccu-primary] Running deterministic boundary diagnostics as appendix/artifact checks only."
  EXP_TAG="$EXP_TAG" bash "$SCRIPT_DIR/run_wccu_stress_and_target_ablation.sh" \
    2>&1 | tee "logs/${EXP_TAG}_boundary_diagnostics.log"
fi

PKG_ARGS=(--tag "$EXP_TAG")
if [[ "$INCLUDE_SOURCE" == "1" ]]; then
  PKG_ARGS+=(--include-source)
fi
bash "$SCRIPT_DIR/package_wccu_experiment_outputs.sh" "${PKG_ARGS[@]}"

echo "[wccu-primary] complete: EXP_TAG=${EXP_TAG}"
