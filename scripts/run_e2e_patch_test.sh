#!/usr/bin/env bash
set -euo pipefail

EXP_TAG=${EXP_TAG:-e2e_patch_$(date +%Y%m%d_%H%M%S)}
INPUT=${INPUT:-data/e2e_patch_tasks.jsonl}
CONDITIONS=${CONDITIONS:-adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated}
RESULT=${RESULT:-results/${EXP_TAG}/e2e_patch_test.json}
ANALYSIS=${ANALYSIS:-analysis/${EXP_TAG}/e2e_patch_test}
WORK_DIR=${WORK_DIR:-runs/${EXP_TAG}/e2e_patch_test}
TIMEOUT_S=${TIMEOUT_S:-120}
LIMIT=${LIMIT:-0}
ROW_TIMEOUT_S=${ROW_TIMEOUT_S:-60}
ISOLATE_ROWS=${ISOLATE_ROWS:-0}

mkdir -p "results/${EXP_TAG}" "${ANALYSIS}" "logs"

EXTRA_ARGS=()
if [[ "${ISOLATE_ROWS}" == "1" ]]; then
  EXTRA_ARGS+=(--isolate-rows)
fi

python -m wccu_eval.eval.run_e2e_patch_test \
  --input "${INPUT}" \
  --conditions "${CONDITIONS}" \
  --out "${RESULT}" \
  --work-dir "${WORK_DIR}" \
  --timeout-s "${TIMEOUT_S}" \
  --limit "${LIMIT}" \
  --row-timeout-s "${ROW_TIMEOUT_S}" \
  "${EXTRA_ARGS[@]}"

python -m wccu_eval.scripts.make_e2e_patch_test_table \
  "${RESULT}" \
  --out-csv "${ANALYSIS}/e2e_patch_test_table.csv" \
  --out-tex "${ANALYSIS}/e2e_patch_test_table.tex" \
  --by-task-type-csv "${ANALYSIS}/e2e_patch_test_by_task_type.csv" \
  --by-task-type-tex "${ANALYSIS}/e2e_patch_test_by_task_type.tex"

zip -r "${EXP_TAG}_e2e_patch_test_outputs.zip" "results/${EXP_TAG}" "${ANALYSIS}" >/dev/null
printf 'Wrote %s_e2e_patch_test_outputs.zip\n' "${EXP_TAG}"
