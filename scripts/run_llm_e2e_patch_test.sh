#!/usr/bin/env bash
set -euo pipefail

EXP_TAG=${EXP_TAG:-llm_e2e_patch_$(date +%Y%m%d_%H%M%S)}
INPUT=${INPUT:-data/e2e_patch_tasks.jsonl}
CONDITIONS=${CONDITIONS:-adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated}
RESULT=${RESULT:-results/${EXP_TAG}/llm_e2e_patch_test.json}
GENERATED_TASKS=${GENERATED_TASKS:-results/${EXP_TAG}/llm_generated_patch_tasks.jsonl}
GEN_LOG=${GEN_LOG:-results/${EXP_TAG}/llm_patch_generation.jsonl}
ANALYSIS=${ANALYSIS:-analysis/${EXP_TAG}/llm_e2e_patch_test}
WORK_DIR=${WORK_DIR:-runs/${EXP_TAG}/llm_e2e_patch_test}
TIMEOUT_S=${TIMEOUT_S:-120}
ROW_TIMEOUT_S=${ROW_TIMEOUT_S:-90}
LIMIT=${LIMIT:-0}
TASK_TYPES=${TASK_TYPES:-}
ISOLATE_ROWS=${ISOLATE_ROWS:-0}
MOCK_FROM_PREPARED=${MOCK_FROM_PREPARED:-0}
PROVIDER=${PROVIDER:-}
MODEL=${MODEL:-}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-2500}
OUTPUT_MODE=${OUTPUT_MODE:-file_edits}

mkdir -p "results/${EXP_TAG}" "${ANALYSIS}" "logs"

EXTRA_ARGS=()
if [[ "${ISOLATE_ROWS}" == "1" ]]; then
  EXTRA_ARGS+=(--isolate-rows)
fi
if [[ "${MOCK_FROM_PREPARED}" == "1" ]]; then
  EXTRA_ARGS+=(--mock-from-prepared)
fi
if [[ -n "${PROVIDER}" ]]; then
  EXTRA_ARGS+=(--provider "${PROVIDER}")
fi
if [[ -n "${MODEL}" ]]; then
  EXTRA_ARGS+=(--model "${MODEL}")
fi

python -m wccu_eval.eval.run_llm_e2e_patch_test \
  --input "${INPUT}" \
  --conditions "${CONDITIONS}" \
  --generated-tasks-out "${GENERATED_TASKS}" \
  --generation-log "${GEN_LOG}" \
  --out "${RESULT}" \
  --work-dir "${WORK_DIR}" \
  --timeout-s "${TIMEOUT_S}" \
  --row-timeout-s "${ROW_TIMEOUT_S}" \
  --limit "${LIMIT}" \
  --task-types "${TASK_TYPES}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
  --output-mode "${OUTPUT_MODE}" \
  "${EXTRA_ARGS[@]}"


python -m wccu_eval.scripts.make_llm_patch_generation_table \
  "${GEN_LOG}" \
  --out-csv "${ANALYSIS}/llm_patch_generation_table.csv" \
  --out-tex "${ANALYSIS}/llm_patch_generation_table.tex" \
  --by-task-type-csv "${ANALYSIS}/llm_patch_generation_by_task_type.csv" \
  --by-task-type-tex "${ANALYSIS}/llm_patch_generation_by_task_type.tex"

python -m wccu_eval.scripts.make_e2e_patch_test_table \
  "${RESULT}" \
  --out-csv "${ANALYSIS}/llm_e2e_patch_test_table.csv" \
  --out-tex "${ANALYSIS}/llm_e2e_patch_test_table.tex" \
  --by-task-type-csv "${ANALYSIS}/llm_e2e_patch_test_by_task_type.csv" \
  --by-task-type-tex "${ANALYSIS}/llm_e2e_patch_test_by_task_type.tex"

zip -r "${EXP_TAG}_llm_e2e_patch_test_outputs.zip" "results/${EXP_TAG}" "${ANALYSIS}" >/dev/null
printf 'Wrote %s_llm_e2e_patch_test_outputs.zip\n' "${EXP_TAG}"
