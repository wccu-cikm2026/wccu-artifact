#!/usr/bin/env bash
set -euo pipefail

EXP_TAG=${EXP_TAG:-llm_e2e_commitment_debug_$(date +%Y%m%d_%H%M%S)}
INPUT=${INPUT:-data/e2e_patch_tasks.jsonl}
LIMIT=${LIMIT:-3}
CONDITIONS=${CONDITIONS:-adaptive_wccu_execution_trace,adaptive_policy}
TASK_TYPES=${TASK_TYPES:-commitment_staleness}
TIMEOUT_S=${TIMEOUT_S:-120}
ROW_TIMEOUT_S=${ROW_TIMEOUT_S:-120}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-4000}
OUTPUT_MODE=${OUTPUT_MODE:-file_edits}
MOCK_FROM_PREPARED=${MOCK_FROM_PREPARED:-0}

export EXP_TAG INPUT LIMIT CONDITIONS TASK_TYPES TIMEOUT_S ROW_TIMEOUT_S MAX_OUTPUT_TOKENS MOCK_FROM_PREPARED OUTPUT_MODE
bash scripts/run_llm_e2e_patch_test.sh
