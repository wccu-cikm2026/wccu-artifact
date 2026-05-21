#!/usr/bin/env bash
# Convert a local/public CooperBench snapshot into WCCU substrate scenarios, then
# run public-dataset-derived external-validity experiments.
#
# Inputs:
#   COOPERBENCH_INPUT=/path/to/already_converted.jsonl      # skips conversion
#   COOPERBENCH_SNAPSHOT=/path/to/hf_snapshot_or_metadata   # converted first
#   COOPERBENCH_RAW_INPUT=/path/to/hf_snapshot_or_metadata  # alias
#
# Common controls:
#   EXP_TAG=wccu_full_YYYYMMDD_HHMMSS
#   WCCU_COOPER_MOCK_LLM=1          # uses provider=mock, no API calls
#   WCCU_COOPER_SUBSET_SIZE=30
#   WCCU_COOPER_COMMITMENT_LIMIT=30
#   WCCU_COOPER_REPETITIONS=1
#
# Outputs:
#   data/$EXP_TAG/cooperbench_*.jsonl
#   results/$EXP_TAG/cooperbench_*_wccu.json
#   analysis/$EXP_TAG/cooperbench_*/tables
#   logs/${EXP_TAG}_cooperbench_*.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_cooper_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${WCCU_COOPER_DATA_DIR:-data/${EXP_TAG}}"
RESULT_DIR="results/${EXP_TAG}"
ANALYSIS_DIR="analysis/${EXP_TAG}"
LOG_DIR="logs"
mkdir -p "$DATA_DIR" "$RESULT_DIR" "$ANALYSIS_DIR" "$LOG_DIR"

COOPERBENCH_INPUT="${COOPERBENCH_INPUT:-}"
COOPERBENCH_SNAPSHOT="${COOPERBENCH_SNAPSHOT:-${COOPERBENCH_RAW_INPUT:-}}"
if [[ -z "$COOPERBENCH_INPUT" && -z "$COOPERBENCH_SNAPSHOT" ]]; then
  for candidate in \
    "data/cooperbench_multirepo_subset50_compact_seed7.jsonl" \
    "data/cooperbench_hf_snapshot" \
    "data/cooperbench_snapshot" \
    "data/cooperbench" \
    "data/CooperBench"; do
    if [[ -e "$candidate" ]]; then
      if [[ "$candidate" == *.jsonl || "$candidate" == *.json ]]; then
        COOPERBENCH_INPUT="$candidate"
      else
        COOPERBENCH_SNAPSHOT="$candidate"
      fi
      break
    fi
  done
fi

if [[ -z "$COOPERBENCH_INPUT" && -z "$COOPERBENCH_SNAPSHOT" ]]; then
  msg="No CooperBench input found. Set COOPERBENCH_INPUT=/path/to/converted.jsonl or COOPERBENCH_SNAPSHOT=/path/to/local_hf_snapshot."
  echo "[wccu-cooper] ${msg}" >&2
  echo "${msg}" > "${RESULT_DIR}/COOPERBENCH_SKIPPED.txt"
  if [[ "${WCCU_REQUIRE_COOPERBENCH:-0}" == "1" ]]; then
    exit 2
  fi
  exit 0
fi

CONVERTED="${COOPERBENCH_CONVERTED:-${DATA_DIR}/cooperbench_converted.jsonl}"
SUBSET_SIZE="${WCCU_COOPER_SUBSET_SIZE:-30}"
SUBSET="${COOPERBENCH_SUBSET:-${DATA_DIR}/cooperbench_subset${SUBSET_SIZE}_seed${WCCU_COOPER_SEED:-7}.jsonl}"
COMMITMENT_LIMIT="${WCCU_COOPER_COMMITMENT_LIMIT:-${SUBSET_SIZE}}"
COMMITMENT_DIAG="${COOPERBENCH_COMMITMENT_DIAG:-${DATA_DIR}/cooperbench_commitment_stale_subset${COMMITMENT_LIMIT}_seed${WCCU_COOPER_SEED:-7}.jsonl}"
SEED="${WCCU_COOPER_SEED:-7}"
FEATURE_MAX_CHARS="${WCCU_COOPER_FEATURE_MAX_CHARS:-900}"
MAX_TASKS="${WCCU_COOPER_MAX_TASKS:-120}"
REPETITIONS="${WCCU_COOPER_REPETITIONS:-1}"
PARALLEL_WORKERS="${WCCU_COOPER_PARALLEL_WORKERS:-3}"
MAX_OUTPUT_TOKENS="${WCCU_COOPER_MAX_OUTPUT_TOKENS:-1800}"
TIMEOUT_SECONDS="${WCCU_COOPER_TIMEOUT_SECONDS:-120}"
MAX_PROVIDER_RETRIES="${WCCU_COOPER_MAX_PROVIDER_RETRIES:-6}"
CERTIFICATE_GUIDANCE="${WCCU_COOPER_CERTIFICATE_GUIDANCE:-guided}"

WORKSPACE_CONDITIONS="${WCCU_COOPER_WORKSPACE_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_readset_occ,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"
COMMITMENT_CONDITIONS="${WCCU_COOPER_COMMITMENT_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_wccu_no_read_validation,adaptive_readset_occ,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"

PROVIDER_FLAGS=()
if [[ "${WCCU_COOPER_MOCK_LLM:-0}" == "1" || "${MOCK_LLM:-0}" == "1" ]]; then
  PROVIDER_FLAGS+=(--provider mock --model mock-llm)
fi

cat > "${RESULT_DIR}/COOPERBENCH_RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_cooperbench_derived.sh
cooperbench_input=${COOPERBENCH_INPUT}
cooperbench_snapshot=${COOPERBENCH_SNAPSHOT}
converted=${CONVERTED}
subset=${SUBSET}
commitment_diag=${COMMITMENT_DIAG}
subset_size=${SUBSET_SIZE}
commitment_limit=${COMMITMENT_LIMIT}
repetitions=${REPETITIONS}
workspace_conditions=${WORKSPACE_CONDITIONS}
commitment_conditions=${COMMITMENT_CONDITIONS}
mock_llm=${WCCU_COOPER_MOCK_LLM:-${MOCK_LLM:-0}}
NOTE

echo "[wccu-cooper] EXP_TAG=${EXP_TAG}"
echo "[wccu-cooper] data=${DATA_DIR}"
echo "[wccu-cooper] results=${RESULT_DIR}"
echo "[wccu-cooper] analysis=${ANALYSIS_DIR}"

if [[ -n "$COOPERBENCH_INPUT" ]]; then
  if [[ ! -f "$COOPERBENCH_INPUT" ]]; then
    echo "[wccu-cooper] Missing COOPERBENCH_INPUT: $COOPERBENCH_INPUT" >&2
    exit 2
  fi
  echo "[wccu-cooper] using already-converted input: $COOPERBENCH_INPUT"
  if [[ "$(cd "$(dirname "$COOPERBENCH_INPUT")" && pwd)/$(basename "$COOPERBENCH_INPUT")" != "$(cd "$(dirname "$CONVERTED")" && pwd)/$(basename "$CONVERTED")" ]]; then
    cp "$COOPERBENCH_INPUT" "$CONVERTED"
  fi
else
  if [[ ! -e "$COOPERBENCH_SNAPSHOT" ]]; then
    echo "[wccu-cooper] Missing COOPERBENCH_SNAPSHOT: $COOPERBENCH_SNAPSHOT" >&2
    exit 2
  fi
  echo "[wccu-cooper] converting local CooperBench snapshot: $COOPERBENCH_SNAPSHOT"
  CONVERT_FLAGS=()
  if [[ "${WCCU_COOPER_REQUIRE_SHARED_FILE:-1}" == "1" ]]; then
    CONVERT_FLAGS+=(--require-shared-file)
  fi
  if [[ "${WCCU_COOPER_FULL_FEATURE_TEXT:-0}" == "1" ]]; then
    CONVERT_FLAGS+=(--full-feature-text)
  fi
  python -m wccu_eval.scripts.convert_cooperbench_dataset \
    --input "$COOPERBENCH_SNAPSHOT" \
    --out "$CONVERTED" \
    --max-tasks "$MAX_TASKS" \
    --feature-max-chars "$FEATURE_MAX_CHARS" \
    --inspect \
    "${CONVERT_FLAGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${EXP_TAG}_cooperbench_convert.log"
fi

python -m wccu_eval.scripts.sample_cooperbench_subset \
  --input "$CONVERTED" \
  --out "$SUBSET" \
  --size "$SUBSET_SIZE" \
  --seed "$SEED" \
  2>&1 | tee "${LOG_DIR}/${EXP_TAG}_cooperbench_sample.log"

python -m wccu_eval.scripts.make_cooperbench_commitment_diagnostics \
  --input "$SUBSET" \
  --out "$COMMITMENT_DIAG" \
  --limit "$COMMITMENT_LIMIT" \
  --seed "$SEED" \
  --inspect \
  2>&1 | tee "${LOG_DIR}/${EXP_TAG}_cooperbench_commitment_diagnostics.log"

if [[ "${WCCU_COOPER_RUN_WORKSPACE:-1}" == "1" ]]; then
  WORKSPACE_OUT="${RESULT_DIR}/cooperbench_workspace_wccu.json"
  WORKSPACE_ANALYSIS="${ANALYSIS_DIR}/cooperbench_workspace_wccu"
  mkdir -p "$WORKSPACE_ANALYSIS"
  echo "[wccu-cooper] running workspace-contention external validation"
  python -m wccu_eval.eval.run_cooperbench_substrate \
    --input "$SUBSET" \
    --condition "$WORKSPACE_CONDITIONS" \
    --repetitions "$REPETITIONS" \
    --reasoning-effort "${WCCU_COOPER_REASONING_EFFORT:-low}" \
    --text-verbosity "${WCCU_COOPER_TEXT_VERBOSITY:-low}" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --parallel-workers "$PARALLEL_WORKERS" \
    --shuffle-cells \
    --max-provider-retries "$MAX_PROVIDER_RETRIES" \
    --retry-backoff-base "${WCCU_COOPER_RETRY_BACKOFF_BASE:-1.0}" \
    --retry-backoff-max "${WCCU_COOPER_RETRY_BACKOFF_MAX:-20.0}" \
    --certificate-guidance "$CERTIFICATE_GUIDANCE" \
    --out "$WORKSPACE_OUT" \
    "${PROVIDER_FLAGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${EXP_TAG}_cooperbench_workspace_wccu.log"

  python -m wccu_eval.scripts.analyze_results "$WORKSPACE_OUT" --out-dir "$WORKSPACE_ANALYSIS"
  python -m wccu_eval.scripts.make_cooperbench_table "$WORKSPACE_OUT" \
    --out-csv "$WORKSPACE_ANALYSIS/cooperbench_workspace_table.csv" \
    --out-tex "$WORKSPACE_ANALYSIS/cooperbench_workspace_table.tex"
  python -m wccu_eval.scripts.make_token_usage_table "$WORKSPACE_OUT" \
    --baseline-condition "${WCCU_COOPER_BASELINE_CONDITION:-adaptive_policy}" \
    --out-csv "$WORKSPACE_ANALYSIS/token_usage_table.csv" \
    --out-tex "$WORKSPACE_ANALYSIS/token_usage_table.tex" \
    --out-json "$WORKSPACE_ANALYSIS/token_usage_table.json" || true
fi

if [[ "${WCCU_COOPER_RUN_COMMITMENT:-1}" == "1" ]]; then
  COMMITMENT_OUT="${RESULT_DIR}/cooperbench_commitment_stale_wccu.json"
  COMMITMENT_ANALYSIS="${ANALYSIS_DIR}/cooperbench_commitment_stale_wccu"
  mkdir -p "$COMMITMENT_ANALYSIS"
  echo "[wccu-cooper] running stale-teammate-commitment external validation"
  python -m wccu_eval.eval.run_cooperbench_substrate \
    --input "$COMMITMENT_DIAG" \
    --condition "$COMMITMENT_CONDITIONS" \
    --repetitions "$REPETITIONS" \
    --reasoning-effort "${WCCU_COOPER_REASONING_EFFORT:-low}" \
    --text-verbosity "${WCCU_COOPER_TEXT_VERBOSITY:-low}" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --parallel-workers "$PARALLEL_WORKERS" \
    --shuffle-cells \
    --max-provider-retries "$MAX_PROVIDER_RETRIES" \
    --retry-backoff-base "${WCCU_COOPER_RETRY_BACKOFF_BASE:-1.0}" \
    --retry-backoff-max "${WCCU_COOPER_RETRY_BACKOFF_MAX:-20.0}" \
    --certificate-guidance "$CERTIFICATE_GUIDANCE" \
    --out "$COMMITMENT_OUT" \
    "${PROVIDER_FLAGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${EXP_TAG}_cooperbench_commitment_stale_wccu.log"

  python -m wccu_eval.scripts.analyze_results "$COMMITMENT_OUT" --out-dir "$COMMITMENT_ANALYSIS"
  python -m wccu_eval.scripts.make_cooperbench_commitment_table "$COMMITMENT_OUT" \
    --out-csv "$COMMITMENT_ANALYSIS/cooperbench_commitment_table.csv" \
    --out-tex "$COMMITMENT_ANALYSIS/cooperbench_commitment_table.tex"
  python -m wccu_eval.scripts.make_wccu_ablation_table "$COMMITMENT_OUT" \
    --out-csv "$COMMITMENT_ANALYSIS/wccu_commitment_ablation.csv" \
    --out-tex "$COMMITMENT_ANALYSIS/wccu_commitment_ablation.tex"
  python -m wccu_eval.scripts.make_token_usage_table "$COMMITMENT_OUT" \
    --baseline-condition "${WCCU_COOPER_BASELINE_CONDITION:-adaptive_policy}" \
    --out-csv "$COMMITMENT_ANALYSIS/token_usage_table.csv" \
    --out-tex "$COMMITMENT_ANALYSIS/token_usage_table.tex" \
    --out-json "$COMMITMENT_ANALYSIS/token_usage_table.json" || true
fi

echo "[wccu-cooper] done. Results are under ${RESULT_DIR}; analysis under ${ANALYSIS_DIR}."
