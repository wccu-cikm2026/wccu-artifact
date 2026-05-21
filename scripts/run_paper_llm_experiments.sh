#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_TAG="${EXP_TAG:-llm_full_$(date +%Y%m%d_%H%M%S)}"
INPUT_PATH="${INPUT_PATH:-data/cooperbench_multirepo_subset50_compact_seed7.jsonl}"
COMMITMENT_CONDITIONS="${COMMITMENT_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_wccu_no_read_validation,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"
LOCK_CONDITIONS="${LOCK_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"

mkdir -p "logs" "results/${EXP_TAG}" "analysis/${EXP_TAG}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Missing INPUT_PATH file: $INPUT_PATH" >&2
  echo "Provide INPUT_PATH=/path/to/cooperbench_multirepo_subset50_compact_seed7.jsonl." >&2
  exit 2
fi

python -m compileall wccu_eval tests \
  2>&1 | tee "logs/${EXP_TAG}_compile.log"
python -m unittest discover -s tests -v \
  2>&1 | tee "logs/${EXP_TAG}_unit_tests.log"

RUN_LLM=1 \
INPUT="$INPUT_PATH" \
DIAG="data/${EXP_TAG}_commitment_stale_subset30_seed7.jsonl" \
OUT="results/${EXP_TAG}/cooperbench_commitment_stale_subset30_wccu_r1_full_ablation.json" \
ANALYSIS="analysis/${EXP_TAG}/cooperbench_commitment_stale_subset30_wccu_r1_full_ablation" \
CONDITIONS="$COMMITMENT_CONDITIONS" \
LIMIT="${LIMIT:-30}" \
SEED="${SEED:-7}" \
REPETITIONS="${REPETITIONS:-1}" \
bash scripts/reproduce_commitment_diagnostic.sh \
  2>&1 | tee "logs/${EXP_TAG}_commitment_ablation_full.log"

RUN_LLM=1 \
INPUT_PATH="$INPUT_PATH" \
RESULT_PATH="results/${EXP_TAG}/cooperbench_multirepo_subset50_wccu_r1.json" \
OUT_DIR="analysis/${EXP_TAG}/cooperbench_multirepo_subset50_wccu_r1" \
CONDITIONS="$LOCK_CONDITIONS" \
REPETITIONS="${REPETITIONS:-1}" \
bash scripts/reproduce_table6.sh \
  2>&1 | tee "logs/${EXP_TAG}_metadata_lock_table6.log"

python -m wccu_eval.eval.run_wccu_stress \
  --cases "${STRESS_CASES:-100}" \
  --writers "${STRESS_WRITERS:-8}" \
  --atom-count "${STRESS_ATOM_COUNT:-64}" \
  --invalidation-prob "${STRESS_INVALIDATION_PROB:-0.35}" \
  --seed "${STRESS_SEED:-7}" \
  --condition adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only \
  --repetitions 1 \
  --out "results/${EXP_TAG}/wccu_randomized_stress_w8_a64_p035_seed7.json" \
  2>&1 | tee "logs/${EXP_TAG}_wccu_randomized_stress_seed7.log"

zip -r "${EXP_TAG}_all_outputs.zip" \
  "results/${EXP_TAG}" \
  "analysis/${EXP_TAG}" \
  logs/${EXP_TAG}_*.log \
  >/dev/null

echo "EXP_TAG=${EXP_TAG}"
echo "Wrote ${EXP_TAG}_all_outputs.zip"
