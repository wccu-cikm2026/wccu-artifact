#!/usr/bin/env bash
# Offline WCCU stress diagnostics plus optional live target-grounding ablations.
# Set WCCU_RUN_TARGET_ABLATION=1 to include LLM target ablations.
# Usage: EXP_TAG=wccu_diag ./scripts/run_wccu_stress_and_target_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_diag_$(date +%Y%m%d_%H%M%S)}"
STRESS_CASES="${WCCU_STRESS_CASES:-100}"
STRESS_REPETITIONS="${WCCU_STRESS_REPETITIONS:-3}"
TARGET_LIMIT="${WCCU_TARGET_LIMIT_PER_FAMILY:-20}"
RUN_TARGET_ABLATION="${WCCU_RUN_TARGET_ABLATION:-0}"
MAX_OUTPUT_TOKENS="${WCCU_MAX_OUTPUT_TOKENS:-1200}"
TIMEOUT_SECONDS="${WCCU_TIMEOUT_SECONDS:-90}"
STRESS_CONDITIONS="adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only"
TARGET_CONDITIONS="adaptive_wccu_execution_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_stress_and_target_ablation.sh
stress_cases=${STRESS_CASES}
stress_repetitions=${STRESS_REPETITIONS}
run_target_ablation=${RUN_TARGET_ABLATION}
target_limit_per_family=${TARGET_LIMIT}
max_output_tokens=${MAX_OUTPUT_TOKENS}
timeout_seconds=${TIMEOUT_SECONDS}
NOTE

run_stress() {
  local name="$1"
  shift
  local out="results/${EXP_TAG}/${name}.json"
  local log="logs/${EXP_TAG}_${name}.log"
  echo "[wccu] running ${name}"
  python -m wccu_eval.eval.run_wccu_stress \
    --cases "$STRESS_CASES" \
    --writers 4 \
    --atom-count 20 \
    --invalidation-prob 0.5 \
    --seed 7 \
    --condition "$STRESS_CONDITIONS" \
    --repetitions "$STRESS_REPETITIONS" \
    --out "$out" \
    "$@" \
    2>&1 | tee "$log"
}

run_stress wccu_randomized_stress
run_stress wccu_randomized_stress_no_witness --no-witness

python -m wccu_eval.scripts.make_stress_table \
  "results/${EXP_TAG}/wccu_randomized_stress.json" \
  "results/${EXP_TAG}/wccu_randomized_stress_no_witness.json" \
  --out-csv "analysis/${EXP_TAG}/wccu_stress.csv" \
  --out-tex "analysis/${EXP_TAG}/wccu_stress.tex"

if [[ "$RUN_TARGET_ABLATION" == "1" ]]; then
  run_target() {
    local name="$1"
    shift
    local out="results/${EXP_TAG}/${name}.json"
    local log="logs/${EXP_TAG}_${name}.log"
    echo "[wccu] running ${name}"
    python -m wccu_eval.eval.run_llm_obligation_benchmark \
      --families freshness,authority,operation,derived_view,witness_gap,safe \
      --limit-per-family "$TARGET_LIMIT" \
      --repetitions 1 \
      --certificate-guidance unguided \
      --condition "$TARGET_CONDITIONS" \
      --max-output-tokens "$MAX_OUTPUT_TOKENS" \
      --timeout-seconds "$TIMEOUT_SECONDS" \
      --out "$out" \
      "$@" \
      2>&1 | tee "$log"
  }

  run_target target_grounding_on
  run_target target_candidates_off --disable-target-candidates
  run_target target_grounding_off --disable-target-grounding

  python -m wccu_eval.scripts.make_target_ablation_table \
    "results/${EXP_TAG}/target_grounding_on.json" \
    --out-csv "analysis/${EXP_TAG}/target_grounding_on.csv" \
    --out-tex "analysis/${EXP_TAG}/target_grounding_on.tex"

  python -m wccu_eval.scripts.make_target_grounding_report \
    "results/${EXP_TAG}/target_candidates_off.json" \
    --out-summary-csv "analysis/${EXP_TAG}/target_candidates_off_summary.csv" \
    --out-cases-jsonl "analysis/${EXP_TAG}/target_candidates_off_cases.jsonl" \
    --out-md "analysis/${EXP_TAG}/target_candidates_off_report.md"

  python -m wccu_eval.scripts.make_target_grounding_report \
    "results/${EXP_TAG}/target_grounding_off.json" \
    --out-summary-csv "analysis/${EXP_TAG}/target_grounding_off_summary.csv" \
    --out-cases-jsonl "analysis/${EXP_TAG}/target_grounding_off_cases.jsonl" \
    --out-md "analysis/${EXP_TAG}/target_grounding_off_report.md"
fi

echo "[wccu] stress diagnostics complete for EXP_TAG=${EXP_TAG}"
