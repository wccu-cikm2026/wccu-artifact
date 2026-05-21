#!/usr/bin/env bash
# Runs guided/minimal/no_hints/unguided certificate-prompt ablations.
# Usage: EXP_TAG=wccu_guidance WCCU_LIMIT_PER_FAMILY=30 ./scripts/run_wccu_certificate_guidance_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_guidance_$(date +%Y%m%d_%H%M%S)}"
FAMILIES="${WCCU_FAMILIES:-freshness,commitment,authority,operation,derived_view,witness_gap,safe}"
LIMIT_PER_FAMILY="${WCCU_LIMIT_PER_FAMILY:-30}"
REPETITIONS="${WCCU_REPETITIONS:-1}"
CONDITIONS="${WCCU_MAIN_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only}"
GUIDANCE_MODES="${WCCU_GUIDANCE_MODES:-guided minimal no_hints unguided}"
MAX_OUTPUT_TOKENS="${WCCU_MAX_OUTPUT_TOKENS:-1200}"
TIMEOUT_SECONDS="${WCCU_TIMEOUT_SECONDS:-90}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_certificate_guidance_ablation.sh
families=${FAMILIES}
limit_per_family=${LIMIT_PER_FAMILY}
repetitions=${REPETITIONS}
conditions=${CONDITIONS}
guidance_modes=${GUIDANCE_MODES}
max_output_tokens=${MAX_OUTPUT_TOKENS}
timeout_seconds=${TIMEOUT_SECONDS}
NOTE

for mode in $GUIDANCE_MODES; do
  out="results/${EXP_TAG}/llm_obligation_${mode}.json"
  log="logs/${EXP_TAG}_llm_obligation_${mode}.log"
  echo "[wccu] running certificate guidance mode: ${mode}"
  python -m wccu_eval.eval.run_llm_obligation_benchmark \
    --families "$FAMILIES" \
    --limit-per-family "$LIMIT_PER_FAMILY" \
    --repetitions "$REPETITIONS" \
    --certificate-guidance "$mode" \
    --condition "$CONDITIONS" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --out "$out" \
    2>&1 | tee "$log"

  python -m wccu_eval.scripts.make_llm_obligation_tables \
    "$out" \
    --out-prefix "analysis/${EXP_TAG}/llm_obligation_${mode}"
done

echo "[wccu] certificate guidance ablation complete for EXP_TAG=${EXP_TAG}"
