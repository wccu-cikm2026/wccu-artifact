#!/usr/bin/env bash
set -euo pipefail

EXP_TAG="${EXP_TAG:-llm_obligation_$(date +%Y%m%d_%H%M%S)}"
FAMILIES="${FAMILIES:-freshness,commitment,authority,operation,derived_view,witness_gap,safe}"
LIMIT_PER_FAMILY="${LIMIT_PER_FAMILY:-5}"
REPETITIONS="${REPETITIONS:-1}"
CONDITIONS="${CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_wccu_no_read_validation,adaptive_readset_occ,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"
CERTIFICATE_GUIDANCE="${CERTIFICATE_GUIDANCE:-unguided}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-90}"
MOCK_LLM="${MOCK_LLM:-0}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

MOCK_FLAG=""
if [[ "${MOCK_LLM}" == "1" ]]; then
  MOCK_FLAG="--mock-llm"
fi

python -m wccu_eval.eval.run_llm_obligation_benchmark \
  --families "${FAMILIES}" \
  --limit-per-family "${LIMIT_PER_FAMILY}" \
  --repetitions "${REPETITIONS}" \
  --condition "${CONDITIONS}" \
  --certificate-guidance "${CERTIFICATE_GUIDANCE}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --out "results/${EXP_TAG}/llm_obligation_benchmark.json" \
  ${MOCK_FLAG} \
  2>&1 | tee "logs/${EXP_TAG}_llm_obligation_benchmark.log"

python - "results/${EXP_TAG}/llm_obligation_benchmark.json" <<'PY'
import json
import sys
path = sys.argv[1]
payload = json.load(open(path, encoding='utf-8'))
generations = payload.get('generations') or []
results = payload.get('results') or []
failed = sum(1 for r in results if r.get('failed'))
print(f"LLM obligation benchmark summary: generations={len(generations)} results={len(results)} failed_results={failed}")
if not generations:
    raise SystemExit("No successful LLM generations were recorded; inspect results/*.errors.jsonl before using the tables.")
PY

python -m wccu_eval.scripts.make_llm_obligation_tables \
  "results/${EXP_TAG}/llm_obligation_benchmark.json" \
  --out-prefix "analysis/${EXP_TAG}/llm_obligation"

python -m wccu_eval.scripts.make_coordination_metrics_table \
  "results/${EXP_TAG}/llm_obligation_benchmark.json" \
  --out-csv "analysis/${EXP_TAG}/coordination_metrics_llm_obligation_table.csv" \
  --out-tex "analysis/${EXP_TAG}/coordination_metrics_llm_obligation_table.tex" \
  --full-tex

zip -qr "${EXP_TAG}_llm_obligation_outputs.zip" \
  "results/${EXP_TAG}" \
  "analysis/${EXP_TAG}" \
  "logs/${EXP_TAG}_llm_obligation_benchmark.log"

echo "Wrote ${EXP_TAG}_llm_obligation_outputs.zip"
