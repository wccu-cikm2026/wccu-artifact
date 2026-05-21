#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_TAG="${EXP_TAG:-latency_$(date +%Y%m%d_%H%M%S)}"
INPUT="${INPUT:-${INPUT_PATH:-data/cooperbench_multirepo_subset50_compact_seed7.jsonl}}"
DIAG="${DIAG:-data/${EXP_TAG}_commitment_stale_subset${LIMIT:-30}_seed${SEED:-7}.jsonl}"
OUT="${OUT:-results/${EXP_TAG}/latency_parallelism_commitment_subset${LIMIT:-30}.json}"
ANALYSIS="${ANALYSIS:-analysis/${EXP_TAG}/latency_parallelism_commitment_subset${LIMIT:-30}}"
CONDITIONS="${CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_policy,uniform_review_gated,serial_adaptive_wccu_execution_trace,serial_adaptive_policy}"
LIMIT="${LIMIT:-30}"
SEED="${SEED:-7}"
REPETITIONS="${REPETITIONS:-1}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
MAX_PROVIDER_RETRIES="${MAX_PROVIDER_RETRIES:-6}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
TEXT_VERBOSITY="${TEXT_VERBOSITY:-low}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1800}"
CERTIFICATE_GUIDANCE="${CERTIFICATE_GUIDANCE:-unguided}"

mkdir -p "$(dirname "$DIAG")" "$(dirname "$OUT")" "$ANALYSIS" logs

if [[ ! -f "$INPUT" ]]; then
  echo "Missing INPUT file: $INPUT" >&2
  exit 2
fi

python -m wccu_eval.scripts.make_cooperbench_commitment_diagnostics \
  --input "$INPUT" \
  --out "$DIAG" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --inspect

python -m wccu_eval.eval.run_cooperbench_substrate \
  --input "$DIAG" \
  --condition "$CONDITIONS" \
  --repetitions "$REPETITIONS" \
  --reasoning-effort "$REASONING_EFFORT" \
  --text-verbosity "$TEXT_VERBOSITY" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --parallel-workers "$PARALLEL_WORKERS" \
  --shuffle-cells \
  --max-provider-retries "$MAX_PROVIDER_RETRIES" \
  --certificate-guidance "$CERTIFICATE_GUIDANCE" \
  --out "$OUT" \
  2>&1 | tee "logs/${EXP_TAG}_latency_parallelism.log"

python -m wccu_eval.scripts.analyze_results "$OUT" --out-dir "$ANALYSIS"
python -m wccu_eval.scripts.make_latency_parallelism_table "$OUT" \
  --out-csv "$ANALYSIS/latency_parallelism_table.csv" \
  --out-tex "$ANALYSIS/latency_parallelism_table.tex"
python -m wccu_eval.scripts.make_cooperbench_commitment_table "$OUT" \
  --out-csv "$ANALYSIS/commitment_table.csv" \
  --out-tex "$ANALYSIS/commitment_table.tex"
python -m wccu_eval.scripts.make_token_usage_table "$OUT" \
  --baseline-condition adaptive_policy \
  --out-csv "$ANALYSIS/token_usage_table.csv" \
  --out-tex "$ANALYSIS/token_usage_table.tex" \
  --out-json "$ANALYSIS/token_usage_table.json"

zip -r "${EXP_TAG}_latency_parallelism_outputs.zip" \
  "$(dirname "$OUT")" \
  "$ANALYSIS" \
  "logs/${EXP_TAG}_latency_parallelism.log" \
  >/dev/null

echo "EXP_TAG=${EXP_TAG}"
echo "Wrote ${EXP_TAG}_latency_parallelism_outputs.zip"
