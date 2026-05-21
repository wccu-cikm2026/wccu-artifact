#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT="${INPUT:-data/cooperbench_multirepo_subset50_compact_seed7.jsonl}"
DIAG="${DIAG:-data/cooperbench_commitment_stale_subset30_seed7.jsonl}"
OUT="${OUT:-results/cooperbench_commitment_stale_subset30_wccu_r1.json}"
ANALYSIS="${ANALYSIS:-analysis/cooperbench_commitment_stale_subset30_wccu_r1}"
CONDITIONS="${CONDITIONS:-adaptive_wccu_execution_trace,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"
LIMIT="${LIMIT:-30}"
SEED="${SEED:-7}"
REPETITIONS="${REPETITIONS:-1}"

mkdir -p "$(dirname "$DIAG")" "$(dirname "$OUT")" "$ANALYSIS"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing INPUT file: $INPUT" >&2
  echo "Set INPUT=/path/to/cooperbench_multirepo_subset50_compact_seed7.jsonl or generate it before running this script." >&2
  exit 2
fi

python -m wccu_eval.scripts.make_cooperbench_commitment_diagnostics \
  --input "$INPUT" \
  --out "$DIAG" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --inspect

if [[ "${RUN_LLM:-0}" == "1" ]]; then
  python -m wccu_eval.eval.run_cooperbench_substrate \
    --input "$DIAG" \
    --condition "$CONDITIONS" \
    --repetitions "$REPETITIONS" \
    --reasoning-effort "${REASONING_EFFORT:-low}" \
    --text-verbosity "${TEXT_VERBOSITY:-low}" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS:-1800}" \
    --parallel-workers "${PARALLEL_WORKERS:-3}" \
    --shuffle-cells \
    --max-provider-retries "${MAX_PROVIDER_RETRIES:-6}" \
    --retry-backoff-base "${RETRY_BACKOFF_BASE:-1.0}" \
    --retry-backoff-max "${RETRY_BACKOFF_MAX:-20.0}" \
    --certificate-guidance "${CERTIFICATE_GUIDANCE:-guided}" \
    --out "$OUT"
else
  echo "Generated $DIAG. Set RUN_LLM=1 to run the LLM experiment."
fi

if [[ ! -f "$OUT" ]]; then
  echo "No result JSON found at $OUT; skipping analysis." >&2
  exit 0
fi

python -m wccu_eval.scripts.analyze_results "$OUT" --out-dir "$ANALYSIS"
python -m wccu_eval.scripts.make_cooperbench_commitment_table "$OUT" \
  --out-csv "$ANALYSIS/commitment_table.csv" \
  --out-tex "$ANALYSIS/commitment_table.tex"
python -m wccu_eval.scripts.make_wccu_ablation_table "$OUT" \
  --out-csv "$ANALYSIS/wccu_commitment_ablation.csv" \
  --out-tex "$ANALYSIS/wccu_commitment_ablation.tex"
python -m wccu_eval.scripts.make_token_usage_table "$OUT" \
  --baseline-condition "${BASELINE_CONDITION:-adaptive_policy}" \
  --out-csv "$ANALYSIS/token_usage_table.csv" \
  --out-tex "$ANALYSIS/token_usage_table.tex" \
  --out-json "$ANALYSIS/token_usage_table.json"

echo "Wrote $OUT and $ANALYSIS"
