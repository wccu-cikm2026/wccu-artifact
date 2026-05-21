#!/usr/bin/env bash
# Multi-model WCCU obligation benchmark. Reads LLM_MODELS from .env unless WCCU_MODEL_SPECS is provided.
# Usage: EXP_TAG=wccu_multi WCCU_MODEL_SPECS='openai:gpt-4.1-mini,gemini:gemini-1.5-flash' ./scripts/run_wccu_multi_model.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_TAG="${EXP_TAG:-wccu_multi_$(date +%Y%m%d_%H%M%S)}"
FAMILIES="${WCCU_FAMILIES:-freshness,commitment,authority,operation,derived_view,witness_gap,safe}"
LIMIT_PER_FAMILY="${WCCU_LIMIT_PER_FAMILY:-30}"
REPETITIONS="${WCCU_REPETITIONS:-1}"
CONDITIONS="${WCCU_MAIN_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_wccu_no_read_validation,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only}"
CERTIFICATE_GUIDANCE="${WCCU_CERTIFICATE_GUIDANCE:-unguided}"
MODEL_SPECS="${WCCU_MODEL_SPECS:-}"
MAX_OUTPUT_TOKENS="${WCCU_MAX_OUTPUT_TOKENS:-1200}"
TIMEOUT_SECONDS="${WCCU_TIMEOUT_SECONDS:-90}"
OUT="results/${EXP_TAG}/multi_model_obligation_${CERTIFICATE_GUIDANCE}.json"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

cat > "results/${EXP_TAG}/RUN_NOTE.txt" <<NOTE
EXP_TAG=${EXP_TAG}
script=run_wccu_multi_model.sh
families=${FAMILIES}
limit_per_family=${LIMIT_PER_FAMILY}
repetitions=${REPETITIONS}
conditions=${CONDITIONS}
certificate_guidance=${CERTIFICATE_GUIDANCE}
model_specs_set=$([[ -n "$MODEL_SPECS" ]] && echo 1 || echo 0)
max_output_tokens=${MAX_OUTPUT_TOKENS}
timeout_seconds=${TIMEOUT_SECONDS}
NOTE

cmd=(
  python -m wccu_eval.eval.run_multi_model_obligation_benchmark
  --families "$FAMILIES"
  --limit-per-family "$LIMIT_PER_FAMILY"
  --repetitions "$REPETITIONS"
  --certificate-guidance "$CERTIFICATE_GUIDANCE"
  --condition "$CONDITIONS"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --timeout-seconds "$TIMEOUT_SECONDS"
  --out "$OUT"
)
if [[ -n "$MODEL_SPECS" ]]; then
  cmd+=(--model-specs "$MODEL_SPECS")
fi

LOG="logs/${EXP_TAG}_multi_model_obligation_${CERTIFICATE_GUIDANCE}.log"
echo "[wccu] running multi-model obligation benchmark"
"${cmd[@]}" 2>&1 | tee "$LOG"

python - "$OUT" "$EXP_TAG" <<'PY'
import json, subprocess, sys
from pathlib import Path
multi_json = Path(sys.argv[1])
tag = sys.argv[2]
payload = json.loads(multi_json.read_text(encoding='utf-8'))
for child in payload.get('child_outputs', []):
    out = child.get('out')
    provider = str(child.get('provider', 'provider')).replace('/', '_').replace(':', '_')
    model = str(child.get('model', 'model')).replace('/', '_').replace(':', '_')
    if not out or not Path(out).exists():
        continue
    prefix = f"analysis/{tag}/multi_model_{provider}_{model}"
    subprocess.run([sys.executable, '-m', 'wccu_eval.scripts.make_llm_obligation_tables', out, '--out-prefix', prefix], check=True)
PY

echo "[wccu] multi-model benchmark complete: ${OUT}"
