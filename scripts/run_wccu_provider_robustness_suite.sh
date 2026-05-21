#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/load_wccu_env.sh"

# Convenience runner for the two provider-robustness experiments:
#  1) Gemini-only replacement of the OpenAI-backed agents.
#  2) Mixed-provider agents over the same shared context store.
# Existing OpenAI-only tags can be passed to the summary step via BASELINE_TAGS.

SEED="${WCCU_COOPER_SEED:-7}"
BASE_TAG_PREFIX="${WCCU_PROVIDER_SUITE_PREFIX:-wccu_provider_robustness}"
GEMINI_TAG="${GEMINI_ONLY_TAG:-${BASE_TAG_PREFIX}_gemini31_seed${SEED}}"
MIXED_TAG="${MIXED_PROVIDER_TAG:-${BASE_TAG_PREFIX}_mixed_openai_gemini_seed${SEED}}"
BASELINE_TAGS="${BASELINE_TAGS:-wccu_frozen_seed7 wccu_frozen_seed13 wccu_frozen_seed7_unguided}"

if [[ "${RUN_GEMINI_ONLY:-1}" != "0" ]]; then
  export EXP_TAG="${GEMINI_TAG}"
  bash scripts/run_wccu_gemini_only_frozen_replay.sh
fi

if [[ "${RUN_MIXED_PROVIDER:-1}" != "0" ]]; then
  export EXP_TAG="${MIXED_TAG}"
  bash scripts/run_wccu_mixed_provider_frozen_replay.sh
fi

SUMMARY_DIR="analysis/${BASE_TAG_PREFIX}_summary"
mkdir -p "${SUMMARY_DIR}"
# shellcheck disable=SC2086
python -m wccu_eval.scripts.make_provider_robustness_summary \
  --tags ${BASELINE_TAGS} "${GEMINI_TAG}" "${MIXED_TAG}" \
  --out-csv "${SUMMARY_DIR}/provider_robustness_summary.csv" \
  --out-json "${SUMMARY_DIR}/provider_robustness_summary.json" \
  --out-md "${SUMMARY_DIR}/provider_robustness_summary.md" || true

echo "Done. Summary, if all tags were available: ${SUMMARY_DIR}/provider_robustness_summary.md"
