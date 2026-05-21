#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/load_wccu_env.sh"

# Gemini-only variant of the true frozen replay protocol.  This is used to
# test whether WCCU works when all agents are backed by Gemini rather than the
# OpenAI model used in the primary run.

export EXP_TAG="${EXP_TAG:-wccu_frozen_gemini31_seed${WCCU_COOPER_SEED:-7}}"
export LLM_PROVIDER="gemini"
export LLM_MODEL="${LLM_MODEL:-${GEMINI_MODEL:-gemini-3.1-flash-lite}}"
export GEMINI_RESPONSE_SCHEMA_MODE="${GEMINI_RESPONSE_SCHEMA_MODE:-sanitize}"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "Missing GEMINI_API_KEY" >&2
  exit 2
fi

bash scripts/run_wccu_true_frozen_replay.sh
