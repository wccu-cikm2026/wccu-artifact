#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/load_wccu_env.sh"

# Mixed-provider variant of the true frozen replay protocol.
# Agent A uses OpenAI and Agent B uses Gemini while both receive projections from
# the same canonical context store and submit WCCUs to the same verifier/policy
# pipeline.  The generated mixed-provider bundle is then replayed across policies
# without further provider calls.

export EXP_TAG="${EXP_TAG:-wccu_frozen_mixed_openai_gemini_seed${WCCU_COOPER_SEED:-7}}"
export OPENAI_AGENT_MODEL="${OPENAI_AGENT_MODEL:-${OPENAI_MODEL:-gpt-5.4-nano}}"
export GEMINI_AGENT_MODEL="${GEMINI_AGENT_MODEL:-gemini-3.1-flash-lite}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Missing OPENAI_API_KEY for coop_agent_a" >&2
  exit 2
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "Missing GEMINI_API_KEY for coop_agent_b" >&2
  exit 2
fi

# The default provider/model only apply to unmatched agents.  Both CooperBench
# agents are explicitly routed below.
export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export LLM_MODEL="${LLM_MODEL:-${OPENAI_AGENT_MODEL}}"
export WCCU_AGENT_MODEL_SPECS="${WCCU_AGENT_MODEL_SPECS:-coop_agent_a=openai:${OPENAI_AGENT_MODEL},coop_agent_b=gemini:${GEMINI_AGENT_MODEL}}"

# Gemini's provider-side structured-output schema is less permissive than
# OpenAI's strict JSON Schema; sanitize by default, and allow callers to set
# GEMINI_RESPONSE_SCHEMA_MODE=none if their endpoint still rejects nested schema.
export GEMINI_RESPONSE_SCHEMA_MODE="${GEMINI_RESPONSE_SCHEMA_MODE:-sanitize}"

bash scripts/run_wccu_true_frozen_replay.sh
