#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/load_wccu_env.sh"

# True frozen replay protocol for CooperBench-derived WCCU evaluation.
# Phase 1: call the real LLM exactly once per scenario/agent under a neutral
# generation condition and save the agent outputs as a frozen bundle.
# Phase 2: replay the same frozen outputs across commit policies without
# provider calls.  The replay JSON has frozen_replay.provider_api_called_in_replay=false.

EXP_TAG="${EXP_TAG:-wccu_frozen_seed${WCCU_COOPER_SEED:-7}}"
SEED="${WCCU_COOPER_SEED:-7}"
SUBSET_SIZE="${WCCU_COOPER_SUBSET_SIZE:-50}"
COMMITMENT_LIMIT="${WCCU_COOPER_COMMITMENT_LIMIT:-50}"
MAX_TASKS="${WCCU_COOPER_MAX_TASKS:-120}"
CONDITIONS="${WCCU_COOPER_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_wccu_no_read_validation,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"
GEN_CONDITION="${WCCU_FROZEN_GENERATION_CONDITION:-adaptive_wccu_execution_trace}"
PARALLEL_WORKERS="${WCCU_COOPER_PARALLEL_WORKERS:-1}"
CERT_GUIDANCE="${WCCU_COOPER_CERTIFICATE_GUIDANCE:-guided}"
AGENT_MODEL_SPECS="${WCCU_AGENT_MODEL_SPECS:-${LLM_AGENT_MODEL_SPECS:-}}"

mkdir -p "data/${EXP_TAG}" "results/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

if [[ -z "${COOPERBENCH_INPUT:-}" ]]; then
  if [[ -z "${COOPERBENCH_SNAPSHOT:-}" ]]; then
    echo "Set COOPERBENCH_INPUT=/path/to/converted.jsonl or COOPERBENCH_SNAPSHOT=/path/to/snapshot" >&2
    exit 2
  fi
  CONVERTED="data/${EXP_TAG}/cooperbench_converted.jsonl"
  python -m wccu_eval.scripts.convert_cooperbench_dataset \
    --input "${COOPERBENCH_SNAPSHOT}" \
    --out "${CONVERTED}" \
    --max-tasks "${MAX_TASKS}"
else
  CONVERTED="${COOPERBENCH_INPUT}"
fi

SUBSET="data/${EXP_TAG}/cooperbench_subset${SUBSET_SIZE}_seed${SEED}.jsonl"
COMMITMENT="data/${EXP_TAG}/cooperbench_commitment_stale_subset${COMMITMENT_LIMIT}_seed${SEED}.jsonl"

python -m wccu_eval.scripts.sample_cooperbench_subset \
  --input "${CONVERTED}" \
  --out "${SUBSET}" \
  --size "${SUBSET_SIZE}" \
  --seed "${SEED}"

python -m wccu_eval.scripts.make_cooperbench_commitment_diagnostics \
  --input "${SUBSET}" \
  --out "${COMMITMENT}" \
  --limit "${COMMITMENT_LIMIT}" \
  --seed "${SEED}"

python -m wccu_eval.scripts.make_cooperbench_dataset_report \
  --converted "${CONVERTED}" \
  --subset "${SUBSET}" \
  --commitment-diag "${COMMITMENT}" \
  --out-json "analysis/${EXP_TAG}/cooperbench_dataset_report.json" \
  --out-csv "analysis/${EXP_TAG}/cooperbench_dataset_report.csv" \
  --out-md "analysis/${EXP_TAG}/cooperbench_dataset_report.md"

# -------------------------
# Workspace frozen protocol
# -------------------------
WORK_GEN="results/${EXP_TAG}/cooperbench_workspace_generation.json"
WORK_BUNDLE="results/${EXP_TAG}/cooperbench_workspace_frozen_bundle.json"
WORK_REPLAY="results/${EXP_TAG}/cooperbench_workspace_frozen_replay.json"

python -m wccu_eval.eval.run_cooperbench_frozen_replay generate \
  --input "${SUBSET}" \
  --out "${WORK_GEN}" \
  --bundle-out "${WORK_BUNDLE}" \
  --generation-condition "${GEN_CONDITION}" \
  --provider "${LLM_PROVIDER:-openai}" \
  --model "${LLM_MODEL:-}" \
  --agent-model-specs "${AGENT_MODEL_SPECS}" \
  --max-output-tokens "${WCCU_COOPER_MAX_OUTPUT_TOKENS:-1800}" \
  --timeout-seconds "${WCCU_COOPER_TIMEOUT_SECONDS:-180}" \
  --max-provider-retries "${WCCU_COOPER_MAX_PROVIDER_RETRIES:-8}" \
  --retry-backoff-max "${WCCU_COOPER_RETRY_BACKOFF_MAX:-30}" \
  --parallel-workers "${PARALLEL_WORKERS}" \
  --certificate-guidance "${CERT_GUIDANCE}" \
  --bundle-id "${EXP_TAG}_workspace"

python -m wccu_eval.eval.run_cooperbench_frozen_replay replay \
  --input "${SUBSET}" \
  --frozen-bundle "${WORK_BUNDLE}" \
  --condition "${CONDITIONS}" \
  --out "${WORK_REPLAY}" \
  --parallel-workers "${PARALLEL_WORKERS}"

python -m wccu_eval.scripts.analyze_results "${WORK_REPLAY}" --out-dir "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay"
python -m wccu_eval.scripts.make_cooperbench_table "${WORK_REPLAY}" --out-csv "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay/cooperbench_table.csv" --out-tex "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay/cooperbench_table.tex"
python -m wccu_eval.scripts.make_cooperbench_table "${WORK_REPLAY}" --out-csv "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay/cooperbench_workspace_table.csv" --out-tex "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay/cooperbench_workspace_table.tex"
python -m wccu_eval.scripts.make_token_usage_table "${WORK_REPLAY}" --out-csv "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay/token_usage_table.csv" --out-tex "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay/token_usage_table.tex" --out-json "analysis/${EXP_TAG}/cooperbench_workspace_frozen_replay/token_usage_table.json"

# ----------------------------
# Commitment frozen protocol
# ----------------------------
COMMIT_GEN="results/${EXP_TAG}/cooperbench_commitment_generation.json"
COMMIT_BUNDLE="results/${EXP_TAG}/cooperbench_commitment_frozen_bundle.json"
COMMIT_REPLAY="results/${EXP_TAG}/cooperbench_commitment_frozen_replay.json"

python -m wccu_eval.eval.run_cooperbench_frozen_replay generate \
  --input "${COMMITMENT}" \
  --out "${COMMIT_GEN}" \
  --bundle-out "${COMMIT_BUNDLE}" \
  --generation-condition "${GEN_CONDITION}" \
  --provider "${LLM_PROVIDER:-openai}" \
  --model "${LLM_MODEL:-}" \
  --agent-model-specs "${AGENT_MODEL_SPECS}" \
  --max-output-tokens "${WCCU_COOPER_MAX_OUTPUT_TOKENS:-1800}" \
  --timeout-seconds "${WCCU_COOPER_TIMEOUT_SECONDS:-180}" \
  --max-provider-retries "${WCCU_COOPER_MAX_PROVIDER_RETRIES:-8}" \
  --retry-backoff-max "${WCCU_COOPER_RETRY_BACKOFF_MAX:-30}" \
  --parallel-workers "${PARALLEL_WORKERS}" \
  --certificate-guidance "${CERT_GUIDANCE}" \
  --bundle-id "${EXP_TAG}_commitment"

python -m wccu_eval.eval.run_cooperbench_frozen_replay replay \
  --input "${COMMITMENT}" \
  --frozen-bundle "${COMMIT_BUNDLE}" \
  --condition "${CONDITIONS}" \
  --out "${COMMIT_REPLAY}" \
  --parallel-workers "${PARALLEL_WORKERS}"

python -m wccu_eval.scripts.analyze_results "${COMMIT_REPLAY}" --out-dir "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay"
python -m wccu_eval.scripts.make_cooperbench_commitment_table "${COMMIT_REPLAY}" --out-csv "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay/cooperbench_commitment_table.csv" --out-tex "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay/cooperbench_commitment_table.tex"
python -m wccu_eval.scripts.make_wccu_ablation_table "${COMMIT_REPLAY}" --out-csv "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay/wccu_commitment_ablation.csv" --out-tex "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay/wccu_commitment_ablation.tex"
python -m wccu_eval.scripts.make_token_usage_table "${COMMIT_REPLAY}" --out-csv "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay/token_usage_table.csv" --out-tex "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay/token_usage_table.tex" --out-json "analysis/${EXP_TAG}/cooperbench_commitment_frozen_replay/token_usage_table.json"
python -m wccu_eval.scripts.make_wccu_real_llm_ablation_table \
  "${WORK_REPLAY}" "${COMMIT_REPLAY}" \
  --out-csv "analysis/${EXP_TAG}/frozen_replay_wccu_ablation.csv" \
  --out-json "analysis/${EXP_TAG}/frozen_replay_wccu_ablation.json" \
  --out-tex "analysis/${EXP_TAG}/frozen_replay_wccu_ablation.tex"

cat > "results/${EXP_TAG}/FROZEN_REPLAY_RUN_NOTE.txt" <<EOF
True frozen replay run: ${EXP_TAG}
Seed: ${SEED}
Generation condition: ${GEN_CONDITION}
Certificate guidance: ${CERT_GUIDANCE}
Agent model specs: ${AGENT_MODEL_SPECS:-<single-provider>}
Conditions replayed: ${CONDITIONS}
Workspace generation: ${WORK_GEN}
Workspace bundle: ${WORK_BUNDLE}
Workspace replay: ${WORK_REPLAY}
Commitment generation: ${COMMIT_GEN}
Commitment bundle: ${COMMIT_BUNDLE}
Commitment replay: ${COMMIT_REPLAY}
Replay phase provider calls: false by construction; see frozen_replay.provider_api_called_in_replay in replay JSON.
EOF

bash ./scripts/package_wccu_experiment_outputs.sh --tag "${EXP_TAG}" || true

echo "Done. Inspect results/${EXP_TAG}/FROZEN_REPLAY_RUN_NOTE.txt"
