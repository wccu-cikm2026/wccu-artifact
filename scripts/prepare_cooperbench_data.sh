#!/usr/bin/env bash
set -euo pipefail

# Prepare CooperBench-derived data for the WCCU artifact without making any
# provider/API calls. The script can either download a HuggingFace dataset
# snapshot, convert an existing local CooperBench snapshot, or reuse an already
# converted JSONL file. It then samples a subset, creates the cross-target
# commitment-staleness diagnostic, and writes a dataset report.

EXP_TAG="${EXP_TAG:-cooperbench_prep_seed${WCCU_COOPER_SEED:-7}}"
SEED="${WCCU_COOPER_SEED:-7}"
SUBSET_SIZE="${WCCU_COOPER_SUBSET_SIZE:-50}"
COMMITMENT_LIMIT="${WCCU_COOPER_COMMITMENT_LIMIT:-50}"
MAX_TASKS="${WCCU_COOPER_MAX_TASKS:-120}"
FEATURE_MAX_CHARS="${WCCU_COOPER_FEATURE_MAX_CHARS:-1400}"

mkdir -p "data/${EXP_TAG}" "analysis/${EXP_TAG}" "logs"

CONVERTED="data/${EXP_TAG}/cooperbench_converted.jsonl"

if [[ -n "${COOPERBENCH_INPUT:-}" ]]; then
  echo "Using existing converted CooperBench JSONL: ${COOPERBENCH_INPUT}"
  CONVERTED="${COOPERBENCH_INPUT}"
elif [[ -n "${COOPERBENCH_SNAPSHOT:-}" ]]; then
  echo "Converting local CooperBench snapshot: ${COOPERBENCH_SNAPSHOT}"
  python -m wccu_eval.scripts.convert_cooperbench_dataset \
    --input "${COOPERBENCH_SNAPSHOT}" \
    --out "${CONVERTED}" \
    --max-tasks "${MAX_TASKS}" \
    --feature-max-chars "${FEATURE_MAX_CHARS}" \
    --inspect
else
  HF_DATASET="${COOPERBENCH_HF_DATASET:-CodeConflict/cooperbench-dataset}"
  HF_SUBDIR="${COOPERBENCH_HF_SUBDIR:-openai_tiktoken_task}"
  HF_REVISION="${COOPERBENCH_HF_REVISION:-main}"
  DOWNLOAD_DIR="${COOPERBENCH_DOWNLOAD_DIR:-data/${EXP_TAG}/cooperbench_hf_snapshot}"
  echo "Downloading and converting HuggingFace dataset: ${HF_DATASET} (${HF_SUBDIR})"
  echo "If this fails, install the optional dependency with: pip install huggingface_hub"
  python -m wccu_eval.scripts.convert_cooperbench_dataset \
    --hf-dataset "${HF_DATASET}" \
    --hf-subdir "${HF_SUBDIR}" \
    --hf-revision "${HF_REVISION}" \
    --download-dir "${DOWNLOAD_DIR}" \
    --out "${CONVERTED}" \
    --max-tasks "${MAX_TASKS}" \
    --feature-max-chars "${FEATURE_MAX_CHARS}" \
    --inspect
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

cat > "data/${EXP_TAG}/COOPERBENCH_DERIVED_DATA_NOTE.txt" <<NOTE
CooperBench-derived data prepared for WCCU artifact experiments.
EXP_TAG: ${EXP_TAG}
Seed: ${SEED}
Converted JSONL: ${CONVERTED}
Workspace subset: ${SUBSET}
Commitment-staleness diagnostic: ${COMMITMENT}
Dataset report: analysis/${EXP_TAG}/cooperbench_dataset_report.md

These files are derived metadata for context-store coordination experiments;
they are not an official CooperBench score or VM/test-harness run.
NOTE

echo "Prepared CooperBench-derived data:"
echo "  converted:  ${CONVERTED}"
echo "  subset:     ${SUBSET}"
echo "  diagnostic: ${COMMITMENT}"
echo "  report:     analysis/${EXP_TAG}/cooperbench_dataset_report.md"
