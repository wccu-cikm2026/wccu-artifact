# WCCU Artifact

This repository contains the anonymized artifact for the paper on **Witness-Carrying Context Updates (WCCUs)**.  The code implements a lightweight context store, WCCU verifier, policy selector, CooperBench-derived workload adapter, live/frozen LLM runners, and analysis scripts used to evaluate dependency-aware context updates.

The artifact is intentionally separated from the development repository. It excludes full packaged experiment-result ZIP files, raw provider logs, and development notes. Reviewers can reproduce the tables by running the scripts below; generated `data/`, `results/`, `analysis/`, and `logs/` directories are ignored by default.

## Repository layout

```text
wccu_eval/      Python package: store, verifier, policies, runners, analysis
scripts/                     Shell entry points for offline checks and paper experiments
data/cooperbench_mini_sample.jsonl
                             Small sample for smoke tests and command validation
tests/                       Unit tests for policy, WCCU, data conversion, and tables
README_REPRODUCE.md          Detailed reproduction workflow
.env.example                 Provider configuration template; no real keys
```

A license file is not included in this ZIP because it should be created from the anonymous artifact-hosting account before submission.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package has no required third-party Python dependencies for the offline tests.

## Quick offline validation

Run these checks before pushing the anonymous repository:

```bash
python -m compileall wccu_eval tests scripts
python -m unittest discover -s tests -v
```

Expected result: all tests pass.  This validates the verifier, policy logic, table builders, and mock/offline paths, but it does not reproduce the live LLM paper numbers.

## No-API smoke run on the included mini sample

```bash
EXP_TAG=artifact_mock_smoke \
COOPERBENCH_INPUT=data/cooperbench_mini_sample.jsonl \
WCCU_COOPER_MOCK_LLM=1 \
WCCU_COOPER_SUBSET_SIZE=2 \
WCCU_COOPER_COMMITMENT_LIMIT=2 \
WCCU_COOPER_REPETITIONS=1 \
./scripts/run_wccu_cooperbench_derived.sh
```

This command exercises dataset sampling, commitment-diagnostic construction, mock proposal generation, WCCU verification, and table generation without calling an external provider.  Its outputs appear under:

```text
results/artifact_mock_smoke/
analysis/artifact_mock_smoke/
logs/
```

## Live LLM configuration

Copy the template and fill in exactly one provider configuration:

```bash
cp .env.example .env
# edit .env
python -m wccu_eval.scripts.check_real_llm_config
python -m wccu_eval.scripts.check_llm_provider --scenario high_risk_rule_change
```

The artifact supports `openai`, `gemini`, and OpenAI-compatible endpoints.  Do not commit `.env` or provider logs.

## CooperBench-derived data preparation

The artifact includes the preprocessing path that turns CooperBench metadata into the workspace-contention and cross-target commitment-staleness fixtures used by the experiments. To download from HuggingFace and prepare the derived JSONL files without any LLM/provider calls:

```bash
pip install huggingface_hub  # optional, only for HuggingFace download
export EXP_TAG=cooperbench_seed7
export COOPERBENCH_HF_DATASET=CodeConflict/cooperbench-dataset
export COOPERBENCH_HF_SUBDIR=openai_tiktoken_task
export WCCU_COOPER_SEED=7
export WCCU_COOPER_SUBSET_SIZE=50
export WCCU_COOPER_COMMITMENT_LIMIT=50
./scripts/prepare_cooperbench_data.sh
```

The same script accepts `COOPERBENCH_SNAPSHOT=/path/to/local/snapshot` or `COOPERBENCH_INPUT=/path/to/converted.jsonl`. It writes the converted dataset, sampled subset, commitment-staleness diagnostic, and dataset report under `data/$EXP_TAG/` and `analysis/$EXP_TAG/`.

## Paper-result reproduction at a glance

The paper uses a two-phase live-generation/fixed-proposal replay protocol for the CooperBench-derived results.  Use the frozen replay runner for the closest reproduction path. The runner can convert/sample data itself from `COOPERBENCH_SNAPSHOT`, or it can use a converted JSONL prepared with `prepare_cooperbench_data.sh`:

```bash
export COOPERBENCH_SNAPSHOT=/path/to/local/cooperbench_snapshot
# or: export COOPERBENCH_INPUT=data/cooperbench_seed7/cooperbench_converted.jsonl
export LLM_PROVIDER=openai
export LLM_MODEL=<model-id>
export EXP_TAG=paper_oai_guided_seed7
export WCCU_COOPER_SEED=7
export WCCU_COOPER_SUBSET_SIZE=50
export WCCU_COOPER_COMMITMENT_LIMIT=50
export WCCU_COOPER_CERTIFICATE_GUIDANCE=guided
./scripts/run_wccu_true_frozen_replay.sh
```

The generation phase calls the provider once per scenario/agent and stores frozen proposal bundles.  The replay phase evaluates the same proposals across commit policies without provider calls.  Key outputs:

```text
results/$EXP_TAG/cooperbench_workspace_frozen_bundle.json
results/$EXP_TAG/cooperbench_workspace_frozen_replay.json
results/$EXP_TAG/cooperbench_commitment_frozen_bundle.json
results/$EXP_TAG/cooperbench_commitment_frozen_replay.json
analysis/$EXP_TAG/cooperbench_commitment_frozen_replay/cooperbench_commitment_table.csv
analysis/$EXP_TAG/frozen_replay_wccu_ablation.csv
```

See `README_REPRODUCE.md` for the full Table 3--5 workflow, including repeated seeds, provider-robustness runs, and controlled obligation diagnostics.

## Result ZIP policy

The original development package contained large result ZIP files.  They are deliberately excluded here to keep the anonymous repo lightweight and avoid leaking raw logs.  To package newly generated outputs after a run:

```bash
./scripts/package_wccu_experiment_outputs.sh --tag $EXP_TAG
```

Use `--include-source` only when preparing a standalone archival bundle, not for the anonymous GitHub repository.
