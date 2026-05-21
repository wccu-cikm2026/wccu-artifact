# WCCU Artifact

This repository contains the anonymized artifact for the paper on
**Witness-Carrying Context Updates (WCCUs)**. The code implements a lightweight
context store, WCCU verifier, policy selector, CooperBench-derived workload
adapter, live/frozen LLM runners, provider-robustness runners, and analysis
scripts used to evaluate dependency-aware context updates.

The artifact excludes full packaged experiment-result ZIP files, raw provider
logs, development notes, and repository history. Reviewers can reproduce the
reported tables by running the scripts below; generated `data/`, `results/`,
`analysis/`, and `logs/` directories are ignored by default.

## Repository layout

```text
wccu_eval/      Python package: store, verifier, policies, runners, analysis
scripts/        Shell entry points for offline checks and paper experiments
data/           Mini sample plus notes; full data is generated locally
tests/          Unit tests for policy, WCCU, conversion, providers, and tables
README_REPRODUCE.md          Detailed reproduction workflow
.env.example                 Provider configuration template; no real keys
```

A license file is not included in this ZIP because it should be created from the
anonymous artifact-hosting account before submission.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package has no required third-party Python dependencies for the offline
unit tests. HuggingFace download support for CooperBench data preparation uses
`huggingface_hub` as an optional dependency.

## Quick offline validation

```bash
python -m compileall wccu_eval tests scripts
python -m unittest discover -s tests -v
bash -n scripts/*.sh scripts/lib/*.sh
```

Expected result: all tests pass. These checks validate the verifier, policy
logic, table builders, mixed-provider routing, shell environment loader, and
mock/offline paths, but they do not reproduce live LLM paper numbers.

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

This command exercises dataset sampling, commitment-diagnostic construction,
mock proposal generation, WCCU verification, and table generation without
calling an external provider.

## Live LLM configuration

Copy the template and fill in a provider configuration:

```bash
cp .env.example .env
# edit .env
python -m wccu_eval.scripts.check_real_llm_config
python -m wccu_eval.scripts.check_llm_provider --scenario high_risk_rule_change
```

Shell entry points automatically load `.env` from the current directory, a
parent directory, or the repository root. You can override the file with
`WCCU_ENV_FILE=/path/to/.env`. The legacy `PCSE_ENV_FILE` variable is also
accepted for compatibility. Do not commit `.env` or provider logs.

## CooperBench-derived data preparation

The artifact includes the preprocessing path that turns CooperBench metadata into
workspace-contention and cross-target commitment-staleness fixtures. To download
from HuggingFace and prepare derived JSONL files without provider calls:

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

The same script accepts `COOPERBENCH_SNAPSHOT=/path/to/local/snapshot` or
`COOPERBENCH_INPUT=/path/to/converted.jsonl`. It writes converted data, a sampled
workspace subset, a commitment-staleness diagnostic, and a dataset report under
`data/$EXP_TAG/` and `analysis/$EXP_TAG/`.

## Paper-result reproduction at a glance

The paper uses a two-phase live-generation/fixed-proposal replay protocol for
CooperBench-derived results. Use the frozen replay runner for the closest
reproduction path:

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

The generation phase calls the provider once per scenario/agent and stores
frozen proposal bundles. The replay phase evaluates the same proposals across
commit policies without provider calls. Key outputs include:

```text
results/$EXP_TAG/cooperbench_workspace_frozen_bundle.json
results/$EXP_TAG/cooperbench_workspace_frozen_replay.json
results/$EXP_TAG/cooperbench_commitment_frozen_bundle.json
results/$EXP_TAG/cooperbench_commitment_frozen_replay.json
analysis/$EXP_TAG/cooperbench_commitment_frozen_replay/cooperbench_commitment_table.csv
analysis/$EXP_TAG/frozen_replay_wccu_ablation.csv
```

For provider-robustness runs, use:

```bash
./scripts/run_wccu_gemini_only_frozen_replay.sh
./scripts/run_wccu_mixed_provider_frozen_replay.sh
# or the convenience wrapper:
./scripts/run_wccu_provider_robustness_suite.sh
```

See `README_REPRODUCE.md` for the Table 3--5 workflow, including repeated seeds,
Gemini-only and mixed-provider runs, and controlled obligation diagnostics.

## Result ZIP policy

The original development package contained large result ZIP files. They are
excluded here to keep the anonymous repo lightweight and avoid leaking raw logs.
To package newly generated outputs after a run:

```bash
./scripts/package_wccu_experiment_outputs.sh --tag $EXP_TAG
```

Use `--include-source` only when preparing a standalone archival bundle, not for
the anonymous GitHub repository.
