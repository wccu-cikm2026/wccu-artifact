# Reproducing the WCCU Experiments

This artifact separates three workflows:

1. **Offline validation**: no API keys, intended for quick reviewer checks.
2. **Frozen CooperBench-derived replay**: live proposal generation followed by
   deterministic replay across policies; this is the closest reproduction path
   for the paper's CooperBench-derived tables.
3. **Controlled obligation diagnostics**: live LLM generations over small typed
   scenarios used to isolate target, authority, operation, freshness, view, and
   witness obligations.

Generated outputs are not committed. They are written to `data/$EXP_TAG`,
`results/$EXP_TAG`, `analysis/$EXP_TAG`, and `logs/`.

## 1. Offline validation

```bash
python -m compileall wccu_eval tests scripts
python -m unittest discover -s tests -v
bash -n scripts/*.sh scripts/lib/*.sh
```

Optional no-API smoke test on the included mini sample:

```bash
EXP_TAG=artifact_mock_smoke \
COOPERBENCH_INPUT=data/cooperbench_mini_sample.jsonl \
WCCU_COOPER_MOCK_LLM=1 \
WCCU_COOPER_SUBSET_SIZE=2 \
WCCU_COOPER_COMMITMENT_LIMIT=2 \
WCCU_COOPER_REPETITIONS=1 \
./scripts/run_wccu_cooperbench_derived.sh
```

## 2. Configure a live provider

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER, LLM_MODEL, and the matching API key
python -m wccu_eval.scripts.check_real_llm_config
python -m wccu_eval.scripts.check_llm_provider --scenario high_risk_rule_change
```

Provider calls are retried on retryable gateway/rate-limit errors. Provider
errors are logged under `results/$EXP_TAG/provider_errors.jsonl` or the path
specified by `LLM_ERROR_LOG_PATH`.

## 3. Prepare CooperBench-derived data

This step makes no LLM/provider calls. It can download the public HuggingFace
CooperBench dataset, convert a local snapshot, or reuse an already converted
JSONL file.

```bash
pip install huggingface_hub  # optional, only for HuggingFace download
export EXP_TAG=cooperbench_seed7
export WCCU_COOPER_SEED=7
export WCCU_COOPER_SUBSET_SIZE=50
export WCCU_COOPER_COMMITMENT_LIMIT=50
export WCCU_COOPER_MAX_TASKS=120
export COOPERBENCH_HF_DATASET=CodeConflict/cooperbench-dataset
export COOPERBENCH_HF_SUBDIR=openai_tiktoken_task
./scripts/prepare_cooperbench_data.sh
```

Alternatively:

```bash
export EXP_TAG=cooperbench_seed7
export COOPERBENCH_SNAPSHOT=/path/to/local/cooperbench_snapshot
./scripts/prepare_cooperbench_data.sh

# or
export EXP_TAG=cooperbench_seed7
export COOPERBENCH_INPUT=/path/to/cooperbench_converted.jsonl
./scripts/prepare_cooperbench_data.sh
```

The script writes:

```text
data/$EXP_TAG/cooperbench_converted.jsonl
data/$EXP_TAG/cooperbench_subset50_seed7.jsonl
data/$EXP_TAG/cooperbench_commitment_stale_subset50_seed7.jsonl
analysis/$EXP_TAG/cooperbench_dataset_report.md
```

These are CooperBench-derived metadata fixtures for context-store coordination
experiments, not official CooperBench VM/test-harness results.

## 4. CooperBench-derived frozen replay for Table 3

Run three 50-scenario OpenAI-backed commitment-staleness runs, matching the
paper's aggregate design: two guided samples and one unguided sample.

```bash
# guided, seed 7
export LLM_PROVIDER=openai
export LLM_MODEL=<openai-model-id>
export COOPERBENCH_SNAPSHOT=/path/to/local/cooperbench_snapshot
export EXP_TAG=table3_oai_guided_seed7
export WCCU_COOPER_SEED=7
export WCCU_COOPER_SUBSET_SIZE=50
export WCCU_COOPER_COMMITMENT_LIMIT=50
export WCCU_COOPER_CERTIFICATE_GUIDANCE=guided
./scripts/run_wccu_true_frozen_replay.sh

# guided, seed 13
export EXP_TAG=table3_oai_guided_seed13
export WCCU_COOPER_SEED=13
export WCCU_COOPER_CERTIFICATE_GUIDANCE=guided
./scripts/run_wccu_true_frozen_replay.sh

# unguided, seed 7
export EXP_TAG=table3_oai_unguided_seed7
export WCCU_COOPER_SEED=7
export WCCU_COOPER_CERTIFICATE_GUIDANCE=unguided
./scripts/run_wccu_true_frozen_replay.sh
```

Per-run commitment tables are written to:

```text
analysis/$EXP_TAG/cooperbench_commitment_frozen_replay/cooperbench_commitment_table.csv
analysis/$EXP_TAG/cooperbench_commitment_frozen_replay/wccu_commitment_ablation.csv
analysis/$EXP_TAG/frozen_replay_wccu_ablation.csv
```

Aggregate per-run summaries:

```bash
python -m wccu_eval.scripts.summarize_paper_runs \
  --glob 'analysis/table3_*/frozen_replay_wccu_ablation.csv' \
  --out-csv analysis/table3_aggregate.csv \
  --out-md analysis/table3_aggregate.md
```

The exact values may differ if provider model versions change; the replay phase
is deterministic once frozen bundles have been generated.

## 5. Provider robustness for Table 4

Gemini-only replacement:

```bash
export GEMINI_API_KEY=...
export LLM_PROVIDER=gemini
export LLM_MODEL=gemini-3.1-flash-lite
export GEMINI_RESPONSE_SCHEMA_MODE=sanitize
export COOPERBENCH_SNAPSHOT=/path/to/local/cooperbench_snapshot
export EXP_TAG=table4_gemini_guided_seed7
./scripts/run_wccu_gemini_only_frozen_replay.sh
```

Mixed OpenAI/Gemini agents over the same context store:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export OPENAI_AGENT_MODEL=<openai-model-id>
export GEMINI_AGENT_MODEL=gemini-3.1-flash-lite
export COOPERBENCH_SNAPSHOT=/path/to/local/cooperbench_snapshot
export EXP_TAG=table4_mixed_openai_gemini_seed7
./scripts/run_wccu_mixed_provider_frozen_replay.sh
```

Convenience wrapper for both robustness runs and a summary file:

```bash
export BASELINE_TAGS="table3_oai_guided_seed7 table3_oai_guided_seed13 table3_oai_unguided_seed7"
./scripts/run_wccu_provider_robustness_suite.sh
```

The summary helper writes `analysis/<suite>_summary/provider_robustness_summary.*`
when all referenced tags are available.

## 6. Controlled live-LLM obligation diagnostic for Table 5

Run guided and unguided variants:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=<openai-model-id>

EXP_TAG=table5_guided \
WCCU_LIMIT_PER_FAMILY=30 \
WCCU_CERTIFICATE_GUIDANCE=guided \
./scripts/run_wccu_main_llm_obligation.sh

EXP_TAG=table5_unguided \
WCCU_LIMIT_PER_FAMILY=30 \
WCCU_CERTIFICATE_GUIDANCE=unguided \
./scripts/run_wccu_main_llm_obligation.sh
```

Per-run outputs:

```text
results/$EXP_TAG/llm_obligation_<guided|unguided>.json
analysis/$EXP_TAG/llm_obligation_<guided|unguided>.csv
analysis/$EXP_TAG/llm_obligation_<guided|unguided>.tex
analysis/$EXP_TAG/coordination_metrics_llm_obligation_<guided|unguided>.csv
```

To inspect a result manually:

```bash
python -m wccu_eval.scripts.make_llm_obligation_tables \
  results/table5_guided/llm_obligation_guided.json \
  --out-prefix analysis/table5_guided/llm_obligation_guided
```

## 7. Packaging newly generated outputs

```bash
./scripts/package_wccu_experiment_outputs.sh --tag $EXP_TAG
```

The package script excludes `.env`, virtual environments, caches, `.git`, and
provider keys. For anonymous submission, inspect any generated ZIP before
uploading it.

## 8. Notes for reviewers

Development notes are intentionally not included in the anonymous artifact. The
supported workflow is offline validation, CooperBench-derived data preparation,
frozen replay for Tables 3--4, controlled diagnostics for Table 5, and optional
packaging of newly generated outputs.
