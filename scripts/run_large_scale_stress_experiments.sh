#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_TAG="${EXP_TAG:-large_stress_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-7 13 21}"
STRESS_CASES="${STRESS_CASES:-250}"
STRESS_WRITER_GRID="${STRESS_WRITER_GRID:-4 8 16 32}"
STRESS_INVALIDATION_PROBS="${STRESS_INVALIDATION_PROBS:-0.00 0.10 0.35 0.60}"
STRESS_ATOM_COUNT="${STRESS_ATOM_COUNT:-128}"
STRESS_REPETITIONS="${STRESS_REPETITIONS:-1}"
STRESS_CONDITIONS="${STRESS_CONDITIONS:-adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only}"
RUN_TESTS="${RUN_TESTS:-1}"

mkdir -p "results/${EXP_TAG}" "analysis/${EXP_TAG}" logs

if [[ "$RUN_TESTS" == "1" ]]; then
  python -m compileall wccu_eval scripts tests \
    2>&1 | tee "logs/${EXP_TAG}_compile.log"
  python -m unittest discover -s tests -v \
    2>&1 | tee "logs/${EXP_TAG}_unit_tests.log"
fi

RESULT_JSONS=()
RESULT_JSONLS=()
CELL_COUNT=0
for seed in $SEEDS; do
  for writers in $STRESS_WRITER_GRID; do
    for p in $STRESS_INVALIDATION_PROBS; do
      OUT="results/${EXP_TAG}/wccu_stress_w${writers}_a${STRESS_ATOM_COUNT}_p${p}_seed${seed}.json"
      echo "=== stress seed=${seed} writers=${writers} p=${p} cases=${STRESS_CASES} ==="
      python -m wccu_eval.eval.run_wccu_stress \
        --cases "$STRESS_CASES" \
        --writers "$writers" \
        --atom-count "$STRESS_ATOM_COUNT" \
        --invalidation-prob "$p" \
        --seed "$seed" \
        --condition "$STRESS_CONDITIONS" \
        --repetitions "$STRESS_REPETITIONS" \
        --out "$OUT" \
        2>&1 | tee "logs/${EXP_TAG}_stress_seed${seed}_w${writers}_p${p}.log"
      RESULT_JSONS+=("$OUT")
      RESULT_JSONLS+=("${OUT%.json}.jsonl")
      CELL_COUNT=$((CELL_COUNT + STRESS_CASES))
    done
  done
done

python -m wccu_eval.scripts.make_stress_table "${RESULT_JSONS[@]}" \
  --out-csv "analysis/${EXP_TAG}/large_stress_aggregate_table.csv" \
  --out-tex "analysis/${EXP_TAG}/large_stress_aggregate_table.tex"

python -m wccu_eval.scripts.make_stress_curve_table "${RESULT_JSONS[@]}" \
  --out-csv "analysis/${EXP_TAG}/large_stress_curve_table.csv" \
  --out-tex "analysis/${EXP_TAG}/large_stress_curve_table.tex"

python -m wccu_eval.scripts.make_coordination_metrics_table "${RESULT_JSONLS[@]}" \
  --out-csv "analysis/${EXP_TAG}/large_stress_coordination_metrics_table.csv" \
  --out-tex "analysis/${EXP_TAG}/large_stress_coordination_metrics_table.tex" \
  --full-tex

python -m wccu_eval.scripts.make_coordination_metrics_table "${RESULT_JSONLS[@]}" \
  --group-by-setting \
  --out-csv "analysis/${EXP_TAG}/large_stress_coordination_curve.csv" \
  --out-tex "analysis/${EXP_TAG}/large_stress_coordination_curve.tex" \
  --full-tex

cat > "analysis/${EXP_TAG}/large_stress_manifest.md" <<EOF
# Large deterministic WCCU stress sweep

- EXP_TAG: ${EXP_TAG}
- Seeds: ${SEEDS}
- Cases per seed/writer/probability cell: ${STRESS_CASES}
- Writer grid: ${STRESS_WRITER_GRID}
- Invalidation probabilities: ${STRESS_INVALIDATION_PROBS}
- Atom count: ${STRESS_ATOM_COUNT}
- Repetitions: ${STRESS_REPETITIONS}
- Conditions: ${STRESS_CONDITIONS}

This script is intended to support the scale/sensitivity claim.  It is larger
than the reviewer-response smoke suite, but it is still deterministic and does
not require an LLM provider.  Use the coordination-curve CSV for Pareto-style
plots and the aggregate table for the main stress summary.
EOF

zip -r "${EXP_TAG}_large_stress_outputs.zip" \
  "results/${EXP_TAG}" "analysis/${EXP_TAG}" logs/${EXP_TAG}_*.log >/dev/null

echo "EXP_TAG=${EXP_TAG}"
echo "Wrote ${EXP_TAG}_large_stress_outputs.zip"
