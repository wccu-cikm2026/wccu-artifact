#!/usr/bin/env bash
set -euo pipefail

EXP_TAG=${EXP_TAG:-e2e_patch_cells_$(date +%Y%m%d_%H%M%S)}
INPUT=${INPUT:-data/e2e_patch_tasks.jsonl}
CONDITIONS=${CONDITIONS:-adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated}
RESULT=${RESULT:-results/${EXP_TAG}/e2e_patch_test.json}
ANALYSIS=${ANALYSIS:-analysis/${EXP_TAG}/e2e_patch_test}
WORK_DIR=${WORK_DIR:-runs/${EXP_TAG}/e2e_patch_test}
TIMEOUT_S=${TIMEOUT_S:-20}
ROW_TIMEOUT_S=${ROW_TIMEOUT_S:-90}
LIMIT=${LIMIT:-0}

mkdir -p "$(dirname "$RESULT")" "$ANALYSIS" "$WORK_DIR/_cell_inputs" "$WORK_DIR/_cell_results" logs
JSONL="${RESULT%.json}.jsonl"
: > "$JSONL"
IFS=',' read -r -a COND_ARR <<< "$CONDITIONS"

row_count=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  row_count=$((row_count + 1))
  if [[ "$LIMIT" != "0" && "$row_count" -gt "$LIMIT" ]]; then
    break
  fi
  task_id=$(python -c 'import json,sys; r=json.loads(sys.stdin.read()); print(r.get("task_id") or r.get("id") or "task")' <<< "$line")
  task_type=$(python -c 'import json,sys; r=json.loads(sys.stdin.read()); print(r.get("task_type") or (r.get("scenario") or {}).get("kind") or "unknown")' <<< "$line")
  for cond in "${COND_ARR[@]}"; do
    cond_trim=$(echo "$cond" | xargs)
    [[ -z "$cond_trim" ]] && continue
    echo "[e2e-cell] task=${row_count} id=${task_id} condition=${cond_trim}"
    cell="${row_count}_${task_id}_${cond_trim}"
    cell_input="$WORK_DIR/_cell_inputs/${cell}.jsonl"
    cell_out="$WORK_DIR/_cell_results/${cell}.json"
    printf '%s\n' "$line" > "$cell_input"
    if ! timeout "$ROW_TIMEOUT_S" python -m wccu_eval.eval.run_e2e_patch_test \
      --input "$cell_input" \
      --conditions "$cond_trim" \
      --out "$cell_out" \
      --work-dir "$WORK_DIR/_cell_work/${cell}" \
      --timeout-s "$TIMEOUT_S" \
      --limit 1 >/tmp/${EXP_TAG}_${cell}.stdout 2>/tmp/${EXP_TAG}_${cell}.stderr; then
      python - "$task_id" "$task_type" "$cond_trim" "$ROW_TIMEOUT_S" >> "$JSONL" <<'PY'
import json, sys
print(json.dumps({
    'kind':'e2e_patch_test_result_v1',
    'task_id':sys.argv[1],
    'task_type':sys.argv[2],
    'condition':sys.argv[3],
    'freshness_pass':False,
    'tests_passed':False,
    'patches_total':0,
    'patches_auto_applied':0,
    'patch_apply_failures':0,
    'patches_held_for_review':0,
    'stale_dependency_accepted_count':0,
    'unsafe_auto_commit_count':0,
    'review_burden_count':0,
    'wccu_intervention_count':0,
    'error':f'cell_timeout_or_failure_after_{sys.argv[4]}s'
}, sort_keys=True))
PY
      continue
    fi
    python - "$cell_out" >> "$JSONL" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
rows=payload.get('results') or []
if not rows:
    raise SystemExit('no rows in cell output')
print(json.dumps(rows[0], sort_keys=True))
PY
  done
done < "$INPUT"

python - "$JSONL" "$RESULT" "$INPUT" "$CONDITIONS" <<'PY'
import json, sys, datetime
jsonl, out, inp, cond = sys.argv[1:]
rows=[json.loads(line) for line in open(jsonl) if line.strip()]
payload={
  'kind':'e2e_patch_test_results_v1',
  'generated_at':datetime.datetime.utcnow().isoformat()+'Z',
  'args':{'input_path':inp,'conditions':cond,'out':out,'cell_runner':True},
  'results':rows,
}
json.dump(payload, open(out,'w'), indent=2, sort_keys=True)
PY

python -m wccu_eval.scripts.make_e2e_patch_test_table \
  "$RESULT" \
  --out-csv "$ANALYSIS/e2e_patch_test_table.csv" \
  --out-tex "$ANALYSIS/e2e_patch_test_table.tex" \
  --by-task-type-csv "$ANALYSIS/e2e_patch_test_by_task_type.csv" \
  --by-task-type-tex "$ANALYSIS/e2e_patch_test_by_task_type.tex"

zip -r "${EXP_TAG}_e2e_patch_test_outputs.zip" "$(dirname "$RESULT")" "$ANALYSIS" >/dev/null
printf 'Wrote %s_e2e_patch_test_outputs.zip\n' "$EXP_TAG"
