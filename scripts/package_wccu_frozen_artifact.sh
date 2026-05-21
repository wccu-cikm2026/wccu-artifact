#!/usr/bin/env bash
set -euo pipefail

RESULT_JSON="${1:-results/llm_obligation_benchmark.json}"
DEST_DIR="${2:-artifact/wccu_frozen_replay_bundle}"
mkdir -p "${DEST_DIR}"

if [[ ! -f "${RESULT_JSON}" ]]; then
  echo "Missing result JSON: ${RESULT_JSON}" >&2
  echo "Run run_llm_obligation_benchmark first, or pass an existing JSON path." >&2
  exit 2
fi

cp "${RESULT_JSON}" "${DEST_DIR}/results.json"
if [[ -f "${RESULT_JSON%.json}.jsonl" ]]; then
  cp "${RESULT_JSON%.json}.jsonl" "${DEST_DIR}/results.jsonl"
fi
python - <<'PY' "${DEST_DIR}"
from __future__ import annotations
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
result_path = root / 'results.json'
payload = json.loads(result_path.read_text())
rows = payload.get('results', [])
proposals = []
for row in rows:
    for run in row.get('agentRuns', []) or []:
        proposals.append({
            'scenario_id': row.get('scenario_id'),
            'family': row.get('llm_obligation_family') or row.get('obligation_kind'),
            'condition': row.get('condition'),
            'repetition': row.get('repetition'),
            'agent_id': run.get('agent_id'),
            'provider': (run.get('llm') or {}).get('provider'),
            'model': (run.get('llm') or {}).get('model'),
            'output': run.get('output'),
            'write_intents': run.get('write_intents') or [],
        })
(root / 'frozen_generated_proposals.json').write_text(json.dumps({'kind':'wccu_frozen_generated_proposals_v1','proposal_count':len(proposals),'proposals':proposals}, indent=2, ensure_ascii=False))
(root / 'manifest.json').write_text(json.dumps({
    'kind': 'wccu_frozen_replay_manifest_v1',
    'source_result_file': str(result_path),
    'result_count': len(rows),
    'proposal_record_count': len(proposals),
    'conditions': sorted({r.get('condition') for r in rows if r.get('condition')}),
    'families': sorted({r.get('llm_obligation_family') or r.get('obligation_kind') for r in rows if r.get('llm_obligation_family') or r.get('obligation_kind')}),
    'replay_note': 'Use results.json/results.jsonl to recompute paper tables; frozen_generated_proposals.json stores model outputs separated from policy replay rows.'
}, indent=2, ensure_ascii=False))
PY
( cd "${DEST_DIR}/.." && zip -qr "$(basename "${DEST_DIR}").zip" "$(basename "${DEST_DIR}")" )
echo "Wrote ${DEST_DIR} and ${DEST_DIR}.zip"
