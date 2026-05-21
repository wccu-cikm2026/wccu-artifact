from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wccu_eval.eval.run_experiment import aggregate
from wccu_eval.eval.run_llm_experiment import _aggregate_llm
from wccu_eval.utils import as_list, ensure_dir, now_iso, write_json


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def combine(paths: list[str | Path], *, out: str | Path) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    external_benchmarks: set[str] = set()
    kinds: set[str] = set()
    for p in paths:
        path = Path(p)
        payload = _load(path)
        kinds.add(str(payload.get('kind') or 'unknown'))
        if payload.get('external_benchmark'):
            external_benchmarks.add(str(payload.get('external_benchmark')))
        inputs.append({
            'path': str(path),
            'kind': payload.get('kind'),
            'generated_at': payload.get('generated_at'),
            'args': payload.get('args', {}),
            'task_count': payload.get('task_count'),
            'scenario_count': payload.get('scenario_count'),
            'result_count': len(as_list(payload.get('results'))),
        })
        for row in as_list(payload.get('results')):
            r = dict(row)
            # Preserve where a row came from.  This is useful for multi-seed and
            # multi-repetition robustness tables without modifying original rows.
            r.setdefault('source_result_file', str(path))
            results.append(r)
    if any('llm' in k or 'external_eval' in k for k in kinds):
        aggregated = _aggregate_llm(results)
    else:
        aggregated = aggregate(results)
    payload = {
        'kind': 'context_substrate_combined_results_v1',
        'generated_at': now_iso(),
        'inputs': inputs,
        'input_count': len(inputs),
        'external_benchmark': ','.join(sorted(external_benchmarks)) if external_benchmarks else '',
        'result_count': len(results),
        'results': results,
        'aggregated': aggregated,
    }
    out_path = Path(out)
    ensure_dir(out_path.parent)
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Combine multiple WCCU result JSON files into one payload for multi-seed analysis.')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    payload = combine(args.inputs, out=args.out)
    print(json.dumps({'ok': True, 'input_count': payload['input_count'], 'result_count': payload['result_count'], 'out': args.out}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
