from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from wccu_eval.eval.run_experiment import REPO_ROOT, aggregate
from wccu_eval.eval.run_wccu_stress import CONDITION_TO_POLICY, generate_stress_scenario
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import append_jsonl, clean, ensure_dir, now_iso, remove_dir, stable_hash, write_json

DEFAULT_CONDITIONS = (
    'adaptive_wccu_execution_trace,'
    'adaptive_wccu_projection_trace,'
    'adaptive_wccu_model_certificate,'
    'adaptive_readset_occ,'
    'adaptive_wccu_no_read_validation,'
    'adaptive_policy,'
    'uniform_snapshot_occ,'
    'uniform_review_gated'
)
DEFAULT_DROP_RATES = '0.0,0.25,0.5,0.75,1.0'


def _split(value: str) -> list[str]:
    return [clean(x) for x in value.split(',') if clean(x)]


def _floats(value: str) -> list[float]:
    return [float(x) for x in _split(value)]


def run_witness_completeness(*, cases: int, writers: int, atom_count: int, invalidation_prob: float, seed: int, conditions: str = DEFAULT_CONDITIONS, drop_rates: str = DEFAULT_DROP_RATES, repetitions: int = 1, out: str = 'results/witness_completeness.json') -> dict[str, Any]:
    condition_ids = _split(conditions)
    for cond in condition_ids:
        if cond not in CONDITION_TO_POLICY:
            raise KeyError(f'Unsupported condition: {cond}')
    rates = _floats(drop_rates)
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    jsonl_path = out_path.with_suffix('.jsonl')
    ensure_dir(out_path.parent)
    if jsonl_path.exists():
        jsonl_path.unlink()
    tmp_root = REPO_ROOT / 'runs' / f"witness_completeness_{stable_hash(f'{out}:{os.getpid()}:{now_iso()}')}"
    remove_dir(tmp_root)
    results: list[dict[str, Any]] = []
    scenarios = [generate_stress_scenario(case_idx=i, writers=writers, atom_count=atom_count, invalidation_prob=invalidation_prob, seed=seed, use_witness=True) for i in range(cases)]
    # Keep payload text free of dependency names, but remove pre-attached raw
    # witnesses so the only read-set source is the runtime witness compiler.
    # This makes the drop-rate sweep test witness completeness rather than
    # lexical fallback recovery.
    for sc in scenarios:
        for spec in (sc.get('agent_outputs') or {}).values():
            for intent in spec.get('intents') or []:
                intent.pop('execution_witness', None)
                intent.pop('read_witness', None)
                intent.pop('projection_witness', None)
                intent['disable_trace_text_fallback'] = True
    for drop_rate in rates:
        for sc in scenarios:
            for cond in condition_ids:
                for rep in range(max(1, repetitions)):
                    run_dir = tmp_root / f"drop{drop_rate:.2f}_{sc['id']}_{cond}_{rep}"
                    root_dir = run_dir / 'context_substrate'
                    remove_dir(run_dir)
                    seed_context(root_dir, sc.get('seed', {}))
                    row = execute_context_policy_parallel(
                        root_dir=root_dir,
                        run_dir=run_dir,
                        scenario=sc,
                        policy_mode=CONDITION_TO_POLICY[cond],
                        condition=cond,
                        agent_runner_config={
                            'witness_compiler_enabled': True,
                            'witness_drop_rate': drop_rate,
                            'witness_seed': f'{seed}:{sc["id"]}:{cond}:{rep}:{drop_rate}',
                            'witness_source_label': 'witness_completeness_runtime_compiler',
                        },
                    )
                    row = {
                        **row,
                        'repetition': rep,
                        'witness_drop_rate': drop_rate,
                        'stress_metadata': sc.get('metadata', {}),
                    }
                    results.append(row)
                    append_jsonl(jsonl_path, row)
    payload = {
        'kind': 'wccu_witness_completeness_results_v1',
        'generated_at': now_iso(),
        'args': {
            'cases': cases,
            'writers': writers,
            'atom_count': atom_count,
            'invalidation_prob': invalidation_prob,
            'seed': seed,
            'conditions': conditions,
            'drop_rates': drop_rates,
            'repetitions': repetitions,
            'out': out,
        },
        'results': results,
        'aggregated': aggregate(results),
    }
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run witness-completeness stress diagnostics by dropping runtime read witnesses deterministically.')
    parser.add_argument('--cases', type=int, default=50)
    parser.add_argument('--writers', type=int, default=8)
    parser.add_argument('--atom-count', type=int, default=64)
    parser.add_argument('--invalidation-prob', type=float, default=0.35)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--conditions', default=DEFAULT_CONDITIONS)
    parser.add_argument('--drop-rates', default=DEFAULT_DROP_RATES)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--out', default='results/witness_completeness.json')
    args = parser.parse_args(argv)
    payload = run_witness_completeness(**vars(args))
    print(json.dumps({'ok': True, 'out': payload['args']['out'], 'result_count': len(payload['results'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
