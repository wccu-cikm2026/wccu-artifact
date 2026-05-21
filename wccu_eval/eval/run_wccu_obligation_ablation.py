from __future__ import annotations

"""Incremental WCCU obligation ablation.

This experiment keeps the same WCCU execution-witness policy but selectively
turns obligation classes on/off at verification time.  It answers the reviewer
question: which WCCU obligations are responsible for catching which failure
families?
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from wccu_eval.eval.run_experiment import REPO_ROOT, aggregate
from wccu_eval.eval.run_obligation_matrix import DEFAULT_KINDS, make_obligation_scenario
from wccu_eval.eval.run_wccu_stress import CONDITION_TO_POLICY
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import append_jsonl, as_dict, as_list, clean, ensure_dir, now_iso, remove_dir, stable_hash, write_json

ALL_OBLIGATIONS = ['O-TARGET', 'O-FRESH', 'O-AUTH', 'O-DELTA', 'O-VIEW']
DEFAULT_STAGES = 'none,fresh,fresh_target,fresh_target_auth,fresh_target_auth_delta,full'
STAGE_ENABLED = {
    'none': [],
    'fresh': ['O-FRESH'],
    'fresh_target': ['O-FRESH', 'O-TARGET'],
    'fresh_target_auth': ['O-FRESH', 'O-TARGET', 'O-AUTH'],
    'fresh_target_auth_delta': ['O-FRESH', 'O-TARGET', 'O-AUTH', 'O-DELTA'],
    'full': ALL_OBLIGATIONS,
}


def _split(value: str) -> list[str]:
    return [clean(x) for x in value.split(',') if clean(x)]


def _events(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in as_list(row.get('wccu_events')):
        ev = as_dict(ev)
        out.extend(as_list(ev.get('errors')))
        out.extend(as_list(ev.get('warnings')))
    return [as_dict(e) for e in out]


def _event_observed(row: dict[str, Any], expected: str) -> bool:
    return any(clean(e.get('kind')) == expected for e in _events(row))


def _score_row(row: dict[str, Any], *, kind: str, expected_event: str) -> dict[str, Any]:
    review_block = int(row.get('review_burden_count') or 0)
    committed = int(as_dict(row.get('commit')).get('committed') or 0)
    if kind == 'freshness':
        issue_accepted = int(row.get('stale_dependency_accepted_count') or 0)
    elif kind in {'target', 'authority', 'delta'}:
        issue_accepted = 0 if review_block > 0 else min(1, committed)
    elif kind == 'view':
        issue_accepted = 0 if _event_observed(row, expected_event) or review_block > 0 else min(1, committed)
    else:
        issue_accepted = 0
    hold_required = 0 if kind == 'view' else 1
    return {
        'expected_event_observed': _event_observed(row, expected_event),
        'ground_truth_issue_count': 1,
        'ground_truth_hold_required_count': hold_required,
        'ground_truth_issue_accepted_count': issue_accepted,
        'ground_truth_problematic_held_count': 1 if hold_required and review_block > 0 and issue_accepted == 0 else 0,
        'ground_truth_total_writes': int(row.get('write_intent_count') or as_dict(row.get('commit')).get('total') or 0),
    }


def _stage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((clean(row.get('obligation_kind')), clean(row.get('obligation_stage'))), []).append(row)
    out: list[dict[str, Any]] = []
    for (kind, stage), group in groups.items():
        issue_count = sum(int(r.get('ground_truth_issue_count') or 0) for r in group)
        accepted = sum(int(r.get('ground_truth_issue_accepted_count') or 0) for r in group)
        held = sum(int(r.get('ground_truth_problematic_held_count') or 0) for r in group)
        out.append({
            'obligation_kind': kind,
            'obligation_stage': stage,
            'n': len(group),
            'issue_count': issue_count,
            'issue_accepted_rate': accepted / issue_count if issue_count else 0.0,
            'problematic_held_rate': held / issue_count if issue_count else 0.0,
            'mean_review_burden_count': sum(int(r.get('review_burden_count') or 0) for r in group) / len(group),
            'enabled_obligations': group[0].get('enabled_obligations', []),
            'disabled_obligations': group[0].get('disabled_obligations', []),
        })
    return sorted(out, key=lambda r: f"{r['obligation_kind']}:{r['obligation_stage']}")


def run_wccu_obligation_ablation(
    *,
    kinds: str = DEFAULT_KINDS,
    stages: str = DEFAULT_STAGES,
    condition: str = 'adaptive_wccu_execution_trace',
    repetitions: int = 1,
    out: str = 'results/wccu_obligation_ablation.json',
) -> dict[str, Any]:
    if condition not in CONDITION_TO_POLICY:
        raise KeyError(f'Unsupported condition: {condition}')
    kind_ids = _split(kinds)
    stage_ids = _split(stages)
    for stage in stage_ids:
        if stage not in STAGE_ENABLED:
            raise KeyError(f'Unsupported stage: {stage}')
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    jsonl_path = out_path.with_suffix('.jsonl')
    ensure_dir(out_path.parent)
    if jsonl_path.exists():
        jsonl_path.unlink()
    tmp_root = REPO_ROOT / 'runs' / f"wccu_obligation_ablation_{stable_hash(f'{out}:{os.getpid()}:{now_iso()}')}"
    remove_dir(tmp_root)
    results: list[dict[str, Any]] = []
    for kind in kind_ids:
        base = make_obligation_scenario(kind)
        expected = clean(as_dict(base.get('metadata')).get('expected_wccu_event'))
        for stage in stage_ids:
            enabled = set(STAGE_ENABLED[stage])
            disabled = [x for x in ALL_OBLIGATIONS if x not in enabled]
            scenario = {
                **base,
                'id': f"{base['id']}_{stage}",
                'wccu_disabled_obligations': disabled,
                'metadata': {**as_dict(base.get('metadata')), 'obligation_stage': stage, 'enabled_obligations': sorted(enabled), 'wccu_disabled_obligations': disabled},
            }
            for rep in range(max(1, repetitions)):
                run_dir = tmp_root / f'{kind}_{stage}_{rep}'
                root_dir = run_dir / 'context_substrate'
                remove_dir(run_dir)
                seed_context(root_dir, scenario.get('seed', {}))
                row = execute_context_policy_parallel(
                    root_dir=root_dir,
                    run_dir=run_dir,
                    scenario=scenario,
                    policy_mode=CONDITION_TO_POLICY[condition],
                    condition=f'{condition}:{stage}',
                    agent_runner_config={'witness_compiler_enabled': True, 'witness_source_label': 'obligation_ablation_runtime_witness'},
                )
                row = {
                    **row,
                    'scenario_id': base.get('id'),
                    'ablation_scenario_id': scenario.get('id'),
                    'condition': condition,
                    'obligation_stage': stage,
                    'enabled_obligations': sorted(enabled),
                    'disabled_obligations': disabled,
                    'obligation_kind': kind,
                    'expected_wccu_event': expected,
                    'repetition': rep,
                    **_score_row(row, kind=kind, expected_event=expected),
                }
                results.append(row)
                append_jsonl(jsonl_path, row)
    payload = {
        'kind': 'wccu_obligation_ablation_results_v1',
        'generated_at': now_iso(),
        'args': {'kinds': kinds, 'stages': stages, 'condition': condition, 'repetitions': repetitions, 'out': out},
        'stage_enabled_obligations': STAGE_ENABLED,
        'results': results,
        'stage_summary': _stage_summary(results),
        'aggregated': aggregate(results),
    }
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run incremental WCCU obligation ablations.')
    parser.add_argument('--kinds', default=DEFAULT_KINDS)
    parser.add_argument('--stages', default=DEFAULT_STAGES)
    parser.add_argument('--condition', default='adaptive_wccu_execution_trace')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--out', default='results/wccu_obligation_ablation.json')
    args = parser.parse_args(argv)
    payload = run_wccu_obligation_ablation(**vars(args))
    print(json.dumps({'ok': True, 'out': payload['args']['out'], 'result_count': len(payload['results'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
