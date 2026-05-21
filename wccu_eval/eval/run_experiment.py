from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from wccu_eval.eval.baselines.prompt_only_serial import run_prompt_only_serial
from wccu_eval.eval.baselines.shared_markdown_parallel import run_shared_markdown_parallel
from wccu_eval.eval.baselines.vector_memory_parallel import run_vector_memory_parallel
from wccu_eval.eval.scenarios import get_scenario, list_scenario_ids
from wccu_eval.scheduler.context_concurrency_policy import PolicyMode
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel, execute_substrate_parallel_group, execute_substrate_serial
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import append_jsonl, clean, ensure_dir, mean, now_iso, remove_dir, stable_hash, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_policy(policy_mode: str, condition: str) -> Callable[..., dict[str, Any]]:
    def runner(**kwargs: Any) -> dict[str, Any]:
        return execute_context_policy_parallel(**kwargs, policy_mode=policy_mode, condition=condition)
    return runner


def _run_serial_policy(policy_mode: str, condition: str) -> Callable[..., dict[str, Any]]:
    def runner(**kwargs: Any) -> dict[str, Any]:
        return execute_substrate_serial(**kwargs, policy_mode=policy_mode, condition=condition)
    return runner

CONDITIONS: dict[str, Callable[..., dict[str, Any]]] = {
    'prompt_only_serial': run_prompt_only_serial,
    'shared_markdown_parallel': run_shared_markdown_parallel,
    'vector_memory_parallel': run_vector_memory_parallel,
    'substrate_serial': execute_substrate_serial,
    'substrate_parallel': execute_substrate_parallel_group,
    'adaptive_policy': _run_policy(PolicyMode.ADAPTIVE, 'adaptive_policy'),
    'adaptive_readset_occ': _run_policy(PolicyMode.ADAPTIVE_READSET_OCC, 'adaptive_readset_occ'),
    'adaptive_wccu': _run_policy(PolicyMode.ADAPTIVE_WCCU, 'adaptive_wccu'),
    'adaptive_wccu_model_certificate': _run_policy(PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE, 'adaptive_wccu_model_certificate'),
    'adaptive_wccu_oracle_dependency': _run_policy(PolicyMode.ADAPTIVE_WCCU_ORACLE_DEPENDENCY, 'adaptive_wccu_oracle_dependency'),
    'adaptive_wccu_projection_trace': _run_policy(PolicyMode.ADAPTIVE_WCCU_PROJECTION_TRACE, 'adaptive_wccu_projection_trace'),
    'adaptive_wccu_no_read_validation': _run_policy(PolicyMode.ADAPTIVE_WCCU_NO_READ_VALIDATION, 'adaptive_wccu_no_read_validation'),
    'adaptive_wccu_unguided_certificate': _run_policy(PolicyMode.ADAPTIVE_WCCU_UNGUIDED_CERTIFICATE, 'adaptive_wccu_unguided_certificate'),
    'adaptive_wccu_execution_trace': _run_policy(PolicyMode.ADAPTIVE_WCCU_EXECUTION_TRACE, 'adaptive_wccu_execution_trace'),
    'serial_adaptive_policy': _run_serial_policy(PolicyMode.ADAPTIVE, 'serial_adaptive_policy'),
    'serial_adaptive_wccu_execution_trace': _run_serial_policy(PolicyMode.ADAPTIVE_WCCU_EXECUTION_TRACE, 'serial_adaptive_wccu_execution_trace'),
    'serial_adaptive_wccu_projection_trace': _run_serial_policy(PolicyMode.ADAPTIVE_WCCU_PROJECTION_TRACE, 'serial_adaptive_wccu_projection_trace'),
    'uniform_snapshot_occ': _run_policy(PolicyMode.UNIFORM_SNAPSHOT_OCC, 'uniform_snapshot_occ'),
    'uniform_pessimistic_lock': _run_policy(PolicyMode.UNIFORM_PESSIMISTIC_LOCK, 'uniform_pessimistic_lock'),
    'uniform_review_gated': _run_policy(PolicyMode.UNIFORM_REVIEW_GATED, 'uniform_review_gated'),
    'uniform_append_only': _run_policy(PolicyMode.UNIFORM_APPEND_ONLY, 'uniform_append_only'),
    'adaptive_no_review_gate': _run_policy(PolicyMode.ADAPTIVE_NO_REVIEW_GATE, 'adaptive_no_review_gate'),
    'adaptive_no_authority_rebase': _run_policy(PolicyMode.ADAPTIVE_NO_AUTHORITY_REBASE, 'adaptive_no_authority_rebase'),
    'adaptive_no_append_only': _run_policy(PolicyMode.ADAPTIVE_NO_APPEND_ONLY, 'adaptive_no_append_only'),
    'adaptive_no_workspace_lock': _run_policy(PolicyMode.ADAPTIVE_NO_WORKSPACE_LOCK, 'adaptive_no_workspace_lock'),
    'adaptive_no_semantic_conflict_detection': _run_policy(PolicyMode.ADAPTIVE_NO_SEMANTIC_CONFLICT_DETECTION, 'adaptive_no_semantic_conflict_detection'),
}


def selected_scenarios(value: str) -> list[str]:
    value = clean(value or 'all')
    if value == 'all':
        return list_scenario_ids()
    return [clean(x) for x in value.split(',') if clean(x)]


def selected_conditions(value: str) -> list[str]:
    value = clean(value or 'all')
    if value == 'all':
        return list(CONDITIONS.keys())
    return [clean(x) for x in value.split(',') if clean(x)]


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault((row.get('scenario_id'), row.get('condition')), []).append(row)
    out = []
    for (scenario_id, condition), rows in groups.items():
        out.append({
            'scenario_id': scenario_id,
            'condition': condition,
            'n': len(rows),
            'task_success_rate': mean([1 if r.get('task_success') else 0 for r in rows]),
            'mean_elapsed_ms': mean([r.get('elapsed_ms') or 0 for r in rows]),
            'mean_context_tokens': mean([r.get('context_tokens') or 0 for r in rows]),
            'mean_conflict_groups': mean([r.get('conflict_groups') or 0 for r in rows]),
            'mean_semantic_conflict_count': mean([r.get('semantic_conflict_count') or 0 for r in rows]),
            'mean_unsafe_auto_commit_count': mean([r.get('unsafe_auto_commit_count') or 0 for r in rows]),
            'mean_review_burden_count': mean([r.get('review_burden_count') or 0 for r in rows]),
            'mean_lock_conflict_count': mean([r.get('lock_conflict_count') or 0 for r in rows]),
            'mean_authority_rebase_count': mean([r.get('authority_rebase_count') or 0 for r in rows]),
            'mean_wccu_blocked_count': mean([r.get('wccu_blocked_count') or 0 for r in rows]),
            'mean_wccu_review_routed_count': mean([r.get('wccu_review_routed_count') or 0 for r in rows]),
            'mean_wccu_intervention_count': mean([(r.get('wccu_intervention_count') if r.get('wccu_intervention_count') is not None else ((r.get('wccu_review_routed_count') or 0) + (r.get('wccu_blocked_count') or 0))) for r in rows]),
            'mean_stale_dependency_count': mean([r.get('stale_dependency_count') or 0 for r in rows]),
            'mean_stale_dependency_accepted_count': mean([r.get('stale_dependency_accepted_count') or 0 for r in rows]),
            'mean_stale_read_validation_ignored_count': mean([r.get('stale_read_validation_ignored_count') or 0 for r in rows]),
            'mean_certificate_invalid_count': mean([r.get('certificate_invalid_count') or 0 for r in rows]),
            'mean_low_target_confidence_count': mean([r.get('low_target_confidence_count') or 0 for r in rows]),
            'mean_authority_insufficient_count': mean([r.get('authority_insufficient_count') or 0 for r in rows]),
            'mean_wrong_target_count': mean([r.get('wrong_target_count') or 0 for r in rows]),
            'mean_weaken_rule_delta_count': mean([r.get('weaken_rule_delta_count') or 0 for r in rows]),
            'mean_semantic_operation_laundering_count': mean([r.get('semantic_operation_laundering_count') or 0 for r in rows]),
            'mean_authority_laundering_count': mean([r.get('authority_laundering_count') or 0 for r in rows]),
            'mean_committed': mean([r.get('commit', {}).get('committed') or 0 for r in rows]),
            'mean_proposals': mean([r.get('commit', {}).get('proposals') or 0 for r in rows]),
            'mean_conflicts': mean([r.get('commit', {}).get('conflicts') or 0 for r in rows]),
        })
    return sorted(out, key=lambda r: f"{r['scenario_id']}:{r['condition']}")


def run_experiment(*, scenario: str = 'all', condition: str = 'all', repetitions: int = 1, out: str = 'results/all_results.json') -> dict[str, Any]:
    scenario_ids = selected_scenarios(scenario)
    condition_ids = selected_conditions(condition)
    repetitions = max(1, int(repetitions))
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    jsonl_path = out_path.with_suffix('.jsonl')
    ensure_dir(out_path.parent)
    if jsonl_path.exists():
        jsonl_path.unlink()
    tmp_root = REPO_ROOT / 'runs' / f"eval_tmp_{stable_hash(f'{out}:{os.getpid()}:{now_iso()}')}"
    remove_dir(tmp_root)
    results: list[dict[str, Any]] = []

    for scenario_id in scenario_ids:
        for condition_id in condition_ids:
            runner = CONDITIONS.get(condition_id)
            if not runner:
                raise KeyError(f'Unknown condition: {condition_id}')
            for rep in range(repetitions):
                sc = get_scenario(scenario_id)
                run_dir = tmp_root / f'{scenario_id}_{condition_id}_{rep}'
                root_dir = run_dir / 'context_substrate'
                remove_dir(run_dir)
                seed_context(root_dir, sc.get('seed', {}))
                row = runner(scenario=sc, root_dir=root_dir, run_dir=run_dir, repetition=rep)
                enriched = {**row, 'repetition': rep}
                results.append(enriched)
                append_jsonl(jsonl_path, enriched)
    aggregated = aggregate(results)
    payload = {'kind': 'wccu_eval_results_v1', 'generated_at': now_iso(), 'args': {'scenario': scenario, 'condition': condition, 'repetitions': repetitions, 'out': str(out)}, 'results': results, 'aggregated': aggregated}
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run deterministic context substrate experiments.')
    parser.add_argument('--scenario', default='all')
    parser.add_argument('--condition', default='all')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--out', default='results/all_results.json')
    args = parser.parse_args(argv)
    payload = run_experiment(scenario=args.scenario, condition=args.condition, repetitions=args.repetitions, out=args.out)
    print(json.dumps({'ok': True, 'out': payload['args']['out'], 'result_count': len(payload['results']), 'aggregate_count': len(payload['aggregated'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
