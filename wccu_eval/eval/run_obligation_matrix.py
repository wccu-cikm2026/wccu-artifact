from __future__ import annotations

"""Deterministic diagnostics for WCCU obligations beyond freshness.

The randomized stale-dependency stress test is intentionally dominated by
freshness.  This module builds small, deterministic scenarios that isolate each
runtime-checkable obligation so the artifact can show where WCCU differs from a
strong read-set OCC baseline.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from wccu_eval.eval.run_experiment import REPO_ROOT, aggregate
from wccu_eval.eval.run_wccu_stress import CONDITION_TO_POLICY
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import append_jsonl, as_dict, as_list, clean, ensure_dir, now_iso, remove_dir, stable_hash, write_json

DEFAULT_CONDITIONS = 'adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated'
DEFAULT_KINDS = 'freshness,target,authority,delta,view'


def _conditions(value: str) -> list[str]:
    return [clean(x) for x in value.split(',') if clean(x)]


def _base_atoms() -> list[dict[str, Any]]:
    return [
        {'id': 'atom_dep_pref', 'atom_type': 'memory', 'title': 'Preference dependency', 'canonical_text_en': 'The user prefers backup URLs in API responses.', 'status': 'active', 'tags': ['pref', 'api']},
        {'id': 'atom_target_memory', 'atom_type': 'memory', 'title': 'Response format target', 'canonical_text_en': 'Use the default response format.', 'status': 'active', 'tags': ['response']},
        {'id': 'atom_policy_lowrisk', 'atom_type': 'memory', 'title': 'Approval wording note', 'canonical_text_en': 'Approval wording is conservative.', 'status': 'active', 'tags': ['approval']},
        {'id': 'atom_view_source', 'atom_type': 'memory', 'title': 'View source', 'canonical_text_en': 'This fact appears in a derived handoff summary.', 'status': 'active', 'tags': ['view']},
    ]


def _certificate(*, target_id: str = 'atom_target_memory', claimed_target_id: str | None = None, deps: list[str] | None = None, actor_authority: str = 'agent', required_authority: str = 'agent', delta_type: str = 'patch_memory', invalidates_views: bool = True) -> dict[str, Any]:
    deps = deps or []
    return {
        'source': 'obligation_matrix_fixture',
        'target_certificate': {'claimed_target_id': claimed_target_id or target_id, 'raw_target': claimed_target_id or target_id, 'confidence': 0.95, 'grounding_rationale': 'fixture'},
        'read_dependencies': [{'target_id': d, 'expected_status': 'active', 'freshness_required': True, 'reason': 'obligation matrix declared dependency'} for d in deps],
        'delta_contract': {'delta_type': delta_type, 'semantic_direction': 'patch', 'invalidates_views': invalidates_views, 'affected_view_ids': [f'target:{target_id}'] if invalidates_views else []},
        'authority_certificate': {'actor_authority': actor_authority, 'required_authority': required_authority},
        'preconditions': {'freshness_required': bool(deps), 'requires_review_if_invalid': True, 'min_target_confidence': 0.55},
    }


def _patch(*, target_id: str = 'atom_target_memory', atom_type: str = 'memory', text: str = 'Patch the response format.', certificate: dict[str, Any] | None = None, authority: str | None = None, intent_type: str = 'patch_atom') -> dict[str, Any]:
    intent: dict[str, Any] = {
        'intent_type': intent_type,
        'payload': {'id': target_id, 'target_id': target_id, 'atom_id': target_id, 'atom_type': atom_type, 'title': f'Patch {target_id}', 'canonical_text_en': text},
    }
    if authority:
        intent['authority'] = authority
    if certificate:
        intent['certificate'] = certificate
    return intent


def _retract(dep_id: str) -> dict[str, Any]:
    return {
        'intent_type': 'retract_atom',
        'authority': 'user',
        'payload': {'id': dep_id, 'target_id': dep_id, 'atom_id': dep_id, 'atom_type': 'memory', 'status': 'retracted', 'title': f'Retract {dep_id}', 'canonical_text_en': f'Retract {dep_id}.'},
    }


def make_obligation_scenario(kind: str) -> dict[str, Any]:
    kind = clean(kind)
    agents = [{'id': 'writer', 'role': 'agent'}]
    outputs: dict[str, Any] = {}
    wccu_deps: dict[str, Any] = {}
    metadata: dict[str, Any] = {'obligation_kind': kind}

    if kind == 'freshness':
        agents.append({'id': 'invalidator', 'role': 'user'})
        outputs['writer'] = {'text': 'Patch target based on a runtime-read preference.', 'intents': [_patch(certificate=_certificate(deps=[]), text='Patch target using runtime-read preference.')], 'read_atoms': ['atom_dep_pref']}
        outputs['invalidator'] = {'text': 'User retracts the dependency.', 'intents': [_retract('atom_dep_pref')]}
        wccu_deps = {'writer': [{'target_id': 'atom_dep_pref', 'expected_status': 'active', 'freshness_required': True, 'reason': 'ground truth for stale-accept counting'}]}
        metadata['expected_wccu_event'] = 'stale_read_dependency'
    elif kind == 'target':
        outputs['writer'] = {'text': 'Patch a target while presenting a wrong target certificate.', 'intents': [_patch(certificate=_certificate(claimed_target_id='atom_wrong_target'), text='Patch target despite wrong target certificate.')]}
        metadata['expected_wccu_event'] = 'wrong_target_certificate'
    elif kind == 'authority':
        # Low-risk retraction: adaptive/read-set OCC do not model authority as an
        # obligation, while WCCU requires user authority for direct retraction.
        outputs['writer'] = {'text': 'Agent tries to retract a memory without user authority.', 'intents': [_patch(intent_type='retract_atom', target_id='atom_target_memory', text='Retract this memory.', certificate=_certificate(target_id='atom_target_memory', actor_authority='agent', required_authority='user', delta_type='retract_memory'))]}
        metadata['expected_wccu_event'] = 'authority_insufficient_for_direct_commit'
    elif kind == 'delta':
        text = 'Relax the approval wording so the agent may run without explicit approval.'
        outputs['writer'] = {'text': text, 'intents': [_patch(target_id='atom_policy_lowrisk', atom_type='memory', text=text, certificate=_certificate(target_id='atom_policy_lowrisk', delta_type='append_evidence'))]}
        metadata['expected_wccu_event'] = 'delta_contract_mismatch'
    elif kind == 'view':
        outputs['writer'] = {'text': 'Patch a source object used by a derived handoff summary.', 'intents': [_patch(target_id='atom_view_source', text='Update the source fact used by a derived summary.', certificate=_certificate(target_id='atom_view_source', delta_type='patch_memory', invalidates_views=True))]}
        metadata['expected_wccu_event'] = 'view_invalidation_required'
    else:
        raise KeyError(f'Unknown obligation kind: {kind}')

    return {
        'id': f'obligation_matrix_{kind}',
        'goal': f'Isolate WCCU obligation: {kind}',
        'task_type': 'wccu_obligation_matrix',
        'default_latency_ms': 0,
        'budget_tokens': 4000,
        'agents': agents,
        'seed': {'atoms': _base_atoms(), 'links': []},
        'agent_outputs': outputs,
        'wccu_read_dependencies': wccu_deps,
        'metadata': metadata,
    }


def run_obligation_matrix(*, kinds: str = DEFAULT_KINDS, conditions: str = DEFAULT_CONDITIONS, repetitions: int = 1, out: str = 'results/obligation_matrix.json') -> dict[str, Any]:
    condition_ids = _conditions(conditions)
    for cond in condition_ids:
        if cond not in CONDITION_TO_POLICY:
            raise KeyError(f'Unsupported condition: {cond}')
    kind_ids = _conditions(kinds)
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    jsonl_path = out_path.with_suffix('.jsonl')
    ensure_dir(out_path.parent)
    if jsonl_path.exists():
        jsonl_path.unlink()
    tmp_root = REPO_ROOT / 'runs' / f"obligation_matrix_{stable_hash(f'{out}:{os.getpid()}:{now_iso()}')}"
    remove_dir(tmp_root)
    results: list[dict[str, Any]] = []
    for kind in kind_ids:
        sc = make_obligation_scenario(kind)
        for cond in condition_ids:
            for rep in range(max(1, repetitions)):
                run_dir = tmp_root / f'{kind}_{cond}_{rep}'
                root_dir = run_dir / 'context_substrate'
                remove_dir(run_dir)
                seed_context(root_dir, sc.get('seed', {}))
                row = execute_context_policy_parallel(
                    root_dir=root_dir,
                    run_dir=run_dir,
                    scenario=sc,
                    policy_mode=CONDITION_TO_POLICY[cond],
                    condition=cond,
                    agent_runner_config={'witness_compiler_enabled': True, 'witness_source_label': 'obligation_matrix_runtime_witness'},
                )
                events = [e for e in as_list(row.get('wccu_events'))]
                expected = clean(sc.get('metadata', {}).get('expected_wccu_event'))
                expected_event_observed = any(
                    clean(as_dict(item).get('kind')) == expected
                    for event in events
                    for item in as_list(event.get('errors')) + as_list(event.get('warnings'))
                )
                review_block = int(row.get('review_burden_count') or 0)
                committed = int(as_dict(row.get('commit')).get('committed') or 0)
                hold_required = 0 if kind == 'view' else 1
                # The obligation matrix has one ground-truth issue per case: the
                # writer update is stale, wrongly targeted, under-authorized,
                # mislabeled, or missing a derived-view invalidation.  Some
                # issues require review/block; the view case is correct if the
                # update commits while invalidating the affected view.
                if kind == 'freshness':
                    issue_accepted = int(row.get('stale_dependency_accepted_count') or 0)
                elif kind in {'target', 'authority', 'delta'}:
                    issue_accepted = 0 if review_block > 0 else min(1, committed)
                elif kind == 'view':
                    issue_accepted = 0 if expected_event_observed or review_block > 0 else min(1, committed)
                else:
                    issue_accepted = 0
                problem_held = 1 if hold_required and review_block > 0 and issue_accepted == 0 else 0
                row = {
                    **row,
                    'repetition': rep,
                    'obligation_kind': kind,
                    'expected_wccu_event': expected,
                    'expected_event_observed': expected_event_observed,
                    'ground_truth_issue_count': 1,
                    'ground_truth_hold_required_count': hold_required,
                    'ground_truth_issue_detected_count': 1 if expected_event_observed else 0,
                    'ground_truth_issue_accepted_count': issue_accepted,
                    'ground_truth_problematic_held_count': problem_held,
                    'ground_truth_total_writes': int(row.get('write_intent_count') or as_dict(row.get('commit')).get('total') or 0),
                }
                results.append(row)
                append_jsonl(jsonl_path, row)
    payload = {'kind': 'wccu_obligation_matrix_results_v1', 'generated_at': now_iso(), 'args': {'kinds': kinds, 'conditions': conditions, 'repetitions': repetitions, 'out': out}, 'results': results, 'aggregated': aggregate(results)}
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run WCCU obligation diagnostics beyond freshness.')
    parser.add_argument('--kinds', default=DEFAULT_KINDS)
    parser.add_argument('--conditions', default=DEFAULT_CONDITIONS)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--out', default='results/obligation_matrix.json')
    args = parser.parse_args(argv)
    payload = run_obligation_matrix(**vars(args))
    print(json.dumps({'ok': True, 'out': payload['args']['out'], 'result_count': len(payload['results'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
