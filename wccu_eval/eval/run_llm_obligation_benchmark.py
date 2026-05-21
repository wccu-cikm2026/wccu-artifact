from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from wccu_eval.agents.deterministic_agent import run_deterministic_agent
from wccu_eval.agents.llm_agent import LlmProviderError, run_llm_agent
from wccu_eval.env import load_dotenv
from wccu_eval.eval.llm_obligation_scenarios import build_llm_obligation_scenarios, expected_dependency_ids, list_llm_obligation_families
from wccu_eval.eval.run_experiment import REPO_ROOT
from wccu_eval.scheduler.context_conflict_resolver import resolve_parallel_write_intents
from wccu_eval.scheduler.context_concurrency_policy import PolicyMode
from wccu_eval.scheduler.dependency_witness import attach_dependency_witness_to_result
from wccu_eval.substrate.context_projection_compiler import compile_projection
from wccu_eval.substrate.context_substrate_store import commit_context_write_intents_batch, seed_context
from wccu_eval.utils import append_jsonl, as_dict, as_list, clean, ensure_dir, mean, now_iso, remove_dir, stable_hash, write_json


DEFAULT_CONDITIONS = [
    'adaptive_wccu_execution_trace',
    'adaptive_wccu_projection_trace',
    'adaptive_wccu_model_certificate',
    'adaptive_wccu_no_read_validation',
    'adaptive_readset_occ',
    'adaptive_policy',
    'uniform_snapshot_occ',
    'uniform_review_gated',
    'uniform_append_only',
]

POLICY_BY_CONDITION = {
    'adaptive_wccu_execution_trace': PolicyMode.ADAPTIVE_WCCU_EXECUTION_TRACE,
    'adaptive_wccu_projection_trace': PolicyMode.ADAPTIVE_WCCU_PROJECTION_TRACE,
    'adaptive_wccu_model_certificate': PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE,
    'adaptive_wccu_no_read_validation': PolicyMode.ADAPTIVE_WCCU_NO_READ_VALIDATION,
    'adaptive_readset_occ': PolicyMode.ADAPTIVE_READSET_OCC,
    'adaptive_policy': PolicyMode.ADAPTIVE,
    'uniform_snapshot_occ': PolicyMode.UNIFORM_SNAPSHOT_OCC,
    'uniform_review_gated': PolicyMode.UNIFORM_REVIEW_GATED,
    'uniform_append_only': PolicyMode.UNIFORM_APPEND_ONLY,
}


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _projection_trace(projection: dict[str, Any]) -> dict[str, Any]:
    return {
        'projection_id': projection.get('projection_id'),
        'snapshot_id': projection.get('snapshot_id'),
        'atoms': [
            {
                'id': a.get('id'),
                'atom_type': a.get('atom_type'),
                'status': a.get('status'),
                'title': a.get('title'),
                'canonical_text_en': a.get('canonical_text_en'),
                'tags': a.get('tags') or [],
            }
            for a in as_list(projection.get('atoms'))
        ],
    }


def _model_cert_dep_ids(agent_result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for intent in as_list(agent_result.get('write_intents')):
        cert = as_dict(intent.get('certificate'))
        for dep in as_list(cert.get('read_dependencies')):
            tid = clean(as_dict(dep).get('target_id') or as_dict(dep).get('atom_id') or as_dict(dep).get('id'))
            if tid:
                ids.append(tid)
    return list(dict.fromkeys(ids))


def _dependency_recall(agent_result: dict[str, Any], scenario: dict[str, Any]) -> float:
    expected = expected_dependency_ids(scenario)
    if not expected:
        return 1.0
    got = set(_model_cert_dep_ids(agent_result))
    return sum(1 for dep in expected if dep in got) / len(expected)


def _dependency_precision(agent_result: dict[str, Any], scenario: dict[str, Any]) -> float:
    expected = set(expected_dependency_ids(scenario))
    got = _model_cert_dep_ids(agent_result)
    if not got:
        return 1.0 if not expected else 0.0
    return sum(1 for dep in got if dep in expected) / len(got)


def _writer_intent_count(agent_result: dict[str, Any], writer: str = 'writer_agent') -> int:
    return len(as_list(agent_result.get('write_intents')))


def _intent_sources(intents: list[dict[str, Any]]) -> set[str]:
    return {clean(i.get('source_agent') or as_dict(i.get('source')).get('agent_id')) for i in intents}


def _events_of_kind(row: dict[str, Any], *kinds: str) -> int:
    wanted = {clean(k) for k in kinds}
    count = 0
    for ev in as_list(row.get('wccu_events')):
        ev = as_dict(ev)
        for bucket in ('errors', 'warnings'):
            for item in as_list(ev.get(bucket)):
                if clean(as_dict(item).get('kind')) in wanted:
                    count += 1
    return count


def _summary_from_resolved(*, scenario: dict[str, Any], condition: str, policy_mode: str, resolved: dict[str, Any], commit_result: dict[str, Any], agent_runs: list[dict[str, Any]], elapsed_ms: int, repetition: int, generation_meta: dict[str, Any]) -> dict[str, Any]:
    writer_id = clean(scenario.get('expected_writer_agent_id') or 'writer_agent')
    writer_has_proposal = any(clean(r.get('agent_id')) == writer_id and as_list(r.get('write_intents')) for r in agent_runs)
    issue_family = clean(scenario.get('expected_issue_type') or scenario.get('llm_obligation_family'))
    hold_required = 1 if writer_has_proposal and issue_family not in {'', 'none', 'safe'} else 0
    committable_sources = _intent_sources(as_list(resolved.get('committable')))
    writer_accepted = writer_id in committable_sources
    issue_accepted = 1 if hold_required and writer_accepted else 0
    problem_held = 1 if hold_required and not writer_accepted else 0
    # Commit records can under-report WCCU review routing when a replayed
    # generated proposal is annotated at the resolver layer.  For evaluation
    # metrics, use the maximum of commit-stage proposals/conflicts and resolver
    # intervention counts so a held problematic update is counted as a held
    # update, not as a precision denominator of zero.
    review_burden = max(
        int(commit_result.get('proposals') or 0) + int(commit_result.get('conflicts') or 0),
        int(resolved.get('review_burden_count') or 0),
        int(resolved.get('wccu_review_routed_count') or 0) + int(resolved.get('wccu_blocked_count') or 0),
        int(resolved.get('readset_occ_review_count') or 0),
    )
    all_intents = [i for r in agent_runs for i in as_list(r.get('write_intents'))]
    wccu_metrics = as_dict(resolved.get('wccu_metrics'))
    row = {
        'kind': 'llm_obligation_benchmark_result_v1',
        'scenario_id': scenario.get('id'),
        'llm_obligation_family': scenario.get('llm_obligation_family'),
        'obligation_kind': scenario.get('llm_obligation_family'),
        'condition': condition,
        'policy_mode': str(policy_mode),
        'repetition': repetition,
        'started_at': now_iso(),
        'elapsed_ms': elapsed_ms,
        'agent_count': len(agent_runs),
        'write_intent_count': len(all_intents),
        'commit': {
            'committed': int(commit_result.get('committed') or 0),
            'proposals': int(commit_result.get('proposals') or 0),
            'conflicts': int(commit_result.get('conflicts') or 0),
            'total': int(commit_result.get('total') or 0),
        },
        'conflict_groups': int(resolved.get('conflict_count') or 0),
        'semantic_conflict_count': int(resolved.get('semantic_conflict_count') or 0),
        'auto_merge_groups': int(resolved.get('auto_merge_count') or 0),
        'lock_conflict_count': int(resolved.get('lock_conflict_count') or 0),
        'authority_rebase_count': int(resolved.get('authority_rebase_count') or 0),
        'review_burden_count': review_burden,
        'unsafe_auto_commit_count': int(resolved.get('unsafe_auto_commit_count') or 0),
        'stale_dependency_accepted_count': int(resolved.get('stale_dependency_accepted_count') or 0),
        'wccu_enabled': bool(wccu_metrics.get('wccu_enabled')),
        'wccu_blocked_count': int(resolved.get('wccu_blocked_count') or 0),
        'wccu_review_routed_count': int(resolved.get('wccu_review_routed_count') or 0),
        'wccu_intervention_count': int(resolved.get('wccu_intervention_count') or 0),
        'readset_occ_stale_count': int(resolved.get('readset_occ_stale_count') or 0),
        'readset_occ_review_count': int(resolved.get('readset_occ_review_count') or 0),
        'certificate_invalid_count': int(wccu_metrics.get('certificate_invalid_count') or 0),
        'certificate_missing_count': int(wccu_metrics.get('certificate_missing_count') or 0),
        'low_target_confidence_count': int(wccu_metrics.get('low_target_confidence_count') or 0),
        'stale_dependency_count': int(wccu_metrics.get('stale_dependency_count') or 0),
        'stale_read_validation_ignored_count': int(wccu_metrics.get('stale_read_validation_ignored_count') or 0),
        'authority_insufficient_count': int(wccu_metrics.get('authority_insufficient_count') or 0),
        'authority_certificate_mismatch_count': int(wccu_metrics.get('authority_certificate_mismatch_count') or 0),
        'view_invalidation_count': int(wccu_metrics.get('view_invalidation_count') or 0),
        'wrong_target_count': int(wccu_metrics.get('wrong_target_count') or 0),
        'weaken_rule_delta_count': int(wccu_metrics.get('weaken_rule_delta_count') or 0),
        'delta_contract_mismatch_count': int(wccu_metrics.get('delta_contract_mismatch_count') or 0),
        'semantic_operation_weakening_count': int(wccu_metrics.get('semantic_operation_weakening_count') or 0),
        'semantic_operation_laundering_count': int(wccu_metrics.get('semantic_operation_laundering_count') or 0),
        'authority_laundering_count': int(wccu_metrics.get('authority_laundering_count') or 0),
        'wccu_certificate_mode': wccu_metrics.get('certificate_mode'),
        'wccu_events': as_list(wccu_metrics.get('events')),
        'stale_read_events': _events_of_kind({'wccu_events': as_list(wccu_metrics.get('events'))}, 'stale_read_dependency'),
        'target_events': _events_of_kind({'wccu_events': as_list(wccu_metrics.get('events'))}, 'wrong_target_certificate', 'low_target_confidence'),
        'authority_events': _events_of_kind({'wccu_events': as_list(wccu_metrics.get('events'))}, 'authority_certificate_mismatch', 'authority_insufficient_for_direct_commit', 'authority_required_understated'),
        'operation_events': _events_of_kind({'wccu_events': as_list(wccu_metrics.get('events'))}, 'delta_contract_mismatch', 'weakening_delta_requires_review', 'semantic_operation_weakening', 'semantic_operation_laundering'),
        'view_events': _events_of_kind({'wccu_events': as_list(wccu_metrics.get('events'))}, 'view_invalidation_required'),
        'ground_truth_total_writes': len(all_intents),
        'ground_truth_issue_count': hold_required,
        'ground_truth_hold_required_count': hold_required,
        'ground_truth_problematic_held_count': problem_held,
        'ground_truth_issue_accepted_count': issue_accepted,
        'ground_truth_problematic_accepted_count': issue_accepted,
        'writer_has_proposal': bool(writer_has_proposal),
        'writer_accepted': bool(writer_accepted),
        'agentRuns': agent_runs,
        'merge_decisions': as_list(resolved.get('decisions')),
        **generation_meta,
    }
    return row


def _mock_llm_result(*, scenario: dict[str, Any], projection: dict[str, Any], provider: str = 'mock', model: str = 'mock-llm') -> dict[str, Any]:
    """Deterministic LLM-shaped proposal for smoke tests only."""
    family = clean(scenario.get('llm_obligation_family'))
    atoms = as_list(as_dict(scenario.get('seed')).get('atoms'))
    target = ''
    if family == 'commitment':
        target = next((clean(a.get('id')) for a in atoms if clean(a.get('atom_type')) == 'workspace_file'), '')
    elif family == 'authority':
        target = next((clean(a.get('id')) for a in atoms if clean(a.get('atom_type')) == 'permission_policy'), '')
    elif family == 'derived_view':
        target = next((clean(a.get('id')) for a in atoms if clean(a.get('atom_type')) == 'rule'), '')
    else:
        target = next((clean(a.get('id')) for a in atoms if clean(a.get('atom_type')) in {'rule', 'memory'} and 'target' in clean(a.get('title')).lower()), '') or clean(atoms[-1].get('id'))
    dep_ids = expected_dependency_ids(scenario)
    text = f'Mock LLM proposal for {family} using context dependency {dep_ids[0] if dep_ids else "none"}.'
    intent_type = 'patch_atom'
    atom_type = 'memory'
    file_path = ''
    if target.startswith('file:'):
        atom_type = 'workspace_file'
        file_path = target[len('file:'):]
    elif 'policy' in target:
        atom_type = 'permission_policy' if family == 'authority' else 'rule'
    elif 'rule' in target:
        atom_type = 'rule'
    # Mock outputs intentionally omit the hidden runtime dependency in
    # witness_gap, mirroring the expected behavior of a model that cannot see
    # an internal tool-read record. Execution-witness WCCU should still catch it.
    cert_dep_ids = [] if family == 'witness_gap' else dep_ids
    cert_reads = [
        {'target_id': d, 'view_id': projection.get('projection_id', ''), 'snapshot_id': projection.get('snapshot_id', ''), 'expected_status': 'active', 'expected_text_hash': '', 'freshness_required': True, 'reason': 'Mock model used this context.'}
        for d in cert_dep_ids
    ]
    intent = {
        'intent_type': intent_type,
        'risk': 'high' if family in {'authority', 'operation'} else 'low',
        'authority': 'agent',
        'commit_mode': 'none',
        'payload': {'id': target, 'target_id': target, 'atom_id': target, 'stream_id': '', 'atom_type': atom_type, 'title': target, 'canonical_text_en': text, 'text_original': '', 'reason': '', 'file_path': file_path, 'tags': [family]},
        'certificate': {
            'schema_version': 'wccu_certificate_v1',
            'certificate_id': f'mock_{scenario.get("id")}',
            'certificate_mode': 'model_certificate',
            'read_dependencies': cert_reads,
            'target_certificate': {'claimed_target_id': target, 'raw_target': target, 'grounding_rationale': 'mock exact target', 'confidence': 1.0},
            'delta_contract': {'delta_type': 'weaken_rule' if family == 'operation' else 'patch_memory', 'semantic_direction': 'weaken' if family == 'operation' else 'patch', 'affected_view_ids': [f'target:{target}'], 'invalidates_views': family == 'derived_view', 'summary': text},
            'authority_certificate': {'actor_authority': 'agent', 'required_authority': 'reviewer' if family in {'authority', 'operation'} else 'agent', 'authority_rationale': 'mock certificate'},
            'preconditions': {'base_snapshot_id': projection.get('snapshot_id', 'ctx_000000'), 'freshness_required': True, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True},
        },
    }
    return {
        'agent_id': 'writer_agent',
        'role': as_list(scenario.get('agents'))[0].get('role', 'planner') if as_list(scenario.get('agents')) else 'planner',
        'projection_id': projection.get('projection_id'),
        'snapshot_id': projection.get('snapshot_id'),
        'output': text,
        'agent_task': as_dict(scenario.get('llm_agent_tasks')).get('writer_agent', ''),
        'certificate_guidance': 'mock',
        'write_intents': [intent],
        'latency_ms': 0,
        'context_tokens': projection.get('metrics', {}).get('context_tokens', 0),
        'llm': {'provider': provider, 'model': model, 'prompt_tokens_est': 0, 'output_tokens_est': 0, 'schema_version': 'mock'},
    }


def _generate_writer_result(*, scenario: dict[str, Any], tmp_root: Path, repetition: int, llm_config: dict[str, Any], mock_llm: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    gen_dir = tmp_root / f'{scenario["id"]}_generation_{repetition}'
    root_dir = gen_dir / 'context_substrate'
    remove_dir(gen_dir)
    seed_context(root_dir, scenario.get('seed', {}))
    writer = as_list(scenario.get('agents'))[0]
    projection = compile_projection(root_dir, snapshot_id='ctx_000000', role=writer.get('role') or writer.get('id'), task_type=scenario.get('task_type') or 'llm_obligation_benchmark', goal=scenario.get('goal', ''), budget_tokens=scenario.get('budget_tokens') or 1400)
    started = time.time()
    if mock_llm:
        result = _mock_llm_result(scenario=scenario, projection=projection)
    else:
        result = run_llm_agent(agent=writer, projection=projection, scenario=scenario, llm_config=llm_config)
    result = attach_dependency_witness_to_result(agent=writer, projection=projection, result=result, scenario=scenario, config=llm_config)
    result = {**result, 'projection_trace': _projection_trace(projection)}
    meta = {
        'llm_generation_success': True,
        'llm_schema_valid': True,
        'llm_proposal_count': _writer_intent_count(result),
        'llm_generation_elapsed_ms': int((time.time() - started) * 1000),
        'llm_model_cert_dependency_recall': _dependency_recall(result, scenario),
        'llm_model_cert_dependency_precision': _dependency_precision(result, scenario),
        'llm_model_cert_dependency_count': len(_model_cert_dep_ids(result)),
        'expected_dependency_count': len(expected_dependency_ids(scenario)),
    }
    return result, meta


def _failure_rows(*, scenario: dict[str, Any], conditions: list[str], repetition: int, error: Exception, generation_elapsed_ms: int) -> list[dict[str, Any]]:
    rows = []
    for condition in conditions:
        rows.append({
            'kind': 'llm_obligation_benchmark_result_v1',
            'scenario_id': scenario.get('id'),
            'llm_obligation_family': scenario.get('llm_obligation_family'),
            'obligation_kind': scenario.get('llm_obligation_family'),
            'condition': condition,
            'policy_mode': str(POLICY_BY_CONDITION.get(condition, condition)),
            'repetition': repetition,
            'failed': True,
            'error_type': type(error).__name__,
            'error': str(error)[:1000],
            'llm_generation_success': False,
            'llm_schema_valid': False,
            'llm_proposal_count': 0,
            'llm_generation_elapsed_ms': generation_elapsed_ms,
            'agent_count': 0,
            'write_intent_count': 0,
            'commit': {'committed': 0, 'proposals': 0, 'conflicts': 0, 'total': 0},
            'review_burden_count': 0,
            'ground_truth_total_writes': 0,
            'ground_truth_issue_count': 0,
            'ground_truth_hold_required_count': 0,
            'ground_truth_problematic_held_count': 0,
            'ground_truth_issue_accepted_count': 0,
        })
    return rows


def run_llm_obligation_benchmark(
    *,
    families: str = 'freshness,commitment,authority,operation,derived_view,witness_gap,safe',
    limit_per_family: int = 5,
    condition: str = ','.join(DEFAULT_CONDITIONS),
    repetitions: int = 1,
    out: str = 'results/llm_obligation_benchmark.json',
    provider: str = '',
    model: str = '',
    temperature: float | None = None,
    max_output_tokens: int = 1200,
    timeout_seconds: int = 90,
    max_parse_retries: int = 1,
    max_provider_retries: int = 4,
    retry_backoff_base: float = 1.0,
    retry_backoff_max: float = 20.0,
    certificate_guidance: str = 'unguided',
    enable_target_candidates: bool = True,
    enable_target_grounding: bool = True,
    mock_llm: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    load_dotenv()
    provider = clean(provider or os.environ.get('LLM_PROVIDER') or 'openai')
    model = clean(model or os.environ.get('LLM_MODEL') or '')
    fams = [clean(f) for f in families.split(',') if clean(f)]
    conditions = [clean(c) for c in condition.split(',') if clean(c)]
    for c in conditions:
        if c not in POLICY_BY_CONDITION:
            raise KeyError(f'Unsupported condition for replay benchmark: {c}')
    scenarios = build_llm_obligation_scenarios(families=fams, limit_per_family=limit_per_family)
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    jsonl_path = out_path.with_suffix('.jsonl')
    error_log_path = out_path.with_suffix('.errors.jsonl')
    ensure_dir(out_path.parent)
    if jsonl_path.exists(): jsonl_path.unlink()
    if error_log_path.exists(): error_log_path.unlink()
    tmp_root = REPO_ROOT / 'runs' / f'llm_obligation_tmp_{stable_hash(f"{out}:{os.getpid()}:{now_iso()}")}'
    remove_dir(tmp_root)
    llm_config = {
        'provider': provider,
        'model': model,
        'temperature': temperature,
        'max_output_tokens': max_output_tokens,
        'timeout_seconds': timeout_seconds,
        'max_parse_retries': max_parse_retries,
        'max_provider_retries': max_provider_retries,
        'retry_backoff_base': retry_backoff_base,
        'retry_backoff_max': retry_backoff_max,
        'error_log_path': str(error_log_path),
        'certificate_guidance': certificate_guidance,
        'enable_target_candidates': enable_target_candidates,
        'enable_target_grounding': enable_target_grounding,
        'witness_compiler_enabled': True,
        'witness_attach_to_all_intents': True,
        'witness_source_label': 'instrumented_llm_context_read_witness',
    }
    results: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    for rep in range(max(1, repetitions)):
        for scenario in scenarios:
            gen_started = time.time()
            try:
                writer_result, gen_meta = _generate_writer_result(scenario=scenario, tmp_root=tmp_root, repetition=rep, llm_config=llm_config, mock_llm=mock_llm)
                generations.append({
                    'scenario_id': scenario['id'],
                    'family': scenario.get('llm_obligation_family'),
                    'repetition': rep,
                    'agent_result': writer_result,
                    **gen_meta,
                })
                concurrent = [_copy_json(r) for r in as_list(scenario.get('concurrent_agent_results'))]
                for cond in conditions:
                    run_dir = tmp_root / f'{scenario["id"]}_{cond}_{rep}'
                    root_dir = run_dir / 'context_substrate'
                    remove_dir(run_dir)
                    seed_context(root_dir, scenario.get('seed', {}))
                    agent_runs = [_copy_json(writer_result)] + [_copy_json(r) for r in concurrent]
                    started = time.time()
                    resolved = resolve_parallel_write_intents(agent_runs, policy_mode=POLICY_BY_CONDITION[cond], scenario=scenario, enable_target_grounding=enable_target_grounding)
                    agent_runs = as_list(resolved.get('grounded_agent_results')) or agent_runs
                    commit_result = commit_context_write_intents_batch(root_dir, as_list(resolved.get('merged_intents')))
                    row = _summary_from_resolved(scenario=scenario, condition=cond, policy_mode=str(POLICY_BY_CONDITION[cond]), resolved=resolved, commit_result=commit_result, agent_runs=agent_runs, elapsed_ms=int((time.time() - started) * 1000), repetition=rep, generation_meta=gen_meta)
                    results.append(row)
                    append_jsonl(jsonl_path, row)
            except Exception as exc:
                if fail_fast:
                    raise
                rows = _failure_rows(scenario=scenario, conditions=conditions, repetition=rep, error=exc, generation_elapsed_ms=int((time.time() - gen_started) * 1000))
                for row in rows:
                    results.append(row)
                    append_jsonl(jsonl_path, row)
                if isinstance(exc, LlmProviderError):
                    append_jsonl(error_log_path, {'scenario_id': scenario.get('id'), 'repetition': rep, 'error': exc.to_dict()})
                else:
                    append_jsonl(error_log_path, {'scenario_id': scenario.get('id'), 'repetition': rep, 'error_type': type(exc).__name__, 'error': str(exc)[:1000]})

    aggregated: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault((clean(row.get('llm_obligation_family')), clean(row.get('condition'))), []).append(row)
    for (family, cond), rows in sorted(groups.items()):
        aggregated.append({
            'family': family,
            'condition': cond,
            'n': len(rows),
            'generation_success_rate': mean([1 if r.get('llm_generation_success') else 0 for r in rows]),
            'schema_valid_rate': mean([1 if r.get('llm_schema_valid') else 0 for r in rows]),
            'proposal_rate': mean([1 if int(r.get('llm_proposal_count') or 0) > 0 else 0 for r in rows]),
            'mean_model_cert_dependency_recall': mean([r.get('llm_model_cert_dependency_recall') or 0 for r in rows if r.get('llm_generation_success')]),
            'mean_model_cert_dependency_precision': mean([r.get('llm_model_cert_dependency_precision') or 0 for r in rows if r.get('llm_generation_success')]),
            'provider_error_count': sum(1 for r in rows if r.get('error_type') == 'LlmProviderError'),
        })
    payload = {
        'kind': 'llm_obligation_benchmark_results_v1',
        'generated_at': now_iso(),
        'args': {'families': families, 'limit_per_family': limit_per_family, 'condition': condition, 'repetitions': repetitions, 'out': str(out), 'provider': provider, 'model': model, 'certificate_guidance': certificate_guidance, 'mock_llm': mock_llm},
        'results': results,
        'generations': generations,
        'aggregated_generation': aggregated,
    }
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run real-LLM generated WCCU obligation benchmark and replay identical proposals across baselines.')
    parser.add_argument('--families', default='freshness,commitment,authority,operation,derived_view,witness_gap,safe', help='Comma-separated family list. Valid: ' + ','.join(list_llm_obligation_families()))
    parser.add_argument('--limit-per-family', type=int, default=5)
    parser.add_argument('--condition', default=','.join(DEFAULT_CONDITIONS))
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--out', default='results/llm_obligation_benchmark.json')
    parser.add_argument('--provider', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-output-tokens', type=int, default=1200)
    parser.add_argument('--timeout-seconds', type=int, default=90)
    parser.add_argument('--max-parse-retries', type=int, default=1)
    parser.add_argument('--max-provider-retries', type=int, default=4)
    parser.add_argument('--retry-backoff-base', type=float, default=1.0)
    parser.add_argument('--retry-backoff-max', type=float, default=20.0)
    parser.add_argument('--certificate-guidance', default='unguided', choices=['guided', 'unguided', 'minimal', 'no_hints'])
    parser.add_argument('--disable-target-candidates', action='store_true')
    parser.add_argument('--disable-target-grounding', action='store_true')
    parser.add_argument('--mock-llm', action='store_true', help='Use deterministic LLM-shaped outputs for smoke tests only; not for paper results.')
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args(argv)
    payload = run_llm_obligation_benchmark(
        families=args.families,
        limit_per_family=args.limit_per_family,
        condition=args.condition,
        repetitions=args.repetitions,
        out=args.out,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        max_parse_retries=args.max_parse_retries,
        max_provider_retries=args.max_provider_retries,
        retry_backoff_base=args.retry_backoff_base,
        retry_backoff_max=args.retry_backoff_max,
        certificate_guidance=args.certificate_guidance,
        enable_target_candidates=not args.disable_target_candidates,
        enable_target_grounding=not args.disable_target_grounding,
        mock_llm=args.mock_llm,
        fail_fast=args.fail_fast,
    )
    print(json.dumps({'ok': True, 'results': len(payload.get('results', [])), 'generations': len(payload.get('generations', [])), 'out': args.out}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
