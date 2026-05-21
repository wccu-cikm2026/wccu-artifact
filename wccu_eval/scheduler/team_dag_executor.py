from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from wccu_eval.agents.deterministic_agent import run_deterministic_agent
from wccu_eval.scheduler.context_conflict_resolver import resolve_parallel_write_intents
from wccu_eval.scheduler.context_concurrency_policy import PolicyMode
from wccu_eval.scheduler.dependency_witness import attach_dependency_witness_to_result
from wccu_eval.substrate.context_projection_compiler import compile_projection
from wccu_eval.substrate.context_substrate_store import commit_context_write_intents_batch, read_context_substrate
from wccu_eval.substrate.handoff_delta_store import append_handoff_delta, build_handoff_delta
from wccu_eval.utils import append_jsonl, as_list, ensure_dir, estimate_tokens, now_iso

AgentRunner = Callable[..., dict[str, Any]]


def _metrics_path(run_dir: str | Path, name: str) -> Path:
    return Path(run_dir) / f'{name}.jsonl'


def _summarize_commit(commit_result: dict[str, Any]) -> dict[str, int]:
    return {
        'committed': int(commit_result.get('committed') or 0),
        'proposals': int(commit_result.get('proposals') or 0),
        'conflicts': int(commit_result.get('conflicts') or 0),
        'total': int(commit_result.get('total') or 0),
    }


def _agent_llm_latency_metrics(agent_runs: list[dict[str, Any]], elapsed_ms: int, *, serial: bool = False) -> dict[str, Any]:
    """Decompose wall-clock latency into agent-call and scheduler components.

    The measured run elapsed time is the authoritative wall-clock value for the
    harness cell. Provider-reported per-agent latencies are used only to compute
    an estimated parallelism benefit: in a parallel speculative run the lower
    bound is the slowest agent call, while a serial queue pays roughly the sum of
    agent calls. These metrics are descriptive and intentionally do not claim
    deterministic provider latency or queueing fairness.
    """
    latencies: list[float] = []
    for run in agent_runs:
        llm = run.get('llm') if isinstance(run.get('llm'), dict) else {}
        value = llm.get('elapsed_ms')
        try:
            n = float(value or 0)
            if n > 0:
                latencies.append(n)
        except Exception:
            pass
    agent_sum = sum(latencies)
    agent_max = max(latencies) if latencies else 0.0
    observed = float(elapsed_ms or 0)
    denominator = observed if observed > 0 else 0.0
    speedup = (agent_sum / denominator) if denominator and agent_sum else 0.0
    efficiency = (speedup / len(latencies)) if latencies else 0.0
    parallel_lower_bound = agent_max if latencies else 0.0
    serial_lower_bound = agent_sum if latencies else 0.0
    return {
        'agent_api_elapsed_ms_sum': int(agent_sum),
        'agent_api_elapsed_ms_max': int(agent_max),
        'agent_api_call_count': len(latencies),
        'parallel_speedup_est': speedup,
        'parallel_efficiency_est': efficiency,
        'parallel_lower_bound_ms': int(parallel_lower_bound),
        'serial_lower_bound_ms': int(serial_lower_bound),
        'scheduler_overhead_ms': int(max(0.0, observed - (agent_sum if serial else agent_max))),
    }


def expected_success(scenario: dict[str, Any], summary: dict[str, Any]) -> bool:
    expected = dict(scenario.get('expected') or {})
    condition = str(summary.get('condition') or summary.get('policy_mode') or '')
    policy_mode = str(summary.get('policy_mode') or '')
    by_condition = scenario.get('expected_by_condition') or {}
    for key in (condition, policy_mode):
        if key in by_condition:
            expected.update(by_condition.get(key) or {})
            break
    if expected.get('requires_proposals') and int(summary.get('commit', {}).get('proposals') or 0) < int(expected['requires_proposals']):
        return False
    if expected.get('max_unsafe_auto_commit_count') is not None and int(summary.get('unsafe_auto_commit_count') or 0) > int(expected['max_unsafe_auto_commit_count']):
        return False
    if expected.get('min_conflict_groups') is not None and int(summary.get('conflict_groups') or 0) < int(expected['min_conflict_groups']):
        return False
    if expected.get('min_authority_rebase_count') is not None and int(summary.get('authority_rebase_count') or 0) < int(expected['min_authority_rebase_count']):
        return False
    if expected.get('min_lock_conflict_count') is not None and int(summary.get('lock_conflict_count') or 0) < int(expected['min_lock_conflict_count']):
        return False
    if expected.get('min_semantic_conflict_count') is not None and int(summary.get('semantic_conflict_count') or 0) < int(expected['min_semantic_conflict_count']):
        return False
    wccu_intervention_count = int(summary.get('wccu_intervention_count') or 0)
    if not wccu_intervention_count:
        wccu_intervention_count = int(summary.get('wccu_review_routed_count') or 0) + int(summary.get('wccu_blocked_count') or 0)
    if expected.get('min_wccu_blocked_count') is not None and int(summary.get('wccu_blocked_count') or 0) < int(expected['min_wccu_blocked_count']):
        return False
    if expected.get('min_wccu_review_routed_count') is not None and int(summary.get('wccu_review_routed_count') or 0) < int(expected['min_wccu_review_routed_count']):
        return False
    if expected.get('min_wccu_intervention_count') is not None and wccu_intervention_count < int(expected['min_wccu_intervention_count']):
        return False
    if expected.get('max_wccu_intervention_count') is not None and wccu_intervention_count > int(expected['max_wccu_intervention_count']):
        return False
    if expected.get('max_stale_dependency_count') is not None and int(summary.get('stale_dependency_count') or 0) > int(expected['max_stale_dependency_count']):
        return False
    if expected.get('min_stale_dependency_count') is not None and int(summary.get('stale_dependency_count') or 0) < int(expected['min_stale_dependency_count']):
        return False
    if expected.get('max_stale_dependency_accepted_count') is not None and int(summary.get('stale_dependency_accepted_count') or 0) > int(expected['max_stale_dependency_accepted_count']):
        return False
    if expected.get('min_stale_dependency_accepted_count') is not None and int(summary.get('stale_dependency_accepted_count') or 0) < int(expected['min_stale_dependency_accepted_count']):
        return False
    if expected.get('min_stale_read_validation_ignored_count') is not None and int(summary.get('stale_read_validation_ignored_count') or 0) < int(expected['min_stale_read_validation_ignored_count']):
        return False
    if expected.get('max_stale_read_validation_ignored_count') is not None and int(summary.get('stale_read_validation_ignored_count') or 0) > int(expected['max_stale_read_validation_ignored_count']):
        return False
    if expected.get('conflicts') is not None:
        return int(summary.get('conflict_groups') or 0) >= int(expected['conflicts'])
    return True


def execute_context_policy_parallel(*, root_dir: str | Path, run_dir: str | Path, scenario: dict[str, Any], agents: list[dict[str, Any]] | None = None, goal: str | None = None, policy_mode: str | PolicyMode = PolicyMode.ADAPTIVE, condition: str | None = None, agent_runner: AgentRunner = run_deterministic_agent, agent_runner_config: dict[str, Any] | None = None, **_ignored: Any) -> dict[str, Any]:
    ensure_dir(run_dir)
    started = time.time()
    agents = agents or scenario.get('agents') or []
    goal = goal if goal is not None else scenario.get('goal', '')
    condition = condition or ('adaptive_policy' if str(policy_mode) == str(PolicyMode.ADAPTIVE) else str(policy_mode))
    base_snapshot_id = read_context_substrate(root_dir)['snapshot_id']
    cfg = agent_runner_config or {}

    def run_one(agent: dict[str, Any]) -> dict[str, Any]:
        projection = compile_projection(
            root_dir,
            snapshot_id=base_snapshot_id,
            role=agent.get('role') or agent.get('id'),
            task_type=agent.get('task_type') or scenario.get('task_type') or 'general_task',
            goal=goal,
            budget_tokens=agent.get('budget_tokens') or scenario.get('budget_tokens') or 1200,
        )
        result = agent_runner(agent=agent, projection=projection, scenario=scenario, llm_config=cfg, agent_runner_config=cfg)
        result = attach_dependency_witness_to_result(agent=agent, projection=projection, result=result, scenario=scenario, config=cfg)
        # Keep a compact projection trace so WCCU projection-trace mode can
        # validate read dependencies without relying on oracle fixtures.
        result = {
            **result,
            'projection_trace': {
                'projection_id': projection['projection_id'],
                'snapshot_id': projection['snapshot_id'],
                'atoms': [
                    {
                        'id': a.get('id'),
                        'atom_type': a.get('atom_type'),
                        'status': a.get('status'),
                        'title': a.get('title'),
                        'canonical_text_en': a.get('canonical_text_en'),
                        'tags': a.get('tags') or [],
                    }
                    for a in projection.get('atoms', [])
                ],
            },
        }
        append_jsonl(_metrics_path(run_dir, 'context_projection_events'), {
            'kind': 'context_projection_event_v1',
            'timestamp': now_iso(),
            'projection_id': projection['projection_id'],
            'snapshot_id': projection['snapshot_id'],
            'role': agent.get('role') or agent.get('id'),
            'context_tokens': projection['metrics']['context_tokens'],
            'selected_atom_count': projection['metrics']['selected_atom_count'],
            'role_context_purity': projection['metrics']['role_context_purity'],
        })
        append_handoff_delta(run_dir, build_handoff_delta(from_agent=agent.get('id'), to_agent='merge_stage', handoff_type='write_intents_ready', snapshot_id=base_snapshot_id, projection_id=projection['projection_id'], delta={'output': result.get('output'), 'write_intent_count': len(result.get('write_intents', []))}))
        return result

    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        agent_runs = list(pool.map(run_one, agents))

    resolved = resolve_parallel_write_intents(agent_runs, policy_mode=policy_mode, scenario=scenario, enable_target_grounding=bool(cfg.get('enable_target_grounding', True)))
    agent_runs = resolved.get('grounded_agent_results', agent_runs)
    commit_result = commit_context_write_intents_batch(root_dir, resolved['merged_intents'])
    elapsed_ms = int((time.time() - started) * 1000)
    all_intents = [i for r in agent_runs for i in r.get('write_intents', [])]
    summary = {
        'kind': 'parallel_execution_result_v2',
        'condition': condition,
        'policy_mode': resolved['policy_mode'],
        'scenario_id': scenario['id'],
        'started_at': now_iso(),
        'elapsed_ms': elapsed_ms,
        'base_snapshot_id': base_snapshot_id,
        'agent_count': len(agent_runs),
        'write_intent_count': len(all_intents),
        'conflict_groups': resolved['conflict_count'],
        'semantic_conflict_count': resolved.get('semantic_conflict_count', 0),
        'auto_merge_groups': resolved['auto_merge_count'],
        'lock_conflict_count': resolved['lock_conflict_count'],
        'authority_rebase_count': resolved['authority_rebase_count'],
        'review_burden_count': int(commit_result.get('proposals') or 0) + int(commit_result.get('conflicts') or 0),
        'commit': _summarize_commit(commit_result),
        'context_tokens': sum(int(r.get('context_tokens') or 0) for r in agent_runs),
        'unsafe_auto_commit_count': resolved['unsafe_auto_commit_count'],
        'stale_write_blocked_count': int(commit_result.get('conflicts') or 0),
        'wccu_enabled': bool(resolved.get('wccu_metrics', {}).get('wccu_enabled')),
        'wccu_blocked_count': int(resolved.get('wccu_blocked_count', resolved.get('wccu_blocked_count')) or 0),
        'wccu_review_routed_count': int(resolved.get('wccu_review_routed_count', resolved.get('wccu_review_routed_count')) or 0),
        'wccu_intervention_count': int(resolved.get('wccu_intervention_count', resolved.get('wccu_intervention_count')) or ((resolved.get('wccu_review_routed_count') or 0) + (resolved.get('wccu_blocked_count') or 0))),
        'readset_occ_stale_count': int(resolved.get('readset_occ_stale_count') or 0),
        'readset_occ_review_count': int(resolved.get('readset_occ_review_count') or 0),
        'stale_dependency_accepted_count': int(resolved.get('stale_dependency_accepted_count') or 0),
        'stale_read_validation_ignored_count': int(resolved.get('wccu_metrics', {}).get('stale_read_validation_ignored_count') or 0),
        'certificate_invalid_count': int(resolved.get('wccu_metrics', {}).get('certificate_invalid_count') or 0),
        'certificate_missing_count': int(resolved.get('wccu_metrics', {}).get('certificate_missing_count') or 0),
        'low_target_confidence_count': int(resolved.get('wccu_metrics', {}).get('low_target_confidence_count') or 0),
        'stale_dependency_count': int(resolved.get('wccu_metrics', {}).get('stale_dependency_count') or 0),
        'authority_insufficient_count': int(resolved.get('wccu_metrics', {}).get('authority_insufficient_count') or 0),
        'authority_certificate_mismatch_count': int(resolved.get('wccu_metrics', {}).get('authority_certificate_mismatch_count') or 0),
        'view_invalidation_count': int(resolved.get('wccu_metrics', {}).get('view_invalidation_count') or 0),
        'wrong_target_count': int(resolved.get('wccu_metrics', {}).get('wrong_target_count') or 0),
        'weaken_rule_delta_count': int(resolved.get('wccu_metrics', {}).get('weaken_rule_delta_count') or 0),
        'delta_contract_mismatch_count': int(resolved.get('wccu_metrics', {}).get('delta_contract_mismatch_count') or 0),
        'wccu_certificate_mode': resolved.get('wccu_metrics', {}).get('certificate_mode'),
        'wccu_events': as_list(resolved.get('wccu_metrics', {}).get('events')),
        'wccu_certificate_mode': resolved.get('wccu_metrics', {}).get('certificate_mode'),
        'wccu_events': as_list(resolved.get('wccu_metrics', {}).get('events')),
        'agentRuns': agent_runs,
        'merge_decisions': resolved['decisions'],
    }
    summary.update(_agent_llm_latency_metrics(agent_runs, elapsed_ms, serial=False))
    summary['task_success'] = expected_success(scenario, summary)
    append_jsonl(_metrics_path(run_dir, 'parallel_execution_metrics'), summary)
    append_jsonl(_metrics_path(run_dir, 'context_conflict_events'), {'timestamp': now_iso(), 'policy_mode': str(policy_mode), 'decisions': resolved['decisions']})
    append_jsonl(_metrics_path(run_dir, 'context_policy_events'), {'timestamp': now_iso(), 'policy_mode': str(policy_mode), 'scenario_id': scenario['id'], 'decisions': [{'target': d['target'], 'decision': d['decision'], 'policy_counts': d['policy_counts']} for d in resolved['decisions']]})
    return summary


def execute_substrate_parallel_group(**kwargs: Any) -> dict[str, Any]:
    return execute_context_policy_parallel(**kwargs, policy_mode=PolicyMode.ADAPTIVE, condition=kwargs.get('condition') or 'substrate_parallel')


def execute_substrate_serial(*, root_dir: str | Path, run_dir: str | Path, scenario: dict[str, Any], agents: list[dict[str, Any]] | None = None, goal: str | None = None, policy_mode: str | PolicyMode = PolicyMode.ADAPTIVE, condition: str | None = None, agent_runner: AgentRunner = run_deterministic_agent, agent_runner_config: dict[str, Any] | None = None, **_ignored: Any) -> dict[str, Any]:
    ensure_dir(run_dir)
    started = time.time()
    agents = agents or scenario.get('agents') or []
    goal = goal if goal is not None else scenario.get('goal', '')
    condition = condition or ('serial_adaptive_policy' if str(policy_mode) == str(PolicyMode.ADAPTIVE) else f'serial_{str(policy_mode)}')
    agent_runs = []
    conflict_groups = 0
    unsafe_auto_commit_count = 0
    review_burden_count = 0
    wccu_blocked_count = 0
    wccu_review_routed_count = 0
    wccu_intervention_count = 0
    stale_dependency_accepted_count = 0
    prior: dict[str, str] = {}
    cfg = agent_runner_config or {}
    for agent in agents:
        substrate = read_context_substrate(root_dir)
        projection = compile_projection(root_dir, snapshot_id=substrate['snapshot_id'], role=agent.get('role') or agent.get('id'), task_type=agent.get('task_type') or scenario.get('task_type') or 'general_task', goal=goal, budget_tokens=agent.get('budget_tokens') or scenario.get('budget_tokens') or 1200)
        result = agent_runner(agent=agent, projection=projection, scenario=scenario, llm_config=cfg, agent_runner_config=cfg)
        result = attach_dependency_witness_to_result(agent=agent, projection=projection, result=result, scenario=scenario, config=cfg)
        for intent in result.get('write_intents', []):
            payload = intent.get('payload') or {}
            target_id = payload.get('id') or payload.get('atom_id') or intent.get('id')
            text = payload.get('canonical_text_en') or payload.get('text_original') or payload.get('reason') or ''
            if target_id and target_id in prior and prior[target_id] != text:
                unsafe_auto_commit_count += 1
            if target_id:
                prior[target_id] = text
        resolved = resolve_parallel_write_intents([result], policy_mode=policy_mode, scenario=scenario, enable_target_grounding=bool(cfg.get('enable_target_grounding', True)))
        commit = commit_context_write_intents_batch(root_dir, resolved['merged_intents'])
        conflict_groups += int(resolved['conflict_count']) + int(commit.get('conflicts') or 0)
        review_burden_count += int(commit.get('proposals') or 0) + int(commit.get('conflicts') or 0)
        wccu_blocked_count += int(resolved.get('wccu_blocked_count') or 0)
        wccu_review_routed_count += int(resolved.get('wccu_review_routed_count') or 0)
        wccu_intervention_count += int(resolved.get('wccu_intervention_count') or ((resolved.get('wccu_review_routed_count') or 0) + (resolved.get('wccu_blocked_count') or 0)))
        stale_dependency_accepted_count += int(resolved.get('stale_dependency_accepted_count') or 0)
        agent_runs.append({**result, 'commit': _summarize_commit(commit), 'merge_decisions': resolved.get('decisions')})
    summary = {
        'kind': 'serial_execution_result_v2',
        'condition': condition,
        'policy_mode': f'{policy_mode}_serial_queue',
        'scenario_id': scenario['id'],
        'elapsed_ms': int((time.time() - started) * 1000),
        'agent_count': len(agent_runs),
        'conflict_groups': conflict_groups,
        'commit': {
            'committed': sum(r['commit']['committed'] for r in agent_runs),
            'proposals': sum(r['commit']['proposals'] for r in agent_runs),
            'conflicts': sum(r['commit']['conflicts'] for r in agent_runs),
            'total': sum(r['commit']['total'] for r in agent_runs),
        },
        'context_tokens': sum(int(r.get('context_tokens') or 0) for r in agent_runs),
        'unsafe_auto_commit_count': unsafe_auto_commit_count,
        'review_burden_count': review_burden_count,
        'wccu_enabled': str(policy_mode).startswith('adaptive_wccu'),
        'wccu_blocked_count': wccu_blocked_count,
        'wccu_review_routed_count': wccu_review_routed_count,
        'wccu_intervention_count': wccu_intervention_count,
        'stale_dependency_accepted_count': stale_dependency_accepted_count,
        'agentRuns': agent_runs,
    }
    summary.update(_agent_llm_latency_metrics(agent_runs, int(summary.get('elapsed_ms') or 0), serial=True))
    summary['task_success'] = expected_success(scenario, summary)
    append_jsonl(_metrics_path(run_dir, 'parallel_execution_metrics'), summary)
    return summary


def approximate_prompt_only_tokens(scenario: dict[str, Any]) -> int:
    return (len(as_list(scenario.get('agents'))) or 1) * estimate_tokens(str(scenario.get('seed', {})) + str(scenario))
