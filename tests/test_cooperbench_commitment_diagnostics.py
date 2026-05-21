from __future__ import annotations

import json
import tempfile
from pathlib import Path

from wccu_eval.external.cooperbench_adapter import cooperbench_task_to_scenario
from wccu_eval.scripts.make_cooperbench_commitment_diagnostics import make_commitment_diagnostics
from wccu_eval.scheduler.context_conflict_resolver import resolve_parallel_write_intents


def _base_task() -> dict:
    return {
        'task_id': 'repo_task0_feature1_feature2',
        'repo': 'repo_task',
        'agent_a_task': 'Feature A writes tests that assume Feature B exposes cache_status API.',
        'agent_b_task': 'Feature B initially promises cache_status API but later changes it to cache_info API.',
        'shared_targets': [
            {'target_id': 'file:tests/test_cache.py', 'file_path': 'tests/test_cache.py', 'target_type': 'workspace_file', 'title': 'tests/test_cache.py'},
            {'target_id': 'file:src/cache.py', 'file_path': 'src/cache.py', 'target_type': 'workspace_file', 'title': 'src/cache.py'},
        ],
        'expected_conflict_type': 'workspace_contention',
    }


def test_make_commitment_diagnostics_marks_tasks():
    rows = make_commitment_diagnostics([_base_task()], limit=1, seed=7)
    assert len(rows) == 1
    row = rows[0]
    assert row['scenario_type'] == 'commitment_stale_dependency'
    assert row['commitment_b_id'].startswith('commitment:repo_task0_feature1_feature2')
    assert row['shared_targets'][0]['file_path'] == 'tests/test_cache.py'


def test_commitment_diagnostic_scenario_has_cross_target_dependency():
    row = make_commitment_diagnostics([_base_task()], limit=1, seed=7)[0]
    scenario = cooperbench_task_to_scenario(row)
    assert scenario['task_type'] == 'cooperbench_commitment_stale_dependency'
    dep_id = scenario['wccu_read_dependencies']['coop_agent_a'][0]['target_id']
    a_intent = scenario['agent_outputs']['coop_agent_a']['intents'][0]
    b_intent = scenario['agent_outputs']['coop_agent_b']['intents'][0]
    assert a_intent['payload']['target_id'].startswith('file:')
    assert b_intent['payload']['target_id'] == dep_id
    assert a_intent['payload']['target_id'] != dep_id


def test_commitment_diagnostic_wccu_blocks_stale_cross_target_dependency():
    row = make_commitment_diagnostics([_base_task()], limit=1, seed=7)[0]
    scenario = cooperbench_task_to_scenario(row)
    a = {'agent_id': 'coop_agent_a', 'role': 'builder', 'write_intents': scenario['agent_outputs']['coop_agent_a']['intents'], 'projection_trace': {'projection_id': 'p', 'snapshot_id': 'ctx_000000', 'atoms': scenario['seed']['atoms']}, 'agent_task': scenario['llm_agent_tasks']['coop_agent_a'], 'output': scenario['agent_outputs']['coop_agent_a']['text']}
    b = {'agent_id': 'coop_agent_b', 'role': 'builder', 'write_intents': scenario['agent_outputs']['coop_agent_b']['intents'], 'projection_trace': {'projection_id': 'p', 'snapshot_id': 'ctx_000000', 'atoms': scenario['seed']['atoms']}, 'agent_task': scenario['llm_agent_tasks']['coop_agent_b'], 'output': scenario['agent_outputs']['coop_agent_b']['text']}
    wccu = resolve_parallel_write_intents([a, b], policy_mode='adaptive_wccu_execution_trace', scenario=scenario, enable_target_grounding=True)
    assert wccu['wccu_metrics']['stale_dependency_count'] >= 1
    assert wccu['wccu_intervention_count'] >= 1
    assert wccu['stale_dependency_accepted_count'] == 0
    base = resolve_parallel_write_intents([a, b], policy_mode='adaptive_policy', scenario=scenario, enable_target_grounding=True)
    assert base['stale_dependency_accepted_count'] >= 1
    assert base['unsafe_auto_commit_count'] >= 1
