from __future__ import annotations

from wccu_eval.eval.run_llm_experiment import build_conditions
from wccu_eval.scripts.make_cooperbench_commitment_table import compute_rows


def test_llm_build_conditions_exposes_modern_wccu_names():
    conds = build_conditions({'provider': 'mock', 'model': 'mock-llm'})
    assert 'adaptive_wccu_execution_trace' in conds
    assert 'adaptive_wccu_projection_trace' in conds
    assert 'adaptive_wccu_model_certificate' in conds


def test_cooperbench_commitment_table_accepts_wccu_condition_names():
    payload = {
        'results': [
            {
                'condition': 'adaptive_wccu_execution_trace',
                'failed': False,
                'task_success': True,
                'stale_dependency_count': 1,
                'stale_dependency_accepted_count': 0,
                'unsafe_auto_commit_count': 0,
                'wccu_intervention_count': 1,
                'review_burden_count': 1,
            },
            {
                'condition': 'adaptive_policy',
                'failed': False,
                'task_success': False,
                'stale_dependency_count': 1,
                'stale_dependency_accepted_count': 1,
                'unsafe_auto_commit_count': 1,
                'review_burden_count': 0,
            },
        ]
    }
    rows = compute_rows(payload)
    by_cond = {r['condition']: r for r in rows}
    assert by_cond['adaptive_wccu_execution_trace']['wccu_freshness_success'] == 1
    assert by_cond['adaptive_policy']['stale_dependency_accepted_count'] == 1
