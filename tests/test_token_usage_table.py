from __future__ import annotations

from wccu_eval.scripts.make_token_usage_table import compute_rows


def _run(condition: str, *, unsafe: int, stale: int, success: bool, total_tokens: int, cached: int = 0, failed: bool = False):
    # Split tokens over two agent calls to resemble parallel agent runs.
    half = total_tokens // 2
    other = total_tokens - half
    return {
        'condition': condition,
        'task_success': success,
        'failed': failed,
        'unsafe_auto_commit_count': unsafe,
        'stale_dependency_accepted_count': stale,
        'stale_dependency_count': 1 if stale == 0 and condition.startswith('adaptive_wccu') else 0,
        'review_burden_count': 1 if condition.startswith('adaptive_wccu') else 0,
        'wccu_intervention_count': 1 if condition.startswith('adaptive_wccu') else 0,
        'elapsed_ms': 100,
        'agentRuns': [
            {'latency_ms': 40, 'llm': {'api_usage': {'input_tokens': half - 20, 'output_tokens': 20, 'total_tokens': half, 'reasoning_tokens': 3, 'cached_input_tokens': cached // 2}}},
            {'latency_ms': 50, 'llm': {'api_usage': {'input_tokens': other - 30, 'output_tokens': 30, 'total_tokens': other, 'reasoning_tokens': 4, 'cached_input_tokens': cached - cached // 2}}},
        ],
    }


def test_token_usage_table_computes_baseline_overhead_and_prevented_unsafe():
    payload = {
        'results': [
            _run('adaptive_policy', unsafe=1, stale=1, success=False, total_tokens=1000, cached=100),
            _run('adaptive_wccu_execution_trace', unsafe=0, stale=0, success=True, total_tokens=1250, cached=200),
        ]
    }
    rows = {r['condition']: r for r in compute_rows(payload, baseline_condition='adaptive_policy')}
    wccu = rows['adaptive_wccu_execution_trace']
    assert wccu['runs'] == 1
    assert wccu['agent_calls'] == 2
    assert wccu['actual_usage_coverage_rate'] == 1.0
    assert wccu['total_tokens_sum'] == 1250
    assert wccu['cached_input_tokens_sum'] == 200
    assert wccu['unsafe_prevented_vs_baseline_count'] == 1
    assert wccu['stale_acceptance_prevented_vs_baseline_count'] == 1
    assert wccu['token_overhead_vs_baseline_pct'] == 25.0
    assert wccu['tokens_per_unsafe_prevented_vs_baseline'] == 1250.0


def test_token_usage_table_falls_back_to_estimates_when_api_usage_missing():
    payload = {
        'results': [
            {
                'condition': 'adaptive_policy',
                'task_success': True,
                'failed': False,
                'unsafe_auto_commit_count': 0,
                'stale_dependency_accepted_count': 0,
                'agentRuns': [
                    {'llm': {'prompt_tokens_est': 123, 'output_tokens_est': 45}},
                ],
            }
        ]
    }
    row = compute_rows(payload, baseline_condition='adaptive_policy')[0]
    assert row['agent_calls'] == 1
    assert row['agent_calls_with_actual_usage'] == 0
    assert row['actual_usage_coverage_rate'] == 0.0
    assert row['total_tokens_sum'] == 168
    assert row['mean_total_tokens_per_run'] == 168.0
