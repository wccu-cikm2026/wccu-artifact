from wccu_eval.scripts.make_cooperbench_table import compute_rows


def test_cooperbench_table_separates_lane_and_safety_pass():
    payload = {
        'external_benchmark': 'cooperbench',
        'results': [
            {
                'condition': 'adaptive_wccu_execution_trace',
                'task_success': True,
                'failed': False,
                'unsafe_auto_commit_count': 0,
                'review_burden_count': 2,
                'commit': {'committed': 0, 'proposals': 2},
                'lock_conflict_count': 1,
                'wccu_intervention_count': 1,
                'certificate_invalid_count': 1,
                'merge_decisions': [{'decision': 'lock_contention_review_required_with_wccu', 'target': 'lock:src/a.py'}],
            },
            {
                'condition': 'uniform_review_gated',
                'task_success': False,
                'failed': False,
                'unsafe_auto_commit_count': 0,
                'review_burden_count': 2,
                'commit': {'committed': 0, 'proposals': 2},
                'lock_conflict_count': 0,
                'merge_decisions': [{'decision': 'review_gated', 'target': 'file:src/a.py'}],
            },
            {
                'condition': 'uniform_append_only',
                'task_success': False,
                'failed': False,
                'unsafe_auto_commit_count': 2,
                'review_burden_count': 0,
                'commit': {'committed': 2, 'proposals': 0},
                'lock_conflict_count': 0,
                'merge_decisions': [{'decision': 'append_only', 'target': 'file:src/a.py'}],
            },
        ],
    }
    rows = {row['condition']: row for row in compute_rows(payload)}
    assert rows['adaptive_wccu_execution_trace']['lane_diagnostic_pass'] == 1
    assert rows['adaptive_wccu_execution_trace']['safety_pass'] == 1
    assert rows['adaptive_wccu_execution_trace']['wccu_intervention_count'] == 1
    assert rows['uniform_review_gated']['lane_diagnostic_pass'] == 0
    assert rows['uniform_review_gated']['safety_pass'] == 1
    assert rows['uniform_append_only']['safety_pass'] == 0
    assert rows['uniform_append_only']['unsafe_auto_commit_count'] == 2
