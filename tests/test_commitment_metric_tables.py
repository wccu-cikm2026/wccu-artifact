from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wccu_eval.scripts.make_cooperbench_commitment_table import compute_rows, write_tex
from wccu_eval.scripts.make_wccu_ablation_table import build_rows as build_wccu_rows


class CommitmentMetricTableTests(unittest.TestCase):
    def test_commitment_table_counts_safe_block_as_freshness_pass(self):
        payload = {
            'results': [
                {
                    'condition': 'adaptive_wccu_execution_trace',
                    'task_success': False,  # old narrow lane diagnostic rejects block
                    'stale_dependency_count': 1,
                    'stale_dependency_accepted_count': 0,
                    'unsafe_auto_commit_count': 0,
                    'wccu_intervention_count': 1,
                    'wccu_blocked_count': 1,
                    'review_burden_count': 1,
                }
            ]
        }
        rows = compute_rows(payload)
        self.assertEqual(rows[0]['diagnostic_pass'], 0)
        self.assertEqual(rows[0]['freshness_pass'], 1)
        self.assertEqual(rows[0]['safety_pass'], 1)
        self.assertEqual(rows[0]['wccu_freshness_success'], 1)
        self.assertEqual(rows[0]['wccu_blocked_count'], 1)
        with tempfile.TemporaryDirectory() as d:
            tex = Path(d) / 'commitment.tex'
            write_tex(tex, rows)
            text = tex.read_text(encoding='utf-8')
            self.assertIn('Freshness', text)
            self.assertIn('1/1', text)

    def test_commitment_table_separates_stale_accept_from_old_task_success(self):
        payload = {
            'results': [
                {
                    'condition': 'adaptive_wccu_model_certificate',
                    'task_success': True,
                    'stale_dependency_count': 1,
                    'stale_dependency_accepted_count': 1,
                    'unsafe_auto_commit_count': 1,
                    'wccu_intervention_count': 0,
                }
            ]
        }
        rows = compute_rows(payload)
        self.assertEqual(rows[0]['diagnostic_pass'], 1)
        self.assertEqual(rows[0]['freshness_pass'], 0)
        self.assertEqual(rows[0]['wccu_freshness_success'], 0)

    def test_wccu_ablation_table_reports_freshness_pass_rate(self):
        payload = {
            'results': [
                {
                    'condition': 'adaptive_wccu_execution_trace',
                    'task_success': False,
                    'stale_dependency_count': 1,
                    'stale_dependency_accepted_count': 0,
                    'unsafe_auto_commit_count': 0,
                    'wccu_blocked_count': 1,
                    'review_burden_count': 1,
                }
            ]
        }
        rows = build_wccu_rows(payload)
        self.assertEqual(rows[0]['raw_task_success_rate'], 0.0)
        self.assertEqual(rows[0]['freshness_pass_rate'], 1.0)
        self.assertEqual(rows[0]['success_rate'], 1.0)
        self.assertEqual(rows[0]['freshness_pass_count'], 1)


if __name__ == '__main__':
    unittest.main()
