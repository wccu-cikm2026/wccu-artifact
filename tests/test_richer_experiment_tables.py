from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wccu_eval.scripts.make_stress_curve_table import build_rows as build_stress_curve_rows
from wccu_eval.scripts.make_target_ablation_table import build_rows as build_target_rows, write_tex as write_target_tex


class RicherExperimentTableTests(unittest.TestCase):
    def test_stress_curve_rows_group_by_setting(self):
        payload = {
            'args': {'seed': 7, 'writers': 4, 'atom_count': 16, 'invalidation_prob': 0.35},
            'results': [
                {
                    'condition': 'adaptive_wccu_execution_trace',
                    'stress_metadata': {'writers': 4, 'atom_count': 16},
                    'stale_dependency_count': 2,
                    'stale_dependency_accepted_count': 0,
                    'unsafe_auto_commit_count': 0,
                    'wccu_intervention_count': 2,
                    'review_burden_count': 2,
                },
                {
                    'condition': 'adaptive_policy',
                    'stress_metadata': {'writers': 4, 'atom_count': 16},
                    'stale_dependency_count': 0,
                    'stale_dependency_accepted_count': 2,
                    'unsafe_auto_commit_count': 2,
                    'review_burden_count': 0,
                },
            ],
        }
        rows = build_stress_curve_rows([payload])
        self.assertEqual(len(rows), 2)
        wccu = [r for r in rows if r['condition'] == 'adaptive_wccu_execution_trace'][0]
        self.assertEqual(wccu['writers'], 4)
        self.assertEqual(wccu['stale_dependencies'], 2)
        self.assertEqual(wccu['stale_accepted'], 0)
        self.assertEqual(wccu['review_per_stale_dependency'], 1.0)

    def test_target_ablation_rows_and_tex(self):
        payload = {
            'results': [
                {
                    'condition': 'adaptive_candidates_with_grounding',
                    'task_success': True,
                    'unsafe_auto_commit_count': 0,
                    'wrong_target_count': 0,
                    'low_target_confidence_count': 1,
                    'review_burden_count': 1,
                    'agentRuns': [
                        {'llm': {'api_usage': {'total_tokens': 100}}, 'write_intents': [
                            {'intent_type': 'patch_atom', 'target_grounding': {'resolved': True}}
                        ]}
                    ],
                }
            ]
        }
        rows = build_target_rows(payload)
        self.assertEqual(rows[0]['diagnostic_pass'], 1)
        self.assertEqual(rows[0]['wrong_or_low_target_events'], 1)
        self.assertEqual(rows[0]['target_grounded_rate'], 1.0)
        with tempfile.TemporaryDirectory() as d:
            tex = Path(d) / 'target.tex'
            write_target_tex(tex, rows)
            self.assertIn('Candidates / grounding', tex.read_text(encoding='utf-8'))
            self.assertIn('100.0\\%', tex.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
