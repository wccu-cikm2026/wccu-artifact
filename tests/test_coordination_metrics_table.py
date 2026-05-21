from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wccu_eval.scripts.coordination_quality import row_quality
from wccu_eval.scripts.make_coordination_metrics_table import build_rows, write_tex


class CoordinationMetricsTableTests(unittest.TestCase):
    def test_stress_metrics_use_latent_dependencies_from_metadata(self):
        payload = {
            'kind': 'wccu_randomized_stress_result',
            'args': {'writers': 4, 'atom_count': 16, 'invalidation_prob': 0.35},
            'results': [
                {
                    'condition': 'adaptive_wccu_execution_trace',
                    'agent_count': 6,
                    'write_intent_count': 6,
                    'stress_metadata': {'writers': 4, 'atom_count': 16, 'invalidated_dependency_count': 2},
                    'review_burden_count': 2,
                    'stale_dependency_count': 2,
                    'stale_dependency_accepted_count': 0,
                    'unsafe_auto_commit_count': 0,
                    'wccu_intervention_count': 2,
                    'commit': {'committed': 4, 'total': 6},
                },
                {
                    'condition': 'uniform_review_gated',
                    'agent_count': 6,
                    'write_intent_count': 6,
                    'stress_metadata': {'writers': 4, 'atom_count': 16, 'invalidated_dependency_count': 2},
                    'review_burden_count': 6,
                    'stale_dependency_accepted_count': 0,
                    'unsafe_auto_commit_count': 0,
                    'commit': {'committed': 0, 'total': 6},
                },
            ],
        }
        rows = build_rows([(payload, None)])
        wccu = [r for r in rows if r['condition'] == 'adaptive_wccu_execution_trace'][0]
        review = [r for r in rows if r['condition'] == 'uniform_review_gated'][0]
        self.assertEqual(wccu['latent_dependencies'], 2)
        self.assertAlmostEqual(wccu['coordination_selectivity'], 1.0)
        self.assertAlmostEqual(wccu['coordination_precision'], 1.0)
        self.assertAlmostEqual(wccu['coordination_recall'], 1.0)
        self.assertAlmostEqual(wccu['coordination_f1'], 1.0)
        self.assertAlmostEqual(wccu['safe_automatic_progress'], 1.0)
        self.assertEqual(review['latent_dependencies'], 2)
        self.assertAlmostEqual(review['coordination_selectivity'], round(2/6, 4))
        self.assertEqual(review['unnecessary_coordination'], 4)
        self.assertAlmostEqual(review['coordination_precision'], round(2/6, 4))
        self.assertAlmostEqual(review['coordination_recall'], 1.0)
        self.assertAlmostEqual(review['over_coordination_rate'], 1.0)

    def test_commitment_metrics_infer_one_latent_dependency(self):
        payload = {
            'results': [
                {
                    'kind': 'parallel_execution_result_v2',
                    'condition': 'adaptive_policy',
                    'scenario_id': 'cooperbench_commitment_x_commitment_stale',
                    'agent_count': 2,
                    'write_intent_count': 2,
                    'review_burden_count': 0,
                    'stale_dependency_accepted_count': 1,
                    'unsafe_auto_commit_count': 1,
                    'commit': {'committed': 2, 'total': 2},
                }
            ]
        }
        rows = build_rows([(payload, None)])
        self.assertEqual(rows[0]['family'], 'commitment_staleness')
        self.assertEqual(rows[0]['latent_dependencies'], 1)
        self.assertEqual(rows[0]['dependency_density'], 0.5)
        self.assertAlmostEqual(rows[0]['unsafe_issue_accept_rate'], 1.0)
        self.assertAlmostEqual(rows[0]['coordination_f1'], 0.0)
        with tempfile.TemporaryDirectory() as d:
            tex = Path(d) / 'coord.tex'
            write_tex(tex, rows, compact=False)
            text = tex.read_text(encoding='utf-8')
            self.assertIn('Commitment', text)
            self.assertIn('Adaptive, no WCCU', text)
            self.assertIn('1.00', text)

    def test_unsafe_rates_are_update_instance_rates_not_event_counts(self):
        q = row_quality({
            'write_intent_count': 1,
            'ground_truth_issue_count': 1,
            'ground_truth_issue_accepted_count': 1,
            'stale_dependency_accepted_count': 1,
            'unsafe_auto_commit_count': 1,
            'review_burden_count': 0,
            'commit': {'committed': 1, 'total': 1},
        })
        self.assertEqual(q['issue_count'], 1)
        self.assertEqual(q['issue_accepted_count'], 1)
        self.assertAlmostEqual(q['unsafe_issue_accept_rate'], 1.0)

    def test_witness_drop_rates_are_not_silently_aggregated(self):
        payload = {
            'kind': 'wccu_witness_completeness_results_v1',
            'results': [
                {
                    'condition': 'adaptive_wccu_execution_trace',
                    'scenario_id': 'wccu_stress_0',
                    'witness_drop_rate': 0.0,
                    'write_intent_count': 2,
                    'stale_dependency_count': 1,
                    'review_burden_count': 1,
                    'stale_dependency_accepted_count': 0,
                    'unsafe_auto_commit_count': 0,
                    'commit': {'committed': 1, 'total': 2},
                },
                {
                    'condition': 'adaptive_wccu_execution_trace',
                    'scenario_id': 'wccu_stress_0',
                    'witness_drop_rate': 1.0,
                    'write_intent_count': 2,
                    'stale_dependency_count': 0,
                    'review_burden_count': 0,
                    'stale_dependency_accepted_count': 1,
                    'unsafe_auto_commit_count': 1,
                    'commit': {'committed': 2, 'total': 2},
                },
            ],
        }
        rows = build_rows([(payload, None)])
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['setting'] for r in rows}, {'drop=0.00', 'drop=1.00'})
        clean = [r for r in rows if r['witness_drop_rate'] == 0.0][0]
        dropped = [r for r in rows if r['witness_drop_rate'] == 1.0][0]
        self.assertAlmostEqual(clean['unsafe_issue_accept_rate'], 0.0)
        self.assertAlmostEqual(dropped['unsafe_issue_accept_rate'], 1.0)

    def test_filter_complete_witness_rows(self):
        payload = {
            'kind': 'wccu_witness_completeness_results_v1',
            'results': [
                {'condition': 'adaptive_wccu_execution_trace', 'scenario_id': 's0', 'witness_drop_rate': 0.0, 'write_intent_count': 1, 'commit': {'total': 1}},
                {'condition': 'adaptive_wccu_execution_trace', 'scenario_id': 's1', 'witness_drop_rate': 0.5, 'write_intent_count': 1, 'commit': {'total': 1}},
            ],
        }
        rows = build_rows([(payload, None)], only_families={'witness_completeness'}, only_witness_drop=0.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['setting'], 'drop=0.00')


if __name__ == '__main__':
    unittest.main()
