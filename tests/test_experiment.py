import unittest

from wccu_eval.eval.run_experiment import run_experiment
from wccu_eval.eval.run_abstract_ablation import run_abstract_ablation


class ExperimentTests(unittest.TestCase):
    def test_small_policy_experiment(self):
        payload = run_experiment(scenario='high_risk_rule_change', condition='adaptive_policy,adaptive_no_review_gate', repetitions=1, out='results/test_py_high_risk.json')
        self.assertEqual(len(payload['results']), 2)
        adaptive = next(r for r in payload['aggregated'] if r['condition'] == 'adaptive_policy')
        no_review = next(r for r in payload['aggregated'] if r['condition'] == 'adaptive_no_review_gate')
        self.assertEqual(adaptive['mean_unsafe_auto_commit_count'], 0)
        self.assertGreater(no_review['mean_unsafe_auto_commit_count'], 0)

    def test_abstract_ablation_claims_pass(self):
        payload = run_abstract_ablation(repetitions=1, out='results/test_py_abstract_ablation.json')
        self.assertTrue(payload['claim_summary']['all_passed'])
        self.assertEqual(payload['claim_summary']['passed'], payload['claim_summary']['total'])


if __name__ == '__main__':
    unittest.main()
