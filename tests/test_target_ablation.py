from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wccu_eval.agents.llm_agent import build_llm_agent_prompt
from wccu_eval.eval.scenarios import get_scenario, list_scenario_ids
from wccu_eval.eval.run_llm_experiment import run_llm_experiment
from wccu_eval.scripts.analyze_results import summarize, write_distribution_files


class TargetAblationTests(unittest.TestCase):
    def test_ambiguous_target_scenario_registered(self):
        self.assertIn('user_correction_ambiguous_targets', list_scenario_ids())
        sc = get_scenario('user_correction_ambiguous_targets')
        ids = [a['id'] for a in sc['seed']['atoms']]
        self.assertIn('atom_pref_backup_url', ids)
        self.assertIn('atom_pref_callback_url', ids)
        self.assertGreaterEqual(len(ids), 5)

    def test_prompt_can_hide_target_candidates(self):
        sc = get_scenario('user_correction_ambiguous_targets')
        projection = {'prompt': 'Context prompt.', 'projection_id': 'p', 'snapshot_id': 's', 'metrics': {}}
        agent = sc['agents'][0]
        shown = build_llm_agent_prompt(agent=agent, projection=projection, scenario=sc, include_target_candidates=True)
        hidden = build_llm_agent_prompt(agent=agent, projection=projection, scenario=sc, include_target_candidates=False)
        self.assertIn('atom_pref_backup_url', shown)
        self.assertNotIn('atom_pref_backup_url', hidden)

    def test_mock_target_ablation_conditions_run(self):
        payload = run_llm_experiment(
            scenario='user_correction_ambiguous_targets',
            condition='adaptive_no_candidates_no_grounding,adaptive_no_candidates_with_grounding,adaptive_candidates_no_grounding,adaptive_candidates_with_grounding',
            repetitions=1,
            provider='mock',
            model='fixture',
            out='/tmp/pcse_target_ablation_test.json',
            parallel_workers=2,
        )
        self.assertEqual(len(payload['results']), 4)
        self.assertEqual(len(payload['aggregated']), 4)
        summary = summarize(payload)
        self.assertEqual(len(summary), 4)

    def test_analysis_distribution_files(self):
        payload = run_llm_experiment(
            scenario='user_correction_ambiguous_targets',
            condition='adaptive_candidates_with_grounding',
            repetitions=1,
            provider='mock',
            model='fixture',
            out='/tmp/pcse_analysis_dist_test.json',
        )
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            write_distribution_files(payload, out_dir)
            self.assertTrue((out_dir / 'target_distribution.csv').exists())
            self.assertTrue((out_dir / 'merge_decision_distribution.csv').exists())
            self.assertTrue((out_dir / 'llm_usage_summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
