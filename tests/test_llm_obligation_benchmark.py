from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wccu_eval.eval.llm_obligation_scenarios import build_llm_obligation_scenario, build_llm_obligation_scenarios
from wccu_eval.eval.run_llm_obligation_benchmark import run_llm_obligation_benchmark
from wccu_eval.scripts.make_llm_obligation_tables import build_decision_rows, build_generation_rows


class LlmObligationBenchmarkTests(unittest.TestCase):
    def test_scenario_generator_covers_requested_families(self) -> None:
        scenarios = build_llm_obligation_scenarios(families=['freshness', 'derived_view'], limit_per_family=2)
        self.assertEqual(len(scenarios), 4)
        self.assertEqual({s['llm_obligation_family'] for s in scenarios}, {'freshness', 'derived_view'})
        self.assertTrue(all(s.get('agents') for s in scenarios))

    def test_authority_is_isolated_from_stale_reads(self) -> None:
        scenario = build_llm_obligation_scenario('authority', 0)
        self.assertFalse(scenario.get('concurrent_agent_results'))
        agent = scenario['agents'][0]
        self.assertEqual(len(agent['read_atoms']), 1)
        self.assertIn('autonomy_note', agent['read_atoms'][0])

    def test_derived_view_reads_view_not_hidden_source(self) -> None:
        scenario = build_llm_obligation_scenario('derived_view', 0)
        agent_reads = scenario['agents'][0]['read_atoms']
        self.assertEqual(len(agent_reads), 1)
        self.assertIn('atom_handoff_', agent_reads[0])
        source = next(a for a in scenario['seed']['atoms'] if a['id'].startswith('atom_source_pref_'))
        self.assertEqual(source.get('role_allowlist'), ['runtime'])
        self.assertEqual(scenario['seed']['links'][0]['type'], 'derived_from')

    def test_witness_gap_hides_runtime_dependency_from_projection(self) -> None:
        scenario = build_llm_obligation_scenario('witness_gap', 0)
        agent_reads = scenario['agents'][0]['read_atoms']
        self.assertEqual(len(agent_reads), 1)
        self.assertIn('runtime_tool_permission', agent_reads[0])
        hidden = next(a for a in scenario['seed']['atoms'] if a['id'] == agent_reads[0])
        self.assertEqual(hidden.get('role_allowlist'), ['runtime'])

    def test_mock_llm_obligation_benchmark_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / 'llm_obligation.json')
            payload = run_llm_obligation_benchmark(
                families='freshness,safe',
                limit_per_family=1,
                condition='adaptive_wccu_execution_trace,uniform_append_only,uniform_review_gated',
                repetitions=1,
                out=out,
                mock_llm=True,
            )
            self.assertEqual(len(payload['generations']), 2)
            self.assertEqual(len(payload['results']), 6)
            generation_rows = build_generation_rows(payload)
            self.assertEqual(len(generation_rows), 2)
            decision_rows = build_decision_rows(payload, by_family=False)
            self.assertEqual(len(decision_rows), 3)
            by_cond = {r['condition']: r for r in decision_rows}
            self.assertEqual(by_cond['adaptive_wccu_execution_trace']['unsafe_issue_accept_rate'], 0.0)
            self.assertGreaterEqual(by_cond['uniform_append_only']['unsafe_issue_accept_rate'], 0.0)

    def test_mock_isolated_families_distinguish_wccu_from_readset_and_model_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / 'llm_obligation.json')
            payload = run_llm_obligation_benchmark(
                families='authority,derived_view,witness_gap',
                limit_per_family=1,
                condition='adaptive_wccu_execution_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,uniform_append_only',
                repetitions=1,
                out=out,
                mock_llm=True,
            )
            rows = build_decision_rows(payload, by_family=True)
            by_key = {(r['family'], r['condition']): r for r in rows}
            self.assertEqual(by_key[('authority', 'adaptive_wccu_execution_trace')]['unsafe_issue_accept_rate'], 0.0)
            self.assertGreater(by_key[('authority', 'adaptive_readset_occ')]['unsafe_issue_accept_rate'], 0.0)
            self.assertEqual(by_key[('derived_view', 'adaptive_wccu_execution_trace')]['unsafe_issue_accept_rate'], 0.0)
            self.assertGreater(by_key[('derived_view', 'adaptive_readset_occ')]['unsafe_issue_accept_rate'], 0.0)
            self.assertEqual(by_key[('witness_gap', 'adaptive_wccu_execution_trace')]['unsafe_issue_accept_rate'], 0.0)
            self.assertGreater(by_key[('witness_gap', 'adaptive_wccu_model_certificate')]['unsafe_issue_accept_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()
