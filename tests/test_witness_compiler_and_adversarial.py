from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wccu_eval.eval.run_adversarial_wccu import run_adversarial_wccu
from wccu_eval.eval.run_witness_completeness import run_witness_completeness
from wccu_eval.scheduler.dependency_witness import compile_dependency_witness
from wccu_eval.utils import read_json


class WitnessCompilerAndAdversarialTests(unittest.TestCase):
    def test_dependency_witness_compiler_collects_and_drops_reads(self) -> None:
        projection = {
            'projection_id': 'proj1',
            'snapshot_id': 'snap1',
            'atoms': [{'id': 'atom_dep', 'status': 'active', 'canonical_text_en': 'dep'}],
        }
        scenario = {'wccu_read_dependencies': {'agent1': [{'target_id': 'atom_dep'}]}, 'agent_outputs': {}}
        result = {'agent_id': 'agent1', 'role': 'agent', 'write_intents': []}
        full = compile_dependency_witness(agent={'id': 'agent1'}, projection=projection, result=result, scenario=scenario, config={'witness_drop_rate': 0.0})
        self.assertEqual(full['read_atoms'], ['atom_dep'])
        dropped = compile_dependency_witness(agent={'id': 'agent1'}, projection=projection, result=result, scenario=scenario, config={'witness_drop_rate': 1.0, 'witness_seed': 'fixed'})
        self.assertEqual(dropped['read_atoms'], [])
        self.assertEqual(dropped['dropped_count'], 1)

    def test_witness_completeness_drop_rate_changes_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out0 = str(Path(tmp) / 'wc.json')
            payload = run_witness_completeness(cases=2, writers=2, atom_count=8, invalidation_prob=0.5, seed=7, conditions='adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy', drop_rates='0,1', repetitions=1, out=out0)
            rows = payload['results']
            wccu0 = [r for r in rows if r['condition'] == 'adaptive_wccu_execution_trace' and r['witness_drop_rate'] == 0.0]
            wccu1 = [r for r in rows if r['condition'] == 'adaptive_wccu_execution_trace' and r['witness_drop_rate'] == 1.0]
            self.assertGreaterEqual(sum(r['stale_dependency_accepted_count'] for r in wccu1), sum(r['stale_dependency_accepted_count'] for r in wccu0))

    def test_adversarial_missing_dependency_distinguishes_model_and_witness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'adv.json')
            payload = run_adversarial_wccu(kinds='missing_dependency', conditions='adaptive_wccu_execution_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy', repetitions=1, out=out)
            by_cond = {r['condition']: r for r in payload['results']}
            self.assertEqual(by_cond['adaptive_wccu_execution_trace']['stale_dependency_accepted_count'], 0)
            self.assertEqual(by_cond['adaptive_readset_occ']['stale_dependency_accepted_count'], 0)
            self.assertGreaterEqual(by_cond['adaptive_wccu_model_certificate']['stale_dependency_accepted_count'], 1)
            self.assertGreaterEqual(by_cond['adaptive_policy']['stale_dependency_accepted_count'], 1)


if __name__ == '__main__':
    unittest.main()

class ObligationMatrixTests(unittest.TestCase):
    def test_obligation_matrix_separates_wccu_from_readset_occ(self) -> None:
        from wccu_eval.eval.run_obligation_matrix import run_obligation_matrix
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'obligation_matrix.json')
            payload = run_obligation_matrix(
                kinds='target,authority,delta,view',
                conditions='adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy',
                repetitions=1,
                out=out,
            )
            rows = {(r['obligation_kind'], r['condition']): r for r in payload['results']}
            for kind in ['target', 'authority', 'delta', 'view']:
                self.assertTrue(rows[(kind, 'adaptive_wccu_execution_trace')]['expected_event_observed'])
            self.assertGreaterEqual(rows[('target', 'adaptive_wccu_execution_trace')]['review_burden_count'], 1)
            self.assertEqual(rows[('target', 'adaptive_readset_occ')]['review_burden_count'], 0)
            self.assertGreaterEqual(rows[('authority', 'adaptive_wccu_execution_trace')]['review_burden_count'], 1)
            self.assertEqual(rows[('authority', 'adaptive_readset_occ')]['review_burden_count'], 0)
            self.assertGreaterEqual(rows[('delta', 'adaptive_wccu_execution_trace')]['review_burden_count'], 1)
            self.assertEqual(rows[('delta', 'adaptive_readset_occ')]['review_burden_count'], 0)
            # Derived-view obligations do not necessarily block a write; they
            # must be recorded for invalidation, which read-set OCC does not do.
            self.assertGreaterEqual(rows[('view', 'adaptive_wccu_execution_trace')]['view_invalidation_count'], 1)
            self.assertEqual(rows[('view', 'adaptive_readset_occ')]['view_invalidation_count'], 0)
            self.assertEqual(rows[('target', 'adaptive_wccu_execution_trace')]['ground_truth_problematic_held_count'], 1)
            self.assertEqual(rows[('target', 'adaptive_readset_occ')]['ground_truth_issue_accepted_count'], 1)
            self.assertEqual(rows[('view', 'adaptive_wccu_execution_trace')]['ground_truth_issue_accepted_count'], 0)
            self.assertEqual(rows[('view', 'adaptive_readset_occ')]['ground_truth_issue_accepted_count'], 1)

class AdversarialTableTests(unittest.TestCase):
    def test_adversarial_table_uses_wccu_labels_and_event_columns(self) -> None:
        from wccu_eval.scripts.make_adversarial_wccu_table import summarize, write_tex
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'adv.json')
            payload = run_adversarial_wccu(
                kinds='wrong_target,fake_authority,misleading_delta',
                conditions='adaptive_wccu_execution_trace,adaptive_policy',
                repetitions=1,
                out=out,
            )
            rows = summarize([out])
            labels = {r['condition_label'] for r in rows}
            self.assertIn('WCCU, execution witness', labels)
            self.assertIn('Adaptive, no WCCU', labels)
            by_key = {(r['adversarial_kind'], r['condition']): r for r in rows}
            self.assertGreaterEqual(by_key[('wrong_target', 'adaptive_wccu_execution_trace')]['wrong_target_events'], 1)
            self.assertGreaterEqual(by_key[('fake_authority', 'adaptive_wccu_execution_trace')]['authority_events'], 1)
            self.assertGreaterEqual(by_key[('misleading_delta', 'adaptive_wccu_execution_trace')]['operation_events'], 1)
            tex = Path(tmp) / 'adv.tex'
            write_tex(str(tex), rows)
            text = tex.read_text(encoding='utf-8')
            self.assertIn('WCCU, execution witness', text)
            self.assertIn('Target ev.', text)
            self.assertIn('Auth ev.', text)
            self.assertIn('Op. ev.', text)
