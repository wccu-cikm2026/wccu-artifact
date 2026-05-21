import tempfile
import unittest
from pathlib import Path

from wccu_eval.agents.llm_agent import run_llm_agent
from wccu_eval.agents.llm_output_schema import normalize_llm_output, parse_json_object_from_text, validate_llm_output_shape
from wccu_eval.eval.run_llm_experiment import run_llm_experiment
from wccu_eval.eval.scenarios import get_scenario
from wccu_eval.substrate.context_projection_compiler import compile_projection
from wccu_eval.substrate.context_substrate_store import seed_context


class LlmMockTests(unittest.TestCase):
    def test_schema_normalization(self):
        raw = {'output': 'ok', 'write_intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_x', 'atom_type': 'memory', 'title': 'X'}}]}
        norm = normalize_llm_output(raw, agent_id='a', projection_id='p', snapshot_id='s')
        self.assertTrue(validate_llm_output_shape(norm)['ok'])
        self.assertEqual(norm['write_intents'][0]['preconditions']['base_snapshot_id'], 's')

    def test_parse_fenced_json(self):
        parsed = parse_json_object_from_text('```json\n{"output":"ok","write_intents":[{"intent_type":"append_event","payload":{"id":"s","atom_type":"event","title":"T"}}]}\n```')
        self.assertEqual(parsed['output'], 'ok')

    def test_mock_llm_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenario = get_scenario('high_risk_rule_change')
            root = Path(tmp) / 'context'
            seed_context(root, scenario['seed'])
            projection = compile_projection(root, role='builder', task_type=scenario['task_type'], goal=scenario['goal'])
            result = run_llm_agent(agent=scenario['agents'][0], projection=projection, scenario=scenario, llm_config={'provider': 'mock', 'model': 'fixture'})
            self.assertEqual(result['llm']['endpoint'], 'mock')
            self.assertGreaterEqual(len(result['write_intents']), 1)

    def test_mock_llm_experiment(self):
        payload = run_llm_experiment(scenario='high_risk_rule_change', condition='adaptive_policy,uniform_snapshot_occ', repetitions=1, provider='mock', model='fixture', out='results/test_py_llm_mock.json')
        self.assertEqual(len(payload['results']), 2)

    def test_certificate_guidance_is_not_forwarded_to_provider(self):
        # Regression test for runs where certificate_guidance is used to change
        # prompt construction but must not be forwarded to call_llm_provider().
        payload = run_llm_experiment(
            scenario='wccu_stale_dependency_lowrisk_cross_target',
            condition='adaptive_wccu_unguided_certificate,adaptive_wccu_execution_trace',
            repetitions=1,
            provider='mock',
            model='fixture',
            out='results/test_py_llm_certificate_guidance.json',
            certificate_guidance='guided',
        )
        self.assertEqual(len(payload['results']), 2)
        self.assertFalse(any(r.get('failed') for r in payload['results']))

    def test_witness_compiler_options_are_not_forwarded_to_provider(self):
        # The obligation benchmark passes witness-compiler settings through
        # llm_config so run_llm_agent can build instrumented projections.  They
        # are harness options, not provider API parameters, and must not be
        # forwarded to call_llm_provider().
        with tempfile.TemporaryDirectory() as tmp:
            scenario = get_scenario('wccu_stale_dependency_lowrisk_cross_target')
            root = Path(tmp) / 'context'
            seed_context(root, scenario['seed'])
            projection = compile_projection(root, role='writer', task_type=scenario['task_type'], goal=scenario['goal'])
            result = run_llm_agent(
                agent={'id': 'writer', 'role': 'writer'},
                projection=projection,
                scenario=scenario,
                llm_config={
                    'provider': 'mock',
                    'model': 'fixture',
                    'witness_compiler_enabled': True,
                    'witness_attach_to_all_intents': True,
                    'witness_source_label': 'test_runtime_witness',
                },
            )
            self.assertEqual(result['llm']['endpoint'], 'mock')
            self.assertGreaterEqual(len(result['write_intents']), 1)



if __name__ == '__main__':
    unittest.main()
