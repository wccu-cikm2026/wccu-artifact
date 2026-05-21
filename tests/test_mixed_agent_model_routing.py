from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from wccu_eval.agents import llm_agent
from wccu_eval.agents.llm_agent import agent_model_override, parse_agent_model_specs, run_llm_agent


class MixedAgentModelRoutingTests(unittest.TestCase):
    def test_parse_agent_model_specs(self):
        specs = parse_agent_model_specs('coop_agent_a=openai:gpt-5.4-nano,coop_agent_b=gemini:gemini-3.1-flash-lite')
        self.assertEqual(specs['coop_agent_a'], {'provider': 'openai', 'model': 'gpt-5.4-nano'})
        self.assertEqual(specs['coop_agent_b'], {'provider': 'gemini', 'model': 'gemini-3.1-flash-lite'})

    def test_agent_model_override_prefers_agent_id(self):
        cfg = {'agent_model_specs': 'builder=gemini:role-model,coop_agent_a=openai:agent-model'}
        override = agent_model_override({'id': 'coop_agent_a', 'role': 'builder'}, cfg)
        self.assertEqual(override, {'provider': 'openai', 'model': 'agent-model'})

    def test_run_llm_agent_uses_per_agent_provider_model(self):
        calls = []

        def fake_call_llm_provider(**kwargs):
            calls.append({'provider': kwargs.get('provider'), 'model': kwargs.get('model')})
            return {
                'text': json.dumps({
                    'output': 'ok',
                    'write_intents': [{
                        'intent_type': 'patch_atom',
                        'risk': 'low',
                        'authority': 'agent',
                        'payload': {
                            'id': 'atom_test',
                            'atom_id': '',
                            'stream_id': '',
                            'atom_type': 'memory',
                            'title': 'Test',
                            'canonical_text_en': 'Test update.',
                            'text_original': '',
                            'reason': '',
                            'file_path': '',
                            'tags': [],
                            'risk': '',
                        },
                    }],
                }),
                'raw': {},
                'endpoint': 'fake',
                'request_options': {},
                'usage': {},
                'http': {},
            }

        projection = {
            'projection_id': 'proj1',
            'snapshot_id': 'snap1',
            'metrics': {'context_tokens': 1},
            'atoms': [],
        }
        scenario = {'id': 'cooperbench_test', 'goal': 'test', 'agents': []}
        cfg = {
            'provider': 'openai',
            'model': 'default-model',
            'agent_model_specs': 'coop_agent_a=openai:gpt-5.4-nano,coop_agent_b=gemini:gemini-3.1-flash-lite',
            'max_parse_retries': 0,
        }
        with patch.object(llm_agent, 'call_llm_provider', side_effect=fake_call_llm_provider):
            a = run_llm_agent(agent={'id': 'coop_agent_a', 'role': 'builder'}, projection=projection, scenario=scenario, llm_config=cfg)
            b = run_llm_agent(agent={'id': 'coop_agent_b', 'role': 'builder'}, projection=projection, scenario=scenario, llm_config=cfg)
        self.assertEqual(calls, [
            {'provider': 'openai', 'model': 'gpt-5.4-nano'},
            {'provider': 'gemini', 'model': 'gemini-3.1-flash-lite'},
        ])
        self.assertEqual(a['llm']['provider'], 'openai')
        self.assertEqual(b['llm']['provider'], 'gemini')
        self.assertTrue(a['llm']['request_options']['mixed_provider_routing'])
        self.assertTrue(b['llm']['request_options']['mixed_provider_routing'])


if __name__ == '__main__':
    unittest.main()
