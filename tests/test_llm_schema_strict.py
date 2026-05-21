import unittest

from wccu_eval.agents.llm_output_schema import (
    LLM_WRITE_INTENT_JSON_SCHEMA,
    assert_openai_strict_schema,
    normalize_llm_output,
)


class LlmStrictSchemaTests(unittest.TestCase):
    def test_schema_is_openai_strict_recursively(self):
        assert_openai_strict_schema(LLM_WRITE_INTENT_JSON_SCHEMA)

    def test_payload_disallows_additional_properties_for_openai(self):
        payload_schema = LLM_WRITE_INTENT_JSON_SCHEMA['properties']['write_intents']['items']['properties']['payload']
        self.assertIs(payload_schema['additionalProperties'], False)
        self.assertIn('id', payload_schema['required'])
        self.assertIn('tags', payload_schema['required'])

    def test_schema_v2_output_with_empty_optional_fields_normalizes(self):
        raw = {
            'output': 'ok',
            'write_intents': [
                {
                    'intent_type': 'patch_atom',
                    'risk': 'high',
                    'authority': 'agent',
                    'commit_mode': 'none',
                    'payload': {
                        'id': 'atom_permission_policy',
                        'atom_id': '',
                        'stream_id': '',
                        'atom_type': 'permission_policy',
                        'title': 'Permission policy',
                        'canonical_text_en': 'Deployment requires approval.',
                        'text_original': '',
                        'reason': '',
                        'file_path': '',
                        'tags': [],
                        'risk': '',
                    },
                }
            ],
        }
        normalized = normalize_llm_output(raw, agent_id='a', projection_id='p', snapshot_id='s')
        self.assertEqual(normalized['write_intents'][0]['risk'], 'high')
        self.assertNotIn('commit_mode', normalized['write_intents'][0])


    def test_gemini_compatible_schema_strips_openai_strict_keywords(self):
        from wccu_eval.agents.llm_output_schema import gemini_compatible_schema

        schema = gemini_compatible_schema(LLM_WRITE_INTENT_JSON_SCHEMA)

        def walk(node):
            if isinstance(node, dict):
                self.assertNotIn('additionalProperties', node)
                self.assertNotIn('$defs', node)
                self.assertNotIn('definitions', node)
                self.assertNotIn('default', node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(schema)
        self.assertEqual(schema['type'], 'object')
        self.assertIn('write_intents', schema['properties'])
        self.assertIn('required', schema)


if __name__ == '__main__':
    unittest.main()
