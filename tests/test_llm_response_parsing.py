from __future__ import annotations

import unittest

from wccu_eval.agents.llm_agent import _extract_openai_responses_text
from wccu_eval.agents.llm_output_schema import parse_json_object_from_text


class LlmResponseParsingTest(unittest.TestCase):
    def test_parse_first_json_object_when_duplicate_json_is_returned(self):
        text = '{"output":"ok","write_intents":[]}\n{"output":"duplicate","write_intents":[]}'
        parsed = parse_json_object_from_text(text)
        self.assertEqual(parsed["output"], "ok")

    def test_parse_first_json_object_with_trailing_commentary(self):
        text = '{"output":"ok","write_intents":[]}\nDone.'
        parsed = parse_json_object_from_text(text)
        self.assertEqual(parsed["output"], "ok")

    def test_openai_responses_text_extractor_deduplicates_output_text_node(self):
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"output":"ok","write_intents":[]}',
                        }
                    ],
                }
            ]
        }
        self.assertEqual(_extract_openai_responses_text(payload), '{"output":"ok","write_intents":[]}')

    def test_openai_responses_text_prefers_output_text_convenience_field(self):
        payload = {
            "output_text": '{"output":"preferred","write_intents":[]}',
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"output":"nested","write_intents":[]}',
                        }
                    ],
                }
            ],
        }
        self.assertEqual(_extract_openai_responses_text(payload), '{"output":"preferred","write_intents":[]}')


if __name__ == "__main__":
    unittest.main()
