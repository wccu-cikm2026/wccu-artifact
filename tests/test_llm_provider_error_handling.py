from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from wccu_eval.agents.llm_agent import LlmProviderError, _post_json
from wccu_eval.utils import read_jsonl


class _FakeResponse:
    status = 200
    code = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode('utf-8')


class LlmProviderErrorHandlingTests(unittest.TestCase):
    def test_retryable_520_is_retried_and_logged(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / 'errors.jsonl'
            calls = {'n': 0}

            def fake_urlopen(req, timeout=0):
                calls['n'] += 1
                if calls['n'] == 1:
                    raise urllib.error.HTTPError(
                        req.full_url,
                        520,
                        'unknown gateway error',
                        hdrs={'cf-ray': 'test-ray'},
                        fp=io.BytesIO(b'{"error":"temporary 520"}'),
                    )
                return _FakeResponse({'ok': True})

            with patch('urllib.request.urlopen', side_effect=fake_urlopen):
                payload = _post_json(
                    'https://api.example.test/v1/responses',
                    body={'model': 'm', 'input': [{'role': 'user', 'content': 'hi'}]},
                    provider='openai',
                    endpoint='responses',
                    max_provider_retries=1,
                    retry_backoff_base=0,
                    retry_backoff_max=0,
                    error_log_path=str(log_path),
                )
            self.assertEqual(payload['ok'], True)
            self.assertEqual(payload['_pcse_http']['attempts'], 2)
            self.assertEqual(calls['n'], 2)
            rows = read_jsonl(log_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status_code'], 520)
            self.assertTrue(rows[0]['will_retry'])
            self.assertEqual(rows[0]['request_id'], 'test-ray')
            self.assertIn('prompt_hash', rows[0]['request'])

    def test_non_retryable_400_is_not_retried(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / 'errors.jsonl'
            calls = {'n': 0}

            def fake_urlopen(req, timeout=0):
                calls['n'] += 1
                raise urllib.error.HTTPError(
                    req.full_url,
                    400,
                    'bad request',
                    hdrs={},
                    fp=io.BytesIO(b'{"error":"bad schema"}'),
                )

            with patch('urllib.request.urlopen', side_effect=fake_urlopen):
                with self.assertRaises(LlmProviderError) as ctx:
                    _post_json(
                        'https://api.example.test/v1/responses',
                        body={'model': 'm', 'input': [{'role': 'user', 'content': 'hi'}]},
                        provider='openai',
                        endpoint='responses',
                        max_provider_retries=3,
                        retry_backoff_base=0,
                        retry_backoff_max=0,
                        error_log_path=str(log_path),
                    )
            self.assertEqual(calls['n'], 1)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertFalse(ctx.exception.retryable)
            rows = read_jsonl(log_path)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]['will_retry'])


if __name__ == '__main__':
    unittest.main()

class GeminiSchemaCompatibilityTests(unittest.TestCase):
    def test_gemini_provider_sends_sanitized_response_schema_by_default(self):
        from wccu_eval.agents.llm_agent import call_llm_provider

        captured = {}

        def fake_post_json(url, **kwargs):
            captured['url'] = url
            captured['body'] = kwargs['body']
            return {
                'candidates': [
                    {
                        'content': {
                            'parts': [
                                {'text': '{"output":"ok","write_intents":[]}'},
                            ]
                        }
                    }
                ],
                '_pcse_http': {'status_code': 200},
            }

        with patch('wccu_eval.agents.llm_agent._post_json', side_effect=fake_post_json):
            result = call_llm_provider(
                provider='gemini',
                model='gemini-2.5-flash',
                prompt='return json',
                api_key='fake-key',
                strict_schema=True,
            )

        schema = captured['body']['generationConfig']['responseSchema']

        def walk(node):
            if isinstance(node, dict):
                self.assertNotIn('additionalProperties', node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(schema)
        self.assertEqual(result['request_options']['response_schema_mode'], 'gemini_sanitized')

    def test_gemini_provider_can_disable_response_schema(self):
        from wccu_eval.agents.llm_agent import call_llm_provider

        captured = {}

        def fake_post_json(url, **kwargs):
            captured['body'] = kwargs['body']
            return {'candidates': [{'content': {'parts': [{'text': '{"output":"ok","write_intents":[]}'}]}}]}

        with patch.dict('os.environ', {'GEMINI_RESPONSE_SCHEMA_MODE': 'none'}, clear=False):
            with patch('wccu_eval.agents.llm_agent._post_json', side_effect=fake_post_json):
                result = call_llm_provider(
                    provider='gemini',
                    model='gemini-2.5-flash',
                    prompt='return json',
                    api_key='fake-key',
                    strict_schema=True,
                )

        self.assertNotIn('responseSchema', captured['body']['generationConfig'])
        self.assertEqual(result['request_options']['response_schema_mode'], 'none')
