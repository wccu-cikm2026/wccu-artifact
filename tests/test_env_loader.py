import os
import tempfile
import unittest
from pathlib import Path

from wccu_eval.env import find_dotenv, load_dotenv, parse_dotenv_text
from wccu_eval.eval.run_llm_experiment import run_llm_experiment


class EnvLoaderTests(unittest.TestCase):
    def setUp(self):
        self._old = {k: os.environ.get(k) for k in ['PCSE_TEST_VALUE', 'PCSE_TEST_QUOTED', 'PCSE_TEST_KEEP', 'WCCU_ENV_FILE', 'LLM_PROVIDER', 'LLM_MODEL']}
        for key in self._old:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_parse_dotenv_text(self):
        parsed = parse_dotenv_text('''\n# comment\nexport PCSE_TEST_VALUE=hello # inline\nPCSE_TEST_QUOTED="hello world"\nPCSE_TEST_SINGLE='single quoted'\n''')
        self.assertEqual(parsed['PCSE_TEST_VALUE'], 'hello')
        self.assertEqual(parsed['PCSE_TEST_QUOTED'], 'hello world')
        self.assertEqual(parsed['PCSE_TEST_SINGLE'], 'single quoted')

    def test_load_dotenv_without_overriding_existing_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / '.env'
            env_path.write_text('PCSE_TEST_KEEP=file\nPCSE_TEST_VALUE=loaded\n', encoding='utf-8')
            os.environ['PCSE_TEST_KEEP'] = 'existing'
            loaded = load_dotenv(env_path)
            self.assertEqual(os.environ['PCSE_TEST_VALUE'], 'loaded')
            self.assertEqual(os.environ['PCSE_TEST_KEEP'], 'existing')
            self.assertIn('PCSE_TEST_VALUE', loaded)
            self.assertNotIn('PCSE_TEST_KEEP', loaded)

    def test_pcse_env_file_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / 'custom.env'
            env_path.write_text('PCSE_TEST_VALUE=from_override\n', encoding='utf-8')
            os.environ['WCCU_ENV_FILE'] = str(env_path)
            self.assertEqual(find_dotenv(), env_path)
            load_dotenv()
            self.assertEqual(os.environ['PCSE_TEST_VALUE'], 'from_override')

    def test_llm_runner_loads_dotenv_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / '.env'
            env_path.write_text('LLM_PROVIDER=mock\nLLM_MODEL=fixture\n', encoding='utf-8')
            os.environ['WCCU_ENV_FILE'] = str(env_path)
            payload = run_llm_experiment(
                scenario='high_risk_rule_change',
                condition='adaptive_policy',
                repetitions=1,
                provider='',
                model='',
                out='results/test_env_loader_llm_mock.json',
            )
            self.assertEqual(payload['args']['provider'], 'mock')
            self.assertEqual(payload['args']['model'], 'fixture')
            self.assertEqual(len(payload['results']), 1)


if __name__ == '__main__':
    unittest.main()
