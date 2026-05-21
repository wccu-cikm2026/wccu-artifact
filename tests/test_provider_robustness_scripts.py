from __future__ import annotations

import unittest
from pathlib import Path


class ProviderRobustnessScriptTests(unittest.TestCase):
    def test_gemini_only_script_uses_gemini_31_flash_lite_default(self):
        text = Path('scripts/run_wccu_gemini_only_frozen_replay.sh').read_text(encoding='utf-8')
        self.assertIn('LLM_PROVIDER="gemini"', text)
        self.assertIn('gemini-3.1-flash-lite', text)
        self.assertIn('GEMINI_RESPONSE_SCHEMA_MODE', text)
        self.assertIn('run_wccu_true_frozen_replay.sh', text)

    def test_provider_suite_runs_gemini_and_mixed_then_summarizes(self):
        text = Path('scripts/run_wccu_provider_robustness_suite.sh').read_text(encoding='utf-8')
        self.assertIn('run_wccu_gemini_only_frozen_replay.sh', text)
        self.assertIn('run_wccu_mixed_provider_frozen_replay.sh', text)
        self.assertIn('make_provider_robustness_summary', text)


if __name__ == '__main__':
    unittest.main()
