from __future__ import annotations

import unittest
from pathlib import Path


class TrueFrozenReplayScriptCliTests(unittest.TestCase):
    def test_true_frozen_replay_script_uses_current_dataset_report_cli(self):
        text = Path('scripts/run_wccu_true_frozen_replay.sh').read_text(encoding='utf-8')
        self.assertIn('--commitment-diag "${COMMITMENT}"', text)
        self.assertIn('--out-json "analysis/${EXP_TAG}/cooperbench_dataset_report.json"', text)
        self.assertIn('--out-csv "analysis/${EXP_TAG}/cooperbench_dataset_report.csv"', text)
        self.assertIn('--out-md "analysis/${EXP_TAG}/cooperbench_dataset_report.md"', text)
        dataset_report_call = text.split('make_cooperbench_dataset_report', 1)[1].split('# -------------------------', 1)[0]
        self.assertNotIn('--out-prefix', dataset_report_call)
        self.assertNotIn('--commitment "${COMMITMENT}"', dataset_report_call)

    def test_true_frozen_replay_script_uses_current_converter_and_sampler_cli(self):
        text = Path('scripts/run_wccu_true_frozen_replay.sh').read_text(encoding='utf-8')
        self.assertIn('--max-tasks "${MAX_TASKS}"', text)
        self.assertIn('--size "${SUBSET_SIZE}"', text)
        self.assertNotIn('--limit "${MAX_TASKS}"', text)

    def test_true_frozen_replay_script_passes_agent_model_specs_to_generation(self):
        text = Path('scripts/run_wccu_true_frozen_replay.sh').read_text(encoding='utf-8')
        self.assertIn('AGENT_MODEL_SPECS=', text)
        self.assertIn('--agent-model-specs "${AGENT_MODEL_SPECS}"', text)

    def test_mixed_provider_script_routes_openai_and_gemini_agents(self):
        text = Path('scripts/run_wccu_mixed_provider_frozen_replay.sh').read_text(encoding='utf-8')
        self.assertIn('coop_agent_a=openai:${OPENAI_AGENT_MODEL}', text)
        self.assertIn('coop_agent_b=gemini:${GEMINI_AGENT_MODEL}', text)
        self.assertIn('gemini-3.1-flash-lite', text)


if __name__ == '__main__':
    unittest.main()
