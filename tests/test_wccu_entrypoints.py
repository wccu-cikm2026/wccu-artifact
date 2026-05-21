from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wccu_eval.eval.run_multi_model_obligation_benchmark import run_multi_model_obligation_benchmark
from wccu_eval.eval.run_wccu_stress import run_wccu_stress
from wccu_eval.scheduler.context_concurrency_policy import normalize_policy_mode


class WccuEntrypointTests(unittest.TestCase):
    def test_mock_multi_model_does_not_require_env_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_multi_model_obligation_benchmark(
                mock_llm=True,
                families='freshness',
                limit_per_family=1,
                condition='adaptive_wccu_execution_trace',
                out=str(Path(tmp) / 'mock.json'),
            )
        self.assertEqual(payload['args']['model_specs'], [{'provider': 'mock', 'model': 'mock-llm'}])
        self.assertEqual(len(payload['aggregated']), 1)

    def test_wccu_stress_entrypoint_returns_wccu_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            new_payload = run_wccu_stress(
                cases=1,
                writers=1,
                atom_count=3,
                invalidation_prob=1.0,
                seed=7,
                condition='adaptive_wccu_execution_trace',
                repetitions=1,
                out=str(Path(tmp) / 'wccu.json'),
            )
        self.assertEqual(new_payload['kind'], 'wccu_randomized_stress_results_v1')
        self.assertEqual(normalize_policy_mode('adaptive_wccu_execution_trace'), 'adaptive_wccu_execution_trace')


if __name__ == '__main__':
    unittest.main()
