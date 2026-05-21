from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wccu_eval.scripts.make_latency_parallelism_table import build_rows, write_tex


class LatencyParallelismTableTest(unittest.TestCase):
    def test_build_rows_parallel_metrics(self) -> None:
        payload = {
            'results': [
                {
                    'condition': 'adaptive_wccu_execution_trace',
                    'elapsed_ms': 1200,
                    'unsafe_auto_commit_count': 0,
                    'stale_dependency_accepted_count': 0,
                    'review_burden_count': 1,
                    'agentRuns': [
                        {'llm': {'elapsed_ms': 1000}},
                        {'llm': {'elapsed_ms': 900}},
                    ],
                },
                {
                    'condition': 'serial_adaptive_wccu_execution_trace',
                    'elapsed_ms': 2100,
                    'unsafe_auto_commit_count': 0,
                    'stale_dependency_accepted_count': 0,
                    'review_burden_count': 1,
                    'agentRuns': [
                        {'llm': {'elapsed_ms': 1000}},
                        {'llm': {'elapsed_ms': 900}},
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'r.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            rows = build_rows([str(path)])
            by_cond = {r['condition']: r for r in rows}
            self.assertEqual(by_cond['adaptive_wccu_execution_trace']['freshness'], '1/1')
            self.assertGreater(by_cond['adaptive_wccu_execution_trace']['mean_speedup_est'], 1.0)
            out_tex = Path(td) / 'lat.tex'
            write_tex(out_tex, rows)
            self.assertIn('Agent-sum s', out_tex.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
