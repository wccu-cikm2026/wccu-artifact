from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from wccu_eval.e2e.patch_test_runner import run_e2e_patch_tests
from wccu_eval.scripts.make_e2e_patch_test_table import summarize as summarize_e2e
from wccu_eval.scripts.make_witness_completeness_table import summarize as summarize_witness


class WitnessMetricAndE2ETests(unittest.TestCase):
    def test_false_negative_rate_uses_detected_plus_accepted_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'wc.json'
            payload = {
                'results': [
                    {'condition': 'adaptive_wccu_execution_trace', 'witness_drop_rate': 0.5, 'failed': False, 'stale_dependency_count': 3, 'stale_dependency_accepted_count': 1, 'unsafe_auto_commit_count': 1},
                    {'condition': 'adaptive_wccu_execution_trace', 'witness_drop_rate': 0.5, 'failed': False, 'stale_dependency_count': 1, 'stale_dependency_accepted_count': 1, 'unsafe_auto_commit_count': 1},
                ]
            }
            p.write_text(json.dumps(payload), encoding='utf-8')
            rows = summarize_witness([str(p)])
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]['false_negative_rate'], 2 / 6)
            self.assertAlmostEqual(rows[0]['witness_recall_est'], 4 / 6)

    def test_e2e_patch_test_applies_auto_committed_patch_and_runs_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / 'tasks.jsonl'
            patch_text = textwrap.dedent('''\
                diff --git a/value.txt b/value.txt
                --- a/value.txt
                +++ b/value.txt
                @@ -1 +1 @@
                -old
                +new
            ''')
            task = {
                'task_id': 'toy_patch',
                'base_files': {'value.txt': 'old\n'},
                'target_id': 'file:value.txt',
                'file_path': 'value.txt',
                'patches': [{'agent_id': 'agent_0', 'patch_text': patch_text}],
                'test_commands': ['python -c "from pathlib import Path; assert Path(\'value.txt\').read_text()==\'new\\n\'"'],
            }
            task_path.write_text(json.dumps(task) + '\n', encoding='utf-8')
            out = root / 'e2e.json'
            payload = run_e2e_patch_tests(input_path=str(task_path), conditions='adaptive_policy', out=str(out), work_dir=str(root / 'runs'), timeout_s=20)
            self.assertEqual(len(payload['results']), 1)
            row = payload['results'][0]
            self.assertTrue(row['tests_passed'])
            self.assertEqual(row['patch_apply_failures'], 0)
            table_rows = summarize_e2e([str(out)])
            self.assertEqual(table_rows[0]['tests_passed'], 1)


if __name__ == '__main__':
    unittest.main()
