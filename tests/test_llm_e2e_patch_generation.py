from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from wccu_eval.e2e.llm_patch_generator import _extract_file_edits, _materialize_edits_to_patch, _extract_patch_text, generate_llm_patch_tasks
from wccu_eval.eval.run_e2e_patch_test import main as run_e2e_main
from scripts.make_e2e_synthetic_tasks import generate_tasks


class LlmE2ePatchGenerationTests(unittest.TestCase):
    def test_extract_patch_text_from_json(self) -> None:
        text = json.dumps({'patch_text': 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n', 'target_files': ['a.py']})
        patch, meta = _extract_patch_text(text)
        self.assertIn('diff --git', patch)
        self.assertEqual(meta['target_files'], ['a.py'])


    def test_file_edit_json_materializes_valid_patch(self) -> None:
        task = {'base_files': {'pkg/a.py': 'x = 1\n'}}
        text = json.dumps({'edits': [{'path': 'pkg/a.py', 'content': 'x = 2\n'}, {'path': 'tests/test_a.py', 'content': 'def test_x():\n    assert 1\n'}]})
        edits, meta = _extract_file_edits(text)
        patch, materialization = _materialize_edits_to_patch(task, edits)
        self.assertEqual(len(edits), 2)
        self.assertIn('diff --git a/pkg/a.py b/pkg/a.py', patch)
        self.assertIn('new file mode 100644', patch)
        self.assertTrue(materialization['materialized_nonempty'])

    def test_generate_mock_from_prepared_and_run_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            patch_dir = root / 'patches'
            tasks = generate_tasks(1, patch_dir, seed=7)
            input_path = root / 'tasks.jsonl'
            with input_path.open('w', encoding='utf-8') as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')
            generated = root / 'generated.jsonl'
            summary = generate_llm_patch_tasks(input_path=input_path, out_path=generated, mock_from_prepared=True, validate_patch=True)
            self.assertEqual(summary['tasks'], 1)
            self.assertEqual(summary['patches'], 2)
            self.assertEqual(summary['parse_success'], 2)
            self.assertEqual(summary['validation_success'], 2)
            out = root / 'e2e.json'
            rc = run_e2e_main(['--input', str(generated), '--conditions', 'adaptive_wccu_execution_trace', '--out', str(out), '--work-dir', str(root / 'runs'), '--timeout-s', '20'])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text())
            self.assertEqual(len(payload['results']), 1)
            self.assertTrue(payload['results'][0]['tests_passed'])


if __name__ == '__main__':
    unittest.main()
