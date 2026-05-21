from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wccu_eval.e2e.llm_patch_generator import generate_llm_patch_tasks
from wccu_eval.scripts.make_llm_patch_generation_table import summarize


class LlmE2eDebugToolsTest(unittest.TestCase):
    def test_task_type_filter_happens_before_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            patch_dir = root / 'patches'
            patch_dir.mkdir()
            # Valid tiny patch files for mock mode.
            (patch_dir / 'a.patch').write_text('diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x=1\n+x=2\n', encoding='utf-8')
            rows = [
                {'task_id': 'ind', 'task_type': 'independent', 'base_files': {'a.py': 'x=1\n'}, 'patches': [{'agent_id': 'a', 'patch_file': 'patches/a.patch'}], 'test_commands': ['python -c "print(1)"']},
                {'task_id': 'com', 'task_type': 'commitment_staleness', 'base_files': {'a.py': 'x=1\n'}, 'patches': [{'agent_id': 'a', 'patch_file': 'patches/a.patch'}], 'test_commands': ['python -c "print(1)"']},
            ]
            inp = root / 'tasks.jsonl'
            inp.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
            out = root / 'generated.jsonl'
            payload = generate_llm_patch_tasks(input_path=inp, out_path=out, limit=1, mock_from_prepared=True, task_types=['commitment_staleness'], validate_patch=False)
            self.assertEqual(payload['tasks'], 1)
            generated = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            self.assertEqual(generated[0]['task_type'], 'commitment_staleness')

    def test_generation_summary_by_task_type(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'gen.jsonl'
            rows = [
                {'task_type': 'independent', 'patch_parse_success': True, 'patch_validation_ok': True, 'llm_generation': {'prompt_tokens_est': 10, 'output_tokens_est': 5}},
                {'task_type': 'commitment_staleness', 'patch_parse_success': True, 'patch_validation_ok': False, 'llm_generation': {'prompt_tokens_est': 20, 'output_tokens_est': 7}},
            ]
            p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
            out = summarize([str(p)], group_by_task_type=True)
            by_type = {r['task_type']: r for r in out}
            self.assertEqual(by_type['independent']['validation_success'], 1)
            self.assertEqual(by_type['commitment_staleness']['validation_success'], 0)


if __name__ == '__main__':
    unittest.main()
