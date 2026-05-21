from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wccu_eval.external.cooperbench_official_converter import load_records_from_path, normalize_official_task, convert_records
from wccu_eval.scheduler.context_concurrency_policy import attach_policy, infer_intent_metadata
from wccu_eval.scheduler.context_conflict_resolver import resolve_parallel_write_intents


class CooperBenchOfficialConverterTest(unittest.TestCase):
    def test_converter_extracts_features_and_shared_file_from_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = root / 'openai_tiktoken_task' / 'task0'
            task.mkdir(parents=True)
            (task / 'metadata.json').write_text(json.dumps({
                'id': 'task0',
                'repository': 'openai_tiktoken',
                'language': 'Python',
                'features': [
                    {'description': 'Add cache status in src/cache.py'},
                    {'description': 'Add invalidation metadata in src/cache.py'},
                ],
                'description': 'Two feature changes touch the cache module.',
            }), encoding='utf-8')
            records = load_records_from_path(root)
            rows = convert_records(records, require_shared_file=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['task_id'], 'task0')
            self.assertEqual(rows[0]['repo'], 'openai_tiktoken')
            self.assertTrue(rows[0]['shared_targets'])
            self.assertEqual(rows[0]['shared_targets'][0]['file_path'], 'src/cache.py')

    def test_lock_scope_normalizes_file_target_forms(self):
        a = attach_policy({
            'intent_type': 'patch_atom',
            'authority': 'builder',
            'payload': {'target_id': 'file:router/core.py', 'atom_id': 'file:router/core.py', 'id': 'file:router/core.py', 'atom_type': 'workspace_file', 'file_path': 'router/core.py', 'canonical_text_en': 'A'},
        }, mode='adaptive_wccu_execution_trace')
        b = attach_policy({
            'intent_type': 'patch_atom',
            'authority': 'builder',
            'payload': {'target_id': 'file:router/core.py', 'atom_id': 'file:router/core.py', 'id': 'file:router/core.py', 'atom_type': 'event', 'file_path': '', 'canonical_text_en': 'B'},
        }, mode='adaptive_wccu_execution_trace')
        self.assertEqual(infer_intent_metadata(a)['lock_scope'], 'router/core.py')
        self.assertEqual(infer_intent_metadata(b)['lock_scope'], 'router/core.py')
        result = resolve_parallel_write_intents([
            {'agent_id': 'a', 'role': 'builder', 'write_intents': [a]},
            {'agent_id': 'b', 'role': 'builder', 'write_intents': [b]},
        ], policy_mode='adaptive_policy', enable_target_grounding=False)
        self.assertEqual(result['lock_conflict_count'], 1)
        self.assertEqual(result['review_burden_count'], 2)

    def test_feature_pool_without_json_converts_feature_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'openai_tiktoken_task' / 'task0'
            f1 = root / 'feature1'
            f2 = root / 'feature2'
            f1.mkdir(parents=True)
            f2.mkdir(parents=True)
            (f1 / 'feature.md').write_text('# Add binary string helper\nModify src/tiktoken/core.py to add binary_str support.\n', encoding='utf-8')
            (f2 / 'feature.md').write_text('# Add octal string helper\nModify src/tiktoken/core.py to add octal_str support.\n', encoding='utf-8')
            (f1 / 'feature.patch').write_text('--- a/src/tiktoken/core.py\n+++ b/src/tiktoken/core.py\n', encoding='utf-8')
            (f2 / 'feature.patch').write_text('--- a/src/tiktoken/core.py\n+++ b/src/tiktoken/core.py\n', encoding='utf-8')
            records = load_records_from_path(root.parent)
            rows = convert_records(records)
            self.assertEqual(len(rows), 1)
            self.assertIn('binary string helper', rows[0]['agent_a_task'])
            self.assertIn('octal string helper', rows[0]['agent_b_task'])
            self.assertTrue(rows[0]['shared_targets'])
            self.assertEqual(rows[0]['shared_targets'][0]['file_path'], 'src/tiktoken/core.py')

class CooperBenchOfficialPatchRegressionTest(unittest.TestCase):
    def test_file_path_grounding_priority_beats_task_text(self):
        from wccu_eval.scheduler.target_grounder import resolve_intent_target
        scenario = {
            'seed': {'atoms': [
                {'id': 'atom_coop_task_x', 'atom_type': 'task', 'title': 'CooperBench task about tests/test_feature1.py and caching', 'canonical_text_en': 'Long task text mentions tests/test_feature1.py many times.'},
                {'id': 'file:tests/test_feature1.py', 'atom_type': 'workspace_file', 'title': 'tests/test_feature1.py', 'canonical_text_en': 'Shared file target.'},
            ]}
        }
        intent = {
            'intent_type': 'patch_atom',
            'payload': {
                'id': 'atom_coop_task_x',
                'target_id': 'atom_coop_task_x',
                'atom_id': 'atom_coop_task_x',
                'atom_type': 'workspace_file',
                'file_path': 'tests/test_feature1.py',
                'title': 'Patch task tests/test_feature1.py for feature tests',
                'canonical_text_en': 'Modify the shared workspace file.',
            },
        }
        grounded = resolve_intent_target(intent, scenario)
        self.assertTrue(grounded['target_grounding']['resolved'])
        self.assertEqual(grounded['payload']['target_id'], 'file:tests/test_feature1.py')
        self.assertEqual(grounded['target_grounding']['method'], 'file_path_priority')

    def test_feature_pool_converter_compacts_long_feature_text_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo_task' / 'task0'
            f1 = root / 'feature1'
            f2 = root / 'feature2'
            f1.mkdir(parents=True)
            f2.mkdir(parents=True)
            long_code = '```python\n' + ('print("x")\n' * 500) + '```\n'
            (f1 / 'feature.md').write_text('# Add feature one\n**Description**: Touch src/a.py.\n' + long_code + '**Files Modified**\n- `src/a.py`\n', encoding='utf-8')
            (f2 / 'feature.md').write_text('# Add feature two\n**Description**: Also touch src/a.py.\n' + long_code + '**Files Modified**\n- `src/a.py`\n', encoding='utf-8')
            (f1 / 'feature.patch').write_text('--- a/src/a.py\n+++ b/src/a.py\n', encoding='utf-8')
            (f2 / 'feature.patch').write_text('--- a/src/a.py\n+++ b/src/a.py\n', encoding='utf-8')
            records = load_records_from_path(root.parent, feature_max_chars=450)
            rows = convert_records(records, feature_max_chars=450)
            self.assertEqual(len(rows), 1)
            self.assertLessEqual(len(rows[0]['agent_a_task']), 470)
            self.assertIn('Files Modified', rows[0]['agent_a_task'])
            self.assertNotIn('print("x")', rows[0]['agent_a_task'])

