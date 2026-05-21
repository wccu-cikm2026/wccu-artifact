import tempfile
import unittest
from pathlib import Path

from wccu_eval.substrate.context_projection_compiler import compile_projection
from wccu_eval.substrate.context_substrate_store import commit_context_write_intents_batch, read_context_substrate, seed_context


class SubstrateTests(unittest.TestCase):
    def test_seed_compile_and_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'context'
            seed_context(root, {'atoms': [{'id': 'atom_goal', 'atom_type': 'task', 'title': 'Goal', 'canonical_text_en': 'Implement tests'}]})
            sub = read_context_substrate(root)
            self.assertEqual(sub['snapshot_id'], 'ctx_000000')
            projection = compile_projection(root, role='researcher', task_type='planning', goal='Implement tests')
            self.assertEqual(projection['snapshot_id'], 'ctx_000000')
            result = commit_context_write_intents_batch(root, [{'intent_type': 'upsert_atom', 'requested_commit_mode': 'auto', 'payload': {'id': 'atom_summary', 'atom_type': 'memory', 'title': 'Summary', 'canonical_text_en': 'Tests matter'}}])
            self.assertEqual(result['committed'], 1)
            self.assertEqual(read_context_substrate(root)['snapshot_id'], 'ctx_000001')


if __name__ == '__main__':
    unittest.main()
