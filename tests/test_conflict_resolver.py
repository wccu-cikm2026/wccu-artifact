import unittest

from wccu_eval.scheduler.context_conflict_resolver import compatible_pair, resolve_parallel_write_intents
from wccu_eval.scheduler.context_concurrency_policy import PolicyMode


def result(agent_id, intent, role='agent'):
    return {'agent_id': agent_id, 'role': role, 'write_intents': [intent]}


class ConflictResolverTests(unittest.TestCase):
    def test_incompatible_same_target_routes_to_review(self):
        a = {'intent_type': 'patch_atom', 'payload': {'id': 'atom_policy', 'atom_type': 'rule', 'title': 'P', 'canonical_text_en': 'Must include backup_url'}}
        b = {'intent_type': 'patch_atom', 'payload': {'id': 'atom_policy', 'atom_type': 'rule', 'title': 'P', 'canonical_text_en': 'Must not include backup_url'}}
        resolved = resolve_parallel_write_intents([result('a', a), result('b', b)], policy_mode=PolicyMode.ADAPTIVE)
        self.assertEqual(resolved['conflict_count'], 1)
        self.assertEqual(len(resolved['conflicted']), 2)

    def test_compatible_low_risk_updates_auto_merge(self):
        a = {'intent_type': 'patch_atom', 'payload': {'id': 'atom_arch', 'atom_type': 'memory', 'title': 'Arch', 'canonical_text_en': 'Same'}}
        b = {'intent_type': 'patch_atom', 'payload': {'id': 'atom_arch', 'atom_type': 'memory', 'title': 'Arch', 'canonical_text_en': 'Same'}}
        resolved = resolve_parallel_write_intents([result('a', a), result('b', b)], policy_mode=PolicyMode.ADAPTIVE)
        self.assertEqual(len(resolved['committable']), 2)
        self.assertEqual(resolved['unsafe_auto_commit_count'], 0)

    def test_uniform_append_only_misses_state_conflict(self):
        a = {'intent_type': 'patch_atom', 'payload': {'id': 'atom_policy', 'atom_type': 'rule', 'title': 'P', 'canonical_text_en': 'A'}}
        b = {'intent_type': 'patch_atom', 'payload': {'id': 'atom_policy', 'atom_type': 'rule', 'title': 'P', 'canonical_text_en': 'B'}}
        resolved = resolve_parallel_write_intents([result('a', a), result('b', b)], policy_mode=PolicyMode.UNIFORM_APPEND_ONLY)
        self.assertGreater(resolved['unsafe_auto_commit_count'], 0)

    def test_compatibility_helper(self):
        self.assertTrue(compatible_pair({'intent_type': 'patch_atom', 'payload': {'id': 'x', 'canonical_text_en': 'same'}}, {'intent_type': 'patch_atom', 'payload': {'id': 'x', 'canonical_text_en': 'same'}}))
        self.assertFalse(compatible_pair({'intent_type': 'patch_atom', 'payload': {'id': 'x', 'canonical_text_en': 'a'}}, {'intent_type': 'patch_atom', 'payload': {'id': 'x', 'canonical_text_en': 'b'}}))


if __name__ == '__main__':
    unittest.main()

class SemanticConflictLaneSeparationTests(unittest.TestCase):
    def test_high_risk_review_gate_preserves_semantic_conflict_label(self):
        a = {'intent_type': 'patch_atom', 'risk': 'high', 'payload': {'id': 'atom_policy', 'atom_type': 'rule', 'title': 'P', 'canonical_text_en': 'API responses must include backup_url'}}
        b = {'intent_type': 'patch_atom', 'risk': 'high', 'payload': {'id': 'atom_policy', 'atom_type': 'rule', 'title': 'P', 'canonical_text_en': 'API responses must not include backup_url'}}
        resolved = resolve_parallel_write_intents([result('a', a), result('b', b)], policy_mode=PolicyMode.ADAPTIVE)
        self.assertEqual(resolved['semantic_conflict_count'], 1)
        self.assertEqual(resolved['conflict_count'], 1)
        self.assertEqual(resolved['decisions'][0]['decision'], 'semantic_conflict_review_gated')
        self.assertTrue(resolved['decisions'][0]['semantic_conflict'])
        self.assertEqual(len(resolved['conflicted']), 2)
