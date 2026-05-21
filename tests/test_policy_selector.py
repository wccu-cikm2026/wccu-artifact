import unittest

from wccu_eval.scheduler.context_concurrency_policy import select_context_concurrency_policy, PolicyMode


class PolicySelectorTests(unittest.TestCase):
    def test_high_risk_permission_uses_review_gate(self):
        intent = {'intent_type': 'patch_atom', 'risk': 'high', 'payload': {'id': 'atom_permission_policy', 'atom_type': 'permission_policy', 'title': 'Policy'}}
        selected = select_context_concurrency_policy(intent, mode=PolicyMode.ADAPTIVE)
        self.assertEqual(selected['isolation_policy'], 'review_gated_serializable')
        self.assertEqual(selected['commit_mode'], 'review_required')

    def test_append_event_uses_append_only(self):
        intent = {'intent_type': 'append_event', 'payload': {'id': 'stream', 'stream_id': 'stream', 'atom_type': 'evidence_event', 'title': 'Evidence'}}
        selected = select_context_concurrency_policy(intent, mode=PolicyMode.ADAPTIVE)
        self.assertEqual(selected['isolation_policy'], 'append_only_causal')
        self.assertEqual(selected['commit_mode'], 'auto')

    def test_user_retraction_uses_authority_rebase(self):
        intent = {'intent_type': 'retract_atom', 'authority': 'user', 'payload': {'id': 'atom_pref', 'atom_type': 'memory', 'title': 'Correction'}}
        selected = select_context_concurrency_policy(intent, mode=PolicyMode.ADAPTIVE)
        self.assertEqual(selected['isolation_policy'], 'authority_interrupt_rebase')

    def test_workspace_patch_uses_lock(self):
        intent = {'intent_type': 'patch_atom', 'payload': {'id': 'atom_file', 'atom_type': 'artifact_plan', 'file_path': 'src/api.ts', 'title': 'Patch'}}
        selected = select_context_concurrency_policy(intent, mode=PolicyMode.ADAPTIVE)
        self.assertEqual(selected['isolation_policy'], 'pessimistic_lock')


if __name__ == '__main__':
    unittest.main()
