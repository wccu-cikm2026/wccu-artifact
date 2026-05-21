from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wccu_eval.eval.scenarios import get_scenario
from wccu_eval.scheduler.context_concurrency_policy import PolicyMode
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel, expected_success
from wccu_eval.substrate.context_substrate_store import seed_context


class WccuTests(unittest.TestCase):
    def _run(self, condition: str, scenario_id: str = 'wccu_stale_dependency_cross_target'):
        sc = get_scenario(scenario_id)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'ctx'
            run = Path(td) / 'run'
            seed_context(root, sc.get('seed', {}))
            return execute_context_policy_parallel(
                root_dir=root,
                run_dir=run,
                scenario=sc,
                policy_mode=condition,
                condition=str(condition),
            )

    def test_wccu_review_routes_cross_target_stale_dependency(self):
        row = self._run(PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE)
        self.assertTrue(row['task_success'])
        self.assertEqual(row['wccu_review_routed_count'], 1)
        self.assertEqual(row['stale_dependency_count'], 1)
        self.assertEqual(row['stale_dependency_accepted_count'], 0)
        self.assertEqual(row['unsafe_auto_commit_count'], 0)
        self.assertIn('wccu_certificate_intervention', {d['decision'] for d in row['merge_decisions']})

    def test_without_wccu_same_target_conflict_does_not_fire(self):
        row = self._run(PolicyMode.ADAPTIVE)
        self.assertFalse(row['task_success'])
        self.assertEqual(row['wccu_blocked_count'], 0)
        self.assertEqual(row['commit']['committed'], 2)
        self.assertEqual(row['stale_dependency_accepted_count'], 1)
        self.assertEqual(row['unsafe_auto_commit_count'], 1)

    def test_wccu_no_read_validation_ablation_logs_but_does_not_block(self):
        row = self._run(PolicyMode.ADAPTIVE_WCCU_NO_READ_VALIDATION)
        self.assertFalse(row['task_success'])
        self.assertEqual(row['stale_dependency_count'], 1)
        self.assertEqual(row['stale_read_validation_ignored_count'], 1)
        self.assertEqual(row['wccu_blocked_count'], 0)
        self.assertEqual(row['wccu_review_routed_count'], 0)
        self.assertEqual(row['commit']['committed'], 2)
        self.assertEqual(row['stale_dependency_accepted_count'], 1)

    def test_lowrisk_cross_target_scenario_is_clean_wccu_ablation(self):
        wccu = self._run(PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE, 'wccu_stale_dependency_lowrisk_cross_target')
        plain = self._run(PolicyMode.ADAPTIVE, 'wccu_stale_dependency_lowrisk_cross_target')
        noread = self._run(PolicyMode.ADAPTIVE_WCCU_NO_READ_VALIDATION, 'wccu_stale_dependency_lowrisk_cross_target')
        self.assertTrue(wccu['task_success'])
        self.assertEqual(wccu['wccu_review_routed_count'], 1)
        self.assertEqual(wccu['stale_dependency_accepted_count'], 0)
        self.assertEqual(wccu['unsafe_auto_commit_count'], 0)
        self.assertFalse(plain['task_success'])
        self.assertEqual(plain['stale_dependency_accepted_count'], 1)
        self.assertEqual(plain['unsafe_auto_commit_count'], 1)
        self.assertFalse(noread['task_success'])
        self.assertEqual(noread['stale_read_validation_ignored_count'], 1)
        self.assertEqual(noread['stale_dependency_accepted_count'], 1)


    def test_wccu_block_counts_as_successful_intervention(self):
        sc = get_scenario('wccu_stale_dependency_lowrisk_cross_target')
        summary = {
            'condition': 'adaptive_wccu_model_certificate',
            'policy_mode': 'adaptive_wccu_model_certificate',
            'commit': {'committed': 1, 'proposals': 0, 'conflicts': 1},
            'unsafe_auto_commit_count': 0,
            'wccu_review_routed_count': 0,
            'wccu_blocked_count': 1,
            'wccu_intervention_count': 1,
            'stale_dependency_count': 1,
            'stale_dependency_accepted_count': 0,
        }
        self.assertTrue(expected_success(sc, summary))

    def test_user_correction_auto_commit_is_not_counted_unsafe_when_it_blocks_stale_write(self):
        row = self._run(PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE, 'wccu_stale_dependency_cross_target')
        self.assertEqual(row['commit']['committed'], 1)
        committed_agents = []
        for decision in row['merge_decisions']:
            if decision['decision'] == 'single_writer':
                committed_agents.extend(decision['agents'])
        self.assertIn('correction_agent', committed_agents)
        self.assertEqual(row['unsafe_auto_commit_count'], 0)

    def test_expected_status_present_is_not_false_stale_when_atom_is_active(self):
        from wccu_eval.scheduler.wccu import verify_certificate

        scenario = {
            'seed': {
                'atoms': [
                    {'id': 'atom_pref_backup_url', 'status': 'active'},
                    {'id': 'atom_response_format_hint', 'status': 'active'},
                ]
            }
        }
        intent = {
            'id': 'intent_present_status',
            'intent_type': 'patch_atom',
            'risk': 'low',
            'authority': 'agent',
            'payload': {
                'id': 'atom_response_format_hint',
                'target_id': 'atom_response_format_hint',
                'atom_id': 'atom_response_format_hint',
                'atom_type': 'memory',
                'canonical_text_en': 'Response examples may include optional URL fields.',
            },
            'preconditions': {'base_snapshot_id': 'ctx_000000'},
            'certificate': {
                'read_dependencies': [
                    {
                        'target_id': 'atom_response_format_hint',
                        'snapshot_id': 'ctx_000000',
                        'expected_status': 'present',
                        'freshness_required': True,
                        'reason': 'Patch the existing response format hint atom.',
                    }
                ],
                'target_certificate': {
                    'claimed_target_id': 'atom_response_format_hint',
                    'confidence': 0.9,
                },
                'preconditions': {
                    'min_target_confidence': 0.6,
                    'requires_review_if_invalid': True,
                },
                'source': 'model_supplied',
            },
        }
        result = verify_certificate(intent, all_intents=[intent], scenario=scenario, certificate_mode='model_certificate')
        self.assertTrue(result['valid'])
        self.assertEqual(result['metrics']['stale_dependency_count'], 0)
        self.assertEqual(result['errors'], [])


if __name__ == '__main__':
    unittest.main()

class WccuGroupAwareCompositionTests(unittest.TestCase):
    def test_wccu_does_not_break_workspace_lock_group(self):
        from wccu_eval.scheduler.context_conflict_resolver import resolve_parallel_write_intents
        from wccu_eval.scheduler.context_concurrency_policy import PolicyMode

        stale_cert = {
            'read_dependencies': [
                {
                    'target_id': 'atom_file_plan_compatible',
                    'snapshot_id': 'ctx_000000',
                    'expected_status': 'active',
                    'freshness_required': True,
                    'reason': 'patch plan depends on the current file plan',
                }
            ],
            'target_certificate': {'claimed_target_id': 'atom_file_plan_compatible', 'confidence': 1.0},
            'preconditions': {'requires_review_if_invalid': True, 'min_target_confidence': 0.5},
            'source': 'model_supplied',
        }
        a = {
            'agent_id': 'builder_a',
            'role': 'builder',
            'write_intents': [{
                'intent_type': 'patch_atom',
                'payload': {'id': 'atom_file_plan_compatible', 'target_id': 'atom_file_plan_compatible', 'atom_type': 'artifact_plan', 'file_path': 'src/cache.ts', 'canonical_text_en': 'same'},
                'certificate': stale_cert,
            }],
        }
        b = {
            'agent_id': 'builder_b',
            'role': 'builder',
            'write_intents': [{
                'intent_type': 'patch_atom',
                'payload': {'id': 'atom_file_plan_compatible', 'target_id': 'atom_file_plan_compatible', 'atom_type': 'artifact_plan', 'file_path': 'src/cache.ts', 'canonical_text_en': 'same'},
            }],
        }
        scenario = {'seed': {'atoms': [{'id': 'atom_file_plan_compatible', 'status': 'active'}]}}
        resolved = resolve_parallel_write_intents([a, b], policy_mode=PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE, scenario=scenario)
        decisions = {d['decision'] for d in resolved['decisions']}
        self.assertIn('lock_contention_review_required_with_wccu', decisions)
        self.assertEqual(len(resolved['committable']), 0)
        self.assertEqual(len(resolved['conflicted']), 2)
        self.assertEqual(resolved['lock_conflict_count'], 1)
        self.assertEqual(resolved['wccu_intervention_count'], 1)

    def test_wccu_does_not_preempt_authority_rebase_group(self):
        from wccu_eval.scheduler.context_conflict_resolver import resolve_parallel_write_intents
        from wccu_eval.scheduler.context_concurrency_policy import PolicyMode

        patch = {
            'agent_id': 'old_snapshot_agent',
            'role': 'planner',
            'write_intents': [{
                'intent_type': 'patch_atom',
                'authority': 'agent',
                'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'canonical_text_en': 'The user strongly prefers backup_url.'},
                'certificate': {
                    'read_dependencies': [{'target_id': 'atom_pref_backup_url', 'snapshot_id': 'ctx_000000', 'expected_status': 'active', 'freshness_required': True, 'reason': 'old preference must still be active'}],
                    'target_certificate': {'claimed_target_id': 'atom_pref_backup_url', 'confidence': 1.0},
                    'preconditions': {'requires_review_if_invalid': True, 'min_target_confidence': 0.5},
                    'source': 'model_supplied',
                },
            }],
        }
        correction = {
            'agent_id': 'correction_agent',
            'role': 'user',
            'write_intents': [{
                'intent_type': 'retract_atom',
                'authority': 'user',
                'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'reason': 'User correction.'},
            }],
        }
        scenario = {'seed': {'atoms': [{'id': 'atom_pref_backup_url', 'status': 'active'}]}}
        resolved = resolve_parallel_write_intents([patch, correction], policy_mode=PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE, scenario=scenario)
        decisions = {d['decision'] for d in resolved['decisions']}
        self.assertIn('authority_interrupt_rebase_with_wccu', decisions)
        self.assertEqual(len(resolved['committable']), 1)
        self.assertEqual(len(resolved['conflicted']), 1)
        self.assertEqual(resolved['authority_rebase_count'], 1)
        self.assertEqual(resolved['wccu_intervention_count'], 1)

class WccuReviewDrivenAdditionsTests(unittest.TestCase):
    def test_unguided_prompt_uses_unguided_task_without_oracle_hint(self):
        from wccu_eval.agents.llm_agent import build_llm_agent_prompt
        from wccu_eval.eval.scenarios import get_scenario

        sc = get_scenario('wccu_stale_dependency_lowrisk_cross_target')
        prompt = build_llm_agent_prompt(
            agent={'id': 'format_agent', 'role': 'planner'},
            projection={'prompt': 'Projection includes atom_pref_backup_url and atom_response_format_hint.'},
            scenario=sc,
            include_target_candidates=True,
            certificate_guidance='unguided',
        )
        self.assertIn('Fill certificate fields from your own understanding', prompt)
        self.assertNotIn('In certificate.read_dependencies, list atom_pref_backup_url', prompt)
        self.assertIn('Patch atom_response_format_hint based on the active memory atom_pref_backup_url. Keep the write low risk.', prompt)

    def test_execution_trace_mode_recovers_dependency_from_agent_task(self):
        from wccu_eval.scheduler.wccu import minimal_certificate

        intent = {
            'id': 'intent_exec_trace',
            'intent_type': 'patch_atom',
            'risk': 'low',
            'authority': 'agent',
            'agent_task': 'Patch the response hint based on active atom_pref_backup_url.',
            'agent_output': 'Used backup_url preference to update examples.',
            'payload': {
                'id': 'atom_response_format_hint',
                'target_id': 'atom_response_format_hint',
                'atom_id': 'atom_response_format_hint',
                'atom_type': 'memory',
                'canonical_text_en': 'Response examples may include optional URL fields.',
            },
            'projection_trace': {
                'projection_id': 'proj_x',
                'snapshot_id': 'ctx_000000',
                'atoms': [
                    {'id': 'atom_pref_backup_url', 'status': 'active', 'title': 'Backup URL response preference', 'canonical_text_en': 'The user prefers backup_url in API responses.', 'tags': ['backup_url']},
                    {'id': 'atom_response_format_hint', 'status': 'active', 'title': 'Response format hint', 'canonical_text_en': 'Response examples may include optional URL fields.', 'tags': ['format']},
                ],
            },
        }
        cert = minimal_certificate(intent, certificate_mode='execution_trace')
        deps = {d['target_id'] for d in cert['read_dependencies']}
        self.assertIn('atom_pref_backup_url', deps)

    def test_dependency_precision_recall_script_counts_oracle_overlap(self):
        from wccu_eval.scripts.make_dependency_precision_recall import compute_rows

        payload = {
            'results': [{
                'scenario_id': 'wccu_stale_dependency_lowrisk_cross_target',
                'condition': 'adaptive_wccu_model_certificate',
                'agentRuns': [{
                    'agent_id': 'format_agent',
                    'write_intents': [{
                        'certificate': {'read_dependencies': [{'target_id': 'atom_pref_backup_url'}]},
                    }],
                }],
            }]
        }
        rows = compute_rows(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['tp'], 1)
        self.assertEqual(rows[0]['fp'], 0)
        self.assertEqual(rows[0]['fn'], 0)
        self.assertEqual(rows[0]['recall'], 1.0)

class WccuDependencyObligationAnalysisTests(unittest.TestCase):
    def test_dependency_analysis_distinguishes_target_recall_from_freshness_obligation(self):
        from wccu_eval.scripts.make_dependency_precision_recall import compute_rows

        payload = {
            'results': [
                {
                    'scenario_id': 'wccu_stale_dependency_lowrisk_cross_target',
                    'condition': 'adaptive_wccu_unguided_certificate',
                    'agentRuns': [
                        {
                            'agent_id': 'format_agent',
                            'write_intents': [
                                {
                                    'id': 'intent_1',
                                    'certificate': {
                                        'read_dependencies': [
                                            {'target_id': 'atom_pref_backup_url', 'freshness_required': False, 'expected_status': 'present'}
                                        ],
                                        'preconditions': {'freshness_required': False, 'requires_review_if_invalid': False},
                                    },
                                }
                            ],
                        }
                    ],
                    'wccu_events': [],
                }
            ]
        }
        row = compute_rows(payload)[0]
        self.assertEqual(row['target_dependency_recall'], 1.0)
        self.assertEqual(row['freshness_obligation_recall'], 0.0)
        self.assertEqual(row['enforced_dependency_recall'], 0.0)
        self.assertEqual(row['missing_freshness_obligation_count'], 1)

    def test_dependency_analysis_counts_enforced_stale_obligation(self):
        from wccu_eval.scripts.make_dependency_precision_recall import compute_rows

        payload = {
            'results': [
                {
                    'scenario_id': 'wccu_stale_dependency_lowrisk_cross_target',
                    'condition': 'adaptive_wccu_projection_trace',
                    'agentRuns': [
                        {
                            'agent_id': 'format_agent',
                            'write_intents': [
                                {
                                    'id': 'intent_1',
                                    'certificate': {
                                        'read_dependencies': [
                                            {'target_id': 'atom_pref_backup_url', 'freshness_required': True, 'expected_status': 'active'}
                                        ],
                                        'preconditions': {'freshness_required': True, 'requires_review_if_invalid': True},
                                    },
                                }
                            ],
                        }
                    ],
                    'wccu_events': [
                        {
                            'intent_id': 'intent_1',
                            'errors': [{'kind': 'stale_read_dependency', 'target_id': 'atom_pref_backup_url'}],
                            'warnings': [],
                        }
                    ],
                }
            ]
        }
        row = compute_rows(payload)[0]
        self.assertEqual(row['target_dependency_recall'], 1.0)
        self.assertEqual(row['freshness_obligation_recall'], 1.0)
        self.assertEqual(row['enforced_dependency_recall'], 1.0)
        self.assertEqual(row['non_reviewing_invalid_precondition_count'], 0)

class WccuAuthorityCorrectionFreshnessToleranceTests(unittest.TestCase):
    def test_user_correction_self_target_read_is_not_stale_against_lower_authority_patch(self):
        from wccu_eval.scheduler.context_conflict_resolver import resolve_parallel_write_intents

        stale_patch = {
            'agent_id': 'old_snapshot_agent',
            'role': 'planner',
            'write_intents': [{
                'id': 'intent_old_patch',
                'intent_type': 'patch_atom',
                'authority': 'agent',
                'payload': {
                    'id': 'atom_pref_backup_url',
                    'target_id': 'atom_pref_backup_url',
                    'atom_id': 'atom_pref_backup_url',
                    'atom_type': 'memory',
                    'canonical_text_en': 'The user strongly prefers backup_url in API responses.',
                    'risk': 'medium',
                },
                'certificate': {
                    'read_dependencies': [{
                        'target_id': 'atom_pref_backup_url',
                        'expected_status': 'active',
                        'freshness_required': True,
                        'reason': 'old preference must remain active',
                    }],
                    'target_certificate': {'claimed_target_id': 'atom_pref_backup_url', 'confidence': 1.0},
                    'preconditions': {'requires_review_if_invalid': True, 'min_target_confidence': 0.5},
                    'source': 'model_supplied',
                },
            }],
        }
        user_correction = {
            'agent_id': 'correction_agent',
            'role': 'user',
            'write_intents': [{
                'id': 'intent_user_retract',
                'intent_type': 'retract_atom',
                'authority': 'user',
                'payload': {
                    'id': 'atom_pref_backup_url',
                    'target_id': 'atom_pref_backup_url',
                    'atom_id': 'atom_pref_backup_url',
                    'atom_type': 'memory',
                    'reason': 'User correction: retract backup_url preference.',
                    'risk': 'medium',
                },
                'certificate': {
                    'read_dependencies': [{
                        'target_id': 'atom_pref_backup_url',
                        'expected_status': 'active',
                        'freshness_required': True,
                        'reason': 'read the memory being corrected',
                    }],
                    'target_certificate': {'claimed_target_id': 'atom_pref_backup_url', 'confidence': 1.0},
                    'authority_certificate': {'actor_authority': 'user', 'required_authority': 'user'},
                    'preconditions': {'requires_review_if_invalid': True, 'min_target_confidence': 0.5},
                    'source': 'model_supplied',
                },
            }],
        }
        scenario = {'seed': {'atoms': [{'id': 'atom_pref_backup_url', 'status': 'active'}]}}
        resolved = resolve_parallel_write_intents(
            [stale_patch, user_correction],
            policy_mode=PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE,
            scenario=scenario,
        )
        by_id = {i['id']: i for i in resolved['merged_intents']}
        correction_metrics = by_id['intent_user_retract']['wccu_verification']['metrics']
        stale_metrics = by_id['intent_old_patch']['wccu_verification']['metrics']
        self.assertEqual(correction_metrics['stale_dependency_count'], 0)
        self.assertEqual(correction_metrics['authority_correction_self_dependency_tolerated_count'], 1)
        self.assertEqual(stale_metrics['stale_dependency_count'], 1)
        self.assertEqual(resolved['stale_dependency_accepted_count'], 0)
        self.assertEqual(resolved['unsafe_auto_commit_count'], 0)
        self.assertIn('authority_interrupt_rebase_with_wccu', {d['decision'] for d in resolved['decisions']})

    def test_lower_authority_self_target_read_is_stale_against_user_correction(self):
        from wccu_eval.scheduler.wccu import verify_certificate

        old_patch = {
            'id': 'intent_old_patch',
            'intent_type': 'patch_atom',
            'authority': 'agent',
            'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_id': 'atom_pref_backup_url', 'atom_type': 'memory'},
            'certificate': {
                'read_dependencies': [{'target_id': 'atom_pref_backup_url', 'expected_status': 'active', 'freshness_required': True}],
                'target_certificate': {'claimed_target_id': 'atom_pref_backup_url', 'confidence': 1.0},
                'preconditions': {'requires_review_if_invalid': True, 'min_target_confidence': 0.5},
                'source': 'model_supplied',
            },
        }
        user_retract = {
            'id': 'intent_user_retract',
            'intent_type': 'retract_atom',
            'authority': 'user',
            'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_id': 'atom_pref_backup_url', 'atom_type': 'memory'},
        }
        scenario = {'seed': {'atoms': [{'id': 'atom_pref_backup_url', 'status': 'active'}]}}
        result = verify_certificate(old_patch, all_intents=[old_patch, user_retract], scenario=scenario)
        self.assertFalse(result['valid'])
        self.assertEqual(result['metrics']['stale_dependency_count'], 1)


class WccuExecutionWitnessTests(unittest.TestCase):
    def test_execution_witness_dependency_is_enforced_without_lexical_overlap(self):
        from wccu_eval.scheduler.wccu import verify_certificate, CERTIFICATE_MODE_EXECUTION_TRACE

        writer = {
        'id': 'intent_writer',
        'actor': 'agent:writer',
        'source_agent': 'writer',
        'intent_type': 'patch_atom',
        'payload': {
        'target_id': 'atom_response_format_hint',
        'atom_id': 'atom_response_format_hint',
        'id': 'atom_response_format_hint',
        'atom_type': 'memory',
        'canonical_text_en': 'Use the compact output profile.',
        },
        'preconditions': {'base_snapshot_id': 'ctx_000000'},
        'projection_trace': {
        'projection_id': 'proj_test',
        'snapshot_id': 'ctx_000000',
        'atoms': [
        {'id': 'atom_hidden_dependency', 'atom_type': 'memory', 'status': 'active', 'title': 'Preference object', 'canonical_text_en': 'The user likes concise answer format.', 'tags': ['preference']},
        {'id': 'atom_response_format_hint', 'atom_type': 'memory', 'status': 'active', 'title': 'Response format', 'canonical_text_en': 'Default response style.', 'tags': ['format']},
        ],
        },
        'execution_witness': {'read_atoms': ['atom_hidden_dependency']},
        }
        invalidator = {
        'id': 'intent_invalidator',
        'actor': 'agent:correction',
        'source_agent': 'correction',
        'intent_type': 'retract_atom',
        'authority': 'user',
        'payload': {
        'target_id': 'atom_hidden_dependency',
        'atom_id': 'atom_hidden_dependency',
        'id': 'atom_hidden_dependency',
        'atom_type': 'memory',
        'status': 'retracted',
        },
        'preconditions': {'base_snapshot_id': 'ctx_000000'},
        }
        scenario = {'seed': {'atoms': [{'id': 'atom_hidden_dependency', 'status': 'active'}]}}
        result = verify_certificate(writer, all_intents=[writer, invalidator], scenario=scenario, certificate_mode=CERTIFICATE_MODE_EXECUTION_TRACE)
        self.assertFalse(result['valid'])
        self.assertEqual(result['metrics']['stale_dependency_count'], 1)
        self.assertEqual(result['certificate']['read_dependencies'][0]['target_id'], 'atom_hidden_dependency')
