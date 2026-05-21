from __future__ import annotations

from copy import deepcopy

SCENARIOS = {
    'snapshot_fanout': {
        'id': 'snapshot_fanout',
        'goal': 'Plan a safe code change with independent research, testing, and risk review.',
        'task_type': 'planning',
        'default_latency_ms': 25,
        'budget_tokens': 900,
        'agents': [
            {'id': 'researcher', 'role': 'researcher'},
            {'id': 'test_planner', 'role': 'test_planner'},
            {'id': 'risk_reviewer', 'role': 'risk_reviewer'},
            {'id': 'synthesizer', 'role': 'synthesizer'},
        ],
        'seed': {
            'atoms': [
                {'id': 'atom_goal', 'atom_type': 'task', 'title': 'Task goal', 'canonical_text_en': 'Implement a feature with tests and risk review.', 'tags': ['planning', 'code']},
                {'id': 'atom_rule_tests', 'atom_type': 'rule', 'title': 'Tests required', 'canonical_text_en': 'Every code change must include a relevant test plan.', 'tags': ['test', 'verification']},
                {'id': 'atom_rule_privacy', 'atom_type': 'rule', 'title': 'Privacy rule', 'canonical_text_en': 'Do not expose private memory or credentials.', 'tags': ['risk', 'privacy']},
                {'id': 'atom_arch', 'atom_type': 'memory', 'title': 'Architecture', 'canonical_text_en': 'The runtime compiles role-specific projections from context atoms.', 'tags': ['architecture', 'context']},
            ],
            'links': [
                {'from': 'atom_goal', 'to': 'atom_rule_tests', 'type': 'requires'},
                {'from': 'atom_goal', 'to': 'atom_rule_privacy', 'type': 'constrained_by'},
            ],
        },
        'agent_outputs': {
            'researcher': {'latency_ms': 35, 'text': 'Finding: context projection should prioritize task and architecture atoms.', 'add_summary_atom': True},
            'test_planner': {'latency_ms': 35, 'text': 'Test plan: verify projections, merges, and stale write rejection.', 'add_summary_atom': True},
            'risk_reviewer': {'latency_ms': 35, 'text': 'Risk: unsafe commits must be routed to review.', 'add_summary_atom': True},
            'synthesizer': {'latency_ms': 35, 'text': 'Synthesis: combine independent findings after the merge stage.', 'add_summary_atom': True},
        },
        'expected': {'max_unsafe_auto_commit_count': 0},
    },
    'append_only_evidence_log': {
        'id': 'append_only_evidence_log',
        'goal': 'Multiple evidence agents append independent observations to the same run log.',
        'task_type': 'evidence_collection',
        'default_latency_ms': 10,
        'agents': [
            {'id': 'evidence_a', 'role': 'researcher'},
            {'id': 'evidence_b', 'role': 'researcher'},
            {'id': 'evidence_c', 'role': 'researcher'},
        ],
        'seed': {'atoms': [{'id': 'atom_question', 'atom_type': 'task', 'title': 'Evidence task', 'canonical_text_en': 'Collect independent evidence observations for a planning decision.', 'tags': ['evidence']}]},
        'agent_outputs': {
            'evidence_a': {'text': 'Observation A: tests cover projection cache hits.', 'intents': [{'intent_type': 'append_event', 'payload': {'id': 'evidence_stream', 'atom_type': 'evidence_event', 'title': 'Observation A', 'canonical_text_en': 'Tests cover projection cache hits.', 'stream_id': 'evidence_stream'}}]},
            'evidence_b': {'text': 'Observation B: conflict resolver emits proposal events.', 'intents': [{'intent_type': 'append_event', 'payload': {'id': 'evidence_stream', 'atom_type': 'evidence_event', 'title': 'Observation B', 'canonical_text_en': 'Conflict resolver emits proposal events.', 'stream_id': 'evidence_stream'}}]},
            'evidence_c': {'text': 'Observation C: policy selector records isolation decisions.', 'intents': [{'intent_type': 'append_event', 'payload': {'id': 'evidence_stream', 'atom_type': 'evidence_event', 'title': 'Observation C', 'canonical_text_en': 'Policy selector records isolation decisions.', 'stream_id': 'evidence_stream'}}]},
        },
        'expected': {'max_unsafe_auto_commit_count': 0},
    },
    'conflict_detection': {
        'id': 'conflict_detection',
        'goal': 'Two agents concurrently update the same policy atom with incompatible semantics.',
        'task_type': 'policy_update',
        'default_latency_ms': 20,
        'agents': [{'id': 'agent_a', 'role': 'policy_writer'}, {'id': 'agent_b', 'role': 'policy_writer'}],
        'seed': {'atoms': [{'id': 'atom_api_policy', 'atom_type': 'rule', 'title': 'API response policy', 'canonical_text_en': 'API responses may include a backup_url when useful.', 'tags': ['api', 'policy']}]},
        'agent_outputs': {
            'agent_a': {'text': 'Patch policy: API responses must include backup_url for recoverability.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_api_policy', 'atom_type': 'rule', 'title': 'API response policy', 'canonical_text_en': 'API responses must include backup_url for recoverability.', 'tags': ['api', 'policy']}}]},
            'agent_b': {'text': 'Patch policy: API responses must not include backup_url for privacy.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_api_policy', 'atom_type': 'rule', 'title': 'API response policy', 'canonical_text_en': 'API responses must not include backup_url for privacy.', 'tags': ['api', 'policy', 'privacy']}}]},
        },
        'expected': {'min_conflict_groups': 1, 'requires_proposals': 2, 'max_unsafe_auto_commit_count': 0},
    },
    'high_risk_rule_change': {
        'id': 'high_risk_rule_change',
        'goal': 'A single agent proposes a high-risk permission policy change with no concurrent conflict.',
        'task_type': 'permission_update',
        'default_latency_ms': 10,
        'agents': [{'id': 'builder', 'role': 'builder'}],
        'seed': {'atoms': [{'id': 'atom_permission_policy', 'atom_type': 'permission_policy', 'title': 'Tool permission policy', 'canonical_text_en': 'Deployment operations require explicit user approval.', 'tags': ['permission', 'deployment']}]},
        'agent_outputs': {'builder': {'text': 'Relax deployment policy for speed.', 'intents': [{'intent_type': 'patch_atom', 'risk': 'high', 'payload': {'id': 'atom_permission_policy', 'atom_type': 'permission_policy', 'title': 'Tool permission policy', 'canonical_text_en': 'Deployment operations may run without explicit approval during task loops.', 'tags': ['permission', 'deployment'], 'risk': 'high'}}]}},
        'expected': {'requires_proposals': 1, 'max_unsafe_auto_commit_count': 0},
    },
    'user_correction_rebase': {
        'id': 'user_correction_rebase',
        'goal': 'A user retracts a memory while an agent still holds an older base snapshot.',
        'task_type': 'correction',
        'default_latency_ms': 15,
        'agents': [{'id': 'old_snapshot_agent', 'role': 'planner'}, {'id': 'correction_agent', 'role': 'user'}],
        'seed': {'atoms': [{'id': 'atom_pref', 'atom_type': 'memory', 'title': 'User preference', 'canonical_text_en': 'The user prefers backup_url in API responses.', 'tags': ['preference', 'api']}]},
        'agent_outputs': {
            'old_snapshot_agent': {'text': 'Use the old preference and strengthen it.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_pref', 'atom_type': 'memory', 'title': 'User preference', 'canonical_text_en': 'The user strongly prefers backup_url in API responses.', 'tags': ['preference', 'api']}}]},
            'correction_agent': {'text': 'Correction: retract the old preference.', 'intents': [{'intent_type': 'retract_atom', 'authority': 'user', 'payload': {'id': 'atom_pref', 'atom_id': 'atom_pref', 'atom_type': 'memory', 'reason': 'User correction: backup_url preference was wrong.'}}]},
        },
        'expected': {'min_conflict_groups': 1, 'min_authority_rebase_count': 1, 'requires_proposals': 1, 'max_unsafe_auto_commit_count': 0},
    },
    'workspace_patch_contention': {
        'id': 'workspace_patch_contention',
        'goal': 'Two coding agents patch the same workspace file plan concurrently.',
        'task_type': 'code_change',
        'default_latency_ms': 30,
        'agents': [{'id': 'builder_a', 'role': 'builder'}, {'id': 'builder_b', 'role': 'builder'}, {'id': 'reviewer', 'role': 'reviewer'}],
        'seed': {'atoms': [
            {'id': 'atom_file_plan', 'atom_type': 'artifact_plan', 'title': 'src/api.ts change plan', 'canonical_text_en': 'src/api.ts should expose getStatus().', 'tags': ['code', 'src/api.ts']},
            {'id': 'atom_test_rule', 'atom_type': 'rule', 'title': 'Test rule', 'canonical_text_en': 'Any src/api.ts change requires a unit test.', 'tags': ['test', 'code']},
        ]},
        'agent_outputs': {
            'builder_a': {'text': 'Changed src/api.ts: getStatus returns ok.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_file_plan', 'atom_type': 'artifact_plan', 'file_path': 'src/api.ts', 'title': 'src/api.ts change plan', 'canonical_text_en': 'src/api.ts should expose getStatus() returning ok.', 'tags': ['code', 'src/api.ts']}}]},
            'builder_b': {'text': 'Changed src/api.ts: getStatus returns health object.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_file_plan', 'atom_type': 'artifact_plan', 'file_path': 'src/api.ts', 'title': 'src/api.ts change plan', 'canonical_text_en': 'src/api.ts should expose getStatus() returning a health object.', 'tags': ['code', 'src/api.ts']}}]},
            'reviewer': {'text': 'Finding: src/api.ts changed; test coverage required.', 'intents': [{'intent_type': 'upsert_atom', 'payload': {'id': 'atom_review_test_required', 'atom_type': 'review_finding', 'title': 'Test coverage required', 'canonical_text_en': 'src/api.ts changed; unit test coverage is required.', 'tags': ['review', 'test']}}]},
        },
        'expected': {'min_conflict_groups': 1, 'requires_proposals': 2, 'max_unsafe_auto_commit_count': 0},
    },
    'low_risk_memory_merge': {
        'id': 'low_risk_memory_merge',
        'goal': 'Two agents make compatible low-risk memory updates from the same snapshot.',
        'task_type': 'memory_update',
        'default_latency_ms': 12,
        'agents': [{'id': 'agent_a', 'role': 'researcher'}, {'id': 'agent_b', 'role': 'researcher'}],
        'seed': {'atoms': [{'id': 'atom_arch', 'atom_type': 'memory', 'title': 'Architecture note', 'canonical_text_en': 'Context projections are compiled before each agent run.', 'tags': ['architecture']}]},
        'agent_outputs': {
            'agent_a': {'text': 'Confirm architecture note.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_arch', 'atom_type': 'memory', 'title': 'Architecture note', 'canonical_text_en': 'Context projections are compiled before each agent run.', 'tags': ['architecture', 'confirmed']}}]},
            'agent_b': {'text': 'Confirm architecture note from another role.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_arch', 'atom_type': 'memory', 'title': 'Architecture note', 'canonical_text_en': 'Context projections are compiled before each agent run.', 'tags': ['architecture', 'verified']}}]},
        },
        'expected': {'max_unsafe_auto_commit_count': 0},
    },
    'workspace_compatible_patch_contention': {
        'id': 'workspace_compatible_patch_contention',
        'goal': 'Two coding agents concurrently patch the same workspace file with compatible-looking semantics.',
        'task_type': 'code_change',
        'default_latency_ms': 30,
        'agents': [{'id': 'builder_a', 'role': 'builder'}, {'id': 'builder_b', 'role': 'builder'}],
        'seed': {'atoms': [{'id': 'atom_file_plan_compatible', 'atom_type': 'artifact_plan', 'title': 'src/cache.ts change plan', 'canonical_text_en': 'src/cache.ts should expose getCacheStatus().', 'tags': ['code', 'src/cache.ts']}]},
        'agent_outputs': {
            'builder_a': {'text': 'Patch src/cache.ts with getCacheStatus returning ready.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_file_plan_compatible', 'atom_type': 'artifact_plan', 'file_path': 'src/cache.ts', 'title': 'src/cache.ts change plan', 'canonical_text_en': 'src/cache.ts should expose getCacheStatus() returning ready.', 'tags': ['code', 'src/cache.ts']}}]},
            'builder_b': {'text': 'Patch src/cache.ts with the same getCacheStatus behavior.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_file_plan_compatible', 'atom_type': 'artifact_plan', 'file_path': 'src/cache.ts', 'title': 'src/cache.ts change plan', 'canonical_text_en': 'src/cache.ts should expose getCacheStatus() returning ready.', 'tags': ['code', 'src/cache.ts', 'compatible']}}]},
        },
        'expected': {'min_conflict_groups': 1, 'requires_proposals': 2, 'max_unsafe_auto_commit_count': 0},
    },
}

SCENARIOS['user_correction_ambiguous_targets'] = {
        'id': 'user_correction_ambiguous_targets',
        'goal': 'A user retracts one URL-related memory while an older agent updates a semantically similar memory; distractor atoms make target grounding non-trivial.',
        'task_type': 'correction',
        'default_latency_ms': 15,
        'agents': [{'id': 'old_snapshot_agent', 'role': 'planner'}, {'id': 'correction_agent', 'role': 'user'}],
        'seed': {'atoms': [
            {'id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL response preference', 'canonical_text_en': 'The user prefers backup_url in API responses.', 'tags': ['preference', 'api', 'backup_url']},
            {'id': 'atom_pref_callback_url', 'atom_type': 'memory', 'title': 'Callback URL webhook preference', 'canonical_text_en': 'The user prefers callback_url in webhook responses.', 'tags': ['preference', 'webhook', 'callback_url']},
            {'id': 'atom_pref_fallback_url', 'atom_type': 'memory', 'title': 'Fallback URL response preference', 'canonical_text_en': 'The user prefers fallback_url in API error responses.', 'tags': ['preference', 'api', 'fallback_url']},
            {'id': 'atom_pref_debug_url', 'atom_type': 'memory', 'title': 'Debug URL logging preference', 'canonical_text_en': 'The user prefers debug_url in internal diagnostic logs.', 'tags': ['preference', 'logs', 'debug_url']},
            {'id': 'atom_pref_backup_email', 'atom_type': 'memory', 'title': 'Backup email preference', 'canonical_text_en': 'The user prefers backup_email in account recovery flows.', 'tags': ['preference', 'account', 'backup_email']},
            {'id': 'atom_api_url_policy', 'atom_type': 'rule', 'title': 'API URL field policy', 'canonical_text_en': 'API responses may include optional URL fields only when requested.', 'tags': ['api', 'policy', 'url']},
        ]},
        'agent_outputs': {
            'old_snapshot_agent': {'text': 'Use the old backup_url preference and strengthen it.', 'intents': [{'intent_type': 'patch_atom', 'payload': {'id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL response preference', 'canonical_text_en': 'The user strongly prefers backup_url in API responses.', 'tags': ['preference', 'api', 'backup_url']}}]},
            'correction_agent': {'text': 'Correction: retract the backup_url preference, not other URL preferences.', 'intents': [{'intent_type': 'retract_atom', 'authority': 'user', 'payload': {'id': 'atom_pref_backup_url', 'atom_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'reason': 'User correction: backup_url preference was wrong.'}}]},
        },
        'expected': {'min_conflict_groups': 1, 'min_authority_rebase_count': 1, 'requires_proposals': 1, 'max_unsafe_auto_commit_count': 0},
    }



SCENARIOS['wccu_stale_dependency_cross_target'] = {
    'id': 'wccu_stale_dependency_cross_target',
    'goal': 'An older agent reads a user preference and writes a different policy atom while a user correction retracts the preference in the same parallel group.',
    'task_type': 'wccu_dependency_validation',
    'default_latency_ms': 15,
    'agents': [{'id': 'policy_agent', 'role': 'planner'}, {'id': 'correction_agent', 'role': 'user'}],
    'seed': {'atoms': [
        {'id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL response preference', 'canonical_text_en': 'The user prefers backup_url in API responses.', 'tags': ['preference', 'api', 'backup_url'], 'status': 'active'},
        {'id': 'atom_api_response_policy', 'atom_type': 'rule', 'title': 'API response policy', 'canonical_text_en': 'API responses may include URL fields only when supported by active user preferences.', 'tags': ['api', 'policy', 'backup_url'], 'status': 'active'},
    ]},
    # Oracle mode reads this fixture. Model-certificate mode ignores it and uses
    # only the model/fixture-provided certificate below. Projection-trace mode
    # derives dependencies from the compiled projection trace.
    'wccu_read_dependencies': {
        'policy_agent': [
            {'target_id': 'atom_pref_backup_url', 'expected_status': 'active', 'freshness_required': True, 'reason': 'Policy patch relies on the backup_url preference still being active.'}
        ]
    },
    'agent_outputs': {
        'policy_agent': {
            'text': 'Patch API policy based on the old backup_url preference.',
            'intents': [{
                'intent_type': 'patch_atom',
                'payload': {'id': 'atom_api_response_policy', 'target_id': 'atom_api_response_policy', 'atom_type': 'rule', 'title': 'API response policy', 'canonical_text_en': 'API responses should include backup_url because the user prefers it.', 'tags': ['api', 'policy', 'backup_url']},
                'certificate': {
                    'schema_version': 'wccu_certificate_v2',
                    'certificate_id': 'fixture_policy_agent_wccu',
                    'certificate_mode': 'model_certificate',
                    'read_dependencies': [{'target_id': 'atom_pref_backup_url', 'view_id': '', 'snapshot_id': 'ctx_000000', 'expected_status': 'active', 'expected_text_hash': '', 'freshness_required': True, 'reason': 'The policy patch relies on the backup_url memory still being active.'}],
                    'target_certificate': {'claimed_target_id': 'atom_api_response_policy', 'raw_target': 'API response policy', 'grounding_rationale': 'fixture exact target', 'confidence': 1.0},
                    'delta_contract': {'delta_type': 'strengthen_rule', 'semantic_direction': 'strengthen', 'affected_view_ids': ['target:atom_api_response_policy'], 'invalidates_views': True, 'summary': 'API responses should include backup_url because the user prefers it.'},
                    'authority_certificate': {'actor_authority': 'agent', 'required_authority': 'agent', 'authority_rationale': 'planner may propose normal policy patches when dependencies are fresh'},
                    'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': True, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True},
                }
            }]
        },
        'correction_agent': {
            'text': 'Correction: retract the backup_url preference.',
            'intents': [{
                'intent_type': 'retract_atom', 'authority': 'user',
                'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'reason': 'User correction: backup_url preference was wrong.'},
                'certificate': {
                    'schema_version': 'wccu_certificate_v2',
                    'certificate_id': 'fixture_correction_agent_wccu',
                    'certificate_mode': 'model_certificate',
                    'read_dependencies': [],
                    'target_certificate': {'claimed_target_id': 'atom_pref_backup_url', 'raw_target': 'backup_url preference', 'grounding_rationale': 'fixture exact target', 'confidence': 1.0},
                    'delta_contract': {'delta_type': 'retract_memory', 'semantic_direction': 'retract', 'affected_view_ids': ['target:atom_pref_backup_url'], 'invalidates_views': True, 'summary': 'Retract the backup_url preference.'},
                    'authority_certificate': {'actor_authority': 'user', 'required_authority': 'user', 'authority_rationale': 'user corrections may retract user memory'},
                    'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': False, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True},
                }
            }]
        },
    },
    'expected': {'min_wccu_review_routed_count': 1, 'min_stale_dependency_count': 1, 'requires_proposals': 1, 'max_unsafe_auto_commit_count': 0, 'max_stale_dependency_accepted_count': 0},
}


SCENARIOS['wccu_stale_dependency_lowrisk_cross_target'] = {
    'id': 'wccu_stale_dependency_lowrisk_cross_target',
    'goal': 'An older agent reads a user preference and writes a different low-risk context atom while a user correction retracts the preference in the same parallel group.',
    'task_type': 'wccu_dependency_validation_lowrisk',
    'default_latency_ms': 15,
    'agents': [{'id': 'format_agent', 'role': 'planner'}, {'id': 'correction_agent', 'role': 'user'}],
    'llm_agent_tasks': {
        'format_agent': 'Patch atom_response_format_hint based on the active memory atom_pref_backup_url. Keep the write low risk. In certificate.read_dependencies, list atom_pref_backup_url and require freshness because this different-target write is unsafe if that memory is retracted. In certificate.authority_certificate set actor_authority=agent and required_authority=agent; do not require builder, reviewer, user, or system authority for this low-risk memory hint patch.',
        'correction_agent': 'Retract atom_pref_backup_url as a user correction. Set authority to user. The correction invalidates any response-format hint that read the old backup_url preference. In certificate.authority_certificate set actor_authority=user and required_authority=user.',
    },
    # Unguided certificate mode removes the explicit answer key from the task.
    # The model still sees the projection and certificate schema, but it must
    # infer read dependencies from what it actually used.
    'llm_agent_tasks_unguided': {
        'format_agent': 'Patch atom_response_format_hint based on the active memory atom_pref_backup_url. Keep the write low risk.',
        'correction_agent': 'Retract atom_pref_backup_url as a user correction. Set authority to user. The correction invalidates response-format behavior derived from the old backup_url preference.',
    },
    'seed': {'atoms': [
        {'id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL response preference', 'canonical_text_en': 'The user prefers backup_url in API responses.', 'tags': ['preference', 'api', 'backup_url'], 'status': 'active'},
        {'id': 'atom_response_format_hint', 'atom_type': 'memory', 'title': 'Response format hint', 'canonical_text_en': 'Response examples may include optional URL fields only when supported by active preferences.', 'tags': ['api', 'format', 'hint'], 'status': 'active'},
    ]},
    # The write target is different from the read dependency.  Same-target
    # conflict detection should not catch this; WCCU read validation should.
    'wccu_read_dependencies': {
        'format_agent': [
            {'target_id': 'atom_pref_backup_url', 'expected_status': 'active', 'freshness_required': True, 'reason': 'The response-format hint relies on the backup_url preference still being active.'}
        ]
    },
    'agent_outputs': {
        'format_agent': {
            'text': 'Patch the response format hint based on the old backup_url preference.',
            'intents': [{
                'intent_type': 'patch_atom',
                'risk': 'low',
                'payload': {'id': 'atom_response_format_hint', 'target_id': 'atom_response_format_hint', 'atom_id': 'atom_response_format_hint', 'atom_type': 'memory', 'title': 'Response format hint', 'canonical_text_en': 'Response examples should include backup_url because the user prefers it.', 'tags': ['api', 'format', 'backup_url'], 'risk': 'low'},
                'certificate': {
                    'schema_version': 'wccu_certificate_v2',
                    'certificate_id': 'fixture_format_agent_lowrisk_wccu',
                    'certificate_mode': 'model_certificate',
                    'read_dependencies': [{'target_id': 'atom_pref_backup_url', 'view_id': '', 'snapshot_id': 'ctx_000000', 'expected_status': 'active', 'expected_text_hash': '', 'freshness_required': True, 'reason': 'The response-format hint relies on the backup_url memory still being active.'}],
                    'target_certificate': {'claimed_target_id': 'atom_response_format_hint', 'raw_target': 'Response format hint', 'grounding_rationale': 'fixture exact target', 'confidence': 1.0},
                    'delta_contract': {'delta_type': 'patch_memory', 'semantic_direction': 'patch', 'affected_view_ids': ['target:atom_response_format_hint'], 'invalidates_views': True, 'summary': 'Response examples should include backup_url because the user prefers it.'},
                    'authority_certificate': {'actor_authority': 'agent', 'required_authority': 'agent', 'authority_rationale': 'planner may update low-risk response-format hints when dependencies are fresh'},
                    'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': True, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True},
                }
            }]
        },
        'correction_agent': {
            'text': 'Correction: retract the backup_url preference.',
            'intents': [{
                'intent_type': 'retract_atom', 'authority': 'user', 'risk': 'high',
                'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'reason': 'User correction: backup_url preference was wrong.', 'risk': 'high'},
                'certificate': {
                    'schema_version': 'wccu_certificate_v2',
                    'certificate_id': 'fixture_lowrisk_correction_wccu',
                    'certificate_mode': 'model_certificate',
                    'read_dependencies': [],
                    'target_certificate': {'claimed_target_id': 'atom_pref_backup_url', 'raw_target': 'backup_url preference', 'grounding_rationale': 'fixture exact target', 'confidence': 1.0},
                    'delta_contract': {'delta_type': 'retract_memory', 'semantic_direction': 'retract', 'affected_view_ids': ['target:atom_pref_backup_url', 'target:atom_response_format_hint'], 'invalidates_views': True, 'summary': 'Retract the backup_url preference.'},
                    'authority_certificate': {'actor_authority': 'user', 'required_authority': 'user', 'authority_rationale': 'user corrections may retract user memory'},
                    'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': False, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True},
                }
            }]
        },
    },
    'expected': {'max_unsafe_auto_commit_count': 0},
    'expected_by_condition': {
        'adaptive_wccu': {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0},
        'adaptive_wccu_model_certificate': {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0},
        'adaptive_wccu_unguided_certificate': {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0},
        'adaptive_wccu_oracle_dependency': {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0},
        'adaptive_wccu_projection_trace': {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0},
        'adaptive_wccu_execution_trace': {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0},
        'adaptive_wccu_no_read_validation': {'min_stale_dependency_count': 1, 'min_stale_dependency_accepted_count': 1, 'min_stale_read_validation_ignored_count': 1},
        'adaptive_policy': {'min_stale_dependency_accepted_count': 1},
        'uniform_snapshot_occ': {'min_stale_dependency_accepted_count': 1},
        'uniform_review_gated': {'requires_proposals': 2, 'max_unsafe_auto_commit_count': 0, 'max_stale_dependency_accepted_count': 0},
    },
}

SCENARIOS['wccu_low_confidence_target'] = {
    'id': 'wccu_low_confidence_target',
    'goal': 'A model proposes a memory patch with a low-confidence target certificate.',
    'task_type': 'wccu_target_validation',
    'default_latency_ms': 10,
    'agents': [{'id': 'agent_a', 'role': 'researcher'}],
    'seed': {'atoms': [{'id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL preference', 'canonical_text_en': 'The user prefers backup_url.', 'tags': ['backup_url']}]},
    'agent_outputs': {
        'agent_a': {'text': 'Patch maybe the backup preference.', 'intents': [{
            'intent_type': 'patch_atom',
            'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL preference', 'canonical_text_en': 'The user may prefer backup_url.', 'tags': ['backup_url']},
            'certificate': {'schema_version': 'wccu_certificate_v2', 'certificate_id': 'low_confidence_target', 'certificate_mode': 'model_certificate', 'read_dependencies': [], 'target_certificate': {'claimed_target_id': 'atom_pref_backup_url', 'raw_target': 'maybe backup preference', 'grounding_rationale': 'ambiguous wording', 'confidence': 0.2}, 'delta_contract': {'delta_type': 'patch_memory', 'semantic_direction': 'patch', 'affected_view_ids': ['target:atom_pref_backup_url'], 'invalidates_views': True, 'summary': 'Maybe patch memory.'}, 'authority_certificate': {'actor_authority': 'agent', 'required_authority': 'agent', 'authority_rationale': 'low-risk memory patch'}, 'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': False, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True}}
        }]}
    },
    'expected': {'min_wccu_intervention_count': 1, 'max_unsafe_auto_commit_count': 0},
}

SCENARIOS['wccu_wrong_target_certificate'] = {
    'id': 'wccu_wrong_target_certificate',
    'goal': 'A certificate claims a different target than the grounded write target.',
    'task_type': 'wccu_target_validation',
    'default_latency_ms': 10,
    'agents': [{'id': 'agent_a', 'role': 'researcher'}],
    'seed': {'atoms': [
        {'id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL preference', 'canonical_text_en': 'The user prefers backup_url.', 'tags': ['backup_url']},
        {'id': 'atom_pref_callback_url', 'atom_type': 'memory', 'title': 'Callback URL preference', 'canonical_text_en': 'The user prefers callback_url.', 'tags': ['callback_url']},
    ]},
    'agent_outputs': {
        'agent_a': {'text': 'Patch backup preference with wrong certificate target.', 'intents': [{
            'intent_type': 'patch_atom',
            'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL preference', 'canonical_text_en': 'The user strongly prefers backup_url.', 'tags': ['backup_url']},
            'certificate': {'schema_version': 'wccu_certificate_v2', 'certificate_id': 'wrong_target_certificate', 'certificate_mode': 'model_certificate', 'read_dependencies': [], 'target_certificate': {'claimed_target_id': 'atom_pref_callback_url', 'raw_target': 'backup URL preference', 'grounding_rationale': 'incorrect model claim', 'confidence': 0.9}, 'delta_contract': {'delta_type': 'patch_memory', 'semantic_direction': 'patch', 'affected_view_ids': ['target:atom_pref_backup_url'], 'invalidates_views': True, 'summary': 'Patch backup memory.'}, 'authority_certificate': {'actor_authority': 'agent', 'required_authority': 'agent', 'authority_rationale': 'low-risk memory patch'}, 'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': False, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True}}
        }]}
    },
    'expected': {'min_wccu_intervention_count': 1, 'max_unsafe_auto_commit_count': 0},
}

SCENARIOS['wccu_policy_weakening_masked_patch'] = {
    'id': 'wccu_policy_weakening_masked_patch',
    'goal': 'A permission policy weakening is submitted as an ordinary patch and should be review-routed by the delta contract.',
    'task_type': 'wccu_delta_validation',
    'default_latency_ms': 10,
    'agents': [{'id': 'builder', 'role': 'builder'}],
    'seed': {'atoms': [{'id': 'atom_permission_policy', 'atom_type': 'permission_policy', 'title': 'Deployment permission policy', 'canonical_text_en': 'Deployment operations require explicit user approval.', 'tags': ['deployment', 'approval']}]},
    'agent_outputs': {
        'builder': {'text': 'Relax deployment approval but mark it as an ordinary patch.', 'intents': [{
            'intent_type': 'patch_atom', 'risk': 'low', 'authority': 'builder',
            'payload': {'id': 'atom_permission_policy', 'target_id': 'atom_permission_policy', 'atom_type': 'permission_policy', 'title': 'Deployment permission policy', 'canonical_text_en': 'Deployment operations may run without explicit approval during task loops.', 'tags': ['deployment', 'approval']},
            'certificate': {'schema_version': 'wccu_certificate_v2', 'certificate_id': 'weakening_masked_patch', 'certificate_mode': 'model_certificate', 'read_dependencies': [], 'target_certificate': {'claimed_target_id': 'atom_permission_policy', 'raw_target': 'deployment permission policy', 'grounding_rationale': 'fixture exact target', 'confidence': 1.0}, 'delta_contract': {'delta_type': 'weaken_rule', 'semantic_direction': 'weaken', 'affected_view_ids': ['target:atom_permission_policy'], 'invalidates_views': True, 'summary': 'Deployment operations may run without explicit approval.'}, 'authority_certificate': {'actor_authority': 'builder', 'required_authority': 'reviewer', 'authority_rationale': 'weakening permission policy requires reviewer authority'}, 'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': False, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True}}
        }]}
    },
    'expected': {'min_wccu_intervention_count': 1, 'max_unsafe_auto_commit_count': 0},
}

SCENARIOS['wccu_derived_view_stale_resurfacing'] = {
    'id': 'wccu_derived_view_stale_resurfacing',
    'goal': 'A derived handoff summary depends on a memory that is retracted in the same parallel group.',
    'task_type': 'wccu_view_invalidation',
    'default_latency_ms': 10,
    'agents': [{'id': 'summary_agent', 'role': 'planner'}, {'id': 'correction_agent', 'role': 'user'}],
    'seed': {'atoms': [
        {'id': 'atom_pref_backup_url', 'atom_type': 'memory', 'title': 'Backup URL preference', 'canonical_text_en': 'The user prefers backup_url.', 'tags': ['backup_url'], 'status': 'active'},
        {'id': 'atom_handoff_summary', 'atom_type': 'handoff_delta', 'title': 'Planning handoff summary', 'canonical_text_en': 'Use current user API URL preferences.', 'tags': ['handoff', 'backup_url'], 'status': 'active'},
    ]},
    'wccu_read_dependencies': {'summary_agent': [{'target_id': 'atom_pref_backup_url', 'view_id': 'handoff:planner', 'expected_status': 'active', 'freshness_required': True, 'reason': 'Handoff summary derived from active backup_url memory.'}]},
    'agent_outputs': {
        'summary_agent': {'text': 'Patch handoff summary based on current backup_url memory.', 'intents': [{
            'intent_type': 'patch_atom', 'payload': {'id': 'atom_handoff_summary', 'target_id': 'atom_handoff_summary', 'atom_type': 'handoff_delta', 'title': 'Planning handoff summary', 'canonical_text_en': 'Use backup_url in API response planning.', 'tags': ['handoff', 'backup_url']},
            'certificate': {'schema_version': 'wccu_certificate_v2', 'certificate_id': 'handoff_summary_wccu', 'certificate_mode': 'model_certificate', 'read_dependencies': [{'target_id': 'atom_pref_backup_url', 'view_id': 'handoff:planner', 'snapshot_id': 'ctx_000000', 'expected_status': 'active', 'expected_text_hash': '', 'freshness_required': True, 'reason': 'Derived summary uses backup_url memory.'}], 'target_certificate': {'claimed_target_id': 'atom_handoff_summary', 'raw_target': 'handoff summary', 'grounding_rationale': 'fixture exact target', 'confidence': 1.0}, 'delta_contract': {'delta_type': 'patch_memory', 'semantic_direction': 'patch', 'affected_view_ids': ['handoff:planner', 'target:atom_handoff_summary'], 'invalidates_views': True, 'summary': 'Patch derived handoff summary.'}, 'authority_certificate': {'actor_authority': 'agent', 'required_authority': 'agent', 'authority_rationale': 'planner may propose handoff summary'}, 'preconditions': {'base_snapshot_id': 'ctx_000000', 'freshness_required': True, 'no_retracted_dependencies': True, 'min_target_confidence': 0.55, 'requires_review_if_invalid': True}}
        }]},
        'correction_agent': {'text': 'Retract backup_url memory.', 'intents': [{'intent_type': 'retract_atom', 'authority': 'user', 'payload': {'id': 'atom_pref_backup_url', 'target_id': 'atom_pref_backup_url', 'atom_id': 'atom_pref_backup_url', 'atom_type': 'memory', 'reason': 'User correction.'}}]},
    },
    'expected': {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0},
}

SCENARIOS['mini_coding_team'] = {**deepcopy(SCENARIOS['workspace_patch_contention']), 'id': 'mini_coding_team'}


def get_scenario(scenario_id: str) -> dict:
    if scenario_id not in SCENARIOS:
        raise KeyError(f'Unknown scenario: {scenario_id}')
    return deepcopy(SCENARIOS[scenario_id])


def list_scenario_ids() -> list[str]:
    return [k for k in SCENARIOS.keys() if k != 'mini_coding_team']
