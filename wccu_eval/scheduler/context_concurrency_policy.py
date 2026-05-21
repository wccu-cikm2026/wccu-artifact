from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from wccu_eval.utils import as_dict, clean, stable_hash


class PolicyMode(StrEnum):
    ADAPTIVE = 'adaptive'
    ADAPTIVE_READSET_OCC = 'adaptive_readset_occ'
    UNIFORM_SNAPSHOT_OCC = 'uniform_snapshot_occ'
    UNIFORM_PESSIMISTIC_LOCK = 'uniform_pessimistic_lock'
    UNIFORM_REVIEW_GATED = 'uniform_review_gated'
    UNIFORM_APPEND_ONLY = 'uniform_append_only'
    ADAPTIVE_NO_REVIEW_GATE = 'adaptive_no_review_gate'
    ADAPTIVE_NO_AUTHORITY_REBASE = 'adaptive_no_authority_rebase'
    ADAPTIVE_NO_APPEND_ONLY = 'adaptive_no_append_only'
    ADAPTIVE_NO_WORKSPACE_LOCK = 'adaptive_no_workspace_lock'
    ADAPTIVE_NO_SEMANTIC_CONFLICT_DETECTION = 'adaptive_no_semantic_conflict_detection'
    ADAPTIVE_WCCU = 'adaptive_wccu'
    ADAPTIVE_WCCU_MODEL_CERTIFICATE = 'adaptive_wccu_model_certificate'
    ADAPTIVE_WCCU_ORACLE_DEPENDENCY = 'adaptive_wccu_oracle_dependency'
    ADAPTIVE_WCCU_PROJECTION_TRACE = 'adaptive_wccu_projection_trace'
    ADAPTIVE_WCCU_NO_READ_VALIDATION = 'adaptive_wccu_no_read_validation'
    ADAPTIVE_WCCU_UNGUIDED_CERTIFICATE = 'adaptive_wccu_unguided_certificate'
    ADAPTIVE_WCCU_EXECUTION_TRACE = 'adaptive_wccu_execution_trace'


APPEND_OPS = {'append_event', 'record_usage', 'record_activity', 'record_observation', 'record_evidence'}
STATE_OPS = {'assert_atom', 'upsert_atom', 'patch_atom', 'retract_atom', 'assert_link', 'upsert_link', 'patch_link', 'link', 'unlink', 'retract_link'}
HIGH_REVIEW_TYPES = {'learned_rule', 'skill_candidate', 'agent_package', 'external_claim', 'financial_claim', 'legal_claim', 'medical_claim', 'permission_policy', 'credential_policy', 'deployment_policy'}
APPEND_TYPES = {'event', 'usage_event', 'activity_event', 'handoff_delta', 'evidence_event', 'observation', 'audit_event', 'review_note'}
WORKSPACE_TYPES = {'artifact_plan', 'workspace_file', 'code_file', 'patch_plan'}
AUTHORITY_RANK = {'system': 50, 'user': 40, 'runtime': 30, 'reviewer': 20, 'builder': 12, 'researcher': 8, 'agent': 10, 'unknown': 0}


def canonical_workspace_lock_scope(value: Any = None, intent: dict[str, Any] | None = None) -> str:
    """Return a stable workspace/file lock scope.

    CooperBench-style tasks can represent the same file as ``file:path.py`` in
    ``target_id``, as ``path.py`` in ``file_path``, or as an already-computed
    ``policy.lock_scope``.  Lock grouping must normalize these forms before
    grouping; otherwise two writes to the same file can split into
    ``lock:path.py`` and ``atom:file:path.py`` groups.
    """
    raw = clean(value)
    if intent is not None:
        intent = as_dict(intent)
        payload = as_dict(intent.get('payload'))
        policy = as_dict(intent.get('policy'))
        target = target_of(intent)
        raw = clean(
            raw
            or payload.get('file_path')
            or payload.get('path')
            or policy.get('lock_scope')
            or payload.get('target_id')
            or payload.get('atom_id')
            or payload.get('id')
            or target.get('id')
        )
    if raw.startswith('atom:file:'):
        raw = raw[len('atom:file:'):]
    elif raw.startswith('file:'):
        raw = raw[len('file:'):]
    elif raw.startswith('lock:'):
        raw = raw[len('lock:'):]
    raw = raw.replace('\\\\', '/').replace('\\', '/')
    while raw.startswith('./'):
        raw = raw[2:]
    raw = raw.strip('/ ')
    return raw


def has_workspace_file_hint(intent: dict[str, Any] | None = None) -> bool:
    intent = as_dict(intent)
    payload = as_dict(intent.get('payload'))
    target_id = clean(payload.get('target_id') or payload.get('atom_id') or payload.get('id'))
    return bool(clean(payload.get('file_path') or payload.get('path')) or target_id.startswith('file:') or target_id.startswith('atom:file:'))


def normalize_policy_mode(mode: str | PolicyMode | None) -> str:
    return clean(mode or PolicyMode.ADAPTIVE).lower()


def target_of(intent: dict[str, Any] | None = None) -> dict[str, str]:
    intent = as_dict(intent)
    payload = as_dict(intent.get('payload'))
    op = clean(intent.get('intent_type') or intent.get('op') or 'assert_atom').lower()
    if op in {'assert_atom', 'upsert_atom', 'patch_atom', 'retract_atom'}:
        return {'kind': 'atom', 'id': clean(payload.get('target_id') or payload.get('atom_id') or payload.get('id') or as_dict(payload.get('atom')).get('id'))}
    if op in {'assert_link', 'upsert_link', 'patch_link', 'link', 'unlink', 'retract_link'}:
        return {'kind': 'link', 'id': clean(payload.get('target_id') or payload.get('link_id') or payload.get('id') or as_dict(payload.get('link')).get('id'))}
    event_key = clean(payload.get('stream_id') or payload.get('target_id') or payload.get('id') or intent.get('id') or stable_hash(intent))
    return {'kind': 'event', 'id': event_key}


def _source_role(intent: dict[str, Any]) -> str:
    explicit = clean(intent.get('role') or intent.get('source_role') or as_dict(intent.get('policy')).get('role')).lower()
    if explicit:
        return explicit
    actor = clean(intent.get('actor') or intent.get('source_agent') or '').lower()
    if re.search('user|correction|human', actor): return 'user'
    if 'review' in actor: return 'reviewer'
    if re.search('build|implement|coder', actor): return 'builder'
    if re.search('research|evidence', actor): return 'researcher'
    if 'system' in actor: return 'system'
    if 'runtime' in actor: return 'runtime'
    return 'agent'


def _risk_level(intent: dict[str, Any]) -> str:
    payload = as_dict(intent.get('payload'))
    policy = as_dict(intent.get('policy'))
    explicit = clean(intent.get('risk') or intent.get('risk_level') or policy.get('risk') or payload.get('risk') or payload.get('risk_level')).lower()
    if explicit:
        return explicit
    text = f"{payload.get('title','')} {payload.get('canonical_text_en','')} {payload.get('text_original','')} {payload.get('reason','')}".lower()
    if re.search(r'credential|api key|secret|deployment|delete|destructive|legal advice|medical advice|investment advice|permission', text):
        return 'high'
    return 'low'


def _context_type(intent: dict[str, Any]) -> str:
    payload = as_dict(intent.get('payload'))
    policy = as_dict(intent.get('policy'))
    explicit = clean(intent.get('context_type') or intent.get('atom_type') or payload.get('atom_type') or payload.get('type') or policy.get('context_type')).lower()
    # A file target is a workspace write even if an LLM emits a loose atom_type
    # such as "task" or "event".  This keeps CooperBench-derived file
    # patches in the lock lane.  High-risk policy-like atom types remain
    # governed by their explicit type unless a concrete file target is present.
    if has_workspace_file_hint(intent):
        return 'workspace_file'
    if explicit:
        return explicit
    op = clean(intent.get('intent_type') or intent.get('op')).lower()
    return 'event' if op in APPEND_OPS else 'memory'


def infer_intent_metadata(intent: dict[str, Any] | None = None) -> dict[str, Any]:
    intent = as_dict(intent)
    payload = as_dict(intent.get('payload'))
    policy = as_dict(intent.get('policy'))
    op = clean(intent.get('intent_type') or intent.get('op') or 'assert_atom').lower()
    ctype = _context_type(intent)
    role = _source_role(intent)
    risk = _risk_level(intent)
    authority = clean(intent.get('authority') or policy.get('authority') or role).lower()
    authority_rank = AUTHORITY_RANK.get(authority, AUTHORITY_RANK.get(role, AUTHORITY_RANK['agent']))
    target = target_of(intent)
    mergeability = clean(intent.get('mergeability') or policy.get('mergeability') or payload.get('mergeability')).lower()
    if not mergeability:
        mergeability = 'commutative' if op in APPEND_OPS or ctype in APPEND_TYPES else 'patchable' if ctype in WORKSPACE_TYPES else 'semantic'
    is_append_only = op in APPEND_OPS or ctype in APPEND_TYPES or mergeability == 'commutative'
    is_state_write = op in STATE_OPS
    high_risk = risk in {'high', 'critical', 'irreversible'} or ctype in HIGH_REVIEW_TYPES
    is_retraction = op in {'retract_atom', 'retract_link'} or clean(payload.get('status')).lower() == 'retracted'
    lock_scope = canonical_workspace_lock_scope(intent=intent) or stable_hash(target)
    return {
        'op': op,
        'context_type': ctype,
        'role': role,
        'risk': risk,
        'authority': authority,
        'authority_rank': authority_rank,
        'target': target,
        'target_key': f"{target.get('kind')}:{target.get('id') or stable_hash(target)}",
        'mergeability': mergeability,
        'is_append_only': is_append_only,
        'is_state_write': is_state_write,
        'high_risk': high_risk,
        'is_retraction': is_retraction,
        'lock_scope': lock_scope,
    }


def _decision(isolation: str, commit_mode: str, reasons: list[str], metadata: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        'isolation_policy': isolation,
        'commit_mode': commit_mode,
        'lane_hint': 'slow' if commit_mode == 'review_required' else 'fast' if isolation == 'append_only_causal' else 'normal',
        'reasons': reasons,
        'context_type': metadata['context_type'],
        'operation_type': metadata['op'],
        'risk': metadata['risk'],
        'authority': metadata['authority'],
        'authority_rank': metadata['authority_rank'],
        'mergeability': metadata['mergeability'],
        'target_key': metadata['target_key'],
        'lock_scope': metadata['lock_scope'],
        **extra,
    }


def select_context_concurrency_policy(intent: dict[str, Any] | None = None, *, mode: str | PolicyMode = PolicyMode.ADAPTIVE) -> dict[str, Any]:
    metadata = infer_intent_metadata(intent)
    mode = normalize_policy_mode(mode)
    if mode == PolicyMode.UNIFORM_APPEND_ONLY:
        return _decision('append_only_causal', 'auto', ['uniform_append_only'], metadata, force_commit_mode='auto', detects_write_conflicts=False)
    if mode == PolicyMode.UNIFORM_REVIEW_GATED:
        return _decision('review_gated_serializable', 'review_required', ['uniform_review_gated'], metadata)
    if mode == PolicyMode.UNIFORM_PESSIMISTIC_LOCK:
        return _decision('pessimistic_lock', 'auto', ['uniform_pessimistic_lock'], metadata, lock_required=True, force_commit_mode='auto')
    if mode == PolicyMode.UNIFORM_SNAPSHOT_OCC:
        return _decision('snapshot_occ', 'auto', ['uniform_snapshot_occ'], metadata, force_commit_mode='auto')

    no_append = mode == PolicyMode.ADAPTIVE_NO_APPEND_ONLY
    no_authority = mode == PolicyMode.ADAPTIVE_NO_AUTHORITY_REBASE
    no_review = mode == PolicyMode.ADAPTIVE_NO_REVIEW_GATE
    no_workspace_lock = mode == PolicyMode.ADAPTIVE_NO_WORKSPACE_LOCK
    no_semantic = mode == PolicyMode.ADAPTIVE_NO_SEMANTIC_CONFLICT_DETECTION

    if metadata['is_append_only']:
        if no_append:
            return _decision('review_gated_serializable', 'review_required', ['ablation_no_append_only_fast_lane'], metadata, ablation='no_append_only_fast_lane')
        return _decision('append_only_causal', 'auto', ['append_only_or_commutative_context'], metadata, force_commit_mode='auto')
    if not no_authority and metadata['authority_rank'] >= AUTHORITY_RANK['user'] and metadata['is_retraction']:
        return _decision('authority_interrupt_rebase', 'auto', ['higher_authority_retraction_or_correction'], metadata, force_commit_mode='auto', authority_interrupt=True)
    if mode == PolicyMode.ADAPTIVE_READSET_OCC:
        # This baseline validates runtime read-set freshness only.  It should not
        # inherit WCCU/adaptive review gates for authority, semantic-operation,
        # or view obligations; otherwise it would no longer be a clean OCC
        # baseline for testing WCCU's non-freshness obligations.
        return _decision('snapshot_occ', 'auto', ['readset_occ_freshness_validation_only'], metadata, force_commit_mode='auto')
    if metadata['high_risk']:
        if no_review:
            return _decision('snapshot_occ', 'auto', ['ablation_no_review_gate_high_risk_treated_as_occ'], metadata, force_commit_mode='auto', ablation='no_review_gate')
        return _decision('review_gated_serializable', 'review_required', ['high_risk_or_policy_context'], metadata)
    if metadata['context_type'] in WORKSPACE_TYPES:
        if no_workspace_lock:
            return _decision('snapshot_occ', 'auto', ['ablation_no_workspace_lock_treated_as_occ'], metadata, force_commit_mode='auto', workspace_lock_disabled=True, ablation='no_workspace_lock')
        return _decision('pessimistic_lock', 'auto', ['workspace_or_patch_context'], metadata, lock_required=True, force_commit_mode='auto')
    return _decision('snapshot_occ', 'auto', ['default_low_risk_snapshot_occ'], metadata, force_commit_mode='auto', semantic_conflict_detection_disabled=no_semantic)


def attach_policy(intent: dict[str, Any], *, mode: str | PolicyMode = PolicyMode.ADAPTIVE) -> dict[str, Any]:
    selected = select_context_concurrency_policy(intent, mode=mode)
    requested = 'review_required' if selected['commit_mode'] == 'review_required' else 'auto'
    return {
        **intent,
        'requested_commit_mode': requested,
        'policy': {
            **as_dict(intent.get('policy')),
            **selected,
            'mode': normalize_policy_mode(mode),
            'commit_mode': requested,
        },
    }


def list_policy_modes() -> list[str]:
    return [m.value for m in PolicyMode]
