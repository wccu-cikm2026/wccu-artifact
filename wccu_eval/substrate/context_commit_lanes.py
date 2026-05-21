from __future__ import annotations

import re
from typing import Any

from wccu_eval.utils import as_dict, clean

FAST_OPS = {'append_event', 'record_usage', 'record_activity', 'invalidate_materialization'}
SLOW_OPS = {'delete_atom', 'destructive_write', 'activate_learned_rule', 'canonical_memory_switch', 'publish_package'}
REVIEW_ATOM_TYPES = {
    'learned_rule', 'skill_candidate', 'agent_package', 'external_claim',
    'financial_claim', 'legal_claim', 'medical_claim', 'permission_policy',
}


def classify_context_commit_lane(intent: dict[str, Any] | None = None) -> dict[str, Any]:
    row = as_dict(intent)
    payload = as_dict(row.get('payload'))
    op = clean(row.get('intent_type') or row.get('op') or row.get('operation') or 'assert_atom').lower()
    policy = as_dict(row.get('policy'))
    requested = clean(row.get('commit_mode')).lower()
    # Normalized write-intents sometimes carry commit_mode='none' from an
    # LLM schema while the resolver has already routed the update to review or
    # block via requested_commit_mode / policy.commit_mode.  Treat 'none' as
    # absent so resolver decisions cannot be accidentally committed at the
    # substrate stage.
    if requested in {'', 'none', 'null'}:
        requested = clean(row.get('requested_commit_mode') or policy.get('commit_mode')).lower()
    risk = clean(row.get('risk') or row.get('risk_level') or policy.get('risk')).lower()
    atom_type = clean(payload.get('atom_type') or row.get('atom_type')).lower()
    text = f"{payload.get('title','')} {payload.get('canonical_text_en','')} {payload.get('text_original','')}".lower()

    if requested in {'blocked', 'block', 'reject'}:
        return {'lane': 'blocked', 'commit_mode': 'blocked', 'reasons': ['requested_block']}
    if requested in {'proposal', 'review_required'}:
        return {'lane': 'slow', 'commit_mode': 'review_required', 'reasons': ['requested_review']}

    force_auto = clean(row.get('force_commit_mode') or policy.get('force_commit_mode')).lower() == 'auto'
    if force_auto or requested == 'auto':
        return {'lane': 'fast' if op in FAST_OPS else 'normal', 'commit_mode': 'auto', 'reasons': ['policy_force_auto' if force_auto else 'requested_auto']}

    slow_reasons: list[str] = []
    if op in SLOW_OPS:
        slow_reasons.append(f'slow_op:{op}')
    if risk in {'high', 'critical'}:
        slow_reasons.append(f'risk:{risk}')
    if atom_type in REVIEW_ATOM_TYPES:
        slow_reasons.append(f'atom_type:{atom_type}')
    if re.search(r'credential|api key|deployment|legal advice|medical advice|investment advice|private memory', text):
        slow_reasons.append('sensitive_text')
    if slow_reasons:
        return {'lane': 'slow', 'commit_mode': 'review_required', 'reasons': slow_reasons}

    if op in FAST_OPS or atom_type in {'event', 'usage_event'}:
        return {'lane': 'fast', 'commit_mode': 'auto', 'reasons': [f'fast_op:{op}']}
    return {'lane': 'normal', 'commit_mode': 'auto', 'reasons': ['normal_low_risk']}
