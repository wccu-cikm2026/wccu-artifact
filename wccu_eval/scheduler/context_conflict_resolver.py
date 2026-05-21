from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from wccu_eval.scheduler.context_concurrency_policy import PolicyMode, attach_policy, canonical_workspace_lock_scope, infer_intent_metadata, normalize_policy_mode
from wccu_eval.scheduler.target_grounder import ground_agent_results
from wccu_eval.scheduler.wccu import attach_and_verify_certificates, certificate_mode_for_policy
from wccu_eval.utils import as_dict, as_list, clean, stable_hash


def _canonical_text(intent: dict[str, Any]) -> str:
    p = as_dict(intent.get('payload'))
    return clean(p.get('canonical_text_en') or p.get('text_original') or p.get('title') or p.get('reason') or '')


def is_retraction(intent: dict[str, Any]) -> bool:
    return bool(infer_intent_metadata(intent).get('is_retraction'))


def is_state_write(intent: dict[str, Any]) -> bool:
    return bool(infer_intent_metadata(intent).get('is_state_write'))


def is_workspace_write(intent: dict[str, Any]) -> bool:
    return infer_intent_metadata(intent).get('context_type') in {'artifact_plan', 'workspace_file', 'code_file', 'patch_plan'}


def compatible_pair(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if not is_state_write(a) or not is_state_write(b):
        return True
    if is_retraction(a) != is_retraction(b):
        return False
    at, bt = _canonical_text(a), _canonical_text(b)
    if at and bt and at != bt:
        return False
    ap, bp = as_dict(a.get('payload')), as_dict(b.get('payload'))
    a_status, b_status = clean(ap.get('status')).lower(), clean(bp.get('status')).lower()
    return not (a_status and b_status and a_status != b_status)


def analyze_semantic_dependency(intents: list[dict[str, Any]]) -> dict[str, Any]:
    if len(intents) <= 1:
        return {'semantic_dependency': False, 'semantic_conflict': False, 'compatible': True}
    state_writes = [i for i in intents if is_state_write(i)]
    if len(state_writes) <= 1:
        return {'semantic_dependency': len(state_writes) == 1, 'semantic_conflict': False, 'compatible': True}
    incompatible_pairs: list[dict[str, Any]] = []
    compatible = True
    for i in range(len(state_writes)):
        for j in range(i + 1, len(state_writes)):
            if not compatible_pair(state_writes[i], state_writes[j]):
                compatible = False
                incompatible_pairs.append({
                    'left_agent': state_writes[i].get('source_agent') or state_writes[i].get('actor') or 'unknown',
                    'right_agent': state_writes[j].get('source_agent') or state_writes[j].get('actor') or 'unknown',
                    'left_text_hash': stable_hash(_canonical_text(state_writes[i]))[:8],
                    'right_text_hash': stable_hash(_canonical_text(state_writes[j]))[:8],
                    'left_is_retraction': is_retraction(state_writes[i]),
                    'right_is_retraction': is_retraction(state_writes[j]),
                })
    return {
        'semantic_dependency': True,
        'semantic_conflict': not compatible,
        'compatible': compatible,
        'state_write_count': len(state_writes),
        'incompatible_pair_count': len(incompatible_pairs),
        'incompatible_pairs': incompatible_pairs,
    }


def _with_review_required(intent: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    return {
        **intent,
        'id': intent.get('id') or f"intent_{stable_hash(reason + ':' + json.dumps(intent, sort_keys=True, ensure_ascii=False))}",
        'commit_mode': 'review_required',
        'requested_commit_mode': 'review_required',
        'policy': {**as_dict(intent.get('policy')), 'commit_mode': 'review_required', 'conflict_reason': reason, **extra},
    }


def _with_blocked(intent: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    return {
        **intent,
        'id': intent.get('id') or f"intent_{stable_hash('blocked:' + reason + ':' + json.dumps(intent, sort_keys=True, ensure_ascii=False))}",
        'commit_mode': 'blocked',
        'requested_commit_mode': 'blocked',
        'policy': {**as_dict(intent.get('policy')), 'commit_mode': 'blocked', 'conflict_reason': reason, **extra},
    }


def _group_key(intent: dict[str, Any]) -> str:
    meta = infer_intent_metadata(intent)
    policy = as_dict(intent.get('policy'))
    if policy.get('isolation_policy') == 'append_only_causal' and policy.get('detects_write_conflicts') is False:
        return f"append_only:{intent.get('id') or stable_hash(intent)}"
    if policy.get('isolation_policy') == 'pessimistic_lock':
        # Workspace writes are coordinated by a canonical file scope, not by the
        # atom that happens to carry the patch plan.  Normalize file:path.py,
        # atom:file:path.py, payload.file_path, and lock:path.py to the same key.
        scope = canonical_workspace_lock_scope(policy.get('lock_scope') or meta.get('lock_scope') or meta.get('target_key'), intent=intent)
        return f"lock:{scope or clean(meta.get('target_key'))}"
    return meta['target_key']


def _strongest_authority(intents: list[dict[str, Any]]) -> float:
    return max(float(infer_intent_metadata(i).get('authority_rank') or 0) for i in intents)


def _summarize_policies(intents: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(as_dict(i.get('policy')).get('isolation_policy', 'unknown') for i in intents))


def _decision_record(target: str, decision: str, intents: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        'target': target,
        'decision': decision,
        'count': len(intents),
        'agents': [i.get('source_agent') or i.get('actor') or 'unknown' for i in intents],
        'policy_counts': _summarize_policies(intents),
        **extra,
    }




def _summarize_wccu_metrics_from_intents(intents: list[dict[str, Any]], *, wccu_enabled: bool, certificate_mode: str) -> dict[str, Any]:
    keys = [
        'certificate_missing_count',
        'certificate_invalid_count',
        'low_target_confidence_count',
        'stale_dependency_count',
        'authority_insufficient_count',
        'authority_certificate_mismatch_count',
        'view_invalidation_count',
        'wrong_target_count',
        'weaken_rule_delta_count',
        'delta_contract_mismatch_count',
        'stale_read_validation_ignored_count',
        'authority_correction_self_dependency_tolerated_count',
        'semantic_operation_weakening_count',
        'semantic_operation_laundering_count',
        'authority_laundering_count',
        'disabled_obligation_event_count',
    ]
    totals: dict[str, Any] = {'wccu_enabled': wccu_enabled, 'certificate_mode': certificate_mode, 'events': []}
    for key in keys:
        totals[key] = 0
    if not wccu_enabled:
        return totals
    for intent in intents:
        v = as_dict(intent.get('wccu_verification'))
        metrics = as_dict(v.get('metrics'))
        for key in keys:
            totals[key] += int(metrics.get(key) or 0)
        cert = as_dict(intent.get('certificate'))
        totals['events'].append({
            'intent_id': intent.get('id'),
            'source_agent': intent.get('source_agent') or intent.get('actor'),
            'target_id': as_dict(intent.get('payload')).get('target_id') or as_dict(intent.get('payload')).get('atom_id') or as_dict(intent.get('payload')).get('id'),
            'valid': v.get('valid'),
            'action': v.get('action'),
            'requires_review': v.get('requires_review'),
            'blocked': v.get('blocked'),
            'errors': as_list(v.get('errors')),
            'warnings': as_list(v.get('warnings')),
            'delta_type': as_dict(cert.get('delta_contract')).get('delta_type'),
            'certificate_source': cert.get('source'),
            'certificate_mode': cert.get('certificate_mode'),
        })
    return totals

def _has_stale_dependency(intent: dict[str, Any]) -> bool:
    metrics = as_dict(as_dict(intent.get('wccu_verification')).get('metrics'))
    return int(metrics.get('stale_dependency_count') or 0) > 0


def _wccu_review_or_block(intent: dict[str, Any]) -> dict[str, Any]:
    v = as_dict(intent.get('wccu_verification'))
    errors = as_list(v.get('errors'))
    warnings = as_list(v.get('warnings'))
    kwargs = {'wccu_errors': errors, 'wccu_warnings': warnings, 'wccu_action': clean(v.get('action'))}
    if v.get('blocked'):
        return _with_blocked(intent, 'wccu_certificate_blocked', **kwargs)
    return _with_review_required(intent, 'wccu_certificate_review_required', **kwargs)


def _wccu_needs_intervention(intent: dict[str, Any]) -> bool:
    v = as_dict(intent.get('wccu_verification'))
    return bool(v.get('blocked') or v.get('requires_review') or (v and not v.get('valid', True)))


def _wccu_group_signal(intents: list[dict[str, Any]], *, force_review: bool = False) -> dict[str, Any]:
    """Summarize WCCU interventions for group-level lane selection.

    WCCU verification is a signal into lane selection, not a pre-filter that
    should remove an intent from its workspace/authority group.  When
    ``force_review`` is true, the group lane has already selected review for
    the affected transaction(s), so even certificates that requested ``block``
    are counted as review-routed for metric/reporting purposes.
    """
    wccu_intents = [i for i in intents if _wccu_needs_intervention(i)]
    if force_review:
        review_count = len(wccu_intents)
        blocked_count = 0
    else:
        review_count = len([i for i in wccu_intents if not as_dict(i.get('wccu_verification')).get('blocked')])
        blocked_count = len([i for i in wccu_intents if as_dict(i.get('wccu_verification')).get('blocked')])
    return {
        'intents': wccu_intents,
        'review_count': review_count,
        'blocked_count': blocked_count,
        'events': [as_dict(i.get('wccu_verification')) for i in wccu_intents],
    }




def _readset_occ_dependency_ids(intent: dict[str, Any]) -> set[str]:
    """Return runtime read-set dependencies without invoking WCCU certificates.

    This supports a stronger baseline: the harness logs concrete read-set
    witnesses and performs OCC-style freshness validation, but it does not check
    target certificates, semantic delta contracts, authority claims, or derived
    view obligations.
    """
    out: set[str] = set()
    for container in [
        as_dict(intent.get('execution_witness')),
        as_dict(intent.get('read_witness')),
        as_dict(intent.get('runtime_witness')),
        as_dict(intent.get('provenance')),
    ]:
        for field in ['read_dependencies', 'read_atoms', 'read_set', 'reads', 'read_views']:
            for row in as_list(container.get(field)):
                row = {'target_id': row} if isinstance(row, str) else as_dict(row)
                tid = clean(row.get('target_id') or row.get('atom_id') or row.get('id') or row.get('view_target_id'))
                if tid:
                    out.add(tid)
    return out


def _mutated_target_ids(intents: list[dict[str, Any]], *, excluding: dict[str, Any] | None = None) -> set[str]:
    out: set[str] = set()
    exclude_id = clean(as_dict(excluding).get('id'))
    for other in intents:
        if excluding is not None and other is excluding:
            continue
        other_id = clean(other.get('id'))
        if exclude_id and other_id and exclude_id == other_id:
            continue
        if not infer_intent_metadata(other).get('is_state_write'):
            continue
        meta = infer_intent_metadata(other)
        tid = clean(as_dict(other.get('payload')).get('target_id') or as_dict(other.get('payload')).get('atom_id') or as_dict(other.get('payload')).get('id') or as_dict(meta.get('target')).get('id'))
        if tid:
            out.add(tid)
    return out


def _has_readset_occ_stale_dependency(intent: dict[str, Any], all_intents: list[dict[str, Any]]) -> bool:
    reads = _readset_occ_dependency_ids(intent)
    if not reads:
        return False
    return bool(reads & _mutated_target_ids(all_intents, excluding=intent))


def _is_unsafe_high_risk_auto_commit(intent: dict[str, Any]) -> bool:
    """Return True only for actually unsafe high-risk auto-commits.

    High-risk user corrections/retractions are allowed to commit through the
    authority-interrupt lane. Earlier versions counted such corrections as
    unsafe merely because the payload carried high risk, which made WCCU rows
    look unsafe even when the only committed write was the user correction and
    the stale dependent write was review-routed.
    """
    policy = as_dict(intent.get('policy'))
    meta = infer_intent_metadata(intent)
    if policy.get('force_commit_mode') != 'auto' or not meta.get('high_risk'):
        return False
    if meta.get('authority_rank', 0) >= 40 and meta.get('is_retraction'):
        return False
    return True

def _oracle_stale_dependency_accepted(committable: list[dict[str, Any]], all_intents: list[dict[str, Any]], scenario: dict[str, Any] | None) -> int:
    """Evaluation-only ground-truth count for stale cross-target writes.

    WCCU modes should detect this from certificates. Non-WCCU baselines do not
    have certificates, but diagnostic scenarios can declare oracle read
    dependencies so the evaluator can count when a stale write was accepted.
    """
    scenario = as_dict(scenario)
    declared = as_dict(scenario.get('wccu_read_dependencies'))
    if not declared:
        return 0
    mutated: set[str] = set()
    for other in all_intents:
        if not infer_intent_metadata(other).get('is_state_write'):
            continue
        p = as_dict(other.get('payload'))
        tid = clean(p.get('target_id') or p.get('atom_id') or p.get('id'))
        if tid:
            mutated.add(tid)
    count = 0
    for intent in committable:
        agent = clean(intent.get('source_agent') or as_dict(intent.get('source')).get('agent_id'))
        for dep in as_list(declared.get(agent)) + as_list(declared.get('*')):
            dep_id = clean(as_dict(dep).get('target_id') or as_dict(dep).get('atom_id') or as_dict(dep).get('id'))
            if dep_id and dep_id in mutated:
                count += 1
                break
    return count

def resolve_parallel_write_intents(agent_results: list[dict[str, Any]], *, policy_mode: str | PolicyMode = PolicyMode.ADAPTIVE, scenario: dict[str, Any] | None = None, enable_target_grounding: bool = True) -> dict[str, Any]:
    mode = normalize_policy_mode(policy_mode)
    if enable_target_grounding:
        agent_results = ground_agent_results(agent_results, scenario)
    all_intents: list[dict[str, Any]] = []
    for result in as_list(agent_results):
        for raw in as_list(result.get('write_intents')):
            intent = attach_policy({
                **raw,
                'source_agent': result.get('agent_id'),
                'role': result.get('role') or raw.get('role'),
                'projection_trace': result.get('projection_trace'),
                'agent_task': result.get('agent_task'),
                'agent_output': result.get('output'),
            }, mode=mode)
            all_intents.append(intent)

    wccu_mode = certificate_mode_for_policy(mode)
    wccu_enforced = wccu_mode != 'disabled'
    readset_occ_enabled = mode == PolicyMode.ADAPTIVE_READSET_OCC
    all_intents, wccu_metrics = attach_and_verify_certificates(all_intents, scenario=scenario, enable_wccu=wccu_enforced, certificate_mode=wccu_mode)
    if mode in {PolicyMode.ADAPTIVE_WCCU_NO_READ_VALIDATION, 'adaptive_wccu_no_read_validation'}:
        # Keep WCCU events and stale-dependency metrics, but do not let stale-read
        # errors affect commit decisions. This creates a clean ablation: the
        # runtime knows a dependency is stale, but ignores that fact when choosing
        # the lane. Other certificate problems, such as wrong targets or
        # insufficient authority, still route to review.
        for intent in all_intents:
            v = as_dict(intent.get('wccu_verification'))
            raw_errors = as_list(v.get('errors'))
            stale_errors = [e for e in raw_errors if clean(as_dict(e).get('kind')) == 'stale_read_dependency']
            filtered_errors = [e for e in raw_errors if clean(as_dict(e).get('kind')) != 'stale_read_dependency']
            metrics = {**as_dict(v.get('metrics'))}
            if stale_errors and not filtered_errors:
                metrics['certificate_invalid_count'] = 0
            metrics['stale_read_validation_ignored_count'] = int(metrics.get('stale_read_validation_ignored_count') or 0) + len(stale_errors)
            filtered_warnings = as_list(v.get('warnings')) + ([{'kind': 'stale_read_dependency_ignored', 'count': len(stale_errors)}] if stale_errors else [])
            needs_review = bool(filtered_errors) or any(clean(as_dict(w).get('kind')) in {'low_target_confidence', 'authority_insufficient_for_direct_commit', 'weakening_delta_requires_review'} for w in filtered_warnings)
            intent['wccu_verification'] = {**v, 'valid': not filtered_errors, 'errors': filtered_errors, 'warnings': filtered_warnings, 'metrics': metrics, 'requires_review': needs_review, 'blocked': False, 'action': 'review_required' if needs_review else 'allow'}
        wccu_metrics = _summarize_wccu_metrics_from_intents(all_intents, wccu_enabled=wccu_enforced, certificate_mode=wccu_mode)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for intent in all_intents:
        groups[_group_key(intent)].append(intent)

    committable: list[dict[str, Any]] = []
    conflicted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    unsafe_auto_commit_count = 0
    review_burden_count = 0
    lock_conflict_count = 0
    authority_rebase_count = 0
    semantic_conflict_count = 0
    wccu_blocked_count = 0
    wccu_review_routed_count = 0
    readset_occ_review_count = 0
    readset_occ_stale_count = 0

    for target, group_intents in groups.items():
        policies = {as_dict(intent.get('policy')).get('isolation_policy', 'snapshot_occ') for intent in group_intents}
        dependency = analyze_semantic_dependency(group_intents)
        semantic_conflict = bool(dependency.get('semantic_conflict'))
        if semantic_conflict:
            semantic_conflict_count += 1

        # WCCU verification annotates intents, but it must not be used as a
        # pre-filter before group-level lane selection.  Authority rebase and
        # workspace locks are group semantics: if a stale/invalid certificate is
        # discovered in such a group, the certificate signal is composed with the
        # group lane rather than removing the invalid intent and letting the
        # remaining valid intent become a misleading single-writer commit.
        wccu_intervened = [i for i in group_intents if wccu_enforced and _wccu_needs_intervention(i)]
        readset_occ_intervened = [i for i in group_intents if readset_occ_enabled and _has_readset_occ_stale_dependency(i, all_intents)]
        readset_occ_stale_count += len(readset_occ_intervened)

        max_auth = _strongest_authority(group_intents)
        high_auth = [i for i in group_intents if float(infer_intent_metadata(i)['authority_rank']) == max_auth and as_dict(i.get('policy')).get('authority_interrupt')]
        if high_auth and len(group_intents) > len(high_auth):
            lower = [i for i in group_intents if i not in high_auth]
            committable.extend(high_auth)
            review = [_with_review_required(i, 'higher_authority_context_rebase_required', rebased_against_authority_rank=max_auth) for i in lower]
            conflicted.extend(review)
            review_burden_count += len(review)
            authority_rebase_count += len(review)
            wccu_lower = [i for i in lower if i in wccu_intervened]
            sig = _wccu_group_signal(wccu_lower, force_review=True)
            wccu_review_routed_count += int(sig['review_count'])
            wccu_blocked_count += int(sig['blocked_count'])
            decision = 'authority_interrupt_rebase_with_wccu' if wccu_lower else 'authority_interrupt_rebase'
            decisions.append(_decision_record(
                target,
                decision,
                group_intents,
                committed_high_authority=len(high_auth),
                rebased_or_reviewed=len(review),
                wccu_review_routed_count=int(sig['review_count']),
                wccu_blocked_count=int(sig['blocked_count']),
                wccu_events=sig['events'],
            ))
            continue

        if 'review_gated_serializable' in policies:
            reason = 'semantic_conflict_review_gated' if semantic_conflict else 'review_gated_policy'
            review = [_with_review_required(i, reason, semantic_conflict=semantic_conflict, semantic_dependency=bool(dependency.get('semantic_dependency'))) for i in group_intents]
            conflicted.extend(review)
            review_burden_count += len(review)
            readset_occ_review_count += len([i for i in review if i in readset_occ_intervened])
            sig = _wccu_group_signal(wccu_intervened, force_review=True)
            wccu_review_routed_count += int(sig['review_count'])
            wccu_blocked_count += int(sig['blocked_count'])
            base_decision = 'semantic_conflict_review_gated' if semantic_conflict else 'review_gated'
            decision = f"{base_decision}_with_wccu" if wccu_intervened else base_decision
            decisions.append(_decision_record(
                target,
                decision,
                review,
                **dependency,
                commit_lane='review_required',
                wccu_review_routed_count=int(sig['review_count']),
                wccu_blocked_count=int(sig['blocked_count']),
                wccu_events=sig['events'],
            ))
            continue

        if 'pessimistic_lock' in policies and len(group_intents) > 1:
            review = [_with_review_required(i, 'pessimistic_lock_contention', lock_scope=as_dict(i.get('policy')).get('lock_scope')) for i in group_intents]
            conflicted.extend(review)
            review_burden_count += len(review)
            readset_occ_review_count += len([i for i in review if i in readset_occ_intervened])
            lock_conflict_count += 1
            sig = _wccu_group_signal(wccu_intervened, force_review=True)
            wccu_review_routed_count += int(sig['review_count'])
            wccu_blocked_count += int(sig['blocked_count'])
            decisions.append(_decision_record(
                target,
                'lock_contention_review_required_with_wccu' if wccu_intervened else 'lock_contention_review_required',
                review,
                lock_scope=as_dict(review[0].get('policy')).get('lock_scope') if review else '',
                wccu_review_routed_count=int(sig['review_count']),
                wccu_blocked_count=int(sig['blocked_count']),
                wccu_events=sig['events'],
            ))
            continue

        # For all remaining cases, WCCU is allowed to intervene at the individual
        # transaction level.  This is the desired behavior for cross-target stale
        # dependencies: the dependent write is routed to review/block while an
        # independent correction can still commit.
        if wccu_intervened:
            routed = [_wccu_review_or_block(i) for i in wccu_intervened]
            conflicted.extend(routed)
            review_routed = [i for i in routed if as_dict(i.get('policy')).get('commit_mode') == 'review_required']
            blocked = [i for i in routed if as_dict(i.get('policy')).get('commit_mode') == 'blocked']
            review_burden_count += len(review_routed)
            wccu_review_routed_count += len(review_routed)
            wccu_blocked_count += len(blocked)
            decisions.append(_decision_record(
                target,
                'wccu_certificate_intervention',
                routed,
                **dependency,
                commit_lane='mixed_wccu_review_or_block',
                wccu_review_routed_count=len(review_routed),
                wccu_blocked_count=len(blocked),
                wccu_events=[as_dict(i.get('wccu_verification')) for i in wccu_intervened],
            ))
            intents = [i for i in group_intents if i not in wccu_intervened]
            if not intents:
                continue
        else:
            intents = group_intents

        if readset_occ_intervened:
            readset_routed = [_with_review_required(i, 'readset_occ_stale_dependency') for i in readset_occ_intervened if i in intents]
            if readset_routed:
                conflicted.extend(readset_routed)
                review_burden_count += len(readset_routed)
                readset_occ_review_count += len(readset_routed)
                decisions.append(_decision_record(
                    target,
                    'readset_occ_dependency_review',
                    readset_routed,
                    commit_lane='readset_occ_review',
                    readset_occ_review_count=len(readset_routed),
                ))
                intents = [i for i in intents if i not in readset_occ_intervened]
                if not intents:
                    continue

        # Recompute policy set and dependency for remaining valid intents.
        policies = {as_dict(intent.get('policy')).get('isolation_policy', 'snapshot_occ') for intent in intents}
        dependency = analyze_semantic_dependency(intents)
        semantic_conflict = bool(dependency.get('semantic_conflict'))

        if policies == {'append_only_causal'}:
            committable.extend(intents)
            decisions.append(_decision_record(target, 'append_only_auto_merge', intents))
            if mode == PolicyMode.UNIFORM_APPEND_ONLY:
                unsafe_auto_commit_count += len([i for i in intents if infer_intent_metadata(i)['is_state_write']])
            continue

        if len(intents) <= 1:
            committable.extend(intents)
            intent = intents[0] if intents else None
            if intent and _is_unsafe_high_risk_auto_commit(intent):
                unsafe_auto_commit_count += 1
            decisions.append(_decision_record(target, 'single_writer', intents))
            continue

        if mode == PolicyMode.ADAPTIVE_NO_SEMANTIC_CONFLICT_DETECTION:
            committable.extend(intents)
            unsafe = len([i for i in intents if infer_intent_metadata(i)['is_state_write']])
            unsafe_auto_commit_count += unsafe
            decisions.append(_decision_record(target, 'semantic_conflict_detection_disabled_auto_commit', intents, unsafe_auto_commit_count=unsafe))
            continue

        compatible = bool(dependency.get('compatible'))
        if compatible:
            committable.extend(intents)
            if mode == PolicyMode.ADAPTIVE_NO_WORKSPACE_LOCK and len(intents) > 1 and any(is_workspace_write(i) for i in intents):
                unsafe = len([i for i in intents if is_workspace_write(i)])
                unsafe_auto_commit_count += unsafe
                decisions.append(_decision_record(target, 'workspace_lock_disabled_auto_merge', intents, unsafe_auto_commit_count=unsafe))
            else:
                decisions.append(_decision_record(target, 'auto_merge_compatible', intents, **dependency, commit_lane='auto'))
        else:
            review = [_with_review_required(i, 'parallel_write_conflict') for i in intents]
            conflicted.extend(review)
            review_burden_count += len(review)
            decisions.append(_decision_record(target, 'conflict_review_required', review, **dependency, commit_lane='review_required'))

    stale_dependency_accepted_count = max(
        len([i for i in committable if _has_stale_dependency(i)]),
        _oracle_stale_dependency_accepted(committable, all_intents, scenario),
    )
    if stale_dependency_accepted_count:
        unsafe_auto_commit_count += stale_dependency_accepted_count
    merged = committable + conflicted
    return {
        'policy_mode': mode,
        'committable': committable,
        'conflicted': conflicted,
        'merged_intents': merged,
        'grounded_agent_results': agent_results,
        'decisions': decisions,
        'conflict_count': len([d for d in decisions if d['decision'] in {'conflict_review_required', 'semantic_conflict_review_gated', 'semantic_conflict_review_gated_with_wccu', 'lock_contention_review_required', 'lock_contention_review_required_with_wccu', 'authority_interrupt_rebase', 'authority_interrupt_rebase_with_wccu', 'wccu_certificate_intervention'}]),
        'semantic_conflict_count': semantic_conflict_count,
        'auto_merge_count': len([d for d in decisions if d['decision'] in {'auto_merge_compatible', 'append_only_auto_merge'}]),
        'lock_conflict_count': lock_conflict_count,
        'authority_rebase_count': authority_rebase_count,
        'review_burden_count': review_burden_count,
        'unsafe_auto_commit_count': unsafe_auto_commit_count,
        'stale_dependency_accepted_count': stale_dependency_accepted_count,
        'wccu_blocked_count': wccu_blocked_count,
        'wccu_review_routed_count': wccu_review_routed_count,
        'wccu_intervention_count': wccu_blocked_count + wccu_review_routed_count,
        'readset_occ_stale_count': readset_occ_stale_count,
        'readset_occ_review_count': readset_occ_review_count,
        'wccu_metrics': wccu_metrics,
        'wccu_blocked_count': wccu_blocked_count,
        'wccu_review_routed_count': wccu_review_routed_count,
        'wccu_intervention_count': wccu_blocked_count + wccu_review_routed_count,
    }
