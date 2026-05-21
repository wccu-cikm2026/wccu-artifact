from __future__ import annotations

import json
import re
from typing import Any

from wccu_eval.scheduler.context_concurrency_policy import AUTHORITY_RANK, infer_intent_metadata
from wccu_eval.scheduler.semantic_operation_verifier import verify_semantic_operation_contract
from wccu_eval.utils import as_dict, as_list, clean, stable_hash

WCCU_SCHEMA_VERSION = 'wccu_certificate_v1'

CERTIFICATE_MODE_MODEL = 'model_certificate'
CERTIFICATE_MODE_ORACLE = 'oracle_dependency'
CERTIFICATE_MODE_PROJECTION_TRACE = 'projection_trace'
CERTIFICATE_MODE_EXECUTION_TRACE = 'execution_trace'
CERTIFICATE_MODE_DISABLED = 'disabled'

DELTA_TYPES = {
    'append_event', 'append_evidence', 'upsert_atom', 'patch_atom', 'patch_memory',
    'retract_atom', 'retract_memory', 'assert_link', 'upsert_link', 'patch_link',
    'retract_link', 'patch_workspace', 'strengthen_rule', 'weaken_rule',
}

WEAKENING_TERMS = (
    'may run without', 'without explicit approval', 'relax', 'weaken',
    'disable approval', 'bypass approval', 'less restrictive', 'remove approval',
)
STRENGTHENING_TERMS = ('must', 'require', 'requires', 'explicit approval', 'prohibit', 'must not', 'shall')
REVIEW_WARNING_KINDS = {
    'low_target_confidence',
    'authority_insufficient_for_direct_commit',
    'authority_certificate_mismatch',
    'authority_required_understated',
    'weakening_delta_requires_review',
    'delta_contract_mismatch',
    'semantic_operation_weakening',
    'semantic_operation_laundering',
    'authority_laundering_detected',
}
REVIEW_ERROR_KINDS = {
    'wrong_target_certificate',
    'stale_read_dependency',
}

OBLIGATION_BY_EVENT_KIND = {
    'wrong_target_certificate': 'O-TARGET',
    'low_target_confidence': 'O-TARGET',
    'stale_read_dependency': 'O-FRESH',
    'authority_insufficient_for_direct_commit': 'O-AUTH',
    'authority_certificate_mismatch': 'O-AUTH',
    'authority_required_understated': 'O-AUTH',
    'weakening_delta_requires_review': 'O-DELTA',
    'delta_contract_mismatch': 'O-DELTA',
    'semantic_operation_weakening': 'O-DELTA',
    'semantic_operation_laundering': 'O-DELTA',
    'authority_laundering_detected': 'O-AUTH',
    'view_invalidation_required': 'O-VIEW',
}

ALL_OBLIGATIONS = {'O-TARGET', 'O-FRESH', 'O-AUTH', 'O-DELTA', 'O-VIEW'}


def _disabled_obligations(scenario: dict[str, Any] | None = None) -> set[str]:
    scenario = as_dict(scenario)
    metadata = as_dict(scenario.get('metadata'))
    raw = as_list(scenario.get('wccu_disabled_obligations')) or as_list(metadata.get('wccu_disabled_obligations'))
    out = {clean(x).upper() for x in raw if clean(x)}
    return {x for x in out if x in ALL_OBLIGATIONS}


def _filter_disabled_obligation_events(events: list[dict[str, Any]], disabled: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not disabled:
        return events, []
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for event in events:
        obligation = OBLIGATION_BY_EVENT_KIND.get(clean(as_dict(event).get('kind')))
        if obligation in disabled:
            filtered.append({**event, 'disabled_obligation': obligation})
        else:
            kept.append(event)
    return kept, filtered


EXISTENCE_STATUS_ALIASES = {'present', 'exists', 'existing', 'available', 'any'}
ACTIVE_LIKE_STATUSES = {'active', 'proposed', 'present'}

TRACE_STOP_TOKENS = {
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'into', 'onto', 'will', 'must',
    'should', 'would', 'could', 'agent', 'task', 'context', 'patch', 'write', 'update',
    'revise', 'target', 'status', 'active', 'proposed', 'memory', 'atom', 'file',
    'feature', 'commitment', 'workspace', 'cooperbench', 'scenario', 'canonical',
}


def _status_compatible(expected_status: str, actual_status: str) -> bool:
    """Return whether an expected dependency status is satisfied.

    LLM-supplied certificates often use natural terms such as ``present`` or
    ``existing`` when they mean that the dependency object should simply still
    exist in the projection.  Treating ``present`` as a literal status caused
    false stale-dependency events like ``status_changed:present->active``.  The
    verifier should only flag status mismatch when the certificate asks for a
    specific lifecycle status such as ``active`` or ``retracted``.
    """
    expected = clean(expected_status).lower()
    actual = clean(actual_status).lower()
    if not expected or actual == 'unknown':
        return True
    if expected in EXISTENCE_STATUS_ALIASES:
        return actual in ACTIVE_LIKE_STATUSES or bool(actual)
    return actual == expected


def certificate_mode_for_policy(policy_mode: str) -> str:
    mode = clean(policy_mode).lower()
    if mode in {'adaptive_wccu_oracle_dependency', 'adaptive_wccu_oracle_dependency'}:
        return CERTIFICATE_MODE_ORACLE
    if mode in {'adaptive_wccu_projection_trace', 'adaptive_wccu_projection_trace'}:
        return CERTIFICATE_MODE_PROJECTION_TRACE
    if mode in {'adaptive_wccu_execution_trace', 'adaptive_wccu_execution_trace'}:
        return CERTIFICATE_MODE_EXECUTION_TRACE
    if mode in {'adaptive_wccu', 'adaptive_wccu_model_certificate', 'adaptive_wccu_no_read_validation', 'adaptive_wccu_unguided_certificate', 'adaptive_wccu', 'adaptive_wccu_model_certificate', 'adaptive_wccu_no_read_validation', 'adaptive_wccu_unguided_certificate'}:
        return CERTIFICATE_MODE_MODEL
    return CERTIFICATE_MODE_DISABLED


def _payload_text(intent: dict[str, Any]) -> str:
    p = as_dict(intent.get('payload'))
    return ' '.join(clean(p.get(k)) for k in ['title', 'canonical_text_en', 'text_original', 'reason', 'file_path']).lower()


def _target_id(intent: dict[str, Any]) -> str:
    p = as_dict(intent.get('payload'))
    return clean(p.get('target_id') or p.get('atom_id') or p.get('id') or p.get('stream_id') or p.get('file_path'))


def _intent_atom_type(intent: dict[str, Any]) -> str:
    return clean(as_dict(intent.get('payload')).get('atom_type')).lower()


def _should_skip_trace_dependency(intent: dict[str, Any], atom: dict[str, Any]) -> bool:
    # Commitment revisions often mention the same domain terms as workspace file
    # paths (e.g. cache, metric, router) but they are not reading the file.  Do
    # not infer a workspace-file dependency for a commitment-target write merely
    # from lexical overlap; this keeps commitment-staleness diagnostics focused
    # on the teammate commitment object rather than over-reviewing the reviser.
    return _intent_atom_type(intent) == 'commitment' and clean(atom.get('atom_type')).lower() == 'workspace_file'


def _text_hash(value: str) -> str:
    return stable_hash(clean(value))[:16]


def _tokenize(value: Any) -> set[str]:
    return {
        t
        for t in re.split(r'[^a-z0-9_]+', clean(value).lower())
        if len(t) > 2 and t not in TRACE_STOP_TOKENS
    }




def _projection_atom_index(intent: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trace = as_dict(intent.get('projection_trace'))
    return {
        clean(as_dict(atom).get('id') or as_dict(atom).get('target_id')): as_dict(atom)
        for atom in as_list(trace.get('atoms'))
        if clean(as_dict(atom).get('id') or as_dict(atom).get('target_id'))
    }


def _dependency_from_atom(
    *,
    target_id: str,
    atom: dict[str, Any] | None,
    snapshot_id: str,
    view_id: str,
    reason: str,
) -> dict[str, Any]:
    atom = as_dict(atom)
    return {
        'target_id': target_id,
        'view_id': view_id,
        'snapshot_id': snapshot_id,
        'expected_status': clean(atom.get('status') or 'active'),
        'expected_text_hash': _text_hash(clean(atom.get('canonical_text_en'))) if clean(atom.get('canonical_text_en')) else '',
        'freshness_required': True,
        'reason': reason,
    }


def _read_dependencies_from_runtime_witness(intent: dict[str, Any], *, witness_field: str = 'runtime_witness') -> list[dict[str, Any]]:
    """Extract dependency obligations from runtime provenance/witness fields.

    This is the preferred source for trace-derived WCCU dependencies.  Unlike the
    lexical fallback below, it is not based on scenario oracle data or model
    self-report.  It consumes compact read witnesses emitted by the harness, for
    example::

        runtime_witness = {
          "read_atoms": ["atom_pref_backup_url"],
          "read_views": [{"target_id": "atom_policy", "view_id": "proj_..."}]
        }

    The helper is deliberately permissive about field names so experiments can
    log witnesses from retrieval, projection, tool reads, or workspace reads
    without rewriting the verifier.
    """
    trace = as_dict(intent.get('projection_trace'))
    atom_index = _projection_atom_index(intent)
    base_snapshot = clean(as_dict(intent.get('preconditions')).get('base_snapshot_id') or trace.get('snapshot_id'))
    projection_id = clean(trace.get('projection_id'))
    target_id = _target_id(intent)
    witness_containers = [
        as_dict(intent.get(witness_field)),
        as_dict(intent.get('execution_witness')),
        as_dict(intent.get('read_witness')),
        as_dict(intent.get('provenance')),
    ]
    # Some diagnostics intentionally remove explicit runtime/projection witnesses
    # while keeping an oracle dependency only for evaluation-time stale-accept
    # counting.  Do not silently recover dependencies from the projection trace
    # in that mode; otherwise ``--no-witness`` stress runs are indistinguishable
    # from full-witness runs.  Normal experiments leave this flag unset and keep
    # the compatibility path for older fixtures whose projection compiler logged
    # reads directly in the trace object.
    if not bool(intent.get('disable_projection_trace_witness') or intent.get('disable_witness_inference')):
        witness_containers.append(trace)

    witnesses = []
    for container in witness_containers:
        witnesses.extend(as_list(container.get('read_dependencies')))
        witnesses.extend(as_list(container.get('read_atoms')))
        witnesses.extend(as_list(container.get('read_set')))
        witnesses.extend(as_list(container.get('reads')))
        witnesses.extend(as_list(container.get('read_views')))

    deps_by_id: dict[str, dict[str, Any]] = {}
    for row in witnesses:
        if isinstance(row, str):
            tid = clean(row)
            row = {'target_id': tid}
        else:
            row = as_dict(row)
            tid = clean(row.get('target_id') or row.get('atom_id') or row.get('id') or row.get('view_target_id'))
        if not tid or tid == target_id:
            continue
        atom = atom_index.get(tid) or as_dict(row.get('atom'))
        deps_by_id[tid] = {
            **_dependency_from_atom(
                target_id=tid,
                atom=atom,
                snapshot_id=clean(row.get('snapshot_id') or base_snapshot),
                view_id=clean(row.get('view_id') or projection_id),
                reason=clean(row.get('reason') or f'{witness_field} read witness'),
            ),
            'expected_status': clean(row.get('expected_status') or atom.get('status') or 'active'),
            'expected_text_hash': clean(row.get('expected_text_hash') or (_text_hash(clean(atom.get('canonical_text_en'))) if clean(atom.get('canonical_text_en')) else '')),
            'freshness_required': bool(row.get('freshness_required', True)),
        }
    return list(deps_by_id.values())


def _read_dependencies_from_trace_text(intent: dict[str, Any], *, include_execution_text: bool) -> list[dict[str, Any]]:
    meta = infer_intent_metadata(intent)
    if meta.get('is_append_only') or (meta.get('authority_rank', 0) >= AUTHORITY_RANK['user'] and meta.get('is_retraction')):
        return []
    target_id = _target_id(intent)
    trace = as_dict(intent.get('projection_trace'))
    atoms = as_list(trace.get('atoms'))
    if not atoms:
        return []
    pieces = [_payload_text(intent)]
    if include_execution_text:
        pieces.extend([
            clean(intent.get('agent_task')),
            clean(intent.get('agent_output')),
            clean(as_dict(intent.get('source')).get('agent_id')),
        ])
    text_tokens = _tokenize(' '.join(pieces))
    if not text_tokens:
        return []
    deps: list[dict[str, Any]] = []
    for atom in atoms:
        atom = as_dict(atom)
        tid = clean(atom.get('id') or atom.get('target_id'))
        if not tid or tid == target_id:
            continue
        if _should_skip_trace_dependency(intent, atom):
            continue
        hay = ' '.join([clean(atom.get('title')), clean(atom.get('canonical_text_en')), ' '.join(as_list(atom.get('tags')))])
        overlap = text_tokens & _tokenize(hay)
        # Fallback text-overlap dependencies are intentionally conservative and
        # vocabulary-agnostic.  Earlier versions used hand-picked implementation
        # terms such as backup_url/cache/router; this made the method look like a
        # fixture-specific heuristic.  The preferred path is explicit runtime
        # witnesses above; text overlap is only a portable recall fallback.
        if not overlap:
            continue
        deps.append(_dependency_from_atom(
            target_id=tid,
            atom=atom,
            snapshot_id=clean(trace.get('snapshot_id') or as_dict(intent.get('preconditions')).get('base_snapshot_id')),
            view_id=clean(trace.get('projection_id')),
            reason='execution trace text-overlap dependency' if include_execution_text else 'projection trace text-overlap dependency',
        ))
    return deps[:8]

def _read_dependencies_from_oracle(intent: dict[str, Any], scenario: dict[str, Any] | None) -> list[dict[str, Any]]:
    scenario = as_dict(scenario)
    agent_id = clean(intent.get('source_agent') or as_dict(intent.get('source')).get('agent_id'))
    declared = as_dict(scenario.get('wccu_read_dependencies'))
    rows = as_list(declared.get(agent_id)) + as_list(declared.get('*'))
    deps: list[dict[str, Any]] = []
    base_snapshot = clean(as_dict(intent.get('preconditions')).get('base_snapshot_id'))
    for row in rows:
        row = as_dict(row)
        tid = clean(row.get('target_id') or row.get('atom_id') or row.get('id'))
        if not tid:
            continue
        deps.append({
            'target_id': tid,
            'view_id': clean(row.get('view_id')),
            'snapshot_id': clean(row.get('snapshot_id') or base_snapshot),
            'expected_status': clean(row.get('expected_status') or 'active'),
            'expected_text_hash': clean(row.get('expected_text_hash')),
            'freshness_required': bool(row.get('freshness_required', True)),
            'reason': clean(row.get('reason') or 'oracle fixture read dependency'),
        })
    return deps


def _read_dependencies_from_projection_trace(intent: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer read dependencies from projection/runtime witnesses.

    The primary source is an explicit read witness from the projection or
    harness.  A vocabulary-agnostic text-overlap fallback is kept only for older
    fixtures that do not log read witnesses.
    """
    witness_deps = _read_dependencies_from_runtime_witness(intent, witness_field='projection_witness')
    if witness_deps:
        return witness_deps[:8]
    if bool(intent.get('disable_trace_text_fallback') or intent.get('disable_text_fallback')):
        return []
    return _read_dependencies_from_trace_text(intent, include_execution_text=False)



def _read_dependencies_from_execution_trace(intent: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer read dependencies from execution witnesses plus fallback text trace.

    Execution-witness mode is the strongest non-oracle mode.  It first consumes
    concrete runtime read witnesses emitted by retrieval/projection/tool reads;
    if none are available, it falls back to projection plus agent task/output
    text for compatibility with older experiment fixtures.
    """
    deps_by_id: dict[str, dict[str, Any]] = {
        d['target_id']: d
        for d in _read_dependencies_from_runtime_witness(intent, witness_field='execution_witness')
        if clean(d.get('target_id'))
    }
    if deps_by_id:
        return list(deps_by_id.values())[:8]
    if bool(intent.get('disable_trace_text_fallback') or intent.get('disable_text_fallback')):
        return []
    return _read_dependencies_from_trace_text(intent, include_execution_text=True)


def _delta_contract(intent: dict[str, Any]) -> dict[str, Any]:
    op = clean(intent.get('intent_type') or 'patch_atom')
    p = as_dict(intent.get('payload'))
    text = _payload_text(intent)
    if op == 'append_event':
        delta_type = 'append_evidence' if 'evidence' in text or p.get('stream_id') else 'append_event'
        direction = 'append'
    elif op == 'retract_atom':
        delta_type = 'retract_memory' if clean(p.get('atom_type')) == 'memory' else 'retract_atom'
        direction = 'retract'
    elif any(term in text for term in WEAKENING_TERMS):
        delta_type = 'weaken_rule'
        direction = 'weaken'
    elif any(term in text for term in STRENGTHENING_TERMS):
        delta_type = 'strengthen_rule'
        direction = 'strengthen'
    elif clean(p.get('file_path')) or clean(p.get('atom_type')) in {'artifact_plan', 'workspace_file', 'code_file', 'patch_plan'}:
        delta_type = 'patch_workspace'
        direction = 'patch'
    elif clean(p.get('atom_type')) == 'memory':
        delta_type = 'patch_memory'
        direction = 'patch'
    else:
        delta_type = op if op in DELTA_TYPES else 'patch_atom'
        direction = 'patch'
    affected_views = []
    if clean(p.get('file_path')):
        affected_views.append(f"workspace:{clean(p.get('file_path'))}")
    if _target_id(intent):
        affected_views.append(f"target:{_target_id(intent)}")
    return {
        'delta_type': delta_type,
        'semantic_direction': direction,
        'affected_view_ids': affected_views[:6],
        'invalidates_views': direction in {'retract', 'weaken', 'patch'} or bool(clean(p.get('file_path'))),
        'summary': clean(p.get('canonical_text_en') or p.get('reason') or p.get('title'))[:240],
    }


def minimal_certificate(intent: dict[str, Any], *, scenario: dict[str, Any] | None = None, certificate_mode: str = CERTIFICATE_MODE_MODEL) -> dict[str, Any]:
    p = as_dict(intent.get('payload'))
    grounding = as_dict(intent.get('target_grounding'))
    target_id = clean(grounding.get('resolved_target_id') or p.get('target_id') or p.get('atom_id') or p.get('id') or p.get('stream_id') or p.get('file_path'))
    meta = infer_intent_metadata(intent)
    if certificate_mode == CERTIFICATE_MODE_ORACLE:
        read_deps = _read_dependencies_from_oracle(intent, scenario)
    elif certificate_mode == CERTIFICATE_MODE_PROJECTION_TRACE:
        read_deps = _read_dependencies_from_projection_trace(intent)
    elif certificate_mode == CERTIFICATE_MODE_EXECUTION_TRACE:
        read_deps = _read_dependencies_from_execution_trace(intent)
    else:
        read_deps = []
    cert = {
        'schema_version': WCCU_SCHEMA_VERSION,
        'certificate_id': f"wccu_{stable_hash(json.dumps({'id': intent.get('id'), 'target': target_id, 'mode': certificate_mode}, sort_keys=True))}",
        'certificate_mode': certificate_mode,
        'read_dependencies': read_deps,
        'target_certificate': {
            'claimed_target_id': target_id,
            'raw_target': clean(p.get('target_id') or p.get('id') or p.get('atom_id') or p.get('file_path') or p.get('stream_id')),
            'grounding_rationale': clean(grounding.get('method') or 'minimal_certificate_from_intent'),
            'confidence': float(grounding.get('score') if grounding.get('score') is not None else (1.0 if grounding.get('resolved') else 0.5)),
        },
        'delta_contract': _delta_contract(intent),
        'authority_certificate': {
            'actor_authority': clean(intent.get('authority') or meta.get('authority') or 'agent'),
            'required_authority': 'user' if meta.get('is_retraction') else 'reviewer' if meta.get('high_risk') or _delta_contract(intent).get('delta_type') == 'weaken_rule' else 'agent',
            'authority_rationale': 'derived from operation type, risk, delta type, and actor role',
        },
        'preconditions': {
            'base_snapshot_id': clean(as_dict(intent.get('preconditions')).get('base_snapshot_id')),
            'freshness_required': bool(read_deps),
            'no_retracted_dependencies': True,
            'min_target_confidence': 0.55,
            'requires_review_if_invalid': True,
        },
    }
    return cert


def normalize_certificate(intent: dict[str, Any], *, scenario: dict[str, Any] | None = None, certificate_mode: str = CERTIFICATE_MODE_MODEL) -> dict[str, Any]:
    raw = as_dict(intent.get('certificate') or intent.get('wccu_certificate'))
    base = minimal_certificate(intent, scenario=scenario, certificate_mode=certificate_mode)
    if not raw:
        base['source'] = 'minimal_from_intent'
        return base

    cert = {**base, **raw}
    cert['schema_version'] = clean(cert.get('schema_version') or WCCU_SCHEMA_VERSION)
    cert['certificate_id'] = clean(cert.get('certificate_id') or base['certificate_id'])
    cert['certificate_mode'] = clean(cert.get('certificate_mode') or certificate_mode)
    model_deps = [
        {
            'target_id': clean(d.get('target_id') or d.get('atom_id') or d.get('id')),
            'view_id': clean(d.get('view_id')),
            'snapshot_id': clean(d.get('snapshot_id') or base['preconditions']['base_snapshot_id']),
            'expected_status': clean(d.get('expected_status') or 'active'),
            'expected_text_hash': clean(d.get('expected_text_hash')),
            'freshness_required': bool(d.get('freshness_required', True)),
            'reason': clean(d.get('reason') or 'model-declared read dependency'),
        }
        for d in as_list(raw.get('read_dependencies'))
        if clean(as_dict(d).get('target_id') or as_dict(d).get('atom_id') or as_dict(d).get('id'))
    ]
    # Explicit modes make the source of dependencies auditable.
    if certificate_mode == CERTIFICATE_MODE_MODEL:
        cert['read_dependencies'] = model_deps
    elif certificate_mode == CERTIFICATE_MODE_ORACLE:
        # Oracle mode is an upper-bound diagnostic: fixture dependencies override
        # missing or incorrect model dependencies.
        cert['read_dependencies'] = base['read_dependencies'] or model_deps
    elif certificate_mode == CERTIFICATE_MODE_PROJECTION_TRACE:
        cert['read_dependencies'] = base['read_dependencies'] or model_deps
    elif certificate_mode == CERTIFICATE_MODE_EXECUTION_TRACE:
        cert['read_dependencies'] = base['read_dependencies'] or model_deps
    else:
        cert['read_dependencies'] = model_deps or base['read_dependencies']

    cert['target_certificate'] = {**base['target_certificate'], **as_dict(raw.get('target_certificate'))}
    cert['target_certificate']['confidence'] = float(cert['target_certificate'].get('confidence') or 0.0)
    cert['delta_contract'] = {**base['delta_contract'], **as_dict(raw.get('delta_contract'))}
    cert['delta_contract']['delta_type'] = clean(cert['delta_contract'].get('delta_type') or base['delta_contract']['delta_type'])
    cert['delta_contract']['affected_view_ids'] = [clean(x) for x in as_list(cert['delta_contract'].get('affected_view_ids')) if clean(x)]
    cert['delta_contract']['invalidates_views'] = bool(cert['delta_contract'].get('invalidates_views'))
    cert['authority_certificate'] = {**base['authority_certificate'], **as_dict(raw.get('authority_certificate'))}
    cert['preconditions'] = {**base['preconditions'], **as_dict(raw.get('preconditions'))}
    cert['preconditions']['freshness_required'] = bool(cert['preconditions'].get('freshness_required'))
    cert['preconditions']['no_retracted_dependencies'] = bool(cert['preconditions'].get('no_retracted_dependencies', True))
    cert['preconditions']['requires_review_if_invalid'] = bool(cert['preconditions'].get('requires_review_if_invalid', True))
    cert['preconditions']['min_target_confidence'] = float(cert['preconditions'].get('min_target_confidence') or 0.55)
    cert['source'] = clean(raw.get('source') or ('model_supplied' if raw else 'minimal_from_intent'))
    return cert


def _scenario_atom_status(scenario: dict[str, Any], target_id: str) -> str:
    for atom in as_list(as_dict(scenario.get('seed')).get('atoms')):
        atom = as_dict(atom)
        if clean(atom.get('id')) == target_id:
            return clean(atom.get('status') or 'active').lower()
    return 'unknown'


def _scenario_atom_by_id(scenario: dict[str, Any], target_id: str) -> dict[str, Any]:
    for atom in as_list(as_dict(scenario.get('seed')).get('atoms')):
        atom = as_dict(atom)
        if clean(atom.get('id')) == target_id:
            return atom
    return {}


def _derived_source_ids_for_view(scenario: dict[str, Any], view_id: str) -> list[str]:
    """Return canonical source objects from which a read view was materialized.

    The verifier uses this to distinguish ordinary read-set OCC from WCCU
    view-provenance checking: an agent may read only a handoff summary H, while
    H is derived from source memory M that is concurrently retracted.  Read-set
    OCC sees only H; WCCU follows the derived_from provenance and treats H as
    stale when M changes.
    """
    view_id = clean(view_id)
    if not view_id:
        return []
    sources: list[str] = []
    seed = as_dict(scenario.get('seed'))
    for link in as_list(seed.get('links')):
        link = as_dict(link)
        if clean(link.get('type')) == 'derived_from' and clean(link.get('to')) == view_id:
            src = clean(link.get('from'))
            if src:
                sources.append(src)
    atom = _scenario_atom_by_id(scenario, view_id)
    structured = as_dict(atom.get('structured'))
    for src in as_list(structured.get('derived_from')) + ([structured.get('derived_from')] if clean(structured.get('derived_from')) else []):
        src = clean(src)
        if src:
            sources.append(src)
    return list(dict.fromkeys(sources))


def _runtime_required_authority_for_intent(intent: dict[str, Any]) -> str:
    meta = infer_intent_metadata(intent)
    runtime_delta = _delta_contract(intent)
    if meta.get('is_retraction'):
        return 'user'
    if meta.get('high_risk') or clean(runtime_delta.get('delta_type')) == 'weaken_rule':
        return 'reviewer'
    return 'agent'


def _higher_authority(a: str, b: str) -> str:
    a = clean(a or 'agent')
    b = clean(b or 'agent')
    return a if AUTHORITY_RANK.get(a, 0) >= AUTHORITY_RANK.get(b, 0) else b


def _mutated_targets_by_other_intents(intent: dict[str, Any], all_intents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    this_id = clean(intent.get('id'))
    for other in all_intents:
        # Only use id equality to skip the current row when ids are present.
        # Deterministic fixtures and some hand-written tests may omit intent ids;
        # treating all empty ids as equal hides parallel mutations from WCCU.
        other_id = clean(other.get('id'))
        if this_id and other_id and other_id == this_id:
            continue
        if other is intent:
            continue
        meta = infer_intent_metadata(other)
        if not meta.get('is_state_write'):
            continue
        tid = clean(as_dict(other.get('payload')).get('target_id') or as_dict(other.get('payload')).get('atom_id') or as_dict(other.get('payload')).get('id') or meta.get('target', {}).get('id'))
        if tid:
            out.setdefault(tid, []).append(other)
    return out



def _is_high_authority_self_target_correction_read(
    *,
    intent: dict[str, Any],
    dependency_target_id: str,
    actual_target_id: str,
    mutated_by_others: dict[str, list[dict[str, Any]]],
) -> bool:
    """Return whether a stale-looking self-target read should be tolerated.

    User corrections commonly read the very atom they retract or correct.  If a
    lower-authority agent concurrently patches that same atom, a purely
    freshness-based WCCU check can make the *correction* look stale.  That is the
    wrong safety interpretation: the high-authority correction is the write that
    should interrupt/rebase the lower-authority stale patch.  We therefore do not
    treat a self-target read on a high-authority retraction/correction as a stale
    dependency when all parallel mutations of that target are lower-authority.

    This exception is deliberately narrow.  It only applies to self-target reads
    by user-or-higher retractions/corrections, and only against lower-authority
    concurrent mutations.  Lower-authority stale writes that read the same target
    still fail WCCU freshness verification.
    """
    dep_tid = clean(dependency_target_id)
    actual_tid = clean(actual_target_id)
    if not dep_tid or dep_tid != actual_tid:
        return False
    meta = infer_intent_metadata(intent)
    if not (meta.get('is_retraction') and float(meta.get('authority_rank') or 0) >= AUTHORITY_RANK['user']):
        return False
    others = as_list(mutated_by_others.get(dep_tid))
    if not others:
        return False
    this_rank = float(meta.get('authority_rank') or 0)
    return all(float(infer_intent_metadata(other).get('authority_rank') or 0) < this_rank for other in others)

def _verification_action(errors: list[dict[str, Any]], warnings: list[dict[str, Any]], cert: dict[str, Any]) -> str:
    if errors:
        return 'review_required' if bool(as_dict(cert.get('preconditions')).get('requires_review_if_invalid', True)) else 'blocked'
    warning_kinds = {clean(as_dict(w).get('kind')) for w in warnings}
    if warning_kinds & REVIEW_WARNING_KINDS:
        return 'review_required'
    return 'allow'


def verify_certificate(intent: dict[str, Any], *, all_intents: list[dict[str, Any]], scenario: dict[str, Any] | None = None, certificate_mode: str = CERTIFICATE_MODE_MODEL) -> dict[str, Any]:
    scenario = as_dict(scenario)
    cert = normalize_certificate(intent, scenario=scenario, certificate_mode=certificate_mode)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics = {
        'certificate_missing_count': 1 if cert.get('source') == 'minimal_from_intent' else 0,
        'certificate_invalid_count': 0,
        'low_target_confidence_count': 0,
        'stale_dependency_count': 0,
        'authority_insufficient_count': 0,
        'authority_certificate_mismatch_count': 0,
        'view_invalidation_count': 1 if cert.get('delta_contract', {}).get('invalidates_views') else 0,
        'wrong_target_count': 0,
        'weaken_rule_delta_count': 1 if clean(as_dict(cert.get('delta_contract')).get('delta_type')) == 'weaken_rule' else 0,
        'delta_contract_mismatch_count': 0,
        'authority_correction_self_dependency_tolerated_count': 0,
        'semantic_operation_weakening_count': 0,
        'semantic_operation_laundering_count': 0,
        'authority_laundering_count': 0,
        'disabled_obligation_event_count': 0,
    }

    tc = as_dict(cert.get('target_certificate'))
    claimed_target = clean(tc.get('claimed_target_id'))
    actual_target = _target_id(intent)
    confidence = float(tc.get('confidence') or 0.0)
    min_conf = float(as_dict(cert.get('preconditions')).get('min_target_confidence') or 0.55)
    if actual_target and claimed_target and actual_target != claimed_target:
        metrics['wrong_target_count'] += 1
        errors.append({'kind': 'wrong_target_certificate', 'claimed_target_id': claimed_target, 'actual_target_id': actual_target})
    if confidence < min_conf and clean(intent.get('intent_type')) != 'append_event':
        metrics['low_target_confidence_count'] += 1
        warnings.append({'kind': 'low_target_confidence', 'confidence': confidence, 'min_target_confidence': min_conf, 'target_id': claimed_target or actual_target})

    meta = infer_intent_metadata(intent)
    semantic_contract = verify_semantic_operation_contract(
        intent,
        target_atom=_scenario_atom_by_id(scenario, actual_target),
        claimed_delta_type=clean(as_dict(cert.get('delta_contract')).get('delta_type')),
    )
    for sem_event in as_list(semantic_contract.get('events')):
        kind = clean(as_dict(sem_event).get('kind'))
        warnings.append(sem_event)
        if kind == 'semantic_operation_weakening':
            metrics['semantic_operation_weakening_count'] += 1
        elif kind == 'semantic_operation_laundering':
            metrics['semantic_operation_laundering_count'] += 1
        elif kind == 'authority_laundering_detected':
            metrics['authority_laundering_count'] += 1

    actor_auth = clean(as_dict(cert.get('authority_certificate')).get('actor_authority') or meta.get('authority') or 'agent')
    actual_actor_auth = clean(meta.get('authority') or 'agent')
    # Many harness roles (planner, analyst, worker) are subtypes of agent but do
    # not appear in the small authority lattice.  Treat unknown role labels as
    # agent authority rather than as rank zero; only flag certificates that
    # over-claim above the runtime-derived authority.
    if actual_actor_auth not in AUTHORITY_RANK:
        actual_actor_auth = 'agent'
    if actor_auth not in AUTHORITY_RANK:
        actor_auth = 'agent'
    claimed_required_auth = clean(as_dict(cert.get('authority_certificate')).get('required_authority') or 'agent')
    runtime_required_auth = _higher_authority(_runtime_required_authority_for_intent(intent), clean(semantic_contract.get('required_authority') or 'agent'))
    required_auth = _higher_authority(claimed_required_auth, runtime_required_auth)
    if AUTHORITY_RANK.get(claimed_required_auth, 0) < AUTHORITY_RANK.get(runtime_required_auth, 0):
        metrics['authority_certificate_mismatch_count'] += 1
        warnings.append({'kind': 'authority_required_understated', 'claimed_required_authority': claimed_required_auth, 'runtime_required_authority': runtime_required_auth})
    if AUTHORITY_RANK.get(actor_auth, 0) > AUTHORITY_RANK.get(actual_actor_auth, 0):
        metrics['authority_certificate_mismatch_count'] += 1
        warnings.append({'kind': 'authority_certificate_mismatch', 'claimed_actor_authority': actor_auth, 'actual_actor_authority': actual_actor_auth})
    effective_actor_auth = actual_actor_auth if AUTHORITY_RANK.get(actor_auth, 0) > AUTHORITY_RANK.get(actual_actor_auth, 0) else actor_auth
    if AUTHORITY_RANK.get(effective_actor_auth, 0) < AUTHORITY_RANK.get(required_auth, 0):
        metrics['authority_insufficient_count'] += 1
        warnings.append({'kind': 'authority_insufficient_for_direct_commit', 'actor_authority': effective_actor_auth, 'required_authority': required_auth})

    delta_type = clean(as_dict(cert.get('delta_contract')).get('delta_type'))
    runtime_delta_type = clean(_delta_contract(intent).get('delta_type'))
    mismatch_sensitive = {'weaken_rule', 'retract_memory', 'retract_atom', 'patch_workspace', 'append_evidence', 'append_event'}
    if delta_type and runtime_delta_type and delta_type != runtime_delta_type and (delta_type in mismatch_sensitive or runtime_delta_type in mismatch_sensitive):
        metrics['delta_contract_mismatch_count'] += 1
        warnings.append({'kind': 'delta_contract_mismatch', 'claimed_delta_type': delta_type, 'runtime_delta_type': runtime_delta_type})
    if delta_type == 'weaken_rule' or runtime_delta_type == 'weaken_rule':
        warnings.append({'kind': 'weakening_delta_requires_review', 'delta_type': delta_type or runtime_delta_type})
    if bool(as_dict(cert.get('delta_contract')).get('invalidates_views')):
        warnings.append({'kind': 'view_invalidation_required', 'affected_view_ids': as_list(as_dict(cert.get('delta_contract')).get('affected_view_ids'))})

    mutated_by_others = _mutated_targets_by_other_intents(intent, all_intents)
    invalidated_by_scenario = set(clean(x) for x in (as_list(scenario.get('wccu_invalidated_targets')) + as_list(scenario.get('wccu_invalidated_targets'))) if clean(x))
    for dep in as_list(cert.get('read_dependencies')):
        dep = as_dict(dep)
        tid = clean(dep.get('target_id'))
        if not tid:
            continue
        stale = False
        reasons: list[str] = []
        if tid in mutated_by_others:
            stale = True
            reasons.append('dependency_mutated_by_parallel_intent')
        if tid in invalidated_by_scenario:
            stale = True
            reasons.append('scenario_marked_dependency_invalidated')
        for src in _derived_source_ids_for_view(scenario, tid):
            if src in mutated_by_others:
                stale = True
                reasons.append(f'derived_view_source_mutated:{src}')
            if src in invalidated_by_scenario:
                stale = True
                reasons.append(f'derived_view_source_invalidated:{src}')
            src_status = _scenario_atom_status(scenario, src)
            if not _status_compatible('active', src_status):
                stale = True
                reasons.append(f'derived_view_source_status_changed:{src}:{src_status}')
        expected_status = clean(dep.get('expected_status')).lower()
        actual_status = _scenario_atom_status(scenario, tid)
        if not _status_compatible(expected_status, actual_status):
            stale = True
            reasons.append(f'status_changed:{expected_status}->{actual_status}')
        if stale and bool(dep.get('freshness_required', True)):
            if _is_high_authority_self_target_correction_read(
                intent=intent,
                dependency_target_id=tid,
                actual_target_id=actual_target,
                mutated_by_others=mutated_by_others,
            ):
                metrics['authority_correction_self_dependency_tolerated_count'] += 1
                continue
            metrics['stale_dependency_count'] += 1
            errors.append({'kind': 'stale_read_dependency', 'target_id': tid, 'reasons': reasons, 'reason': clean(dep.get('reason'))})

    disabled = _disabled_obligations(scenario)
    raw_errors, raw_warnings = list(errors), list(warnings)
    errors, disabled_errors = _filter_disabled_obligation_events(errors, disabled)
    warnings, disabled_warnings = _filter_disabled_obligation_events(warnings, disabled)
    disabled_events = disabled_errors + disabled_warnings
    metrics['disabled_obligation_event_count'] = len(disabled_events)

    valid = not errors
    if errors:
        metrics['certificate_invalid_count'] = 1
    else:
        metrics['certificate_invalid_count'] = 0
    action = _verification_action(errors, warnings, cert)
    obligation_failures = sorted({OBLIGATION_BY_EVENT_KIND.get(clean(as_dict(e).get('kind'))) for e in errors + warnings if OBLIGATION_BY_EVENT_KIND.get(clean(as_dict(e).get('kind')))})
    return {
        'valid': valid,
        'action': action,
        'requires_review': action == 'review_required',
        'blocked': action == 'blocked',
        'certificate': cert,
        'errors': errors,
        'warnings': warnings,
        'obligation_failures': obligation_failures,
        'disabled_obligations': sorted(disabled),
        'disabled_events': disabled_events,
        'raw_errors': raw_errors,
        'raw_warnings': raw_warnings,
        'metrics': metrics,
    }


def attach_and_verify_certificates(
    intents: list[dict[str, Any]],
    *,
    scenario: dict[str, Any] | None = None,
    enable_wccu: bool = True,
    certificate_mode: str = CERTIFICATE_MODE_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not enable_wccu:
        return intents, {
            'wccu_enabled': False,
            'certificate_mode': CERTIFICATE_MODE_DISABLED,
            'certificate_invalid_count': 0,
            'low_target_confidence_count': 0,
            'stale_dependency_count': 0,
            'authority_insufficient_count': 0,
            'authority_certificate_mismatch_count': 0,
            'view_invalidation_count': 0,
            'wrong_target_count': 0,
            'weaken_rule_delta_count': 0,
            'delta_contract_mismatch_count': 0,
            'authority_correction_self_dependency_tolerated_count': 0,
            'certificate_missing_count': 0,
            'semantic_operation_weakening_count': 0,
            'semantic_operation_laundering_count': 0,
            'authority_laundering_count': 0,
            'disabled_obligation_event_count': 0,
            'events': [],
        }
    annotated: list[dict[str, Any]] = []
    totals = {
        'wccu_enabled': True,
        'certificate_mode': certificate_mode,
        'certificate_invalid_count': 0,
        'low_target_confidence_count': 0,
        'stale_dependency_count': 0,
        'authority_insufficient_count': 0,
        'authority_certificate_mismatch_count': 0,
        'view_invalidation_count': 0,
        'wrong_target_count': 0,
        'weaken_rule_delta_count': 0,
        'delta_contract_mismatch_count': 0,
        'authority_correction_self_dependency_tolerated_count': 0,
        'certificate_missing_count': 0,
        'semantic_operation_weakening_count': 0,
        'semantic_operation_laundering_count': 0,
        'authority_laundering_count': 0,
        'disabled_obligation_event_count': 0,
        'events': [],
    }
    for intent in intents:
        result = verify_certificate(intent, all_intents=intents, scenario=scenario, certificate_mode=certificate_mode)
        for key, value in result['metrics'].items():
            totals[key] = int(totals.get(key, 0)) + int(value)
        event = {
            'intent_id': intent.get('id'),
            'source_agent': intent.get('source_agent') or intent.get('actor'),
            'target_id': _target_id(intent),
            'valid': result['valid'],
            'action': result['action'],
            'requires_review': result['requires_review'],
            'blocked': result['blocked'],
            'errors': result['errors'],
            'warnings': result['warnings'],
            'obligation_failures': result.get('obligation_failures', []),
            'disabled_obligations': result.get('disabled_obligations', []),
            'disabled_events': result.get('disabled_events', []),
            'delta_type': result['certificate'].get('delta_contract', {}).get('delta_type'),
            'certificate_source': result['certificate'].get('source'),
            'certificate_mode': result['certificate'].get('certificate_mode'),
        }
        totals['events'].append(event)
        verification_payload = {
            'valid': result['valid'],
            'action': result['action'],
            'requires_review': result['requires_review'],
            'blocked': result['blocked'],
            'errors': result['errors'],
            'warnings': result['warnings'],
            'obligation_failures': result.get('obligation_failures', []),
            'disabled_obligations': result.get('disabled_obligations', []),
            'disabled_events': result.get('disabled_events', []),
            'metrics': result['metrics'],
        }
        annotated.append({
            **intent,
            'certificate': result['certificate'],
            'wccu_verification': verification_payload,
        })
    return annotated, totals
