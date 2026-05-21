from __future__ import annotations

import re
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean


def _tokens(value: str) -> set[str]:
    value = clean(value).lower()
    value = re.sub(r'ctx_[0-9a-fA-F_:-]+', ' ', value)
    value = re.sub(r'[^a-z0-9_]+', ' ', value)
    stop = {'the','a','an','to','in','of','and','or','for','with','user','users','context','atom','memory','preference','policy'}
    return {t for t in value.split() if len(t) > 2 and t not in stop}


def target_candidates_for_scenario(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for atom in as_list(as_dict(scenario.get('seed')).get('atoms')):
        atom = as_dict(atom)
        target_id = clean(atom.get('id'))
        if not target_id:
            continue
        aliases = [target_id, clean(atom.get('title')), clean(atom.get('canonical_text_en')), clean(atom.get('atom_type'))]
        aliases.extend(clean(t) for t in as_list(atom.get('tags')))
        candidates.append({
            'target_id': target_id,
            'atom_type': clean(atom.get('atom_type')),
            'title': clean(atom.get('title')),
            'canonical_text_en': clean(atom.get('canonical_text_en')),
            'tags': [clean(t) for t in as_list(atom.get('tags')) if clean(t)],
            'aliases': [a for a in aliases if a],
        })
    return candidates


def _raw_target_strings(intent: dict[str, Any]) -> list[str]:
    p = as_dict(intent.get('payload'))
    values = [
        clean(p.get('id')),
        clean(p.get('atom_id')),
        clean(p.get('target_id')),
        clean(p.get('stream_id')),
        clean(p.get('file_path')),
        clean(p.get('title')),
        clean(p.get('canonical_text_en')),
        clean(p.get('text_original')),
        clean(p.get('reason')),
    ]
    values.extend(clean(t) for t in as_list(p.get('tags')))
    return [v for v in values if v]


def _same_id(raw: str, target_id: str) -> bool:
    raw = clean(raw)
    target_id = clean(target_id)
    if not raw or not target_id:
        return False
    if raw == target_id:
        return True
    parts = [p for p in re.split(r'[:/\\\s]+', raw) if p]
    return target_id in parts or raw.endswith(':' + target_id) or raw.endswith('/' + target_id)




def _canonical_file_hint(value: str) -> str:
    value = clean(value)
    if not value:
        return ''
    if value.startswith('atom:file:'):
        value = value[len('atom:file:'):]
    elif value.startswith('file:'):
        value = value[len('file:'):]
    elif value.startswith('lock:'):
        value = value[len('lock:'):]
    value = value.replace('\\\\', '/').replace('\\', '/')
    while value.startswith('./'):
        value = value[2:]
    return value.strip('/ ')


def _file_hint_from_intent(intent: dict[str, Any]) -> str:
    p = as_dict(intent.get('payload'))
    for raw in (p.get('file_path'), p.get('path'), p.get('target_id'), p.get('atom_id'), p.get('id')):
        hint = _canonical_file_hint(clean(raw))
        if hint and (clean(raw).startswith(('file:', 'atom:file:', 'lock:')) or '/' in hint or '.' in hint):
            return hint
    return ''


def _candidate_file_hints(candidate: dict[str, Any]) -> list[str]:
    values = [clean(candidate.get('target_id')), clean(candidate.get('title'))]
    values.extend(clean(a) for a in as_list(candidate.get('aliases')))
    values.extend(clean(t) for t in as_list(candidate.get('tags')))
    hints: list[str] = []
    for value in values:
        hint = _canonical_file_hint(value)
        if hint and (value.startswith(('file:', 'atom:file:', 'lock:')) or '/' in hint or '.' in hint):
            hints.append(hint)
    return list(dict.fromkeys(hints))


def resolve_intent_target(intent: dict[str, Any], scenario: dict[str, Any], *, min_score: float = 0.24) -> dict[str, Any]:
    """Resolve an LLM-produced free-form target to a stable substrate atom id.

    This is deliberately deterministic and inspectable. It is not a semantic
    embedding model; it gives us an ablation-friendly target grounding layer
    between LLM write-intent extraction and concurrency control.
    """
    intent_type = clean(intent.get('intent_type'))
    if intent_type == 'append_event':
        return {**intent, 'target_grounding': {'method': 'skipped_append_event', 'resolved': False}}

    candidates = target_candidates_for_scenario(scenario)
    if not candidates:
        return {**intent, 'target_grounding': {'method': 'no_candidates', 'resolved': False}}

    raw_values = _raw_target_strings(intent)

    # Workspace/file writes should be grounded by their concrete file path before
    # lexical matching.  CooperBench-derived prompts contain long task text, so
    # the task atom can otherwise win an alias match even when payload.file_path
    # clearly identifies the shared file to lock.
    file_hint = _file_hint_from_intent(intent)
    if file_hint:
        for c in candidates:
            if file_hint in _candidate_file_hints(c):
                return _apply_grounding(intent, c, method='file_path_priority', score=1.0, raw_values=raw_values)

    for raw in raw_values:
        for c in candidates:
            if _same_id(raw, c['target_id']):
                return _apply_grounding(intent, c, method='exact_or_suffix_id', score=1.0, raw_values=raw_values)

    raw_blob = ' '.join(raw_values)
    raw_tokens = _tokens(raw_blob)
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for c in candidates:
        cand_blob = ' '.join(c.get('aliases') or [])
        cand_tokens = _tokens(cand_blob)
        if not cand_tokens:
            continue
        overlap = len(raw_tokens & cand_tokens)
        score = overlap / max(1, min(len(raw_tokens), len(cand_tokens)))
        # Strong bonus for distinctive domain terms such as backup_url / deployment.
        distinctive = {'backup_url', 'callback_url', 'fallback_url', 'debug_url', 'backup_email', 'deployment', 'approval', 'api', 'cache', 'status'}
        if raw_tokens & cand_tokens & distinctive:
            score += 0.2
        if score > best[0]:
            best = (score, c)
    score, candidate = best
    if candidate and score >= min_score:
        return _apply_grounding(intent, candidate, method='lexical_alias_match', score=round(score, 4), raw_values=raw_values)
    return {**intent, 'target_grounding': {'method': 'unresolved', 'resolved': False, 'score': round(score, 4), 'raw_values': raw_values, 'candidate_count': len(candidates)}}


def _apply_grounding(intent: dict[str, Any], candidate: dict[str, Any], *, method: str, score: float, raw_values: list[str]) -> dict[str, Any]:
    p = {**as_dict(intent.get('payload'))}
    before = {'id': clean(p.get('id')), 'atom_id': clean(p.get('atom_id')), 'target_id': clean(p.get('target_id'))}
    target_id = clean(candidate.get('target_id'))
    # Keep append stream IDs untouched; for state writes canonicalize target id.
    p['id'] = target_id
    p['atom_id'] = target_id
    p['target_id'] = target_id
    if not p.get('atom_type') and candidate.get('atom_type'):
        p['atom_type'] = candidate['atom_type']
    if (clean(candidate.get('atom_type')) == 'workspace_file' or target_id.startswith('file:')) and not clean(p.get('file_path')):
        hint = _canonical_file_hint(target_id) or _canonical_file_hint(clean(candidate.get('title')))
        if hint:
            p['file_path'] = hint
    return {
        **intent,
        'payload': p,
        'target_grounding': {
            'resolved': True,
            'method': method,
            'score': score,
            'raw_target': before,
            'resolved_target_id': target_id,
            'candidate_title': clean(candidate.get('title')),
            'raw_values': raw_values[:12],
        },
    }


def ground_agent_results(agent_results: list[dict[str, Any]], scenario: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not scenario:
        return agent_results
    grounded = []
    for result in as_list(agent_results):
        result = {**as_dict(result)}
        result['write_intents'] = [resolve_intent_target(as_dict(intent), scenario) for intent in as_list(result.get('write_intents'))]
        grounded.append(result)
    return grounded
