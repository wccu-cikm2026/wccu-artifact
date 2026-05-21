from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from wccu_eval.substrate.context_substrate_store import read_context_snapshot, read_context_substrate
from wccu_eval.utils import as_list, clean, estimate_tokens, stable_hash


def _tokenize(text: Any = '') -> set[str]:
    return {t for t in re.split(r'[^a-z0-9가-힣_]+', clean(text).lower()) if len(t) > 2}


def _visible_to_role(atom: dict[str, Any], role: str) -> bool:
    if atom.get('status') and atom.get('status') != 'active':
        return False
    allow = [clean(x).lower() for x in as_list(atom.get('role_allowlist')) if clean(x)]
    return not allow or clean(role).lower() in allow


def _compact_atom(atom: dict[str, Any]) -> str:
    return f"[{atom.get('atom_type')}] {atom.get('title')}: {atom.get('canonical_text_en') or atom.get('text_original') or ''}"


def _overlap_score(query_tokens: set[str], atom: dict[str, Any]) -> int:
    hay = _tokenize(f"{atom.get('atom_type','')} {atom.get('title','')} {atom.get('canonical_text_en','')} {atom.get('text_original','')} {' '.join(as_list(atom.get('tags')))}")
    score = sum(1 for tok in query_tokens if tok in hay)
    if any(clean(tag).lower() in query_tokens for tag in as_list(atom.get('tags'))):
        score += 2
    return score


def compile_projection(root_dir: str | Path, *, snapshot_id: str = '', role: str = 'agent', task_type: str = 'general_task', goal: str = '', budget_tokens: int = 1200, atom_limit: int = 24) -> dict[str, Any]:
    substrate = read_context_substrate(root_dir)
    snapshot = read_context_snapshot(root_dir, snapshot_id or substrate['snapshot_id'])
    if not snapshot:
        raise FileNotFoundError(f'snapshot not found: {snapshot_id}')
    q_tokens = _tokenize(f'{role} {task_type} {goal}')
    scored = [
        {'atom': atom, 'score': _overlap_score(q_tokens, atom)}
        for atom in as_list(snapshot.get('atoms'))
        if _visible_to_role(atom, role)
    ]
    scored.sort(key=lambda row: (-row['score'], str(row['atom'].get('id'))))
    selected: list[dict[str, Any]] = []
    tokens = 0
    for row in scored:
        atom = row['atom']
        atom_tokens = estimate_tokens(_compact_atom(atom))
        if len(selected) >= atom_limit:
            break
        if selected and tokens + atom_tokens > budget_tokens:
            continue
        selected.append(atom)
        tokens += atom_tokens
    atom_ids = {a.get('id') for a in selected}
    links = [l for l in as_list(snapshot.get('links')) if l.get('status') == 'active' and l.get('from') in atom_ids and l.get('to') in atom_ids]
    lines = [
        '[CONTEXT PROJECTION]',
        f"snapshot_id: {snapshot.get('snapshot_id')}",
        f'role: {role}',
        f'task_type: {task_type}',
        f'goal: {goal}',
        '',
        'Relevant atoms:',
    ]
    lines.extend([f'- {_compact_atom(a)}' for a in selected] or ['- none'])
    if links:
        lines.append('\nRelevant links:')
        lines.extend([f"- {l.get('from')} --{l.get('type')}--> {l.get('to')}" for l in links])
    prompt = '\n'.join([line for line in lines if line != ''])
    projection_key = f"{snapshot.get('snapshot_id')}:{role}:{task_type}:{goal}:{prompt}"
    projection_id = f"proj_{stable_hash(projection_key)}"
    return {
        'kind': 'compiled_context_projection_v1',
        'projection_id': projection_id,
        'snapshot_id': snapshot.get('snapshot_id'),
        'role': role,
        'task_type': task_type,
        'goal': goal,
        'atoms': selected,
        'links': links,
        'prompt': prompt,
        'metrics': {
            'context_tokens': estimate_tokens(prompt),
            'selected_atom_count': len(selected),
            'selected_link_count': len(links),
            'role_context_purity': (sum(1 for a in selected if _visible_to_role(a, role)) / len(selected)) if selected else 1,
        },
    }
