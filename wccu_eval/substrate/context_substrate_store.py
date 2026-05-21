from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wccu_eval.substrate.context_commit_lanes import classify_context_commit_lane
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, now_iso, read_json, write_json, append_jsonl, stable_hash, slugify


def snapshot_id(version: int) -> str:
    return f"ctx_{int(version):06d}"


def paths_for(root_dir: str | Path) -> dict[str, Path]:
    root = Path(root_dir)
    return {
        'root': root,
        'manifest': root / 'substrate_manifest.json',
        'operations': root / 'operations.jsonl',
        'proposals': root / 'proposals.jsonl',
        'atoms': root / 'atoms_current.json',
        'links': root / 'links_current.json',
        'invalidations': root / 'materialization_invalidations.jsonl',
        'snapshots': root / 'snapshots',
    }


def default_manifest() -> dict[str, Any]:
    ts = now_iso()
    return {
        'kind': 'context_substrate_manifest_v1',
        'created_at': ts,
        'updated_at': ts,
        'latest_version': 0,
        'latest_snapshot_id': 'ctx_000000',
        'operation_count': 0,
        'proposal_count': 0,
    }


def normalize_atom(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    row = as_dict(raw)
    atom_type = clean(row.get('atom_type') or row.get('type') or 'memory').lower()
    title = clean(row.get('title') or row.get('name') or atom_type)
    canonical = clean(row.get('canonical_text_en') or row.get('canonical') or row.get('text') or row.get('content') or '')
    atom_id = clean(row.get('id') or row.get('atom_id')) or f"atom_{slugify(atom_type)}_{stable_hash(title + ':' + canonical)}"
    created = clean(row.get('created_at')) or now_iso()
    scope = as_dict(row.get('scope'))
    return {
        'kind': 'semantic_atom_v1',
        'id': atom_id,
        'atom_type': atom_type,
        'status': clean(row.get('status') or 'active').lower(),
        'title': title,
        'canonical_text_en': canonical,
        'text_original': clean(row.get('text_original') or row.get('original') or ''),
        'structured': as_dict(row.get('structured')),
        'tags': list(dict.fromkeys([clean(x) for x in as_list(row.get('tags')) if clean(x)])),
        'scope': scope,
        'visibility': clean(row.get('visibility') or scope.get('visibility') or 'team'),
        'role_allowlist': [clean(x).lower() for x in as_list(row.get('role_allowlist') or scope.get('role_allowlist')) if clean(x)],
        'evidence_refs': [clean(x) for x in as_list(row.get('evidence_refs') or row.get('evidence')) if clean(x)],
        'source_ref': clean(row.get('source_ref') or ''),
        'confidence': float(row['confidence']) if 'confidence' in row and str(row.get('confidence')).replace('.', '', 1).isdigit() else row.get('confidence'),
        'version': int(row.get('version') or 1),
        'created_at': created,
        'updated_at': clean(row.get('updated_at')) or created,
    }


def normalize_link(raw: dict[str, Any] | None = None) -> dict[str, Any] | None:
    row = as_dict(raw)
    source = clean(row.get('from') or row.get('from_id') or row.get('source'))
    target = clean(row.get('to') or row.get('to_id') or row.get('target'))
    if not source or not target:
        return None
    rel = clean(row.get('type') or row.get('relation') or 'related_to').lower()
    link_id = clean(row.get('id') or row.get('link_id')) or f"link_{slugify(rel)}_{stable_hash(source + ':' + rel + ':' + target)}"
    created = clean(row.get('created_at')) or now_iso()
    return {
        'kind': 'semantic_link_v1',
        'id': link_id,
        'from': source,
        'to': target,
        'type': rel,
        'status': clean(row.get('status') or 'active').lower(),
        'weight': row.get('weight'),
        'metadata': as_dict(row.get('metadata')),
        'evidence_refs': [clean(x) for x in as_list(row.get('evidence_refs') or row.get('evidence')) if clean(x)],
        'version': int(row.get('version') or 1),
        'created_at': created,
        'updated_at': clean(row.get('updated_at')) or created,
    }


def ensure_context_substrate(root_dir: str | Path) -> dict[str, Path]:
    paths = paths_for(root_dir)
    ensure_dir(paths['root'])
    ensure_dir(paths['snapshots'])
    if read_json(paths['manifest'], None) is None:
        write_json(paths['manifest'], default_manifest())
    if read_json(paths['atoms'], None) is None:
        write_json(paths['atoms'], [])
    if read_json(paths['links'], None) is None:
        write_json(paths['links'], [])
    initial = paths['snapshots'] / 'ctx_000000.json'
    if read_json(initial, None) is None:
        write_json(initial, {'kind': 'context_snapshot_v1', 'snapshot_id': 'ctx_000000', 'version': 0, 'created_at': now_iso(), 'atoms': [], 'links': []})
    return paths


def read_context_substrate(root_dir: str | Path) -> dict[str, Any]:
    paths = ensure_context_substrate(root_dir)
    manifest = read_json(paths['manifest'], default_manifest())
    atoms = [normalize_atom(a) for a in as_list(read_json(paths['atoms'], []))]
    links = [l for l in (normalize_link(l) for l in as_list(read_json(paths['links'], []))) if l]
    return {
        'kind': 'context_substrate_v1',
        'root_dir': str(root_dir),
        'manifest': manifest,
        'snapshot_id': manifest.get('latest_snapshot_id') or snapshot_id(manifest.get('latest_version') or 0),
        'version': int(manifest.get('latest_version') or 0),
        'atoms': atoms,
        'links': links,
    }


def read_context_snapshot(root_dir: str | Path, sid: str = '') -> dict[str, Any] | None:
    paths = ensure_context_substrate(root_dir)
    manifest = read_json(paths['manifest'], default_manifest())
    sid = clean(sid) or manifest.get('latest_snapshot_id') or 'ctx_000000'
    return read_json(paths['snapshots'] / f'{sid}.json', None)


def write_snapshot(root_dir: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_context_substrate(root_dir)
    version = int(state.get('version') or 0)
    sid = snapshot_id(version)
    snap = {
        'kind': 'context_snapshot_v1',
        'snapshot_id': sid,
        'version': version,
        'created_at': now_iso(),
        'atom_count': len(as_list(state.get('atoms'))),
        'link_count': len(as_list(state.get('links'))),
        'atoms': as_list(state.get('atoms')),
        'links': as_list(state.get('links')),
    }
    write_json(paths['snapshots'] / f'{sid}.json', snap)
    return snap


def seed_context(root_dir: str | Path, seed: dict[str, Any]) -> dict[str, Any]:
    ensure_context_substrate(root_dir)
    atoms = [normalize_atom(a) for a in as_list(seed.get('atoms'))]
    links = [l for l in (normalize_link(l) for l in as_list(seed.get('links'))) if l]
    state = {'version': 0, 'atoms': atoms, 'links': links}
    paths = paths_for(root_dir)
    write_json(paths['atoms'], atoms)
    write_json(paths['links'], links)
    snap = write_snapshot(root_dir, state)
    manifest = default_manifest()
    manifest.update({'latest_version': 0, 'latest_snapshot_id': snap['snapshot_id']})
    write_json(paths['manifest'], manifest)
    return read_context_substrate(root_dir)


def _operation_from_intent(intent: dict[str, Any], version: int, status: str, lane_result: dict[str, Any], errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = as_dict(intent.get('payload') or intent)
    op = clean(intent.get('intent_type') or intent.get('op') or 'assert_atom').lower()
    return {
        'kind': 'context_operation_v1',
        'id': clean(intent.get('id') or intent.get('intent_id')) or f"op_{stable_hash(str(version) + ':' + json.dumps(intent, sort_keys=True, ensure_ascii=False))}",
        'op': op,
        'version': version,
        'actor': clean(intent.get('actor') or 'runtime'),
        'timestamp': now_iso(),
        'status': status,
        'lane': lane_result.get('lane'),
        'commit_mode': lane_result.get('commit_mode'),
        'lane_reasons': lane_result.get('reasons'),
        'payload': payload,
        'preconditions': as_dict(intent.get('preconditions')),
        'source': as_dict(intent.get('source')),
        'policy': as_dict(intent.get('policy')),
        'target_grounding': as_dict(intent.get('target_grounding')),
        'certificate': as_dict(intent.get('certificate')),
        'wccu_verification': as_dict(intent.get('wccu_verification')),
        'errors': errors or [],
    }


def _apply_operation(operation: dict[str, Any], atoms: list[dict[str, Any]], links: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    op = operation.get('op')
    payload = as_dict(operation.get('payload'))
    ts = operation.get('timestamp') or now_iso()
    by_atom = {a['id']: a for a in atoms}
    by_link = {l['id']: l for l in links}
    if op in {'assert_atom', 'upsert_atom', 'patch_atom'}:
        atom = normalize_atom({**payload, **as_dict(payload.get('atom'))})
        prev = by_atom.get(atom['id'], {})
        merged = {**prev, **atom}
        merged['structured'] = {**as_dict(prev.get('structured')), **as_dict(atom.get('structured'))}
        merged['tags'] = list(dict.fromkeys(as_list(prev.get('tags')) + as_list(atom.get('tags'))))
        merged['evidence_refs'] = list(dict.fromkeys(as_list(prev.get('evidence_refs')) + as_list(atom.get('evidence_refs'))))
        merged['version'] = int(prev.get('version') or 0) + 1
        merged['updated_at'] = ts
        by_atom[atom['id']] = merged
    elif op == 'retract_atom':
        atom_id = clean(payload.get('atom_id') or payload.get('id'))
        prev = by_atom.get(atom_id)
        if prev:
            by_atom[atom_id] = {**prev, 'status': 'retracted', 'retraction_reason': clean(payload.get('reason') or ''), 'version': int(prev.get('version') or 0) + 1, 'updated_at': ts}
    elif op in {'assert_link', 'upsert_link', 'patch_link', 'link'}:
        link = normalize_link({**payload, **as_dict(payload.get('link'))})
        if link:
            prev = by_link.get(link['id'], {})
            merged = {**prev, **link}
            merged['metadata'] = {**as_dict(prev.get('metadata')), **as_dict(link.get('metadata'))}
            merged['version'] = int(prev.get('version') or 0) + 1
            merged['updated_at'] = ts
            by_link[link['id']] = merged
    elif op in {'unlink', 'retract_link'}:
        link_id = clean(payload.get('link_id') or payload.get('id'))
        prev = by_link.get(link_id)
        if prev:
            by_link[link_id] = {**prev, 'status': 'retracted', 'version': int(prev.get('version') or 0) + 1, 'updated_at': ts}
    return list(by_atom.values()), list(by_link.values())


def commit_context_write_intents_batch(root_dir: str | Path, intents: list[dict[str, Any]]) -> dict[str, Any]:
    paths = ensure_context_substrate(root_dir)
    substrate = read_context_substrate(root_dir)
    atoms = list(substrate['atoms'])
    links = list(substrate['links'])
    committed_ops: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    version = int(substrate.get('version') or 0)

    for intent in intents:
        lane = classify_context_commit_lane(intent)
        version += 1
        op = _operation_from_intent(intent, version, 'pending', lane)
        if lane.get('commit_mode') == 'blocked':
            blocked = {**op, 'status': 'blocked'}
            conflicts.append(blocked)
            append_jsonl(paths['proposals'], blocked)
            continue
        if lane.get('commit_mode') == 'review_required':
            proposal = {**op, 'status': 'proposal'}
            proposals.append(proposal)
            append_jsonl(paths['proposals'], proposal)
            continue
        # In this experiment harness, merge-stage conflict decisions are encoded as review_required.
        op['status'] = 'committed'
        atoms, links = _apply_operation(op, atoms, links)
        committed_ops.append(op)
        append_jsonl(paths['operations'], op)

    final_version = int(substrate.get('version') or 0) + len(committed_ops)
    snap = write_snapshot(root_dir, {'version': final_version, 'atoms': atoms, 'links': links})
    manifest = {**as_dict(substrate.get('manifest'))}
    manifest.update({
        'updated_at': now_iso(),
        'operation_count': int(manifest.get('operation_count') or 0) + len(committed_ops),
        'proposal_count': int(manifest.get('proposal_count') or 0) + len(proposals) + len(conflicts),
        'latest_version': final_version,
        'latest_snapshot_id': snap['snapshot_id'],
    })
    write_json(paths['manifest'], manifest)
    write_json(paths['atoms'], atoms)
    write_json(paths['links'], links)
    return {
        'kind': 'context_commit_batch_result_v1',
        'committed': len(committed_ops),
        'proposals': len(proposals),
        'conflicts': len(conflicts),
        'total': len(intents),
        'snapshot_id': snap['snapshot_id'],
        'operations': committed_ops,
        'proposal_records': proposals,
        'conflict_records': conflicts,
    }
