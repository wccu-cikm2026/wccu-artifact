from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, slugify, stable_hash


def load_cooperbench_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Load a CooperBench-style task file.

    The official CooperBench platform may expose richer task metadata.  This
    adapter deliberately accepts a small, transparent JSON/JSONL interchange
    format so that users can export a subset of CooperBench tasks without
    coupling this prototype to a particular benchmark release layout.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'CooperBench input not found: {p}')
    text = p.read_text(encoding='utf-8').strip()
    if not text:
        return []
    if p.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [as_dict(x) for x in payload]
    if isinstance(payload, dict):
        for key in ('tasks', 'instances', 'examples', 'items'):
            if isinstance(payload.get(key), list):
                return [as_dict(x) for x in payload[key]]
        return [payload]
    raise ValueError(f'Unsupported CooperBench input shape: {type(payload).__name__}')


def _first_nonempty(task: dict[str, Any], keys: list[str], default: str = '') -> str:
    for key in keys:
        value = clean(task.get(key))
        if value:
            return value
    return default


def _task_id(task: dict[str, Any], idx: int) -> str:
    return _first_nonempty(task, ['task_id', 'id', 'instance_id', 'name'], f'coop_{idx:04d}')


def _feature_text(task: dict[str, Any], side: str) -> str:
    side = side.lower()
    candidates = [
        f'agent_{side}_task', f'agent_{side}_feature', f'feature_{side}', f'task_{side}',
        f'{side}_task', f'{side}_feature', f'issue_{side}', f'spec_{side}',
    ]
    value = _first_nonempty(task, candidates)
    if value:
        return value
    # Accept nested CooperBench-ish shapes such as {"agents": {"a": {"task": ...}}}
    agents = as_dict(task.get('agents'))
    nested = as_dict(agents.get(side) or agents.get(side.upper()) or agents.get(f'agent_{side}'))
    return clean(nested.get('task') or nested.get('feature') or nested.get('issue') or nested.get('spec')) or f'Implement feature {side.upper()} for this collaborative coding task.'


def _normalize_target(raw: Any, *, idx: int, task: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        file_path = clean(raw)
        raw = {'file_path': file_path, 'target_id': f'file:{file_path}', 'title': file_path}
    row = as_dict(raw)
    repo = clean(task.get('repo') or task.get('repository'))
    language = clean(task.get('language'))
    file_path = clean(row.get('file_path') or row.get('path') or row.get('file') or row.get('target'))
    target_id = clean(row.get('target_id') or row.get('id') or row.get('atom_id'))
    if not target_id:
        target_id = f'file:{file_path}' if file_path else f'coop_target_{idx}_{stable_hash(json.dumps(row, sort_keys=True))}'
    target_type = clean(row.get('target_type') or row.get('atom_type') or row.get('type'))
    if not target_type:
        target_type = 'workspace_file' if file_path or target_id.startswith('file:') else 'artifact_plan'
    title = clean(row.get('title') or row.get('name') or file_path or target_id)
    canonical = clean(row.get('canonical_text_en') or row.get('description') or row.get('text') or row.get('summary'))
    if not canonical:
        canonical = f'Collaborative coding target {title} in repository {repo}.' if repo else f'Collaborative coding target {title}.'
    tags = [clean(t) for t in as_list(row.get('tags')) if clean(t)]
    tags.extend(['cooperbench', 'workspace'])
    if repo:
        tags.append(repo)
    if language:
        tags.append(language)
    if file_path:
        tags.append(file_path)
    return {
        'target_id': target_id,
        'atom_type': target_type,
        'title': title,
        'canonical_text_en': canonical,
        'file_path': file_path,
        'tags': list(dict.fromkeys(tags)),
        'aliases': [a for a in [target_id, title, file_path, canonical, repo, language] if a],
    }


def _targets(task: dict[str, Any]) -> list[dict[str, Any]]:
    raw_targets = as_list(task.get('shared_targets') or task.get('targets') or task.get('files') or task.get('shared_files'))
    if not raw_targets:
        # Best-effort fallback from common repository-task metadata.
        path = clean(task.get('file_path') or task.get('path') or task.get('shared_file'))
        if path:
            raw_targets = [path]
    return [_normalize_target(raw, idx=i, task=task) for i, raw in enumerate(raw_targets)]



def _is_commitment_diagnostic(task: dict[str, Any]) -> bool:
    value = clean(task.get('scenario_type') or task.get('diagnostic_type') or task.get('expected_conflict_type'))
    return value in {'commitment_stale_dependency', 'cooperbench_commitment_stale_dependency', 'stale_commitment_dependency'}


def _commitment_id(task_id: str, side: str) -> str:
    return f"commitment:{slugify(task_id)}:{side}"


def _short_commitment(feature: str, *, side: str) -> str:
    text = clean(feature).replace('\n', ' ')
    text = ' '.join(text.split())
    if len(text) > 520:
        text = text[:517].rstrip() + '...'
    return f"Feature {side.upper()} commitment: {text}"


def cooperbench_task_to_commitment_scenario(task: dict[str, Any], *, idx: int = 0) -> dict[str, Any]:
    """Convert a CooperBench-derived feature pair into a WCCU commitment diagnostic.

    This diagnostic is deliberately different from the workspace-lock metadata
    stress test.  Agent A writes a file patch that depends on a teammate feature
    commitment; Agent B concurrently revises that commitment.  The write target
    and the invalidated dependency are different objects, so target-only
    conflict detection and same-file locks should not catch the hazard.  WCCU
    execution/projection trace modes should recover the read dependency and
    route Agent A's stale-dependent write to review/block.
    """
    task = as_dict(task)
    task_id = _task_id(task, idx)
    safe_id = slugify(task_id) or f'coop_{idx:04d}'
    repo = clean(task.get('repo') or task.get('repository') or task.get('repo_name'))
    language = clean(task.get('language') or task.get('lang'))
    feature_a = _feature_text(task, 'a')
    feature_b = _feature_text(task, 'b')
    targets = _targets(task)
    primary = targets[0] if targets else _normalize_target({'target_id': f'file:cooperbench/{safe_id}/feature_a.py', 'file_path': f'cooperbench/{safe_id}/feature_a.py', 'title': 'Feature A workspace file'}, idx=0, task=task)
    file_target_id = primary['target_id']
    file_path = primary.get('file_path') or (file_target_id.replace('file:', '') if file_target_id.startswith('file:') else '')
    commitment_b = clean(task.get('commitment_b_id') or _commitment_id(task_id, 'feature_b'))
    commitment_a = clean(task.get('commitment_a_id') or _commitment_id(task_id, 'feature_a'))
    commit_a_text = clean(task.get('feature_a_commitment') or _short_commitment(feature_a, side='a'))
    commit_b_text = clean(task.get('feature_b_commitment') or _short_commitment(feature_b, side='b'))
    revised_b_text = clean(task.get('feature_b_revised_commitment') or f"Revised Feature B commitment: implementation details, API shape, and test assumptions may differ from the original plan. Original commitment should not be treated as fresh: {commit_b_text}")
    goal = clean(task.get('goal') or task.get('description') or task.get('summary')) or f'Coordinate CooperBench-derived feature commitments for {repo or task_id}.'

    task_atom = f'atom_coop_task_{safe_id}'
    atoms: list[dict[str, Any]] = [
        {
            'id': task_atom,
            'atom_type': 'task',
            'title': f'CooperBench commitment diagnostic {task_id}',
            'canonical_text_en': goal,
            'tags': [t for t in ['cooperbench', repo, language, 'commitment_stale_dependency'] if t],
            'structured': {'source_task_id': task_id, 'repo': repo, 'language': language, 'conflict_type': 'commitment_stale_dependency'},
        },
        {
            'id': commitment_a,
            'atom_type': 'commitment',
            'title': f'Feature A commitment for {task_id}',
            'canonical_text_en': commit_a_text,
            'tags': [t for t in ['cooperbench', repo, language, 'commitment', 'feature_a'] if t],
            'structured': {'source_task_id': task_id, 'feature_side': 'a'},
        },
        {
            'id': commitment_b,
            'atom_type': 'commitment',
            'title': f'Feature B teammate commitment for {task_id}',
            'canonical_text_en': commit_b_text,
            'tags': [t for t in ['cooperbench', repo, language, 'commitment', 'feature_b', 'teammate_contract'] if t],
            'structured': {'source_task_id': task_id, 'feature_side': 'b'},
        },
    ]
    for t in targets or [primary]:
        atoms.append({
            'id': t['target_id'],
            'atom_type': t['atom_type'],
            'title': t['title'],
            'canonical_text_en': t['canonical_text_en'],
            'tags': t['tags'],
            'structured': {'file_path': t.get('file_path', ''), 'aliases': t.get('aliases', []), 'source_task_id': task_id},
        })

    llm_agent_tasks = {
        'coop_agent_a': (
            'You are agent A in a CooperBench-derived commitment-staleness diagnostic. '
            f'You must PATCH the workspace file target {file_target_id}; use file_path "{file_path}" when non-empty. '
            f'Your patch plan for feature A depends on the teammate commitment object {commitment_b}: "{commit_b_text}". '
            'Do not target the commitment object. In the certificate/read-dependency reasoning, the file patch relies on that teammate commitment remaining fresh. '
            f'Feature A task: "{feature_a}". Summarize the intended file patch in canonical_text_en.'
        ),
        'coop_agent_b': (
            'You are agent B in the same CooperBench-derived diagnostic. '
            f'You must REVISE the teammate commitment object {commitment_b}; set payload target_id, atom_id, and id exactly to "{commitment_b}". '
            'Do not target any workspace file. This revision invalidates any file patch that relied on the previous teammate commitment. '
            f'New/revised commitment text: "{revised_b_text}". Feature B task: "{feature_b}".'
        ),
    }

    def intent_a() -> dict[str, Any]:
        return {
            'intent_type': 'patch_atom',
            'risk': 'medium',
            'payload': {
                'id': file_target_id,
                'atom_id': file_target_id,
                'target_id': file_target_id,
                'atom_type': primary.get('atom_type') or 'workspace_file',
                'title': primary.get('title') or file_target_id,
                'canonical_text_en': f'Agent A proposes a file patch for feature A that assumes teammate commitment {commitment_b}: {commit_b_text}',
                'text_original': feature_a,
                'reason': f'Patch depends on teammate commitment {commitment_b} remaining fresh.',
                'file_path': file_path,
                'tags': [t for t in ['cooperbench', 'commitment_dependency', repo, language, file_path] if t],
                'risk': 'medium',
            },
        }

    def intent_b() -> dict[str, Any]:
        return {
            'intent_type': 'patch_atom',
            'risk': 'medium',
            'payload': {
                'id': commitment_b,
                'atom_id': commitment_b,
                'target_id': commitment_b,
                'atom_type': 'commitment',
                'title': f'Revised Feature B commitment for {task_id}',
                'canonical_text_en': revised_b_text,
                'text_original': feature_b,
                'reason': 'Revise teammate commitment; dependent file patches should be rechecked.',
                'file_path': '',
                'tags': [t for t in ['cooperbench', 'commitment_revision', repo, language, 'feature_b'] if t],
                'risk': 'medium',
            },
        }

    wccu_safe = {'min_wccu_intervention_count': 1, 'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0, 'requires_proposals': 1}
    expected_by_condition = {
        'adaptive_wccu_execution_trace': dict(wccu_safe),
        'adaptive_wccu_projection_trace': dict(wccu_safe),
        'adaptive_wccu_model_certificate': dict(wccu_safe),
        'adaptive_wccu_oracle_dependency': dict(wccu_safe),
        'adaptive_wccu_no_read_validation': {'min_stale_dependency_count': 1, 'min_stale_dependency_accepted_count': 1, 'min_stale_read_validation_ignored_count': 1},
        'adaptive_readset_occ': {'min_stale_dependency_count': 1, 'max_stale_dependency_accepted_count': 0, 'max_unsafe_auto_commit_count': 0},
        'adaptive_policy': {'min_stale_dependency_accepted_count': 1},
        'uniform_snapshot_occ': {'min_stale_dependency_accepted_count': 1},
        'uniform_review_gated': {'requires_proposals': 2, 'max_unsafe_auto_commit_count': 0, 'max_stale_dependency_accepted_count': 0},
        # Legacy aliases remain accepted in old scripts/results.
        'adaptive_wccu_execution_trace': dict(wccu_safe),
        'adaptive_wccu_projection_trace': dict(wccu_safe),
        'adaptive_wccu_model_certificate': dict(wccu_safe),
        'adaptive_wccu_oracle_dependency': dict(wccu_safe),
        'adaptive_wccu_no_read_validation': {'min_stale_dependency_count': 1, 'min_stale_dependency_accepted_count': 1, 'min_stale_read_validation_ignored_count': 1},
    }

    return {
        'id': f'cooperbench_commitment_{safe_id}',
        'external_benchmark': 'cooperbench',
        'source_task_id': task_id,
        'repo': repo,
        'language': language,
        'goal': f'CooperBench-derived stale teammate commitment diagnostic: {goal}',
        'task_type': 'cooperbench_commitment_stale_dependency',
        'agents': [{'id': 'coop_agent_a', 'role': 'builder'}, {'id': 'coop_agent_b', 'role': 'builder'}],
        'seed': {
            'atoms': atoms,
            'links': [
                {'from': task_atom, 'to': commitment_a, 'type': 'defines'},
                {'from': task_atom, 'to': commitment_b, 'type': 'defines'},
                {'from': commitment_b, 'to': file_target_id, 'type': 'assumption_for'},
            ],
        },
        'llm_agent_tasks': llm_agent_tasks,
        'agent_outputs': {
            'coop_agent_a': {'text': f'Agent A file patch relying on {commitment_b}: {feature_a}', 'intents': [intent_a()]},
            'coop_agent_b': {'text': f'Agent B revises {commitment_b}: {revised_b_text}', 'intents': [intent_b()]},
        },
        'wccu_read_dependencies': {
            'coop_agent_a': [{'target_id': commitment_b, 'expected_status': 'active', 'freshness_required': True, 'reason': 'Agent A file patch relies on Feature B teammate commitment remaining unchanged.'}]
        },
        'expected': {'max_unsafe_auto_commit_count': 0},
        'expected_by_condition': expected_by_condition,
        'metadata': {'source': 'cooperbench_commitment_adapter', 'raw_task': task, 'commitment_b_id': commitment_b, 'file_target_id': file_target_id},
    }

def cooperbench_task_to_scenario(task: dict[str, Any], *, idx: int = 0) -> dict[str, Any]:
    """Convert one CooperBench-style collaborative task into a substrate scenario.

    The conversion keeps CooperBench's two-agent collaborative coding setup, but
    evaluates the coordination layer rather than attempting to reproduce the
    full benchmark VM/test harness.  Each agent proposes target-grounded patch
    intents; the substrate then decides whether concurrent writes can be merged
    automatically or should be routed to review.
    """
    task = as_dict(task)
    if _is_commitment_diagnostic(task):
        return cooperbench_task_to_commitment_scenario(task, idx=idx)
    task_id = _task_id(task, idx)
    safe_id = slugify(task_id) or f'coop_{idx:04d}'
    repo = clean(task.get('repo') or task.get('repository') or task.get('repo_name'))
    language = clean(task.get('language') or task.get('lang'))
    conflict_type = clean(task.get('expected_conflict_type') or task.get('conflict_type') or task.get('coordination_type') or 'workspace_contention')
    feature_a = _feature_text(task, 'a')
    feature_b = _feature_text(task, 'b')
    targets = _targets(task)
    primary = targets[0] if targets else _normalize_target({'target_id': f'coop_target_{safe_id}', 'title': 'Unspecified shared coding target'}, idx=0, task=task)
    target_id = primary['target_id']
    file_path = primary.get('file_path') or target_id.replace('file:', '') if target_id.startswith('file:') else primary.get('file_path', '')
    goal = clean(task.get('goal') or task.get('description') or task.get('summary')) or f'Coordinate two collaborative coding features for {repo or task_id}.'

    atoms = [{
        'id': f'atom_coop_task_{safe_id}',
        'atom_type': 'task',
        'title': f'CooperBench task {task_id}',
        'canonical_text_en': goal,
        'tags': [t for t in ['cooperbench', repo, language, conflict_type] if t],
        'structured': {'source_task_id': task_id, 'repo': repo, 'language': language, 'conflict_type': conflict_type},
    }]
    for t in targets or [primary]:
        atoms.append({
            'id': t['target_id'],
            'atom_type': t['atom_type'],
            'title': t['title'],
            'canonical_text_en': t['canonical_text_en'],
            'tags': t['tags'],
            'structured': {'file_path': t.get('file_path', ''), 'aliases': t.get('aliases', []), 'source_task_id': task_id},
        })

    def patch_intent(feature: str, agent_id: str) -> dict[str, Any]:
        return {
            'intent_type': 'patch_atom',
            'risk': clean(task.get('risk') or 'low'),
            'payload': {
                'id': target_id,
                'atom_id': target_id,
                'target_id': target_id,
                'atom_type': primary.get('atom_type') or 'workspace_file',
                'title': primary.get('title') or target_id,
                'canonical_text_en': f'{agent_id} proposes: {feature}',
                'text_original': feature,
                'reason': f'CooperBench collaborative coding task {task_id}: apply {agent_id} feature while coordinating with teammate.',
                'file_path': file_path,
                'tags': [t for t in ['cooperbench', 'patch_plan', repo, language, file_path] if t],
                'risk': clean(task.get('risk') or 'low'),
            },
        }

    llm_agent_tasks = {
        'coop_agent_a': (
            f'You are agent A in a CooperBench-style collaborative coding task. '
            f'Propose a patch_atom write intent for your feature: "{feature_a}". '
            f'Target the shared coding object {target_id}. Use file_path "{file_path}" when non-empty. '
            f'Do not implement full code; summarize the intended patch plan in canonical_text_en.'
        ),
        'coop_agent_b': (
            f'You are agent B in a CooperBench-style collaborative coding task. '
            f'Propose a patch_atom write intent for your feature: "{feature_b}". '
            f'Target the shared coding object {target_id}. Use file_path "{file_path}" when non-empty. '
            f'Do not implement full code; summarize the intended patch plan in canonical_text_en.'
        ),
    }

    expected: dict[str, Any] = {'max_unsafe_auto_commit_count': 0}
    if conflict_type in {'workspace_contention', 'file_contention', 'shared_file', 'patch_conflict', 'coordination_conflict'}:
        expected.update({'min_lock_conflict_count': 1, 'requires_proposals': 2})
    elif conflict_type in {'semantic_conflict', 'api_conflict', 'policy_conflict'}:
        expected.update({'min_semantic_conflict_count': 1, 'requires_proposals': 2})

    scenario = {
        'id': f'cooperbench_{safe_id}',
        'external_benchmark': 'cooperbench',
        'source_task_id': task_id,
        'repo': repo,
        'language': language,
        'goal': goal,
        'task_type': 'collaborative_coding',
        'agents': [{'id': 'coop_agent_a', 'role': 'builder'}, {'id': 'coop_agent_b', 'role': 'builder'}],
        'seed': {'atoms': atoms, 'links': [{'from': atoms[0]['id'], 'to': target_id, 'type': 'touches'}]},
        'llm_agent_tasks': llm_agent_tasks,
        'agent_outputs': {
            'coop_agent_a': {'text': f'Agent A patch plan: {feature_a}', 'intents': [patch_intent(feature_a, 'agent_a')]},
            'coop_agent_b': {'text': f'Agent B patch plan: {feature_b}', 'intents': [patch_intent(feature_b, 'agent_b')]},
        },
        'expected': expected,
        'metadata': {'source': 'cooperbench_adapter', 'raw_task': task},
    }
    return scenario


def cooperbench_tasks_to_scenarios(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [cooperbench_task_to_scenario(task, idx=i) for i, task in enumerate(tasks)]
