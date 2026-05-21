from __future__ import annotations

import difflib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from wccu_eval.agents.llm_agent import call_llm_provider
from wccu_eval.agents.llm_output_schema import parse_json_object_from_text
from wccu_eval.e2e.patch_test_runner import _apply_unified_patch, _materialize_repo, _patch_text, _load_rows
from wccu_eval.env import load_dotenv
from wccu_eval.utils import append_jsonl, as_dict, as_list, clean, deep_clone, ensure_dir, estimate_tokens, now_iso, stable_hash, write_json


def _task_id(task: dict[str, Any], idx: int) -> str:
    return clean(task.get('task_id') or task.get('id') or f'e2e_{idx:04d}')


def _task_type(task: dict[str, Any]) -> str:
    scenario = as_dict(task.get('scenario'))
    value = clean(task.get('task_type') or scenario.get('kind'))
    aliases = {'workspace_lock': 'shared_file_lock'}
    return aliases.get(value, value or 'unknown')


def _truncate(text: str, limit: int = 3000) -> str:
    text = text or ''
    if len(text) <= limit:
        return text
    head = text[: max(0, limit // 2)]
    tail = text[-max(0, limit // 2) :]
    return head + '\n# ... [truncated] ...\n' + tail




def _normalize_edit_path(path: str) -> str:
    value = clean(path).replace('\\', '/')
    if value.startswith('a/') or value.startswith('b/'):
        value = value[2:]
    value = value.lstrip('/')
    parts = [p for p in value.split('/') if p not in {'', '.'}]
    if any(p == '..' for p in parts):
        raise ValueError(f'Unsafe edit path: {path}')
    if not parts:
        raise ValueError('Edit path is empty')
    return '/'.join(parts)


def _extract_file_edits(provider_text: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    meta: dict[str, Any] = {}
    parsed = parse_json_object_from_text(provider_text)
    meta = as_dict(parsed)
    raw_edits = meta.get('edits') or meta.get('file_edits') or meta.get('files') or []
    edits: list[dict[str, str]] = []
    for item in as_list(raw_edits):
        d = as_dict(item)
        path = clean(d.get('path') or d.get('file') or d.get('filename'))
        if not path:
            continue
        if d.get('content') is not None:
            content = str(d.get('content'))
        elif d.get('final_content') is not None:
            content = str(d.get('final_content'))
        elif d.get('new_content') is not None:
            content = str(d.get('new_content'))
        else:
            continue
        edits.append({'path': _normalize_edit_path(path), 'content': content if content.endswith('\n') else content + '\n'})
    return edits, meta


def _base_files_for_patch_materialization(task: dict[str, Any]) -> dict[str, str]:
    return { _normalize_edit_path(str(k)): str(v) for k, v in as_dict(task.get('base_files')).items() }


def _single_file_diff(path: str, before: str, after: str, *, existed: bool) -> str:
    if before == after:
        return ''
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    old_name = f'a/{path}' if existed else '/dev/null'
    new_name = f'b/{path}'
    diff_lines = list(difflib.unified_diff(before_lines, after_lines, fromfile=old_name, tofile=new_name, lineterm=''))
    normalized: list[str] = []
    for line in diff_lines:
        if not line.endswith('\n'):
            line += '\n'
        normalized.append(line)
    header = f'diff --git a/{path} b/{path}\n'
    if not existed:
        header += 'new file mode 100644\n'
    return header + ''.join(normalized)


def _materialize_edits_to_patch(task: dict[str, Any], edits: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    base_files = _base_files_for_patch_materialization(task)
    parts: list[str] = []
    target_files: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for edit in edits:
        try:
            path = _normalize_edit_path(edit.get('path', ''))
            if path in seen:
                errors.append(f'duplicate_edit:{path}')
                continue
            seen.add(path)
            after = str(edit.get('content', ''))
            if not after.endswith('\n'):
                after += '\n'
            existed = path in base_files
            before = base_files.get(path, '')
            patch = _single_file_diff(path, before, after, existed=existed)
            if patch:
                parts.append(patch)
                target_files.append(path)
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
    return '\n'.join(parts), {'edit_count': len(edits), 'target_files': target_files, 'materialization_errors': errors, 'materialized_nonempty': bool(parts)}
def _base_file_block(task: dict[str, Any], *, max_chars_per_file: int = 3000, max_total_chars: int = 24000) -> str:
    base_files = as_dict(task.get('base_files'))
    parts: list[str] = []
    total = 0
    for path in sorted(base_files):
        text = str(base_files[path])
        snippet = _truncate(text, max_chars_per_file)
        block = f"### {path}\n```\n{snippet}\n```\n"
        if total + len(block) > max_total_chars:
            parts.append('# ... additional files omitted for prompt budget ...\n')
            break
        parts.append(block)
        total += len(block)
    return '\n'.join(parts)


def _agent_task_text(task: dict[str, Any], patch: dict[str, Any], *, agent_id: str) -> str:
    explicit = clean(patch.get('llm_task') or patch.get('agent_task') or patch.get('instruction'))
    if explicit:
        return explicit
    task_type = clean(task.get('task_type') or as_dict(task.get('scenario')).get('kind'))
    desc = clean(patch.get('description'))
    targets = ', '.join(clean(x) for x in as_list(patch.get('workspace_targets')) if clean(x))
    if task_type == 'independent':
        return f"Implement this independent feature and include minimal tests. Feature: {desc}. Target files: {targets}."
    if task_type == 'shared_file_lock':
        return f"Implement this feature in the shared file and include minimal tests. Feature: {desc}. Target files: {targets}."
    if task_type == 'commitment_staleness':
        if patch.get('depends_on'):
            return f"Implement the client-side feature and include a test. Assume the teammate commitment you read is still valid. Feature: {desc}. Target files: {targets}."
        if patch.get('revises'):
            return f"Revise the API contract as described and preserve the existing base tests. Include only necessary changes. Feature: {desc}. Target files: {targets}."
    if task_type == 'target_ambiguity':
        return f"Resolve the ambiguous target using the target candidates and implement the requested config change with a test. Feature: {desc}. Target files: {targets}."
    return desc or f'Implement the requested patch for agent {agent_id}. Target files: {targets}.'




def build_llm_file_edit_prompt(*, task: dict[str, Any], patch: dict[str, Any], idx: int) -> str:
    task_id = _task_id(task, idx)
    agent_id = clean(patch.get('agent_id') or patch.get('agent') or 'agent')
    scenario = as_dict(task.get('scenario'))
    task_type = clean(task.get('task_type') or scenario.get('kind') or 'unknown')
    targets = [clean(x) for x in as_list(patch.get('workspace_targets') or patch.get('targets')) if clean(x)]
    deps = [clean(x if isinstance(x, str) else as_dict(x).get('target_id') or as_dict(x).get('id')) for x in as_list(patch.get('depends_on') or patch.get('read_dependencies'))]
    revises = [clean(x if isinstance(x, str) else as_dict(x).get('target_id') or as_dict(x).get('id')) for x in as_list(patch.get('revises') or patch.get('invalidates') or patch.get('updates_commitments'))]
    candidates = as_list(patch.get('target_candidates') or scenario.get('target_candidates'))
    task_text = _agent_task_text(task, patch, agent_id=agent_id)
    files = _base_file_block(task)
    return f"""
You are agent {agent_id} in a controlled multi-agent patch/test evaluation.
Generate a minimal FILE-EDIT proposal for the repository files below.

Task id: {task_id}
Task type: {task_type}
Your objective:
{task_text}

Workspace targets you should edit if possible: {json.dumps(targets, ensure_ascii=False)}
Read/commitment dependencies you rely on: {json.dumps([d for d in deps if d], ensure_ascii=False)}
Commitments you revise or invalidate: {json.dumps([r for r in revises if r], ensure_ascii=False)}
Target candidates, if any: {json.dumps(candidates, ensure_ascii=False)}

Repository files:
{files}

Return exactly one JSON object with these fields:
{{
  "edits": [
    {{"path": "relative/path.py", "content": "complete final file content after your edit"}}
  ],
  "notes": "short explanation",
  "target_files": ["relative/path.py"],
  "read_dependencies": ["dependency ids you actually used"]
}}

Rules:
- Return JSON only. No Markdown fences.
- Do NOT return a unified diff or apply_patch block.
- For every changed file, include the complete final file content, not a fragment.
- Use paths exactly as shown in the repository files, or a new test path under tests/.
- Add or update tests when your feature changes behavior.
- Keep edits minimal; do not rewrite unrelated files.
""".strip()

def build_llm_patch_prompt(*, task: dict[str, Any], patch: dict[str, Any], idx: int) -> str:
    task_id = _task_id(task, idx)
    agent_id = clean(patch.get('agent_id') or patch.get('agent') or 'agent')
    scenario = as_dict(task.get('scenario'))
    task_type = clean(task.get('task_type') or scenario.get('kind') or 'unknown')
    targets = [clean(x) for x in as_list(patch.get('workspace_targets') or patch.get('targets')) if clean(x)]
    deps = [clean(x if isinstance(x, str) else as_dict(x).get('target_id') or as_dict(x).get('id')) for x in as_list(patch.get('depends_on') or patch.get('read_dependencies'))]
    revises = [clean(x if isinstance(x, str) else as_dict(x).get('target_id') or as_dict(x).get('id')) for x in as_list(patch.get('revises') or patch.get('invalidates') or patch.get('updates_commitments'))]
    candidates = as_list(patch.get('target_candidates') or scenario.get('target_candidates'))
    task_text = _agent_task_text(task, patch, agent_id=agent_id)
    files = _base_file_block(task)
    return f"""
You are agent {agent_id} in a controlled multi-agent patch/test evaluation.
Generate a minimal unified diff patch for the repository files below.

Task id: {task_id}
Task type: {task_type}
Your objective:
{task_text}

Workspace targets you should edit if possible: {json.dumps(targets, ensure_ascii=False)}
Read/commitment dependencies you rely on: {json.dumps([d for d in deps if d], ensure_ascii=False)}
Commitments you revise or invalidate: {json.dumps([r for r in revises if r], ensure_ascii=False)}
Target candidates, if any: {json.dumps(candidates, ensure_ascii=False)}

Repository files:
{files}

Return exactly one JSON object with these fields:
{{
  "patch_text": "<git-apply-compatible unified diff>",
  "notes": "short explanation",
  "target_files": ["relative/path.py"],
  "read_dependencies": ["dependency ids you actually used"]
}}

Rules:
- Return JSON only. No Markdown fences.
- patch_text must be a valid unified diff that can be applied with `git apply` from the repository root.
- Use paths exactly as shown in the repository files.
- Add or update tests when your feature changes behavior.
- Keep the patch minimal; do not rewrite unrelated files.
""".strip()


def _extract_patch_text(provider_text: str) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    try:
        parsed = parse_json_object_from_text(provider_text)
        meta = as_dict(parsed)
        patch = clean(meta.get('patch_text') or meta.get('diff') or meta.get('patch'))
        if patch:
            return (patch if patch.endswith('\n') else patch + '\n'), meta
    except Exception as exc:
        meta['json_parse_error'] = str(exc)
    text = provider_text or ''
    if '```' in text:
        chunks = text.split('```')
        for i, chunk in enumerate(chunks):
            body = chunk
            if i % 2 == 1:
                if body.lstrip().startswith('diff'):
                    body = body.lstrip()[4:].lstrip('\n') if body.lstrip().startswith('diff\n') else body
                if 'diff --git ' in body or ('--- a/' in body and '+++ b/' in body):
                    return (body.strip() + '\n'), meta
    if 'diff --git ' in text or ('--- a/' in text and '+++ b/' in text):
        start = text.find('diff --git ')
        if start < 0:
            start = text.find('--- a/')
        return (text[start:].strip() + '\n'), meta
    return '', meta


def _provider_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    load_dotenv()
    cfg = cfg or {}
    return {
        'provider': clean(cfg.get('provider') or os.environ.get('LLM_PROVIDER') or 'openai'),
        'model': clean(cfg.get('model') or os.environ.get('LLM_MODEL') or os.environ.get('OPENAI_MODEL') or os.environ.get('GEMINI_MODEL') or ''),
        'temperature': cfg.get('temperature'),
        'max_output_tokens': int(cfg.get('max_output_tokens') or os.environ.get('LLM_E2E_MAX_OUTPUT_TOKENS') or 2500),
        'timeout_seconds': int(cfg.get('timeout_seconds') or os.environ.get('LLM_E2E_TIMEOUT_SECONDS') or 120),
        'max_provider_retries': int(cfg.get('max_provider_retries') or os.environ.get('LLM_MAX_PROVIDER_RETRIES') or 4),
        'reasoning_effort': clean(cfg.get('reasoning_effort') or os.environ.get('LLM_REASONING_EFFORT') or os.environ.get('OPENAI_REASONING_EFFORT') or 'low'),
        'text_verbosity': clean(cfg.get('text_verbosity') or os.environ.get('LLM_TEXT_VERBOSITY') or os.environ.get('OPENAI_TEXT_VERBOSITY') or 'low'),
        'error_log_path': clean(cfg.get('error_log_path') or os.environ.get('LLM_ERROR_LOG_PATH')),
    }


def generate_patch_for_agent(*, task: dict[str, Any], patch: dict[str, Any], idx: int, base_dir: Path, llm_config: dict[str, Any] | None = None, mock_from_prepared: bool = False, validate_patch: bool = True, timeout_s: int = 60, output_mode: str = 'file_edits') -> dict[str, Any]:
    agent_id = clean(patch.get('agent_id') or patch.get('agent') or 'agent')
    cfg = _provider_config(llm_config)
    mode = clean(output_mode or os.environ.get('LLM_E2E_OUTPUT_MODE') or 'file_edits')
    prompt = build_llm_file_edit_prompt(task=task, patch=patch, idx=idx) if mode in {'file_edits','edits','json_edits'} else build_llm_patch_prompt(task=task, patch=patch, idx=idx)
    provider = clean(cfg.get('provider')).lower()
    started_at = now_iso()
    if mock_from_prepared or provider in {'mock', 'fixture'}:
        text = _patch_text(patch, base_dir=base_dir)
        meta = {'mock_from_prepared': True}
        provider_result: dict[str, Any] = {'endpoint': 'prepared_patch', 'usage': {}, 'http': {}, 'text': text}
        patch_text = text
    else:
        provider_result = call_llm_provider(
            provider=cfg['provider'],
            model=cfg['model'],
            prompt=prompt,
            scenario={'id': _task_id(task, idx), 'task_type': task.get('task_type')},
            agent={'id': agent_id, 'role': 'builder'},
            temperature=cfg.get('temperature'),
            max_output_tokens=int(cfg['max_output_tokens']),
            timeout_seconds=int(cfg['timeout_seconds']),
            strict_schema=False,
            max_provider_retries=int(cfg['max_provider_retries']),
            error_log_path=clean(cfg.get('error_log_path')),
            reasoning_effort=clean(cfg.get('reasoning_effort')),
            text_verbosity=clean(cfg.get('text_verbosity')),
        )
        raw_text = provider_result.get('text', '')
        if mode in {'file_edits', 'edits', 'json_edits'}:
            try:
                edits, meta = _extract_file_edits(raw_text)
                patch_text, materialization = _materialize_edits_to_patch(task, edits)
                meta['file_edit_mode'] = True
                meta['edit_parse_success'] = bool(edits)
                meta['edits'] = edits
                meta['materialization'] = materialization
            except Exception as exc:
                meta = {'file_edit_mode': True, 'edit_parse_success': False, 'edit_parse_error': f'{type(exc).__name__}: {exc}'}
                patch_text = ''
        else:
            patch_text, meta = _extract_patch_text(raw_text)
            meta['file_edit_mode'] = False
    parse_success = bool(patch_text)
    edit_parse_success = bool(as_dict(meta).get('edit_parse_success')) if as_dict(meta).get('file_edit_mode') else False
    validation: dict[str, Any] = {'attempted': False}
    if parse_success and validate_patch:
        try:
            with tempfile.TemporaryDirectory(prefix='llm_e2e_validate_') as tmp:
                repo, _ = _materialize_repo(task, base_dir=base_dir, work_dir=Path(tmp))
                validation = _apply_unified_patch(repo, patch_text, timeout_s=timeout_s)
                validation['attempted'] = True
        except Exception as exc:
            validation = {'attempted': True, 'ok': False, 'error': str(exc)}
    return {
        'agent_id': agent_id,
        'patch_text': patch_text,
        'patch_parse_success': parse_success,
        'edit_parse_success': edit_parse_success,
        'materialized_patch_valid': bool(validation.get('ok')) if validation.get('attempted') else False,
        'patch_validation': validation,
        'llm_generation': {
            'started_at': started_at,
            'provider': cfg.get('provider'),
            'model': cfg.get('model'),
            'endpoint': provider_result.get('endpoint'),
            'prompt_hash': stable_hash(prompt),
            'prompt_tokens_est': estimate_tokens(prompt),
            'output_tokens_est': estimate_tokens(provider_result.get('text', '')),
            'api_usage': provider_result.get('usage', {}),
            'provider_http': provider_result.get('http', {}),
            'request_options': provider_result.get('request_options', {}),
            'output_mode': mode,
            'metadata': meta,
        },
    }


def generate_llm_patch_tasks(*, input_path: str | Path, out_path: str | Path, limit: int = 0, llm_config: dict[str, Any] | None = None, mock_from_prepared: bool = False, validate_patch: bool = True, timeout_s: int = 60, generation_log_path: str | Path = '', task_types: list[str] | None = None, output_mode: str = 'file_edits') -> dict[str, Any]:
    in_path = Path(input_path)
    rows = _load_rows(str(in_path))
    allowed_types = {clean(t).lower() for t in (task_types or []) if clean(t)}
    if allowed_types:
        rows = [r for r in rows if _task_type(r).lower() in allowed_types]
    if limit and limit > 0:
        rows = rows[:limit]
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    generated_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for idx, task in enumerate(rows):
        task_id = _task_id(task, idx)
        generated = deep_clone(task)
        generated['source_task_id'] = task_id
        generated['task_type'] = _task_type(task)
        generated['patch_generation_mode'] = 'mock_prepared' if mock_from_prepared else 'llm_generated'
        new_patches: list[dict[str, Any]] = []
        for patch in [as_dict(p) for p in as_list(task.get('patches'))]:
            gen = generate_patch_for_agent(task=task, patch=patch, idx=idx, base_dir=in_path.parent, llm_config=llm_config, mock_from_prepared=mock_from_prepared, validate_patch=validate_patch, timeout_s=timeout_s, output_mode=output_mode)
            new_patch = deep_clone(patch)
            new_patch.pop('patch_file', None)
            new_patch['patch_text'] = gen['patch_text']
            new_patch['llm_generation'] = gen['llm_generation']
            new_patch['patch_parse_success'] = gen['patch_parse_success']
            new_patch['edit_parse_success'] = gen.get('edit_parse_success', False)
            new_patch['materialized_patch_valid'] = gen.get('materialized_patch_valid', False)
            new_patch['patch_validation'] = gen['patch_validation']
            new_patches.append(new_patch)
            log_row = {'kind': 'llm_e2e_patch_generation_v1', 'task_id': task_id, 'task_type': _task_type(task), 'agent_id': gen['agent_id'], 'patch_parse_success': gen['patch_parse_success'], 'edit_parse_success': gen.get('edit_parse_success', False), 'materialized_patch_valid': gen.get('materialized_patch_valid', False), 'patch_validation_ok': as_dict(gen.get('patch_validation')).get('ok'), 'output_mode': clean(as_dict(gen.get('llm_generation')).get('output_mode')), 'llm_generation': gen['llm_generation']}
            generation_rows.append(log_row)
            if generation_log_path:
                append_jsonl(generation_log_path, log_row)
        generated['patches'] = new_patches
        generated_rows.append(generated)
    with out_path.open('w', encoding='utf-8') as f:
        for row in generated_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    payload = {'kind': 'llm_e2e_patch_generation_results_v1', 'generated_at': now_iso(), 'input_path': str(in_path), 'out_path': str(out_path), 'task_types': sorted(allowed_types) if allowed_types else [], 'tasks': len(generated_rows), 'patches': len(generation_rows), 'parse_success': sum(1 for r in generation_rows if r.get('patch_parse_success')), 'validation_success': sum(1 for r in generation_rows if r.get('patch_validation_ok')), 'edit_parse_success': sum(1 for r in generation_rows if r.get('edit_parse_success')), 'materialized_patch_valid': sum(1 for r in generation_rows if r.get('materialized_patch_valid')), 'output_mode': output_mode, 'mock_from_prepared': mock_from_prepared, 'generation_rows': generation_rows}
    write_json(str(out_path) + '.summary.json', payload)
    return payload
