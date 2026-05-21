from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from wccu_eval.external.cooperbench_adapter import load_cooperbench_tasks
from wccu_eval.utils import as_dict, as_list, clean, slugify, stable_hash

FEATURE_A_KEYS = [
    'agent_a_task', 'agent_a_feature', 'feature_a', 'task_a', 'a_task', 'a_feature',
    'feature1', 'feature_1', 'task1', 'task_1', 'issue_a', 'spec_a', 'prompt_a',
]
FEATURE_B_KEYS = [
    'agent_b_task', 'agent_b_feature', 'feature_b', 'task_b', 'b_task', 'b_feature',
    'feature2', 'feature_2', 'task2', 'task_2', 'issue_b', 'spec_b', 'prompt_b',
]
DESC_KEYS = ['description', 'summary', 'goal', 'task', 'task_description', 'problem_statement', 'instructions']
REPO_KEYS = ['repo', 'repository', 'repo_name', 'project', 'library']
LANG_KEYS = ['language', 'lang']
FILE_KEYS = [
    'shared_files', 'shared_targets', 'files', 'target_files', 'relevant_files',
    'modified_files', 'conflict_files', 'touched_files', 'paths', 'file_paths',
]

FILE_RE = re.compile(r'(?<![\w.-])([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|cs|cpp|c|h|hpp|rb|php|swift|scala|sql|yaml|yml|json|toml|md))(?![\w.-])')
DIFF_FILE_RE = re.compile(r'^(?:\+\+\+|---)\s+[ab]/(.+)$', re.MULTILINE)
DEFAULT_FEATURE_MAX_CHARS = 1400



def _read_json(path: Path) -> Any | None:
    try:
        if path.suffix.lower() in {'.json', '.jsonl'}:
            if path.suffix.lower() == '.jsonl':
                return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return None




def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def _feature_title_and_body(text: str, fallback: str) -> tuple[str, str]:
    """Extract a compact title/body from CooperBench feature.md style text."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    title = ''
    body_lines: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        if not title and stripped.startswith('#'):
            title = stripped.lstrip('#').strip()
            continue
        if stripped:
            body_lines.append(stripped)
    if not title:
        for ln in body_lines:
            if len(ln) <= 120:
                title = ln
                break
    body = '\n'.join(body_lines).strip()
    return clean(title or fallback), clean(body or title or fallback)




def _strip_markdown_noise(text: str) -> str:
    # Keep prose and file lists, but remove long fenced code blocks that make
    # agent prompts large and increase malformed/truncated JSON risk.
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)
    text = re.sub(r'`([^`]{1,80})`', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _compact_feature_text(title: str, body: str, *, max_chars: int = DEFAULT_FEATURE_MAX_CHARS) -> str:
    """Return a compact feature spec suitable for LLM prompts.

    Official CooperBench feature.md files can contain long PR descriptions and
    code examples.  For our substrate-level validation we do not need the full
    code-level prompt; we need enough text for the two agents to emit a target-
    grounded patch plan.  This compaction reduces JSON truncation without
    changing the target/file structure used by the runtime.
    """
    title = clean(title)
    body = _strip_markdown_noise(clean(body))
    paths = _dedupe_paths(FILE_RE.findall(body))
    lines = [ln.strip(' -*') for ln in body.splitlines() if ln.strip()]
    selected: list[str] = []
    if title:
        selected.append(f'Title: {title}')
    # Preserve high-signal PR fields and the first few explanatory lines.
    important_patterns = re.compile(r'(description|problem|solution|files modified|technical background|requirement|behavior|bug|fix)', re.I)
    for ln in lines:
        if important_patterns.search(ln) or len(selected) < 6:
            if ln not in selected:
                selected.append(ln)
        if len('\n'.join(selected)) > max_chars * 0.75:
            break
    if paths:
        selected.append('Files Modified: ' + ', '.join(paths[:8]))
    text = clean('\n'.join(selected) or title or body)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(' ', 1)[0].rstrip() + ' ...'
    return text


def _compact_row_feature_text(value: str, *, max_chars: int) -> str:
    if not value:
        return ''
    title, body = _feature_title_and_body(value, value[:80])
    return _compact_feature_text(title, body, max_chars=max_chars)

def _feature_dirs(task_dir: Path) -> list[Path]:
    dirs = []
    for d in task_dir.iterdir() if task_dir.exists() else []:
        if d.is_dir() and re.match(r'features?[_-]?\d+$', d.name, re.I):
            # CooperBench feature folders contain feature.md plus patches.
            if (d / 'feature.md').exists() or any(d.glob('*.patch')) or any(d.glob('*.diff')):
                dirs.append(d)
    def key(d: Path):
        m = re.search(r'(\d+)$', d.name)
        return (int(m.group(1)) if m else 10**9, d.name)
    return sorted(dirs, key=key)


def _paths_from_feature_dir(feature_dir: Path) -> list[str]:
    paths: list[str] = []
    for fp in feature_dir.glob('feature.md'):
        paths.extend(FILE_RE.findall(_read_text(fp)))
    for patch in list(feature_dir.glob('*.patch')) + list(feature_dir.glob('*.diff')):
        text = _read_text(patch)
        paths.extend(DIFF_FILE_RE.findall(text))
        paths.extend(FILE_RE.findall(text))
    return _dedupe_paths(paths)


def _record_from_feature_pair(task_dir: Path, fa: Path, fb: Path, *, feature_max_chars: int = DEFAULT_FEATURE_MAX_CHARS, full_feature_text: bool = False) -> dict[str, Any]:
    feature_a_text = _read_text(fa / 'feature.md')
    feature_b_text = _read_text(fb / 'feature.md')
    title_a, body_a = _feature_title_and_body(feature_a_text, fa.name)
    title_b, body_b = _feature_title_and_body(feature_b_text, fb.name)
    prompt_a = clean(f'{title_a}\n{body_a}') if full_feature_text else _compact_feature_text(title_a, body_a, max_chars=feature_max_chars)
    prompt_b = clean(f'{title_b}\n{body_b}') if full_feature_text else _compact_feature_text(title_b, body_b, max_chars=feature_max_chars)
    paths_a = _paths_from_feature_dir(fa)
    paths_b = _paths_from_feature_dir(fb)
    overlap = sorted(set(paths_a) & set(paths_b))
    union = _dedupe_paths(paths_a + paths_b)
    task_name = task_dir.name
    repo_name = task_dir.parent.name if task_dir.parent and task_dir.parent.name else ''
    return {
        'task_id': f'{repo_name}_{task_name}_{fa.name}_{fb.name}',
        'repo': repo_name,
        'description': f'CooperBench feature-pair task from {repo_name}/{task_name}: {title_a} + {title_b}.',
        'agent_a_task': prompt_a,
        'agent_b_task': prompt_b,
        'features': [
            {'id': fa.name, 'title': title_a, 'description': prompt_a, 'full_description_chars': len(body_a), 'files': paths_a},
            {'id': fb.name, 'title': title_b, 'description': prompt_b, 'full_description_chars': len(body_b), 'files': paths_b},
        ],
        'shared_files': overlap or union[:4],
        'expected_conflict_type': 'workspace_contention' if (overlap or union) else 'coordination_conflict',
        'metadata': {
            'source_layout': 'cooperbench_feature_pool',
            'source_dir': str(task_dir),
            'feature_a_dir': str(fa),
            'feature_b_dir': str(fb),
            'overlap_file_count': len(overlap),
            'union_file_count': len(union),
            'feature_max_chars': feature_max_chars,
            'full_feature_text': full_feature_text,
        },
    }

def _flatten(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, list):
        for x in obj:
            yield from _flatten(x)
    elif isinstance(obj, dict):
        for key in ('tasks', 'instances', 'examples', 'items', 'data'):
            if isinstance(obj.get(key), list):
                yield from _flatten(obj[key])
                return
        yield obj


def _first(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        v = row.get(key)
        if isinstance(v, str) and clean(v):
            return clean(v)
    return ''


def _nested_feature(row: dict[str, Any], side: str) -> str:
    agents = as_dict(row.get('agents'))
    nested = as_dict(agents.get(side) or agents.get(side.upper()) or agents.get(f'agent_{side}'))
    return clean(nested.get('task') or nested.get('feature') or nested.get('issue') or nested.get('spec') or nested.get('prompt'))


def _text_blobs(row: dict[str, Any]) -> list[str]:
    blobs: list[str] = []
    def walk(x: Any):
        if isinstance(x, str):
            if len(x) < 20000:
                blobs.append(x)
        elif isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)
    walk(row)
    return blobs


def _paths_from_value(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, str):
        paths.extend(FILE_RE.findall(value))
    elif isinstance(value, list):
        for x in value:
            paths.extend(_paths_from_value(x))
    elif isinstance(value, dict):
        p = clean(value.get('file_path') or value.get('path') or value.get('file') or value.get('filename') or value.get('target'))
        if p:
            paths.append(p)
        for key in FILE_KEYS:
            if key in value:
                paths.extend(_paths_from_value(value[key]))
    return paths


def _paths_from_task_dir(task_dir: Path) -> list[str]:
    paths: list[str] = []
    for diff in list(task_dir.rglob('*.diff')) + list(task_dir.rglob('*.patch')):
        try:
            text = diff.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        paths.extend(DIFF_FILE_RE.findall(text))
    for sh in task_dir.glob('check_merge_conflicts.sh'):
        try:
            paths.extend(FILE_RE.findall(sh.read_text(encoding='utf-8', errors='ignore')))
        except Exception:
            pass
    return paths


def _dedupe_paths(paths: list[str]) -> list[str]:
    cleaned: list[str] = []
    for p in paths:
        p = clean(p).replace('\\', '/')
        while p.startswith('./'):
            p = p[2:]
        p = p.strip('/ ')
        if p.startswith('a/') or p.startswith('b/'):
            p = p[2:]
        if not p or p.startswith('http') or p.endswith('.json'):
            continue
        cleaned.append(p)
    return list(dict.fromkeys(cleaned))


def _shared_targets(row: dict[str, Any], task_dir: Path | None = None, *, max_targets: int = 4) -> list[dict[str, Any]]:
    paths: list[str] = []
    for key in FILE_KEYS:
        if key in row:
            paths.extend(_paths_from_value(row[key]))
    # Candidate paths from textual fields; useful for official metadata that
    # stores feature descriptions without a normalized shared_files column.
    for blob in _text_blobs(row):
        paths.extend(FILE_RE.findall(blob))
    if task_dir is not None:
        paths.extend(_paths_from_task_dir(task_dir))
    paths = _dedupe_paths(paths)
    if not paths:
        return []
    # Prefer files that appear multiple times, because they are more likely to be
    # shared/coordination-relevant.  Preserve deterministic ordering.
    counts = {p: paths.count(p) for p in dict.fromkeys(paths)}
    ranked = sorted(counts, key=lambda p: (-counts[p], p))[:max_targets]
    return [{'target_id': f'file:{p}', 'target_type': 'workspace_file', 'file_path': p, 'title': p, 'description': f'CooperBench-derived shared file target {p}.'} for p in ranked]


def _infer_conflict_type(row: dict[str, Any], targets: list[dict[str, Any]]) -> str:
    explicit = clean(row.get('expected_conflict_type') or row.get('conflict_type') or row.get('coordination_type'))
    if explicit:
        return explicit
    text = json.dumps(row, ensure_ascii=False).lower()
    if any(t in text for t in ['prohibit', 'must not', 'forbid', 'privacy', 'incompatible policy', 'semantic conflict']):
        return 'semantic_conflict'
    if targets:
        return 'workspace_contention'
    return 'coordination_conflict'


def normalize_official_task(row: dict[str, Any], *, idx: int = 0, task_dir: Path | None = None, feature_max_chars: int = DEFAULT_FEATURE_MAX_CHARS, full_feature_text: bool = False) -> dict[str, Any]:
    row = as_dict(row)
    task_id = clean(row.get('task_id') or row.get('id') or row.get('instance_id') or row.get('name'))
    if not task_id and task_dir is not None:
        task_id = f'{task_dir.parent.name}_{task_dir.name}'
    task_id = task_id or f'cooperbench_{idx:04d}'
    repo = _first(row, REPO_KEYS)
    if not repo and task_dir is not None and task_dir.parent.name != '.':
        repo = task_dir.parent.name
    language = _first(row, LANG_KEYS)
    feature_a = _first(row, FEATURE_A_KEYS) or _nested_feature(row, 'a') or _nested_feature(row, '1')
    feature_b = _first(row, FEATURE_B_KEYS) or _nested_feature(row, 'b') or _nested_feature(row, '2')
    features = as_list(row.get('features') or row.get('tasks') or row.get('issues'))
    if not feature_a and len(features) >= 1:
        feature_a = clean(as_dict(features[0]).get('task') or as_dict(features[0]).get('feature') or as_dict(features[0]).get('description') or features[0])
    if not feature_b and len(features) >= 2:
        feature_b = clean(as_dict(features[1]).get('task') or as_dict(features[1]).get('feature') or as_dict(features[1]).get('description') or features[1])
    if not feature_a:
        feature_a = 'Implement feature A from the CooperBench task metadata.'
    if not feature_b:
        feature_b = 'Implement feature B from the CooperBench task metadata.'
    if not full_feature_text:
        feature_a = _compact_row_feature_text(feature_a, max_chars=feature_max_chars)
        feature_b = _compact_row_feature_text(feature_b, max_chars=feature_max_chars)
    targets = _shared_targets(row, task_dir=task_dir)
    description = _first(row, DESC_KEYS)
    if not description:
        description = f'CooperBench task {task_id}: coordinate two feature changes.'
    conflict_type = _infer_conflict_type(row, targets)
    return {
        'task_id': task_id,
        'repo': repo,
        'language': language,
        'description': description,
        'agent_a_task': feature_a,
        'agent_b_task': feature_b,
        'shared_targets': targets,
        'expected_conflict_type': conflict_type,
        'metadata': {
            'converter': 'cooperbench_official_converter',
            'source_dir': str(task_dir) if task_dir else '',
            'source_keys': sorted(row.keys()),
            'raw_hash': stable_hash(json.dumps(row, sort_keys=True, ensure_ascii=False))[:12],
        },
    }


def load_records_from_path(path: str | Path, *, feature_max_chars: int = DEFAULT_FEATURE_MAX_CHARS, full_feature_text: bool = False) -> list[tuple[dict[str, Any], Path | None]]:
    p = Path(path)
    if p.is_file():
        tasks = load_cooperbench_tasks(p)
        return [(as_dict(t), None) for t in tasks]
    if not p.exists():
        raise FileNotFoundError(f'Input path does not exist: {p}')
    records: list[tuple[dict[str, Any], Path | None]] = []
    # Prefer task-like directories.  If a directory contains multiple JSON files,
    # merge their top-level dictionaries so no single release layout is assumed.
    candidate_dirs = [d for d in p.rglob('*') if d.is_dir() and re.match(r'task\d+|task[_-]?\w+|features?[_-]?\d+', d.name, re.I)]
    if not candidate_dirs:
        candidate_dirs = [p]
    seen_files: set[Path] = set()
    for d in candidate_dirs:
        merged: dict[str, Any] = {}
        found = False
        for jf in list(d.glob('*.json')) + list(d.glob('*.jsonl')):
            obj = _read_json(jf)
            if obj is None:
                continue
            seen_files.add(jf.resolve())
            rows = list(_flatten(obj))
            if len(rows) == 1 and isinstance(rows[0], dict):
                merged.update(rows[0])
                found = True
            else:
                for row in rows:
                    records.append((row, d))
                    found = True
        if found and merged:
            records.append((merged, d))
            continue
        # Official CooperBench HF snapshots can be feature-pool directories with
        # no task-level JSON: task0/feature1/feature.md, feature.patch, tests.patch.
        # In that layout each pair of feature folders forms a cooperative task.
        fdirs = _feature_dirs(d)
        if len(fdirs) >= 2:
            for i in range(len(fdirs)):
                for j in range(i + 1, len(fdirs)):
                    records.append((_record_from_feature_pair(d, fdirs[i], fdirs[j], feature_max_chars=feature_max_chars, full_feature_text=full_feature_text), d))
            found = True
    # Fallback: collect standalone JSON/JSONL files not already handled.
    for jf in list(p.rglob('*.json')) + list(p.rglob('*.jsonl')):
        if jf.resolve() in seen_files:
            continue
        obj = _read_json(jf)
        if obj is None:
            continue
        for row in _flatten(obj):
            records.append((row, jf.parent))
    # Deduplicate by task-like id/hash.
    dedup: dict[str, tuple[dict[str, Any], Path | None]] = {}
    for i, (row, d) in enumerate(records):
        key = clean(row.get('task_id') or row.get('id') or row.get('instance_id')) or (f'{d}:{i}' if d else stable_hash(row))
        dedup[key] = (row, d)
    return list(dedup.values())


def download_hf_dataset(repo_id: str, *, subdir: str = '', revision: str = 'main', local_dir: str = 'data/cooperbench_hf_snapshot') -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('huggingface_hub is required for --hf-dataset. Install with: pip install huggingface_hub') from exc
    allow_patterns = [f'{subdir.rstrip("/")}/**'] if subdir else None
    path = snapshot_download(repo_id=repo_id, repo_type='dataset', revision=revision, local_dir=local_dir, allow_patterns=allow_patterns)
    return Path(path) / subdir if subdir else Path(path)


def convert_records(records: list[tuple[dict[str, Any], Path | None]], *, max_tasks: int = 0, require_shared_file: bool = False, feature_max_chars: int = DEFAULT_FEATURE_MAX_CHARS, full_feature_text: bool = False) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for idx, (row, d) in enumerate(records):
        task = normalize_official_task(row, idx=idx, task_dir=d, feature_max_chars=feature_max_chars, full_feature_text=full_feature_text)
        if require_shared_file and not task.get('shared_targets'):
            continue
        converted.append(task)
        if max_tasks and len(converted) >= max_tasks:
            break
    return converted


def write_jsonl(rows: list[dict[str, Any]], out: str | Path) -> None:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Convert official/local CooperBench metadata into WCCU CooperBench-style JSONL.')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', help='Local CooperBench metadata file or extracted dataset directory.')
    src.add_argument('--hf-dataset', help='HuggingFace dataset id, e.g. CodeConflict/cooperbench-dataset.')
    parser.add_argument('--hf-subdir', default='', help='Optional subdir/config to download, e.g. openai_tiktoken_task.')
    parser.add_argument('--hf-revision', default='main')
    parser.add_argument('--download-dir', default='data/cooperbench_hf_snapshot')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-tasks', type=int, default=0)
    parser.add_argument('--feature-max-chars', type=int, default=DEFAULT_FEATURE_MAX_CHARS, help='Compact each feature prompt to this many characters before conversion. Use 0 with --full-feature-text to keep full text.')
    parser.add_argument('--full-feature-text', action='store_true', help='Keep full CooperBench feature text in agent prompts. This can increase JSON parse failures.')
    parser.add_argument('--require-shared-file', action='store_true', help='Drop tasks for which no shared file can be inferred.')
    parser.add_argument('--inspect', action='store_true', help='Print a short conversion summary to stdout.')
    args = parser.parse_args(argv)
    if args.hf_dataset:
        input_path = download_hf_dataset(args.hf_dataset, subdir=args.hf_subdir, revision=args.hf_revision, local_dir=args.download_dir)
    else:
        input_path = Path(args.input)
    records = load_records_from_path(input_path, feature_max_chars=args.feature_max_chars, full_feature_text=args.full_feature_text)
    rows = convert_records(records, max_tasks=args.max_tasks, require_shared_file=args.require_shared_file, feature_max_chars=args.feature_max_chars, full_feature_text=args.full_feature_text)
    write_jsonl(rows, args.out)
    summary = {
        'ok': True,
        'input': str(input_path),
        'raw_records': len(records),
        'converted': len(rows),
        'with_shared_targets': sum(1 for r in rows if r.get('shared_targets')),
        'feature_max_chars': args.feature_max_chars,
        'full_feature_text': bool(args.full_feature_text),
        'out': args.out,
        'sample_task_ids': [r.get('task_id') for r in rows[:5]],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
