from __future__ import annotations

import ast
import json
import multiprocessing as mp
import shutil
import sys
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from wccu_eval.external.cooperbench_adapter import cooperbench_task_to_commitment_scenario, cooperbench_task_to_scenario, load_cooperbench_tasks
from wccu_eval.eval.run_wccu_stress import CONDITION_TO_POLICY
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, now_iso, read_json, remove_dir, write_json, append_jsonl, stable_hash

AUTO_DECISIONS = {
    'single_writer',
    'auto_merge_compatible',
    'append_only_auto_merge',
}

REVIEW_DECISION_PREFIXES = (
    'wccu_certificate_intervention',
    'readset_occ_dependency_review',
    'lock_contention_review_required',
    'semantic_conflict_review_gated',
    'review_gated',
    'conflict_review_required',
    'authority_interrupt_rebase',
)


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == '.jsonl':
        rows: list[dict[str, Any]] = []
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(as_dict(json.loads(line)))
        return rows
    payload = read_json(p, {})
    if isinstance(payload, list):
        return [as_dict(x) for x in payload]
    for key in ('tasks', 'items', 'examples', 'instances'):
        if isinstance(payload.get(key), list):
            return [as_dict(x) for x in payload[key]]
    return [as_dict(payload)]


def _task_id(task: dict[str, Any], idx: int) -> str:
    return clean(task.get('task_id') or task.get('id') or task.get('instance_id') or task.get('name') or f'e2e_{idx:04d}')


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    # Prefer paths relative to the input file, but fall back to the current
    # working directory.  Generated e2e task files often live under data/ while
    # patch files live under patches/.
    from_input = (base_dir / p).resolve()
    if from_input.exists():
        return from_input
    from_cwd = p.resolve()
    if from_cwd.exists():
        return from_cwd
    return from_input


def _materialize_repo(task: dict[str, Any], *, base_dir: Path, work_dir: Path) -> tuple[Path, str]:
    repo_dir = work_dir / 'repo'
    ensure_dir(repo_dir.parent)
    repo_path = clean(task.get('repo_path') or task.get('repository_path'))
    repo_archive = clean(task.get('repo_archive') or task.get('repository_archive'))
    base_files = as_dict(task.get('base_files'))
    if repo_path:
        src = _resolve_path(repo_path, base_dir=base_dir)
        if not src.exists():
            raise FileNotFoundError(f'repo_path does not exist: {src}')
        shutil.copytree(src, repo_dir, ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
        return repo_dir, 'repo_path'
    if repo_archive:
        src = _resolve_path(repo_archive, base_dir=base_dir)
        if not src.exists():
            raise FileNotFoundError(f'repo_archive does not exist: {src}')
        ensure_dir(repo_dir)
        if src.suffix.lower() == '.zip':
            with zipfile.ZipFile(src) as zf:
                zf.extractall(repo_dir)
        elif src.suffix.lower() in {'.tgz', '.gz', '.tar'} or src.name.endswith('.tar.gz'):
            with tarfile.open(src) as tf:
                tf.extractall(repo_dir)
        else:
            raise ValueError(f'Unsupported repo archive format: {src}')
        # If the archive contains one top-level directory, use it as the repo root.
        children = [p for p in repo_dir.iterdir() if p.name not in {'.DS_Store'}]
        if len(children) == 1 and children[0].is_dir():
            return children[0], 'repo_archive_nested'
        return repo_dir, 'repo_archive'
    if base_files:
        ensure_dir(repo_dir)
        for rel, text in base_files.items():
            dest = repo_dir / str(rel)
            ensure_dir(dest.parent)
            dest.write_text(str(text), encoding='utf-8')
        return repo_dir, 'base_files'
    raise ValueError('Task must provide repo_path, repo_archive, or base_files')


def _command(cmd: str, *, cwd: Path, timeout_s: int) -> dict[str, Any]:
    started = now_iso()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), shell=True, text=True, capture_output=True, timeout=timeout_s)
        return {
            'command': cmd,
            'started_at': started,
            'exit_code': proc.returncode,
            'ok': proc.returncode == 0,
            'stdout_tail': proc.stdout[-4000:],
            'stderr_tail': proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            'command': cmd,
            'started_at': started,
            'exit_code': 124,
            'ok': False,
            'timeout': True,
            'stdout_tail': (exc.stdout or '')[-4000:] if isinstance(exc.stdout, str) else '',
            'stderr_tail': (exc.stderr or '')[-4000:] if isinstance(exc.stderr, str) else '',
        }




def _discover_tests(repo: Path) -> dict[str, Any]:
    """Return test files and top-level test functions visible after patch application."""
    tests_dir = repo / 'tests'
    files: list[dict[str, Any]] = []
    if not tests_dir.exists():
        return {'files': [], 'file_count': 0, 'test_function_count': 0}
    total_functions = 0
    for path in sorted(tests_dir.glob('test_*.py')):
        rel = path.relative_to(repo).as_posix()
        funcs: list[str] = []
        parse_error = ''
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                    funcs.append(node.name)
        except Exception as exc:
            parse_error = f'{type(exc).__name__}: {exc}'
        total_functions += len(funcs)
        item: dict[str, Any] = {'path': rel, 'test_functions': funcs, 'test_function_count': len(funcs)}
        if parse_error:
            item['parse_error'] = parse_error
        files.append(item)
    return {'files': files, 'file_count': len(files), 'test_function_count': total_functions}


def _paths_from_patch_text(patch_text: str) -> list[str]:
    out: list[str] = []
    for line in (patch_text or '').splitlines():
        line = line.strip()
        if line.startswith('+++ b/'):
            value = line[len('+++ b/'):].strip()
        elif line.startswith('--- a/'):
            value = line[len('--- a/'):].strip()
        else:
            continue
        if value and value != '/dev/null' and value not in out:
            out.append(value)
    return out


def _audit_paths(task: dict[str, Any], patches: list[dict[str, Any]], *, base_dir: Path) -> list[str]:
    paths: list[str] = []
    def add(value: Any) -> None:
        v = clean(value)
        if v and not v.startswith('commitment:'):
            v = v.replace('file:', '')
            if v not in paths:
                paths.append(v)
    for value in as_dict(task.get('base_files')).keys():
        if str(value).startswith(('pkg/', 'tests/')):
            add(str(value))
    scenario = as_dict(task.get('scenario'))
    for value in as_list(scenario.get('workspace_targets') or scenario.get('target_candidates')):
        add(value)
    for patch in patches:
        for value in _patch_targets(patch):
            add(value)
        meta = as_dict(as_dict(patch.get('llm_generation')).get('metadata'))
        mat = as_dict(meta.get('materialization'))
        for value in as_list(mat.get('target_files')):
            add(value)
        try:
            text = _patch_text(patch, base_dir=base_dir)
            for value in _paths_from_patch_text(text):
                add(value)
        except Exception:
            pass
    return paths


def _snapshot_files(repo: Path, paths: list[str], *, max_chars: int = 12000) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in paths:
        rel = clean(rel).replace('file:', '')
        if not rel:
            continue
        p = repo / rel
        item: dict[str, Any] = {'path': rel, 'exists': p.exists()}
        if p.exists() and p.is_file():
            try:
                text = p.read_text(encoding='utf-8')
                item['sha1'] = stable_hash(text)
                item['size_chars'] = len(text)
                item['content_tail'] = text[-max_chars:]
            except Exception as exc:
                item['read_error'] = f'{type(exc).__name__}: {exc}'
        out.append(item)
    return out


def _commitment_probe(repo: Path, task: dict[str, Any], *, timeout_s: int) -> dict[str, Any]:
    """Audit whether a commitment-staleness combination is visibly broken."""
    task_type = clean(task.get('task_type') or as_dict(task.get('scenario')).get('kind'))
    if task_type != 'commitment_staleness':
        return {'attempted': False, 'reason': 'not_commitment_staleness'}
    scenario = as_dict(task.get('scenario'))
    targets = [clean(x).replace('file:', '') for x in as_list(scenario.get('workspace_targets'))]
    client = next((t for t in targets if '/client_' in t or t.startswith('pkg/client_')), '')
    api = next((t for t in targets if '/api_' in t or t.startswith('pkg/api_')), '')
    if not client and (repo / 'pkg').exists():
        for pth in (repo / 'pkg').glob('client_*.py'):
            client = pth.relative_to(repo).as_posix()
            break
    if not api and (repo / 'pkg').exists():
        for pth in (repo / 'pkg').glob('api_*.py'):
            api = pth.relative_to(repo).as_posix()
            break
    if not client:
        return {'attempted': False, 'reason': 'no_client_module'}
    client_mod = client[:-3].replace('/', '.') if client.endswith('.py') else client.replace('/', '.')
    api_mod = api[:-3].replace('/', '.') if api.endswith('.py') else api.replace('/', '.')
    script = f'''
import importlib, json, pathlib, sys
root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
out = {{"attempted": True, "client_module": {client_mod!r}, "api_module": {api_mod!r}}}
raw = {{"id": "7", "name": "Ada"}}
try:
    client = importlib.import_module({client_mod!r})
    out["has_render_user_name"] = hasattr(client, "render_user_name")
    if out["has_render_user_name"]:
        try:
            out["render_user_name_value"] = client.render_user_name(raw)
            out["render_user_name_ok"] = True
        except Exception as exc:
            out["render_user_name_ok"] = False
            out["render_user_name_error"] = type(exc).__name__ + ": " + str(exc)
    if {bool(api_mod)!r}:
        try:
            api = importlib.import_module({api_mod!r})
            if hasattr(api, "normalize_user"):
                user = api.normalize_user(raw)
                out["normalize_user_type"] = type(user).__name__
                out["normalize_user_is_dict"] = isinstance(user, dict)
        except Exception as exc:
            out["api_probe_error"] = type(exc).__name__ + ": " + str(exc)
except Exception as exc:
    out["probe_error"] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(out, sort_keys=True))
'''
    started = now_iso()
    try:
        proc = subprocess.run([sys.executable, '-c', script], cwd=str(repo), text=True, capture_output=True, timeout=timeout_s)
        result = {
            'command': f'{sys.executable} -c <commitment_probe>',
            'started_at': started,
            'exit_code': proc.returncode,
            'ok': proc.returncode == 0,
            'stdout_tail': proc.stdout[-4000:],
            'stderr_tail': proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            'command': f'{sys.executable} -c <commitment_probe>',
            'started_at': started,
            'exit_code': 124,
            'ok': False,
            'timeout': True,
            'stdout_tail': (exc.stdout or '')[-4000:] if isinstance(exc.stdout, str) else '',
            'stderr_tail': (exc.stderr or '')[-4000:] if isinstance(exc.stderr, str) else '',
        }
    payload: dict[str, Any] = {'attempted': True, 'command_result': result}
    text = clean(result.get('stdout_tail'))
    try:
        payload.update(as_dict(json.loads(text.splitlines()[-1])))
    except Exception as exc:
        payload['parse_error'] = f'{type(exc).__name__}: {exc}'
    payload['detected_stale_contract_failure'] = bool(payload.get('has_render_user_name') and payload.get('render_user_name_ok') is False)
    return payload

def _patch_text(patch: dict[str, Any], *, base_dir: Path) -> str:
    if patch.get('patch_text') is not None and str(patch.get('patch_text')):
        text = str(patch.get('patch_text'))
        return text if text.endswith('\n') else text + '\n'
    patch_file = clean(patch.get('patch_file') or patch.get('file'))
    if not patch_file:
        raise ValueError('Patch entry must provide patch_text or patch_file')
    p = _resolve_path(patch_file, base_dir=base_dir)
    if not p.exists():
        raise FileNotFoundError(f'patch_file does not exist: {p}')
    return p.read_text(encoding='utf-8')


def _apply_unified_patch(repo: Path, patch_text: str, *, timeout_s: int) -> dict[str, Any]:
    check = subprocess.run(['git', 'apply', '--check', '--whitespace=nowarn', '-'], cwd=str(repo), input=patch_text, text=True, capture_output=True, timeout=timeout_s)
    if check.returncode == 0:
        apply = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-'], cwd=str(repo), input=patch_text, text=True, capture_output=True, timeout=timeout_s)
        return {
            'tool': 'git apply',
            'ok': apply.returncode == 0,
            'check_exit_code': check.returncode,
            'apply_exit_code': apply.returncode,
            'stdout_tail': apply.stdout[-4000:],
            'stderr_tail': apply.stderr[-4000:],
        }
    # If the patch does not apply cleanly, treat it as a downstream merge
    # conflict.  Earlier versions fell back to the system `patch` utility, but
    # that can introduce platform-specific interactive behavior on malformed or
    # conflicting multi-file patches.  The e2e diagnostic should be deterministic:
    # git-apply success means auto-applied; git-apply failure means merge/apply
    # failure for the selected commit policy.
    return {
        'tool': 'git apply --check',
        'ok': False,
        'check_exit_code': check.returncode,
        'apply_exit_code': check.returncode,
        'stdout_tail': check.stdout[-4000:],
        'stderr_tail': check.stderr[-4000:],
    }



def _safe_repo_path(repo: Path, rel_path: str) -> Path:
    rel = clean(rel_path).replace('file:', '').lstrip('/')
    if not rel or rel.startswith('..') or '/..' in rel:
        raise ValueError(f'unsafe edit path: {rel_path}')
    dest = (repo / rel).resolve()
    root = repo.resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'edit path escapes repo: {rel_path}') from exc
    return dest


def _file_edits_from_patch(patch: dict[str, Any]) -> list[dict[str, str]]:
    """Extract structured file edits from an LLM-generated patch entry.

    File-edit mode keeps the semantic code generation separate from patch
    serialization.  When these edits are present, the e2e runner applies them
    directly to the workspace instead of relying on serialized patch text.
    """
    candidates: list[Any] = []
    if isinstance(patch.get('file_edits'), list):
        candidates.append(patch.get('file_edits'))
    meta = as_dict(as_dict(patch.get('llm_generation')).get('metadata'))
    if isinstance(meta.get('edits'), list):
        candidates.append(meta.get('edits'))
    if isinstance(meta.get('file_edits'), list):
        candidates.append(meta.get('file_edits'))
    for items in candidates:
        edits: list[dict[str, str]] = []
        for item in as_list(items):
            d = as_dict(item)
            path = clean(d.get('path') or d.get('file') or d.get('target'))
            if d.get('content') is not None:
                content = str(d.get('content'))
            elif d.get('new_content') is not None:
                content = str(d.get('new_content'))
            else:
                continue
            if path:
                edits.append({'path': path, 'content': content})
        if edits:
            return edits
    return []


def _apply_file_edits(repo: Path, edits: list[dict[str, str]]) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    errors: list[str] = []
    for edit in edits:
        rel = clean(edit.get('path'))
        try:
            dest = _safe_repo_path(repo, rel)
            before_exists = dest.exists()
            before_text = dest.read_text(encoding='utf-8') if before_exists and dest.is_file() else ''
            after_text = str(edit.get('content') or '')
            ensure_dir(dest.parent)
            dest.write_text(after_text, encoding='utf-8')
            changed.append({
                'path': dest.relative_to(repo.resolve()).as_posix(),
                'before_exists': before_exists,
                'before_sha1': stable_hash(before_text) if before_exists else '',
                'after_sha1': stable_hash(after_text),
                'changed': (not before_exists) or before_text != after_text,
                'size_chars': len(after_text),
            })
        except Exception as exc:
            errors.append(f'{rel}: {type(exc).__name__}: {exc}')
    return {
        'tool': 'file_edits',
        'ok': not errors,
        'changed_files': changed,
        'edit_count': len(edits),
        'errors': errors,
        'stdout_tail': '',
        'stderr_tail': '\n'.join(errors)[-4000:],
    }


def _apply_patch_entry(repo: Path, patch: dict[str, Any], *, base_dir: Path, timeout_s: int) -> dict[str, Any]:
    edits = _file_edits_from_patch(patch)
    if edits:
        return _apply_file_edits(repo, edits)
    text = _patch_text(patch, base_dir=base_dir)
    return _apply_unified_patch(repo, text, timeout_s=timeout_s)



def _file_atom(file_path: str) -> dict[str, Any]:
    file_path = clean(file_path)
    target_id = file_path if file_path.startswith('file:') else f'file:{file_path}'
    return {
        'id': target_id,
        'atom_type': 'workspace_file',
        'title': file_path,
        'canonical_text_en': f'Workspace file {file_path}',
        'file_path': file_path.replace('file:', ''),
        'tags': ['e2e_patch_test', 'workspace'],
    }


def _commitment_atom(atom_id: str, *, text: str = '') -> dict[str, Any]:
    atom_id = clean(atom_id)
    return {
        'id': atom_id,
        'atom_type': 'commitment',
        'title': atom_id,
        'canonical_text_en': clean(text) or f'Teammate commitment {atom_id}',
        'tags': ['e2e_patch_test', 'commitment'],
    }


def _patch_targets(patch: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for value in as_list(patch.get('workspace_targets') or patch.get('targets') or patch.get('files')):
        value = clean(value)
        if value:
            out.append(value)
    if not out:
        raw = clean(patch.get('file_path') or patch.get('target') or patch.get('target_id'))
        if raw:
            out.append(raw.replace('file:', ''))
    return out or ['workspace.patch']


def _patch_target_id(patch: dict[str, Any]) -> str:
    targets = _patch_targets(patch)
    primary = targets[0]
    return primary if primary.startswith('file:') else f'file:{primary}'


def _synthesize_e2e_scenario(task: dict[str, Any], *, idx: int) -> dict[str, Any]:
    """Build a full substrate scenario from an e2e patch/test JSONL row.

    The e2e input format intentionally describes repository artifacts and patch
    files, not the internal substrate scenario.  This compiler maps prepared
    patch proposals to context write intents so the same policy machinery can
    decide which patches are safe to auto-apply downstream.
    """
    tid = _task_id(task, idx)
    patches = [as_dict(p) for p in as_list(task.get('patches'))]
    scenario_meta = as_dict(task.get('scenario'))
    kind = clean(task.get('task_type') or scenario_meta.get('kind') or 'e2e_patch_test')
    agents: list[dict[str, Any]] = []
    atoms_by_id: dict[str, dict[str, Any]] = {}
    agent_outputs: dict[str, dict[str, Any]] = {}
    wccu_reads: dict[str, list[dict[str, Any]]] = {}

    # Seed workspace file atoms from all patch targets and scenario targets.
    for patch in patches:
        for target in _patch_targets(patch):
            atom = _file_atom(target)
            atoms_by_id[atom['id']] = atom
    for target in as_list(scenario_meta.get('workspace_targets') or scenario_meta.get('target_candidates')):
        if clean(target):
            atom = _file_atom(clean(target))
            atoms_by_id[atom['id']] = atom

    # Seed commitment atoms and WCCU read dependencies from patch metadata.
    for patch in patches:
        agent_id = clean(patch.get('agent_id') or patch.get('agent') or f'agent_{len(agents)}')
        for dep in as_list(patch.get('depends_on') or patch.get('read_dependencies') or patch.get('read_atoms')):
            dep_id = clean(dep if isinstance(dep, str) else as_dict(dep).get('target_id') or as_dict(dep).get('id') or as_dict(dep).get('atom_id'))
            if not dep_id:
                continue
            atoms_by_id.setdefault(dep_id, _commitment_atom(dep_id, text=f'{agent_id} depends on {dep_id}'))
            wccu_reads.setdefault(agent_id, []).append({'target_id': dep_id, 'freshness_required': True, 'reason': 'e2e patch depends on teammate commitment'})
        for dep in as_list(patch.get('revises') or patch.get('invalidates') or patch.get('updates_commitments')):
            dep_id = clean(dep if isinstance(dep, str) else as_dict(dep).get('target_id') or as_dict(dep).get('id') or as_dict(dep).get('atom_id'))
            if dep_id:
                atoms_by_id.setdefault(dep_id, _commitment_atom(dep_id, text=f'{agent_id} may revise {dep_id}'))

    # Scenario-level commitment id is a convenient shorthand.
    commitment_id = clean(scenario_meta.get('commitment_atom_id'))
    if commitment_id:
        atoms_by_id.setdefault(commitment_id, _commitment_atom(commitment_id))

    for i, patch in enumerate(patches):
        agent_id = clean(patch.get('agent_id') or patch.get('agent') or f'agent_{i}')
        role = clean(patch.get('role') or ('builder' if 'agent' in agent_id else agent_id))
        agents.append({'id': agent_id, 'role': role, 'task_type': kind})
        revises = [clean(x if isinstance(x, str) else as_dict(x).get('target_id') or as_dict(x).get('id') or as_dict(x).get('atom_id')) for x in as_list(patch.get('revises') or patch.get('invalidates') or patch.get('updates_commitments'))]
        revises = [r for r in revises if r]
        if not revises and commitment_id and agent_id == clean(scenario_meta.get('revising_agent_id')):
            revises = [commitment_id]
        if revises:
            target_id = revises[0]
            payload = {
                'id': target_id,
                'target_id': target_id,
                'atom_id': target_id,
                'atom_type': 'commitment',
                'title': f'{agent_id} revised teammate commitment',
                'canonical_text_en': f'{agent_id} revises commitment {target_id}; downstream assumptions based on its prior version may be stale.',
                'status': 'active',
                'structured': {'e2e_patch_file': patch.get('patch_file'), 'revises': revises},
            }
            intent = {'intent_type': 'patch_atom', 'risk': 'low', 'payload': payload, 'disable_trace_text_fallback': True}
        else:
            target_id = _patch_target_id(patch)
            file_path = target_id.replace('file:', '')
            payload = {
                'id': target_id,
                'target_id': target_id,
                'atom_id': target_id,
                'atom_type': 'workspace_file',
                'title': file_path,
                'canonical_text_en': clean(patch.get('description')) or f'{agent_id} proposes a repository patch for {file_path}.',
                'file_path': file_path,
                'structured': {'e2e_patch_file': patch.get('patch_file'), 'workspace_targets': _patch_targets(patch)},
            }
            intent = {'intent_type': 'patch_atom', 'risk': clean(patch.get('risk') or 'low'), 'payload': payload, 'disable_trace_text_fallback': True}
        reads = wccu_reads.get(agent_id, [])
        if reads:
            intent['execution_witness'] = {'read_dependencies': reads}
            intent['read_witness'] = {'read_dependencies': reads}
        agent_outputs[agent_id] = {
            'text': clean(patch.get('description')) or f'{agent_id} prepared patch proposal for {tid}.',
            'intents': [intent],
            'read_dependencies': reads,
        }

    # Some rows express dependency/revision via scenario fields rather than patch fields.
    dependent = clean(scenario_meta.get('dependent_agent_id'))
    if commitment_id and dependent and not wccu_reads.get(dependent):
        wccu_reads.setdefault(dependent, []).append({'target_id': commitment_id, 'freshness_required': True, 'reason': 'scenario-level teammate commitment dependency'})
        if dependent in agent_outputs:
            agent_outputs[dependent]['read_dependencies'] = wccu_reads[dependent]
            intent = agent_outputs[dependent]['intents'][0]
            intent['execution_witness'] = {'read_dependencies': wccu_reads[dependent]}
            intent['read_witness'] = {'read_dependencies': wccu_reads[dependent]}

    expected: dict[str, Any] = {'max_unsafe_auto_commit_count': 0}
    if commitment_id:
        expected['min_stale_dependency_count'] = 1
    return {
        'id': tid,
        'task_type': f'e2e_{kind}',
        'goal': clean(task.get('goal') or scenario_meta.get('expected_relation') or f'Evaluate prepared multi-agent patches for {tid}.'),
        'agents': agents,
        'seed': {'atoms': list(atoms_by_id.values())},
        'agent_outputs': agent_outputs,
        'wccu_read_dependencies': wccu_reads,
        'target_candidates': as_list(scenario_meta.get('target_candidates')),
        'expected': expected,
        'metadata': {'task_type': kind, 'source_task_id': tid, 'e2e_scenario': scenario_meta},
    }


def scenario_from_task(task: dict[str, Any], *, idx: int) -> dict[str, Any]:
    scenario = as_dict(task.get('scenario'))
    if scenario and all(k in scenario for k in ('id', 'agents', 'seed', 'agent_outputs')):
        return scenario
    scenario_type = clean(task.get('scenario_type') or task.get('diagnostic_type') or task.get('expected_conflict_type'))
    if scenario_type in {'commitment_stale_dependency', 'cooperbench_commitment_stale_dependency', 'stale_commitment_dependency'}:
        return cooperbench_task_to_commitment_scenario(task, idx=idx)
    if task.get('agent_a_task') or task.get('agent_b_task') or task.get('shared_targets'):
        return cooperbench_task_to_scenario(task, idx=idx)
    return _synthesize_e2e_scenario(task, idx=idx)

def _auto_agents_from_row(row: dict[str, Any], *, apply_reviewed: bool = False) -> set[str]:
    all_agents = {clean(a.get('agent_id')) for a in as_list(row.get('agentRuns')) if clean(a.get('agent_id'))}
    held: set[str] = set()
    auto: set[str] = set()
    for d in as_list(row.get('merge_decisions')):
        decision = clean(as_dict(d).get('decision'))
        agents = {clean(a) for a in as_list(as_dict(d).get('agents')) if clean(a)}
        if decision in AUTO_DECISIONS:
            auto.update(agents)
        elif decision.startswith(REVIEW_DECISION_PREFIXES) or 'review' in decision or 'block' in decision or 'intervention' in decision:
            if apply_reviewed:
                auto.update(agents)
            else:
                held.update(agents)
        else:
            # Conservative: unknown decisions are not automatically applied.
            held.update(agents)
    return (all_agents if apply_reviewed else auto - held)


def run_one_task(task: dict[str, Any], *, idx: int, condition: str, work_root: Path, base_dir: Path, apply_reviewed: bool = False, timeout_s: int = 120) -> dict[str, Any]:
    if condition not in CONDITION_TO_POLICY:
        raise KeyError(f'Unsupported condition: {condition}')
    task_id = _task_id(task, idx)
    run_dir = work_root / f'{idx:04d}_{task_id}_{condition}'
    remove_dir(run_dir)
    ensure_dir(run_dir)
    repo, repo_source = _materialize_repo(task, base_dir=base_dir, work_dir=run_dir)
    scenario = scenario_from_task(task, idx=idx)
    root_dir = run_dir / 'context_substrate'
    seed_context(root_dir, scenario.get('seed', {}))
    policy_row = execute_context_policy_parallel(
        root_dir=root_dir,
        run_dir=run_dir / 'substrate_run',
        scenario=scenario,
        policy_mode=CONDITION_TO_POLICY[condition],
        condition=condition,
    )
    auto_agents = _auto_agents_from_row(policy_row, apply_reviewed=apply_reviewed)
    patches = [as_dict(p) for p in as_list(task.get('patches'))]
    patch_results: list[dict[str, Any]] = []
    patches_policy_auto = 0
    patches_apply_attempted = 0
    for i, patch in enumerate(patches):
        agent_id = clean(patch.get('agent_id') or patch.get('agent') or f'agent_{i}')
        if agent_id not in auto_agents:
            patch_results.append({'agent_id': agent_id, 'policy_auto': False, 'applied': False, 'held_for_review': True, 'ok': True})
            continue
        patches_policy_auto += 1
        patches_apply_attempted += 1
        try:
            result = _apply_patch_entry(repo, patch, base_dir=base_dir, timeout_s=timeout_s)
            patch_results.append({'agent_id': agent_id, 'policy_auto': True, 'applied': bool(result.get('ok')), 'apply_attempted': True, 'held_for_review': False, **result})
        except Exception as exc:
            patch_results.append({'agent_id': agent_id, 'policy_auto': True, 'applied': False, 'apply_attempted': True, 'held_for_review': False, 'ok': False, 'error': str(exc)})
    patch_apply_failures = sum(1 for p in patch_results if p.get('apply_attempted') and not p.get('ok'))
    patch_apply_success = sum(1 for p in patch_results if p.get('apply_attempted') and p.get('ok'))
    held = sum(1 for p in patch_results if p.get('held_for_review'))
    setup_results = [_command(clean(cmd), cwd=repo, timeout_s=timeout_s) for cmd in as_list(task.get('setup_commands')) if clean(cmd)]
    test_cmds = [clean(cmd) for cmd in as_list(task.get('test_commands')) if clean(cmd)] or [clean(task.get('test_command'))]
    test_cmds = [c for c in test_cmds if c]
    # If every patch is held for review, the downstream repository is unchanged.
    # Treat tests as not attempted/passing for the gated auto-apply path instead
    # of spending time on base-repo test execution.
    test_discovery_before = _discover_tests(repo)
    if patch_apply_success == 0 and held == len(patch_results) and not apply_reviewed:
        test_results = []
        tests_passed = True
        test_execution_skipped_reason = 'all_patches_held_for_review'
    else:
        test_results = [_command(cmd, cwd=repo, timeout_s=timeout_s) for cmd in test_cmds]
        tests_passed = bool(test_results) and all(t.get('ok') for t in test_results)
        test_execution_skipped_reason = ''
    test_discovery_after = _discover_tests(repo)
    audit_paths = _audit_paths(task, patches, base_dir=base_dir)
    final_file_snapshots = _snapshot_files(repo, audit_paths)
    commitment_probe = _commitment_probe(repo, task, timeout_s=timeout_s)
    merge_conflicts = patch_apply_failures
    row = {
        'kind': 'e2e_patch_test_result_v1',
        'task_id': task_id,
        'task_type': clean(task.get('task_type') or as_dict(task.get('scenario')).get('kind') or as_dict(scenario.get('metadata')).get('task_type') or 'unknown'),
        'condition': condition,
        'repo_source': repo_source,
        'scenario_id': scenario.get('id'),
        'apply_reviewed': apply_reviewed,
        'patches_total': len(patches),
        'patches_policy_auto': patches_policy_auto,
        'patches_apply_attempted': patches_apply_attempted,
        'patches_applied_successfully': patch_apply_success,
        # Backward-compatible alias: actual successful applications.
        'patches_auto_applied': patch_apply_success,
        'patch_apply_failures': patch_apply_failures,
        'patches_held_for_review': held,
        'merge_conflict_count': merge_conflicts,
        'tests_passed': tests_passed,
        'test_commands_total': len(test_results),
        'test_failures': sum(1 for t in test_results if not t.get('ok')),
        'auto_agents': sorted(auto_agents),
        'freshness_pass': int(policy_row.get('unsafe_auto_commit_count') or 0) == 0 and int(policy_row.get('stale_dependency_accepted_count') or 0) == 0,
        'stale_dependency_accepted_count': int(policy_row.get('stale_dependency_accepted_count') or 0),
        'unsafe_auto_commit_count': int(policy_row.get('unsafe_auto_commit_count') or 0),
        'review_burden_count': int(policy_row.get('review_burden_count') or 0),
        'wccu_intervention_count': int(policy_row.get('wccu_intervention_count') or 0),
        'readset_occ_review_count': int(policy_row.get('readset_occ_review_count') or 0),
        'policy_commit': policy_row.get('commit'),
        'patch_results': patch_results,
        'setup_results': setup_results,
        'test_results': test_results,
        'test_execution_skipped_reason': test_execution_skipped_reason,
        'test_discovery_before': test_discovery_before,
        'test_discovery_after': test_discovery_after,
        'final_file_snapshots': final_file_snapshots,
        'commitment_probe': commitment_probe,
        'commitment_probe_detected_stale_failure': bool(as_dict(commitment_probe).get('detected_stale_contract_failure')),
    }
    return row




def _run_one_task_child(result_path: str, task: dict[str, Any], idx: int, condition: str, work_root: str, base_dir: str, apply_reviewed: bool, timeout_s: int) -> None:
    try:
        row = run_one_task(task, idx=idx, condition=condition, work_root=Path(work_root), base_dir=Path(base_dir), apply_reviewed=apply_reviewed, timeout_s=timeout_s)
        write_json(result_path, {'ok': True, 'row': row})
    except Exception as exc:
        write_json(result_path, {'ok': False, 'error': f'{type(exc).__name__}: {exc}'})


def _failure_row(task: dict[str, Any], *, idx: int, condition: str, error: str, apply_reviewed: bool = False) -> dict[str, Any]:
    return {
        'kind': 'e2e_patch_test_result_v1',
        'task_id': _task_id(task, idx),
        'task_type': clean(task.get('task_type') or as_dict(task.get('scenario')).get('kind') or 'unknown'),
        'condition': condition,
        'apply_reviewed': apply_reviewed,
        'patches_total': len(as_list(task.get('patches'))),
        'patches_policy_auto': 0,
        'patches_apply_attempted': 0,
        'patches_applied_successfully': 0,
        'patches_auto_applied': 0,
        'patch_apply_failures': 0,
        'patches_held_for_review': len(as_list(task.get('patches'))),
        'merge_conflict_count': 0,
        'tests_passed': False,
        'test_commands_total': 0,
        'test_failures': 1,
        'auto_agents': [],
        'freshness_pass': False,
        'stale_dependency_accepted_count': 0,
        'unsafe_auto_commit_count': 0,
        'review_burden_count': 0,
        'wccu_intervention_count': 0,
        'readset_occ_review_count': 0,
        'error': error,
    }


def _run_one_task_isolated(task: dict[str, Any], *, idx: int, condition: str, work_root: Path, base_dir: Path, apply_reviewed: bool, timeout_s: int, row_timeout_s: int) -> dict[str, Any]:
    """Run one task/condition in a fresh Python interpreter.

    Patch/test subprocesses can interact badly with long-lived Python workers on
    some CI images.  A fresh interpreter per cell is slower but deterministic and
    keeps row-level timeouts enforceable.
    """
    ensure_dir(work_root / '_row_inputs')
    ensure_dir(work_root / '_row_results')
    safe_task = _task_id(task, idx)
    input_path = work_root / '_row_inputs' / f'{idx:04d}_{safe_task}_{condition}.jsonl'
    output_path = work_root / '_row_results' / f'{idx:04d}_{safe_task}_{condition}.json'
    cell_work = work_root / '_row_work' / f'{idx:04d}_{safe_task}_{condition}'
    input_path.write_text(json.dumps(task, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    if output_path.exists():
        output_path.unlink()
    cmd = [
        sys.executable,
        '-m', 'wccu_eval.eval.run_e2e_patch_test',
        '--input', str(input_path),
        '--conditions', condition,
        '--out', str(output_path),
        '--work-dir', str(cell_work),
        '--timeout-s', str(timeout_s),
        '--limit', '1',
    ]
    if apply_reviewed:
        cmd.append('--apply-reviewed')
    stdout_path = output_path.with_suffix('.stdout.txt')
    stderr_path = output_path.with_suffix('.stderr.txt')
    try:
        with stdout_path.open('w', encoding='utf-8') as stdout_f, stderr_path.open('w', encoding='utf-8') as stderr_f:
            proc = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, stdout=stdout_f, stderr=stderr_f, timeout=row_timeout_s)
    except subprocess.TimeoutExpired:
        return _failure_row(task, idx=idx, condition=condition, error=f'row_timeout_after_{row_timeout_s}s', apply_reviewed=apply_reviewed)
    stdout_tail = stdout_path.read_text(encoding='utf-8')[-2000:] if stdout_path.exists() else ''
    stderr_tail = stderr_path.read_text(encoding='utf-8')[-2000:] if stderr_path.exists() else ''
    if proc.returncode != 0:
        return _failure_row(task, idx=idx, condition=condition, error=(stderr_tail or stdout_tail or f'row_process_exit_{proc.returncode}')[-1000:], apply_reviewed=apply_reviewed)
    payload = as_dict(read_json(output_path, {}))
    rows = [r for r in as_list(payload.get('results')) if isinstance(r, dict)]
    if not rows:
        return _failure_row(task, idx=idx, condition=condition, error='row_process_produced_no_results', apply_reviewed=apply_reviewed)
    row = as_dict(rows[0])
    row['row_process_stdout_tail'] = stdout_tail
    row['row_process_stderr_tail'] = stderr_tail
    return row


def run_e2e_patch_tests(*, input_path: str, conditions: str, out: str, work_dir: str = 'runs/e2e_patch_test', apply_reviewed: bool = False, timeout_s: int = 120, limit: int = 0, isolate_rows: bool = False, row_timeout_s: int = 60) -> dict[str, Any]:
    in_path = Path(input_path).resolve()
    rows = _load_rows(in_path)
    if limit:
        rows = rows[:limit]
    condition_ids = [clean(c) for c in conditions.split(',') if clean(c)]
    out_path = Path(out).resolve()
    ensure_dir(out_path.parent)
    jsonl_path = out_path.with_suffix('.jsonl')
    if jsonl_path.exists():
        jsonl_path.unlink()
    work_root = Path(work_dir).resolve()
    ensure_dir(work_root)
    results: list[dict[str, Any]] = []
    for idx, task in enumerate(rows):
        task_name = _task_id(task, idx)
        for condition in condition_ids:
            print(f'[e2e] task={idx + 1}/{len(rows)} id={task_name} condition={condition}', flush=True)
            result = (_run_one_task_isolated(task, idx=idx, condition=condition, work_root=work_root, base_dir=in_path.parent, apply_reviewed=apply_reviewed, timeout_s=timeout_s, row_timeout_s=row_timeout_s) if isolate_rows else run_one_task(task, idx=idx, condition=condition, work_root=work_root, base_dir=in_path.parent, apply_reviewed=apply_reviewed, timeout_s=timeout_s))
            results.append(result)
            append_jsonl(jsonl_path, result)
    payload = {
        'kind': 'e2e_patch_test_results_v1',
        'generated_at': now_iso(),
        'args': {'input_path': str(in_path), 'conditions': conditions, 'out': str(out_path), 'work_dir': str(work_root), 'apply_reviewed': apply_reviewed, 'timeout_s': timeout_s, 'limit': limit, 'isolate_rows': isolate_rows, 'row_timeout_s': row_timeout_s},
        'results': results,
    }
    write_json(out_path, payload)
    return payload
