from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from wccu_eval.external.cooperbench_adapter import load_cooperbench_tasks
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, slugify


def _has_file_target(row: dict[str, Any]) -> bool:
    for target in as_list(row.get('shared_targets') or row.get('targets') or row.get('files')):
        target = as_dict(target)
        if clean(target.get('file_path') or target.get('path') or target.get('file') or target.get('target_id')).startswith('file:') or clean(target.get('file_path') or target.get('path') or target.get('file')):
            return True
    return False


def _first_file_target(row: dict[str, Any]) -> dict[str, Any]:
    for target in as_list(row.get('shared_targets') or row.get('targets') or row.get('files')):
        target = as_dict(target)
        fp = clean(target.get('file_path') or target.get('path') or target.get('file'))
        tid = clean(target.get('target_id') or target.get('id') or target.get('atom_id'))
        if fp or tid.startswith('file:'):
            if not fp and tid.startswith('file:'):
                fp = tid.replace('file:', '', 1)
            return {**target, 'target_id': tid or f'file:{fp}', 'file_path': fp}
    return {}


def make_commitment_diagnostics(rows: list[dict[str, Any]], *, limit: int = 0, seed: int = 7, require_shared_file: bool = True) -> list[dict[str, Any]]:
    candidates = [as_dict(r) for r in rows]
    if require_shared_file:
        candidates = [r for r in candidates if _has_file_target(r)]
    rng = random.Random(seed)
    # Stable random subset without destroying reproducibility across Python hash seeds.
    candidates.sort(key=lambda r: clean(r.get('task_id') or r.get('id') or r.get('name')))
    rng.shuffle(candidates)
    if limit and limit > 0:
        candidates = candidates[:limit]
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        task_id = clean(row.get('task_id') or row.get('id') or f'coop_commitment_{idx:04d}')
        safe_id = slugify(task_id) or f'coop_commitment_{idx:04d}'
        primary = _first_file_target(row)
        shared_targets = [primary] if primary else as_list(row.get('shared_targets'))[:1]
        # Preserve the original task metadata but mark it so the adapter emits a
        # cross-target stale-commitment diagnostic instead of a same-file lock test.
        out.append({
            **row,
            'task_id': f'{task_id}_commitment_stale',
            'source_task_id': task_id,
            'scenario_type': 'commitment_stale_dependency',
            'expected_conflict_type': 'commitment_stale_dependency',
            'shared_targets': shared_targets,
            'commitment_a_id': f'commitment:{safe_id}:feature_a',
            'commitment_b_id': f'commitment:{safe_id}:feature_b',
            'description': clean(row.get('description') or row.get('goal') or f'CooperBench-derived commitment stale dependency diagnostic for {task_id}.'),
        })
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create CooperBench-derived cross-target stale teammate-commitment diagnostics.')
    parser.add_argument('--input', required=True, help='CooperBench-derived JSON/JSONL task file, e.g. data/cooperbench_multirepo_subset50_compact_seed7.jsonl')
    parser.add_argument('--out', required=True, help='Output JSONL diagnostic task file.')
    parser.add_argument('--limit', type=int, default=30, help='Maximum number of diagnostics to emit. Use 0 for all candidates.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--allow-missing-shared-file', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args(argv)
    rows = load_cooperbench_tasks(args.input)
    out = make_commitment_diagnostics(rows, limit=args.limit, seed=args.seed, require_shared_file=not args.allow_missing_shared_file)
    write_jsonl(args.out, out)
    if args.inspect:
        print(json.dumps({
            'ok': True,
            'input': args.input,
            'raw_records': len(rows),
            'converted': len(out),
            'out': args.out,
            'sample_task_ids': [r.get('task_id') for r in out[:5]],
            'scenario_type': 'commitment_stale_dependency',
        }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
