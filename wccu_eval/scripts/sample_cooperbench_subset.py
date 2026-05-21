from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from wccu_eval.external.cooperbench_adapter import load_cooperbench_tasks
from wccu_eval.utils import clean


def _task_score(task: dict[str, Any]) -> int:
    text = json.dumps(task, ensure_ascii=False).lower()
    score = 0
    for term in ['shared', 'same file', 'conflict', 'coordination', 'api', 'policy', 'cache', 'router', 'workspace', 'patch']:
        if term in text:
            score += 1
    if task.get('shared_targets'):
        score += 4
    if clean(task.get('expected_conflict_type')):
        score += 2
    return score


def sample_tasks(tasks: list[dict[str, Any]], *, size: int = 30, seed: int = 7, prefer_conflict: bool = True) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ranked = sorted(tasks, key=lambda t: (_task_score(t) if prefer_conflict else 0, clean(t.get('task_id'))), reverse=True)
    # Take a larger candidate pool so the subset is not simply sorted by score.
    pool_size = min(len(ranked), max(size * 3, size))
    pool = ranked[:pool_size]
    rng.shuffle(pool)
    chosen = pool[:min(size, len(pool))]
    return sorted(chosen, key=lambda t: clean(t.get('task_id')))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Sample a 30--50 task CooperBench-style subset for downstream validation.')
    parser.add_argument('--input', required=True, help='CooperBench-style JSON/JSONL metadata file.')
    parser.add_argument('--out', required=True, help='Output JSONL subset path.')
    parser.add_argument('--size', type=int, default=30, help='Recommended range: 30--50.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--no-prefer-conflict', action='store_true')
    args = parser.parse_args(argv)
    tasks = load_cooperbench_tasks(args.input)
    subset = sample_tasks(tasks, size=args.size, seed=args.seed, prefer_conflict=not args.no_prefer_conflict)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for task in subset:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    print(json.dumps({'ok': True, 'input_count': len(tasks), 'output_count': len(subset), 'out': str(out), 'seed': args.seed}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
