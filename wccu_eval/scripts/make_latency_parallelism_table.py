from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir

LABELS = {
    'adaptive_wccu_execution_trace': 'WCCU, execution witness',
    'adaptive_wccu_projection_trace': 'WCCU, projection witness',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_review_gated': 'Review-gated',
    'serial_adaptive_policy': 'Serial adaptive',
    'serial_adaptive_wccu_execution_trace': 'Serial WCCU, execution',
    'serial_adaptive_wccu_projection_trace': 'Serial WCCU, projection',
}

ORDER = [
    'adaptive_wccu_execution_trace',
    'adaptive_wccu_projection_trace',
    'adaptive_policy',
    'uniform_review_gated',
    'serial_adaptive_wccu_execution_trace',
    'serial_adaptive_policy',
]


def _num(value: Any) -> float:
    try:
        n = float(value or 0)
        return n if math.isfinite(n) else 0.0
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, int(math.ceil(0.95 * len(xs))) - 1)
    return xs[idx]


def _agent_latencies(row: dict[str, Any]) -> list[float]:
    vals = []
    for ar in as_list(row.get('agentRuns')):
        llm = as_dict(ar.get('llm'))
        n = _num(llm.get('elapsed_ms'))
        if n > 0:
            vals.append(n)
    return vals


def _fresh(row: dict[str, Any]) -> bool:
    return not row.get('failed') and int(row.get('unsafe_auto_commit_count') or 0) == 0 and int(row.get('stale_dependency_accepted_count') or 0) == 0


def build_rows(paths: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        for row in as_list(payload.get('results')):
            groups.setdefault(clean(row.get('condition')), []).append(row)

    rows: list[dict[str, Any]] = []
    for condition, records in groups.items():
        if not records:
            continue
        wall = [_num(r.get('elapsed_ms')) for r in records]
        agent_sum = []
        agent_max = []
        speedups = []
        overhead = []
        for r in records:
            lat = _agent_latencies(r)
            s = sum(lat)
            m = max(lat) if lat else 0.0
            w = _num(r.get('elapsed_ms'))
            agent_sum.append(_num(r.get('agent_api_elapsed_ms_sum')) or s)
            agent_max.append(_num(r.get('agent_api_elapsed_ms_max')) or m)
            speedups.append(_num(r.get('parallel_speedup_est')) or ((s / w) if s and w else 0.0))
            serial_like = condition.startswith('serial_')
            lower = s if serial_like else m
            overhead.append(_num(r.get('scheduler_overhead_ms')) or max(0.0, w - lower))
        freshness = sum(1 for r in records if _fresh(r))
        stale_accepted = sum(int(r.get('stale_dependency_accepted_count') or 0) for r in records)
        unsafe = sum(int(r.get('unsafe_auto_commit_count') or 0) for r in records)
        review = sum(int(r.get('review_burden_count') or 0) for r in records)
        rows.append({
            'condition': condition,
            'label': LABELS.get(condition, condition),
            'runs': len(records),
            'freshness': f'{freshness}/{len(records)}',
            'stale_accepted': stale_accepted,
            'unsafe': unsafe,
            'review_block': review,
            'mean_wall_ms': _mean(wall),
            'median_wall_ms': median(wall) if wall else 0.0,
            'p95_wall_ms': _p95(wall),
            'mean_agent_sum_ms': _mean(agent_sum),
            'mean_agent_max_ms': _mean(agent_max),
            'mean_speedup_est': _mean(speedups),
            'mean_scheduler_overhead_ms': _mean(overhead),
        })
    order = {c: i for i, c in enumerate(ORDER)}
    rows.sort(key=lambda r: (order.get(r['condition'], 999), r['condition']))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_s(ms: float) -> str:
    return f'{ms/1000:.2f}'


def write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        r'Condition & Fresh. & Unsafe & Review & Wall s & Agent-sum s & Speedup \\',
        r'\midrule',
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['freshness']} & {int(r['unsafe'])} & {int(r['review_block'])} & "
            f"{_fmt_s(float(r['mean_wall_ms']))} & {_fmt_s(float(r['mean_agent_sum_ms']))} & {float(r['mean_speedup_est']):.2f} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Summarize wall-clock latency and estimated parallelism from LLM experiment JSON files.')
    parser.add_argument('results', nargs='+')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', required=True)
    args = parser.parse_args(argv)
    rows = build_rows(args.results)
    write_csv(Path(args.out_csv), rows)
    write_tex(Path(args.out_tex), rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
