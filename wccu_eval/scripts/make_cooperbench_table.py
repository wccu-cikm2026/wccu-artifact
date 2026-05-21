from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir

DEFAULT_ORDER = [
    'adaptive_wccu_execution_trace',
    'adaptive_wccu_projection_trace',
    'adaptive_readset_occ',
    'adaptive_policy',
    'uniform_snapshot_occ',
    'uniform_review_gated',
    'uniform_append_only',
]

DISPLAY_NAMES = {
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_wccu_projection_trace': 'WCCU, projection trace',
    'adaptive_readset_occ': 'Read-set OCC',
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_snapshot_occ': 'Snapshot OCC',
    'uniform_review_gated': 'Review-gated',
    'uniform_append_only': 'Append-only',
}


def _iter_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [as_dict(r) for r in as_list(payload.get('results'))]


def _has_lock_lane(row: dict[str, Any]) -> bool:
    if float(row.get('lock_conflict_count') or 0) > 0:
        return True
    for d in as_list(row.get('merge_decisions')):
        decision = clean(as_dict(d).get('decision'))
        target = clean(as_dict(d).get('target'))
        if 'lock_contention' in decision or target.startswith('lock:'):
            return True
    return False


def _has_expected_lane(row: dict[str, Any]) -> bool:
    # In CooperBench-derived metadata scenarios, task_success is a mechanism-level
    # expected-outcome flag, not repository-level task completion.  Preserve that
    # diagnostic definition while exposing the raw lock-lane signal separately.
    if row.get('task_success'):
        return True
    return False


def _safety_pass(row: dict[str, Any]) -> bool:
    if row.get('failed'):
        return False
    return float(row.get('unsafe_auto_commit_count') or 0) <= 0


def _sum_commit(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(float(as_dict(r.get('commit')).get(key) or 0) for r in rows))


def compute_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_rows(payload):
        groups[clean(row.get('condition'))].append(row)
    order = DEFAULT_ORDER + sorted(c for c in groups if c not in DEFAULT_ORDER)
    out: list[dict[str, Any]] = []
    for condition in order:
        rows = groups.get(condition)
        if not rows:
            continue
        n = len(rows)
        lane_pass = sum(1 for r in rows if _has_expected_lane(r))
        safety_pass = sum(1 for r in rows if _safety_pass(r))
        failed = sum(1 for r in rows if r.get('failed'))
        lock_lane = sum(1 for r in rows if _has_lock_lane(r))
        unsafe = int(sum(float(r.get('unsafe_auto_commit_count') or 0) for r in rows))
        review = int(sum(float(r.get('review_burden_count') or 0) for r in rows))
        wccu_intervention = int(sum(float(r.get('wccu_intervention_count') or 0) for r in rows))
        cert_invalid = int(sum(float(r.get('certificate_invalid_count') or 0) for r in rows))
        decisions: dict[str, int] = defaultdict(int)
        for r in rows:
            for d in as_list(r.get('merge_decisions')):
                decisions[clean(as_dict(d).get('decision') or 'unknown')] += 1
        out.append({
            'condition': condition,
            'display_name': DISPLAY_NAMES.get(condition, condition),
            'runs': n,
            'failed_runs': failed,
            'lane_diagnostic_pass': lane_pass,
            'lane_diagnostic_rate': lane_pass / n if n else 0.0,
            'safety_pass': safety_pass,
            'safety_rate': safety_pass / n if n else 0.0,
            'lock_lane_selected': lock_lane,
            'lock_lane_rate': lock_lane / n if n else 0.0,
            'unsafe_auto_commit_count': unsafe,
            'review_routed_writes': review,
            'committed_writes': _sum_commit(rows, 'committed'),
            'proposal_writes': _sum_commit(rows, 'proposals'),
            'wccu_intervention_count': wccu_intervention,
            'certificate_invalid_count': cert_invalid,
            'merge_decisions': json.dumps(dict(sorted(decisions.items())), sort_keys=True),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Condition & Lane pass & Safety pass & Unsafe & Review \\',
        r'\midrule',
    ]
    for row in rows:
        name = row['display_name'].replace('_', r'\_')
        lines.append(f"{name} & {row['lane_diagnostic_pass']}/{row['runs']} & {row['safety_pass']}/{row['runs']} & {row['unsafe_auto_commit_count']} & {row['review_routed_writes']} " + '\\\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create a CooperBench-derived metadata stress-test table with lane and safety pass separated.')
    parser.add_argument('input', help='CooperBench substrate result JSON')
    parser.add_argument('--out-csv', default='analysis/cooperbench_table6.csv')
    parser.add_argument('--out-tex', default='')
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding='utf-8'))
    rows = compute_rows(payload)
    write_csv(Path(args.out_csv), rows)
    if args.out_tex:
        write_tex(Path(args.out_tex), rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
