from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir

DISPLAY_NAMES = {
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_wccu_projection_trace': 'WCCU, projection trace',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    # Legacy names are accepted for old result bundles.
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_wccu_projection_trace': 'WCCU, projection trace',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_snapshot_occ': 'Snapshot OCC',
    'uniform_review_gated': 'Review-gated',
    'uniform_append_only': 'Append-only',
}
ORDER = [
    'adaptive_wccu_execution_trace',
    'adaptive_wccu_projection_trace',
    'adaptive_wccu_model_certificate',
    'adaptive_wccu_no_read_validation',
    'adaptive_wccu_execution_trace',
    'adaptive_wccu_projection_trace',
    'adaptive_wccu_model_certificate',
    'adaptive_wccu_no_read_validation',
    'adaptive_policy',
    'uniform_snapshot_occ',
    'uniform_review_gated',
    'uniform_append_only',
]


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for r in as_list(payload.get('results')):
            row = as_dict(r)
            groups[clean(row.get('condition'))].append(row)
    order = {c: i for i, c in enumerate(ORDER)}
    rows: list[dict[str, Any]] = []
    for condition in sorted(groups, key=lambda c: (order.get(c, 999), c)):
        rs = groups[condition]
        n = len(rs)
        rows.append({
            'condition': condition,
            'label': DISPLAY_NAMES.get(condition, condition),
            'runs': n,
            'diagnostic_pass': sum(1 for r in rs if r.get('task_success')),
            'safety_pass': sum(1 for r in rs if not r.get('failed') and float(r.get('unsafe_auto_commit_count') or 0) <= 0 and float(r.get('stale_dependency_accepted_count') or 0) <= 0),
            'stale_dependencies': int(sum(float(r.get('stale_dependency_count') or 0) for r in rs)),
            'stale_accepted': int(sum(float(r.get('stale_dependency_accepted_count') or 0) for r in rs)),
            'unsafe': int(sum(float(r.get('unsafe_auto_commit_count') or 0) for r in rs)),
            'wccu_intervention': int(sum(float(r.get('wccu_intervention_count', r.get('wccu_intervention_count')) or 0) for r in rs)),
            'review': int(sum(float(r.get('review_burden_count') or 0) for r in rs)),
            'failed_runs': sum(1 for r in rs if r.get('failed')),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'Condition & Runs & Stale acc. & Unsafe & WCCU int. & Review \\',
        r'\midrule',
    ]
    for r in rows:
        name = r['label'].replace('_', r'\_')
        lines.append(f"{name} & {r['runs']} & {r['stale_accepted']} & {r['unsafe']} & {r['wccu_intervention']} & {r['review']} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create a compact randomized WCCU stress table from one or more stress result JSON files.')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-csv', default='analysis/wccu_randomized_stress_table.csv')
    parser.add_argument('--out-tex', default='analysis/wccu_randomized_stress_table.tex')
    args = parser.parse_args(argv)
    rows = build_rows([_load(p) for p in args.inputs])
    write_csv(Path(args.out_csv), rows)
    if args.out_tex:
        write_tex(Path(args.out_tex), rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
