from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.scripts.make_stress_table import DISPLAY_NAMES, ORDER
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stress_key(payload: dict[str, Any], row: dict[str, Any]) -> tuple[str, int, int, float]:
    args = as_dict(payload.get('args'))
    meta = as_dict(row.get('stress_metadata'))
    seed = int(args.get('seed') or 0)
    writers = int(meta.get('writers') or args.get('writers') or 0)
    atoms = int(meta.get('atom_count') or args.get('atom_count') or 0)
    prob = float(args.get('invalidation_prob') or 0.0)
    return str(seed), writers, atoms, prob


def build_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for r in as_list(payload.get('results')):
            row = as_dict(r)
            seed, writers, atoms, prob = _stress_key(payload, row)
            groups[(seed, writers, atoms, prob, clean(row.get('condition')))].append(row)
    order = {c: i for i, c in enumerate(ORDER)}
    out: list[dict[str, Any]] = []
    for (seed, writers, atoms, prob, condition), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][3], order.get(item[0][4], 999), item[0][4], item[0][0])):
        runs = len(rows)
        stale_deps = int(sum(float(r.get('stale_dependency_count') or 0) for r in rows))
        review = int(sum(float(r.get('review_burden_count') or 0) for r in rows))
        stale_acc = int(sum(float(r.get('stale_dependency_accepted_count') or 0) for r in rows))
        unsafe = int(sum(float(r.get('unsafe_auto_commit_count') or 0) for r in rows))
        wccu_int = int(sum(float(r.get('wccu_intervention_count') or 0) for r in rows))
        out.append({
            'seed': seed,
            'writers': writers,
            'atom_count': atoms,
            'invalidation_prob': prob,
            'condition': condition,
            'label': DISPLAY_NAMES.get(condition, condition),
            'runs': runs,
            'stale_dependencies': stale_deps,
            'stale_accepted': stale_acc,
            'unsafe': unsafe,
            'wccu_intervention': wccu_int,
            'review': review,
            'review_per_stale_dependency': round(review / stale_deps, 4) if stale_deps else 0.0,
            'safety_pass': sum(1 for r in rows if not r.get('failed') and float(r.get('unsafe_auto_commit_count') or 0) <= 0 and float(r.get('stale_dependency_accepted_count') or 0) <= 0),
            'failed_runs': sum(1 for r in rows if r.get('failed')),
        })
    return out


def build_pivot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact pivot for plots: one row per setting/condition."""
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
        r'\begin{tabular}{rrlrrrr}',
        r'\toprule',
        r'Writers & $p_{inv}$ & Condition & Stale acc. & Unsafe & WCCU int. & Review \\',
        r'\midrule',
    ]
    for r in rows:
        name = str(r['label']).replace('_', r'\_')
        lines.append(f"{r['writers']} & {float(r['invalidation_prob']):.2f} & {name} & {r['stale_accepted']} & {r['unsafe']} & {r['wccu_intervention']} & {r['review']} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create a curve/sweep table from randomized WCCU stress result JSON files.')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-csv', default='analysis/wccu_stress_curve_table.csv')
    parser.add_argument('--out-tex', default='analysis/wccu_stress_curve_table.tex')
    args = parser.parse_args(argv)
    rows = build_rows([_load(p) for p in args.inputs])
    write_csv(Path(args.out_csv), rows)
    if args.out_tex:
        write_tex(Path(args.out_tex), rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
