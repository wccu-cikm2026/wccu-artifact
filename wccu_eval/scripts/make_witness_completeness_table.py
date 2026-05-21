from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_list, clean, ensure_dir, read_json, read_jsonl

LABELS = {
    'adaptive_wccu_execution_trace': 'WCCU, execution witness',
    'adaptive_wccu_projection_trace': 'WCCU, projection witness',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_readset_occ': 'Read-set OCC',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_snapshot_occ': 'Snapshot OCC',
    'uniform_review_gated': 'Review-gated',
}
ORDER = list(LABELS)


def _rows_from_path(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == '.jsonl':
        return [r for r in read_jsonl(p) if isinstance(r, dict)]
    data = read_json(p, {})
    return [r for r in as_list(data.get('results')) if isinstance(r, dict)]


def _fresh(row: dict[str, Any]) -> bool:
    return not row.get('failed') and int(row.get('unsafe_auto_commit_count') or 0) == 0 and int(row.get('stale_dependency_accepted_count') or 0) == 0


def summarize(paths: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for path in paths:
        for row in _rows_from_path(path):
            groups.setdefault((float(row.get('witness_drop_rate') or 0.0), clean(row.get('condition'))), []).append(row)
    out: list[dict[str, Any]] = []
    for (drop, cond), rows in groups.items():
        runs = len(rows)
        stale = sum(int(r.get('stale_dependency_count') or 0) for r in rows)
        stale_acc = sum(int(r.get('stale_dependency_accepted_count') or 0) for r in rows)
        unsafe = sum(int(r.get('unsafe_auto_commit_count') or 0) for r in rows)
        wccu_int = sum(int(r.get('wccu_intervention_count') or 0) for r in rows)
        readset_int = sum(int(r.get('readset_occ_review_count') or 0) for r in rows)
        review = sum(int(r.get('review_burden_count') or 0) for r in rows)
        fresh = sum(1 for r in rows if _fresh(r))
        denom = stale + stale_acc
        false_negative_rate = (stale_acc / denom) if denom else 0.0
        witness_recall_est = (stale / denom) if denom else 0.0
        out.append({
            'witness_drop_rate': drop,
            'condition': cond,
            'condition_label': LABELS.get(cond, cond),
            'runs': runs,
            'freshness_pass': fresh,
            'stale_dependencies_detected': stale,
            'stale_accepted': stale_acc,
            'unsafe': unsafe,
            'wccu_interventions': wccu_int,
            'readset_occ_interventions': readset_int,
            'review_block': review,
            'witness_recall_est': witness_recall_est,
            'false_negative_rate': false_negative_rate,
        })
    return sorted(out, key=lambda r: (r['witness_drop_rate'], ORDER.index(r['condition']) if r['condition'] in ORDER else 999, r['condition']))


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    fields = ['witness_drop_rate', 'condition_label', 'runs', 'freshness_pass', 'stale_dependencies_detected', 'stale_accepted', 'unsafe', 'wccu_interventions', 'readset_occ_interventions', 'review_block', 'witness_recall_est', 'false_negative_rate']
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def write_tex(path: str, rows: list[dict[str, Any]], *, compact: bool = True) -> None:
    ensure_dir(Path(path).parent)
    fields = ['Drop', 'Condition', 'Fresh.', 'Stale acc.', 'Unsafe', 'Review/block']
    lines = ['\\begin{tabular}{rlrrrr}', '\\toprule', ' & '.join(fields) + ' \\\\', '\\midrule']
    for r in rows:
        # Compact default: only show witness/readset/adaptive/OCC/review at drop rates.
        if compact and r['condition'] not in {'adaptive_wccu_execution_trace', 'adaptive_readset_occ', 'adaptive_policy', 'uniform_snapshot_occ', 'uniform_review_gated'}:
            continue
        line = f"{r['witness_drop_rate']:.2f} & {r['condition_label']} & {int(r['freshness_pass'])}/{int(r['runs'])} & {int(r['stale_accepted'])} & {int(r['unsafe'])} & {int(r['review_block'])} " + r"\\"
        lines.append(line)
    lines.extend(['\\bottomrule', '\\end{tabular}', ''])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Make witness-completeness analysis tables.')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', required=True)
    parser.add_argument('--full-tex', action='store_true')
    args = parser.parse_args(argv)
    rows = summarize(args.inputs)
    write_csv(args.out_csv, rows)
    write_tex(args.out_tex, rows, compact=not args.full_tex)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
