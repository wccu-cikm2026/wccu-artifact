from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from wccu_eval.scripts.coordination_quality import row_quality
from wccu_eval.utils import as_list, clean, ensure_dir, read_json, read_jsonl

LABELS = {
    'adaptive_wccu_execution_trace': 'WCCU',
    'adaptive_readset_occ': 'Read-set OCC',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_review_gated': 'Review-gated',
}
KIND_LABELS = {
    'freshness': 'Freshness',
    'target': 'Target',
    'authority': 'Authority',
    'delta': 'Operation label',
    'view': 'Derived view',
}
KIND_ORDER = ['freshness', 'target', 'authority', 'delta', 'view']
COND_ORDER = ['adaptive_wccu_execution_trace', 'adaptive_readset_occ', 'adaptive_policy', 'uniform_review_gated']


def _rows_from_path(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == '.jsonl':
        return [r for r in read_jsonl(p) if isinstance(r, dict)]
    data = read_json(p, {})
    return [r for r in as_list(data.get('results')) if isinstance(r, dict)]


def summarize(paths: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in paths:
        for row in _rows_from_path(path):
            groups.setdefault((clean(row.get('obligation_kind')), clean(row.get('condition'))), []).append(row)
    out: list[dict[str, Any]] = []
    for (kind, cond), rows in groups.items():
        runs = len(rows)
        event_obs = sum(1 for r in rows if bool(r.get('expected_event_observed')))
        review = sum(int(r.get('review_burden_count') or 0) for r in rows)
        stale_acc = sum(int(r.get('stale_dependency_accepted_count') or 0) for r in rows)
        unsafe = sum(int(r.get('unsafe_auto_commit_count') or 0) for r in rows)
        committed = sum(int((r.get('commit') or {}).get('committed') or 0) for r in rows)
        view = sum(int(r.get('view_invalidation_count') or 0) for r in rows)
        quality_rows = [row_quality(r) for r in rows]
        total_issues = sum(int(q['issue_count']) for q in quality_rows)
        total_holds_required = sum(int(q['hold_required_count']) for q in quality_rows)
        total_held = sum(int(q['review_block_count']) for q in quality_rows)
        total_problem_held = sum(int(q['problematic_held_count']) for q in quality_rows)
        total_false_hold = sum(int(q['false_hold_count']) for q in quality_rows)
        total_issue_accepted = sum(int(q['issue_accepted_count']) for q in quality_rows)
        total_safe = sum(int(q['safe_write_count']) for q in quality_rows)
        total_safe_auto = sum(int(q['safe_auto_commit_count']) for q in quality_rows)
        precision = total_problem_held / total_held if total_held else (1.0 if total_issues == 0 else 0.0)
        recall = total_problem_held / total_holds_required if total_holds_required else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        safe_progress = total_safe_auto / total_safe if total_safe else 1.0
        overcoord = total_false_hold / total_safe if total_safe else 0.0
        coverage = 1.0 - (total_issue_accepted / total_issues) if total_issues else 1.0
        out.append({
            'obligation_kind': kind,
            'obligation_label': KIND_LABELS.get(kind, kind),
            'condition': cond,
            'condition_label': LABELS.get(cond, cond),
            'runs': runs,
            'expected_event_observed': event_obs,
            'review_block': review,
            'stale_accepted': stale_acc,
            'unsafe': unsafe,
            'committed': committed,
            'view_invalidations': view,
            'issue_count': total_issues,
            'issue_accepted': total_issue_accepted,
            'coordination_precision': round(precision, 4),
            'coordination_recall': round(recall, 4),
            'coordination_f1': round(f1, 4),
            'safe_automatic_progress': round(safe_progress, 4),
            'over_coordination_rate': round(overcoord, 4),
            'obligation_coverage': round(coverage, 4),
        })
    return sorted(out, key=lambda r: (KIND_ORDER.index(r['obligation_kind']) if r['obligation_kind'] in KIND_ORDER else 999, COND_ORDER.index(r['condition']) if r['condition'] in COND_ORDER else 999, r['condition']))


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    fields = ['obligation_label', 'condition_label', 'runs', 'expected_event_observed', 'review_block', 'issue_count', 'issue_accepted', 'obligation_coverage', 'coordination_precision', 'coordination_recall', 'coordination_f1', 'safe_automatic_progress', 'over_coordination_rate', 'stale_accepted', 'unsafe', 'committed', 'view_invalidations']
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def write_tex(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    # Compact paper table: one row per obligation with key conditions side-by-side.
    by_kind: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_kind.setdefault(r['obligation_kind'], {})[r['condition']] = r
    lines = [
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Obl. & WCCU det. & WCCU cov. & RS-OCC cov. & No-WCCU unsafe \\',
        r'\midrule',
    ]
    for kind in KIND_ORDER:
        group = by_kind.get(kind, {})
        wccu = group.get('adaptive_wccu_execution_trace', {})
        rs = group.get('adaptive_readset_occ', {})
        no = group.get('adaptive_policy', {})
        lines.append(
            f"{KIND_LABELS.get(kind, kind)} & {int(wccu.get('expected_event_observed') or 0)}/{int(wccu.get('runs') or 0)} & "
            f"{float(wccu.get('obligation_coverage') or 0):.2f} & {float(rs.get('obligation_coverage') or 0):.2f} & "
            f"{int(no.get('issue_accepted') or 0)} " + r'\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Summarize WCCU obligation-matrix diagnostics.')
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--out-csv')
    parser.add_argument('--out-tex')
    args = parser.parse_args(argv)
    rows = summarize(args.paths)
    if args.out_csv:
        write_csv(args.out_csv, rows)
    if args.out_tex:
        write_tex(args.out_tex, rows)
    if not args.out_csv and not args.out_tex:
        for row in rows:
            print(row)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
