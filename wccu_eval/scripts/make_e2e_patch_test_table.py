from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, read_json, read_jsonl

LABELS = {
    'adaptive_wccu_execution_trace': 'WCCU, execution witness',
    'adaptive_readset_occ': 'Read-set OCC',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_review_gated': 'Review-gated',
    'uniform_snapshot_occ': 'Snapshot OCC',
}
ORDER = list(LABELS)
TASK_ORDER = ['independent', 'shared_file_lock', 'commitment_staleness', 'target_ambiguity', 'workspace_lock']


def _rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == '.jsonl':
        return [r for r in read_jsonl(p) if isinstance(r, dict)]
    data = read_json(p, {})
    return [r for r in as_list(data.get('results')) if isinstance(r, dict)]


def _task_type(row: dict[str, Any]) -> str:
    value = clean(row.get('task_type'))
    if value and value != 'unknown':
        return value
    task_id = clean(row.get('task_id'))
    for prefix, typ in [
        ('toy_independent', 'independent'),
        ('toy_shared_lock', 'shared_file_lock'),
        ('toy_commitment', 'commitment_staleness'),
        ('toy_target_ambiguity', 'target_ambiguity'),
    ]:
        if task_id.startswith(prefix):
            return typ
    return value or 'unknown'


def summarize(paths: list[str], *, group_by_task_type: bool = False) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in paths:
        for row in _rows(path):
            cond = clean(row.get('condition'))
            task_type = _task_type(row) if group_by_task_type else 'all'
            groups.setdefault((task_type, cond), []).append(row)
    out: list[dict[str, Any]] = []
    for (task_type, cond), rows in groups.items():
        n = len(rows)
        patches_total = sum(int(r.get('patches_total') or 0) for r in rows)
        policy_auto = sum(int(r.get('patches_policy_auto') if r.get('patches_policy_auto') is not None else r.get('patches_auto_applied') or 0) for r in rows)
        attempted = sum(int(r.get('patches_apply_attempted') if r.get('patches_apply_attempted') is not None else r.get('patches_auto_applied') or 0) for r in rows)
        applied_success = sum(int(r.get('patches_applied_successfully') if r.get('patches_applied_successfully') is not None else r.get('patches_auto_applied') or 0) for r in rows)
        auto = applied_success
        held = sum(int(r.get('patches_held_for_review') or 0) for r in rows)
        patch_fail = sum(int(r.get('patch_apply_failures') or 0) for r in rows)
        tests = sum(1 for r in rows if r.get('tests_passed'))
        freshness = sum(1 for r in rows if r.get('freshness_pass'))
        stale = sum(int(r.get('stale_dependency_accepted_count') or 0) for r in rows)
        unsafe = sum(int(r.get('unsafe_auto_commit_count') or 0) for r in rows)
        review = sum(int(r.get('review_burden_count') or 0) for r in rows)
        useful_safe_runs = sum(1 for r in rows if r.get('tests_passed') and r.get('freshness_pass') and int(r.get('patches_auto_applied') or 0) > 0 and int(r.get('patch_apply_failures') or 0) == 0)
        probe_failures = sum(1 for r in rows if r.get('commitment_probe_detected_stale_failure'))
        discovered_tests = sum(int(as_dict(r.get('test_discovery_after')).get('test_function_count') or 0) for r in rows)
        out.append({
            'task_type': task_type,
            'condition': cond,
            'condition_label': LABELS.get(cond, cond),
            'runs': n,
            'patches_total': patches_total,
            'freshness_pass': freshness,
            'tests_passed': tests,
            'patch_apply_failures': patch_fail,
            'patches_policy_auto': policy_auto,
            'patches_apply_attempted': attempted,
            'patches_applied_successfully': applied_success,
            'patches_auto_applied': applied_success,
            'patches_held_for_review': held,
            'auto_progress_rate': applied_success / patches_total if patches_total else 0.0,
            'stale_accepted': stale,
            'unsafe': unsafe,
            'review_block': review,
            'useful_safe_runs': useful_safe_runs,
            'commitment_probe_failures': probe_failures,
            'discovered_test_functions': discovered_tests,
        })
    return sorted(out, key=lambda r: (
        TASK_ORDER.index(r['task_type']) if r['task_type'] in TASK_ORDER else 999,
        ORDER.index(r['condition']) if r['condition'] in ORDER else 999,
        r['task_type'], r['condition'],
    ))


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    fields = [
        'task_type','condition_label','runs','patches_total','freshness_pass','tests_passed','patch_apply_failures',
        'patches_policy_auto','patches_apply_attempted','patches_applied_successfully','patches_auto_applied','patches_held_for_review','auto_progress_rate','stale_accepted','unsafe','review_block','useful_safe_runs','commitment_probe_failures','discovered_test_functions'
    ]
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            row = dict(row)
            row['auto_progress_rate'] = f"{float(row.get('auto_progress_rate') or 0):.2f}"
            w.writerow({k: row.get(k) for k in fields})


def write_tex(path: str, rows: list[dict[str, Any]], *, include_task_type: bool = False) -> None:
    ensure_dir(Path(path).parent)
    if include_task_type:
        lines = ['\\begin{tabular}{llrrrrrrr}', '\\toprule', 'Task type & Condition & Fresh. & Tests & Patch fail & Applied & Held & Stale & Probe fail \\\\', '\\midrule']
        last_task = None
        for r in rows:
            task = r['task_type'].replace('_', '\\_')
            if last_task is not None and task != last_task:
                lines.append('\\addlinespace')
            lines.append(f"{task} & {r['condition_label']} & {int(r['freshness_pass'])}/{int(r['runs'])} & {int(r['tests_passed'])}/{int(r['runs'])} & {int(r['patch_apply_failures'])} & {int(r['patches_auto_applied'])} & {int(r['patches_held_for_review'])} & {int(r['stale_accepted'])} & {int(r.get('commitment_probe_failures') or 0)} " + r"\\")
            last_task = task
        lines.extend(['\\bottomrule','\\end{tabular}',''])
    else:
        lines = ['\\begin{tabular}{lrrrrrrrr}', '\\toprule', 'Condition & Fresh. & Tests & Patch fail & Applied & Held & Stale & Unsafe & Probe fail \\\\', '\\midrule']
        for r in rows:
            lines.append(f"{r['condition_label']} & {int(r['freshness_pass'])}/{int(r['runs'])} & {int(r['tests_passed'])}/{int(r['runs'])} & {int(r['patch_apply_failures'])} & {int(r['patches_auto_applied'])} & {int(r['patches_held_for_review'])} & {int(r['stale_accepted'])} & {int(r['unsafe'])} & {int(r.get('commitment_probe_failures') or 0)} " + r"\\")
        lines.extend(['\\bottomrule','\\end{tabular}',''])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Make end-to-end patch/test summary table.')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', required=True)
    parser.add_argument('--by-task-type-csv')
    parser.add_argument('--by-task-type-tex')
    args = parser.parse_args(argv)
    rows = summarize(args.inputs, group_by_task_type=False)
    write_csv(args.out_csv, rows)
    write_tex(args.out_tex, rows)
    by_type_rows = summarize(args.inputs, group_by_task_type=True)
    if args.by_task_type_csv:
        write_csv(args.by_task_type_csv, by_type_rows)
    if args.by_task_type_tex:
        write_tex(args.by_task_type_tex, by_type_rows, include_task_type=True)
    print(json.dumps({'ok': True, 'rows': len(rows), 'by_task_type_rows': len(by_type_rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
