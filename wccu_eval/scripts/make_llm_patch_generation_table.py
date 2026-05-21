from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, read_json, read_jsonl

TASK_ORDER = ['independent', 'shared_file_lock', 'commitment_staleness', 'target_ambiguity', 'workspace_lock', 'unknown']


def _rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == '.jsonl':
            rows.extend([r for r in read_jsonl(p) if isinstance(r, dict)])
        else:
            data = read_json(p, {})
            rows.extend([r for r in as_list(as_dict(data).get('generation_rows')) if isinstance(r, dict)])
    return rows


def _validation_ok(row: dict[str, Any]) -> bool:
    if 'patch_validation_ok' in row:
        return bool(row.get('patch_validation_ok'))
    val = as_dict(row.get('patch_validation'))
    return bool(val.get('ok'))


def summarize(paths: list[str], *, group_by_task_type: bool = False) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(paths):
        task_type = clean(row.get('task_type')) or 'unknown'
        key = task_type if group_by_task_type else 'all'
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in groups.items():
        n = len(rows)
        parsed = sum(1 for r in rows if r.get('patch_parse_success'))
        edit_parsed = sum(1 for r in rows if r.get('edit_parse_success'))
        materialized = sum(1 for r in rows if r.get('materialized_patch_valid'))
        valid = sum(1 for r in rows if _validation_ok(r))
        failed = n - valid
        prompt_tokens_est = sum(int(as_dict(r.get('llm_generation')).get('prompt_tokens_est') or 0) for r in rows)
        output_tokens_est = sum(int(as_dict(r.get('llm_generation')).get('output_tokens_est') or 0) for r in rows)
        out.append({
            'task_type': key,
            'patches': n,
            'parse_success': parsed,
            'edit_parse_success': edit_parsed,
            'materialized_patch_valid': materialized,
            'validation_success': valid,
            'validation_failures': failed,
            'parse_rate': parsed / n if n else 0.0,
            'edit_parse_rate': edit_parsed / n if n else 0.0,
            'materialized_rate': materialized / n if n else 0.0,
            'validation_rate': valid / n if n else 0.0,
            'prompt_tokens_est': prompt_tokens_est,
            'output_tokens_est': output_tokens_est,
        })
    return sorted(out, key=lambda r: TASK_ORDER.index(r['task_type']) if r['task_type'] in TASK_ORDER else 999)


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    fields = ['task_type', 'patches', 'parse_success', 'edit_parse_success', 'materialized_patch_valid', 'validation_success', 'validation_failures', 'parse_rate', 'edit_parse_rate', 'materialized_rate', 'validation_rate', 'prompt_tokens_est', 'output_tokens_est']
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            row = dict(row)
            row['parse_rate'] = f"{float(row.get('parse_rate') or 0):.2f}"
            row['edit_parse_rate'] = f"{float(row.get('edit_parse_rate') or 0):.2f}"
            row['materialized_rate'] = f"{float(row.get('materialized_rate') or 0):.2f}"
            row['validation_rate'] = f"{float(row.get('validation_rate') or 0):.2f}"
            w.writerow({k: row.get(k) for k in fields})


def write_tex(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    lines = [
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        r'Task type & Patches & Parsed & Edits & Materialized & Valid & Valid rate \\',
        r'\midrule',
    ]
    for r in rows:
        task = str(r['task_type']).replace('_', r'\_')
        lines.append(f"{task} & {int(r['patches'])} & {int(r['parse_success'])} & {int(r.get('edit_parse_success') or 0)} & {int(r.get('materialized_patch_valid') or 0)} & {int(r['validation_success'])} & {float(r['validation_rate']):.2f} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Summarize LLM e2e patch generation parse/validation rates.')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', required=True)
    parser.add_argument('--by-task-type-csv')
    parser.add_argument('--by-task-type-tex')
    args = parser.parse_args(argv)
    rows = summarize(args.inputs, group_by_task_type=False)
    write_csv(args.out_csv, rows)
    write_tex(args.out_tex, rows)
    by_type = summarize(args.inputs, group_by_task_type=True)
    if args.by_task_type_csv:
        write_csv(args.by_task_type_csv, by_type)
    if args.by_task_type_tex:
        write_tex(args.by_task_type_tex, by_type)
    print(json.dumps({'ok': True, 'rows': len(rows), 'by_task_type_rows': len(by_type), 'out_csv': args.out_csv}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
