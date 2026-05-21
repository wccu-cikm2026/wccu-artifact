from __future__ import annotations

"""Summarize WCCU obligation ablations from live-LLM result files.

This script is intentionally analysis-only: it does not create synthetic rows or
run a mock model.  It reads result JSON files produced by the CooperBench-derived
or live LLM obligation runners and reports how each live condition behaves under
its obligation/witness configuration.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, mean, write_json


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    payload = json.loads(p.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        return [as_dict(x) for x in payload]
    return [as_dict(x) for x in as_list(payload.get('results'))]


def _condition_label(condition: str) -> str:
    c = clean(condition)
    mapping = {
        'adaptive_wccu_execution_trace': 'WCCU: execution witness',
        'adaptive_wccu_projection_trace': 'WCCU: projection witness',
        'adaptive_wccu_model_certificate': 'WCCU: model certificate only',
        'adaptive_wccu_unguided_certificate': 'WCCU: unguided model certificate',
        'adaptive_wccu_no_read_validation': 'WCCU ablation: no freshness check',
        'adaptive_readset_occ': 'Read-set OCC: freshness only',
        'uniform_snapshot_occ': 'Snapshot OCC: target/snapshot only',
        'adaptive_policy': 'Adaptive policy: no WCCU obligations',
        'uniform_review_gated': 'Uniform review',
        'uniform_append_only': 'Append-only',
    }
    return mapping.get(c, c)


def _obligation_surface(condition: str) -> str:
    c = clean(condition)
    if c == 'adaptive_wccu_execution_trace':
        return 'read+target+op+auth+fresh+view; execution witnesses'
    if c == 'adaptive_wccu_projection_trace':
        return 'read+target+op+auth+fresh+view; projection witnesses'
    if c == 'adaptive_wccu_model_certificate':
        return 'read+target+op+auth+fresh+view; model-supplied certificate only'
    if c == 'adaptive_wccu_unguided_certificate':
        return 'read+target+op+auth+fresh+view; unguided model certificate'
    if c == 'adaptive_wccu_no_read_validation':
        return 'target+op+auth+view; freshness disabled'
    if c == 'adaptive_readset_occ':
        return 'freshness over runtime read set only'
    if c == 'uniform_snapshot_occ':
        return 'same-target/snapshot validation only'
    if c == 'adaptive_policy':
        return 'type/risk/lock policy only; no WCCU obligations'
    if c == 'uniform_review_gated':
        return 'all writes held for review'
    if c == 'uniform_append_only':
        return 'all writes directly appended/committed'
    return 'custom'


def _scenario_family(row: dict[str, Any]) -> str:
    return clean(
        row.get('llm_obligation_family')
        or row.get('obligation_kind')
        or row.get('task_type')
        or row.get('expected_conflict_type')
        or row.get('scenario_family')
        or row.get('external_benchmark')
        or 'all'
    )


def _issue_accept(row: dict[str, Any]) -> float:
    # CooperBench commitment rows use stale_dependency_accepted_count; LLM
    # obligation rows also expose ground_truth_issue_accepted_count.  Use the
    # strongest available live-run signal without reading oracle-only files.
    for key in ('ground_truth_issue_accepted_count', 'stale_dependency_accepted_count', 'unsafe_auto_commit_count'):
        value = row.get(key)
        if value is not None:
            return 1.0 if int(value or 0) > 0 else 0.0
    return 0.0


def _safe_progress(row: dict[str, Any]) -> float | None:
    # Safe-update rows in the obligation benchmark mark ground_truth_safe_count;
    # CooperBench workspace rows do not have a ground-truth safe class, so leave
    # blank rather than inventing a safe denominator.
    if int(row.get('ground_truth_safe_count') or 0) > 0:
        return 1.0 if int(row.get('commit', {}).get('committed') or row.get('commit_committed') or 0) > 0 else 0.0
    return None


def summarize(paths: list[str], *, group_by_family: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _load_rows(path):
            if row.get('failed'):
                continue
            row = dict(row)
            row['_source_file'] = str(path)
            rows.append(row)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cond = clean(row.get('condition') or row.get('policy_mode'))
        if not cond:
            continue
        family = _scenario_family(row) if group_by_family else 'all'
        groups[(family, cond)].append(row)
    out: list[dict[str, Any]] = []
    for (family, cond), group in sorted(groups.items()):
        issue_vals = [_issue_accept(r) for r in group]
        safe_vals = [_safe_progress(r) for r in group]
        safe_vals = [v for v in safe_vals if v is not None]
        out.append({
            'family': family,
            'condition': cond,
            'label': _condition_label(cond),
            'obligation_surface': _obligation_surface(cond),
            'n': len(group),
            'issue_accept_rate': mean(issue_vals),
            'safe_auto_commit_rate': mean(safe_vals) if safe_vals else '',
            'mean_review_burden': mean([int(r.get('review_burden_count') or 0) for r in group]),
            'mean_wccu_interventions': mean([int(r.get('wccu_intervention_count') or 0) for r in group]),
            'mean_stale_dependencies': mean([int(r.get('stale_dependency_count') or 0) for r in group]),
            'provider_error_count': sum(1 for r in group if r.get('error_type') == 'LlmProviderError'),
        })
    return out


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    keys = list(rows[0].keys()) if rows else ['family', 'condition', 'n']
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    lines = [
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Family & Condition & N & Issue acc. & Review/run & WCCU int./run \\',
        r'\midrule',
    ]
    row_end = chr(92) * 2
    for r in rows:
        fam = str(r['family']).replace('_', r'\_')
        label = str(r['label']).replace('_', r'\_')
        issue = f"{float(r['issue_accept_rate']):.3f}" if r['issue_accept_rate'] != '' else '--'
        review = f"{float(r['mean_review_burden']):.2f}"
        intr = f"{float(r['mean_wccu_interventions']):.2f}"
        lines.append(f"{fam} & {label} & {r['n']} & {issue} & {review} & {intr} " + row_end)
    lines += [r'\bottomrule', r'\end{tabular}']
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Make WCCU obligation/witness ablation table from live-LLM CooperBench or obligation results.')
    parser.add_argument('inputs', nargs='+', help='Result JSON/JSONL files from live LLM runs.')
    parser.add_argument('--group-by-family', action='store_true')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-json', default='')
    parser.add_argument('--out-tex', default='')
    args = parser.parse_args(argv)
    rows = summarize(args.inputs, group_by_family=args.group_by_family)
    write_csv(rows, args.out_csv)
    if args.out_tex:
        write_tex(rows, args.out_tex)
    if args.out_json:
        write_json(Path(args.out_json), {'kind': 'wccu_live_llm_ablation_summary_v1', 'inputs': args.inputs, 'group_by_family': args.group_by_family, 'rows': rows})
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_json': args.out_json, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
