from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir

DISPLAY_NAMES = {
    'adaptive_no_candidates_no_grounding': 'No candidates / no grounding',
    'adaptive_no_candidates_with_grounding': 'No candidates / grounding',
    'adaptive_candidates_no_grounding': 'Candidates / no grounding',
    'adaptive_candidates_with_grounding': 'Candidates / grounding',
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_policy': 'Adaptive, no WCCU',
}
ORDER = [
    'adaptive_no_candidates_no_grounding',
    'adaptive_no_candidates_with_grounding',
    'adaptive_candidates_no_grounding',
    'adaptive_candidates_with_grounding',
    'adaptive_policy',
    'adaptive_wccu_execution_trace',
]


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _iter_intents(row: dict[str, Any]):
    for run in as_list(row.get('agentRuns')):
        for intent in as_list(run.get('write_intents')):
            yield run, as_dict(intent)


def _token_total(row: dict[str, Any]) -> int:
    total = 0
    for run in as_list(row.get('agentRuns')):
        usage = as_dict(as_dict(run.get('llm')).get('api_usage'))
        total += int(usage.get('total_tokens') or usage.get('total') or 0)
    return total


def build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in as_list(payload.get('results')):
        groups[clean(as_dict(r).get('condition'))].append(as_dict(r))
    order = {c: i for i, c in enumerate(ORDER)}
    out: list[dict[str, Any]] = []
    for condition in sorted(groups, key=lambda c: (order.get(c, 999), c)):
        rows = groups[condition]
        intents = grounded = unresolved = target_events = 0
        for row in rows:
            target_events += int(float(row.get('wrong_target_count') or 0) + float(row.get('low_target_confidence_count') or 0))
            for _, intent in _iter_intents(row):
                if clean(intent.get('intent_type')) == 'append_event':
                    continue
                tg = as_dict(intent.get('target_grounding'))
                if tg:
                    intents += 1
                    if tg.get('resolved'):
                        grounded += 1
                    else:
                        unresolved += 1
        n = len(rows)
        out.append({
            'condition': condition,
            'label': DISPLAY_NAMES.get(condition, condition),
            'runs': n,
            'diagnostic_pass': sum(1 for r in rows if r.get('task_success')),
            'safety_pass': sum(1 for r in rows if not r.get('failed') and float(r.get('unsafe_auto_commit_count') or 0) <= 0),
            'unsafe': int(sum(float(r.get('unsafe_auto_commit_count') or 0) for r in rows)),
            'review': int(sum(float(r.get('review_burden_count') or 0) for r in rows)),
            'wrong_or_low_target_events': target_events,
            'target_grounded_rate': round(grounded / intents, 4) if intents else 0.0,
            'target_unresolved_rate': round(unresolved / intents, 4) if intents else 0.0,
            'tokens_per_run': round(sum(_token_total(r) for r in rows) / n, 2) if n else 0.0,
            'failed_runs': sum(1 for r in rows if r.get('failed')),
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
        r'Condition & Pass & Unsafe & Target events & Grounded \\',
        r'\midrule',
    ]
    for r in rows:
        name = str(r['label']).replace('_', r'\_')
        lines.append(f"{name} & {r['diagnostic_pass']}/{r['runs']} & {r['unsafe']} & {r['wrong_or_low_target_events']} & {100*float(r['target_grounded_rate']):.1f}\\% " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create a compact target-candidate/grounding ablation table from an LLM result JSON.')
    parser.add_argument('input')
    parser.add_argument('--out-csv', default='analysis/target_ablation_table.csv')
    parser.add_argument('--out-tex', default='analysis/target_ablation_table.tex')
    args = parser.parse_args(argv)
    rows = build_rows(_load(args.input))
    write_csv(Path(args.out_csv), rows)
    if args.out_tex:
        write_tex(Path(args.out_tex), rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
