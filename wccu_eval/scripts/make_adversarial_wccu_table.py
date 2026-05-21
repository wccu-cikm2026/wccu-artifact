from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_list, clean, ensure_dir, read_json, read_jsonl

LABELS = {
    'adaptive_wccu_execution_trace': 'WCCU, execution witness',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_readset_occ': 'Read-set OCC',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_review_gated': 'Review-gated',
}

# Event groups used in the LaTeX table.  The verifier may emit a narrow
# mismatch event or a policy-level review event; both are useful for showing
# that WCCU explains *why* an update was held rather than only saying that it
# was held.
STALE_EVENT_KINDS = {'stale_read_dependency'}
TARGET_EVENT_KINDS = {'wrong_target_certificate'}
AUTH_EVENT_KINDS = {'authority_certificate_mismatch', 'authority_insufficient_for_direct_commit'}
OP_EVENT_KINDS = {'delta_contract_mismatch', 'weakening_delta_requires_review'}
VIEW_EVENT_KINDS = {'view_invalidation_required'}


def _rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == '.jsonl':
        return [r for r in read_jsonl(p) if isinstance(r, dict)]
    return [r for r in as_list(read_json(p, {}).get('results')) if isinstance(r, dict)]


def _event_kinds(row: dict[str, Any]) -> list[str]:
    kinds: list[str] = []
    for event in as_list(row.get('wccu_events')):
        for e in as_list(event.get('errors')) + as_list(event.get('warnings')):
            k = clean(e.get('kind'))
            if k:
                kinds.append(k)
    return kinds


def _has_any_event(row: dict[str, Any], kinds: set[str]) -> bool:
    return bool(set(_event_kinds(row)) & kinds)


def _count_rows_with_event(rows: list[dict[str, Any]], kinds: set[str]) -> int:
    return sum(1 for r in rows if _has_any_event(r, kinds))


def summarize(inputs: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in inputs:
        for row in _rows(path):
            groups.setdefault((clean(row.get('adversarial_kind') or row.get('scenario_id')), clean(row.get('condition'))), []).append(row)
    out=[]
    for (kind, cond), rows in groups.items():
        stale_events = _count_rows_with_event(rows, STALE_EVENT_KINDS)
        target_events = _count_rows_with_event(rows, TARGET_EVENT_KINDS)
        auth_events = _count_rows_with_event(rows, AUTH_EVENT_KINDS)
        operation_events = _count_rows_with_event(rows, OP_EVENT_KINDS)
        view_events = _count_rows_with_event(rows, VIEW_EVENT_KINDS)
        explicit_event_hits = stale_events + target_events + auth_events + operation_events + view_events
        out.append({
            'adversarial_kind': kind,
            'condition': cond,
            'condition_label': LABELS.get(cond, cond),
            'runs': len(rows),
            'freshness_pass': sum(1 for r in rows if int(r.get('unsafe_auto_commit_count') or 0) == 0 and int(r.get('stale_dependency_accepted_count') or 0) == 0),
            'stale_accepted': sum(int(r.get('stale_dependency_accepted_count') or 0) for r in rows),
            'unsafe': sum(int(r.get('unsafe_auto_commit_count') or 0) for r in rows),
            'review_block': sum(int(r.get('review_burden_count') or 0) for r in rows),
            'wccu_interventions': sum(int(r.get('wccu_intervention_count') or 0) for r in rows),
            'stale_read_events': stale_events,
            'wrong_target_events': target_events,
            'authority_events': auth_events,
            'authority_mismatch_events': auth_events,  # Backward-compatible CSV column name.
            'operation_events': operation_events,
            'delta_mismatch_events': operation_events,  # Backward-compatible CSV column name.
            'view_events': view_events,
            'explicit_event_hits': explicit_event_hits,
        })
    return sorted(out, key=lambda r: (r['adversarial_kind'], r['condition_label']))


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    fields = [
        'adversarial_kind','condition','condition_label','runs','freshness_pass',
        'stale_accepted','unsafe','review_block','wccu_interventions',
        'stale_read_events','wrong_target_events','authority_events',
        'authority_mismatch_events','operation_events','delta_mismatch_events',
        'view_events','explicit_event_hits',
    ]
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k) for k in fields})


def _tex_escape(s: Any) -> str:
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def write_tex(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    lines=[
        r'\begin{tabular}{llrrrrrrrr}',
        r'\toprule',
        r'Attack & Condition & Fresh. & Unsafe & Review & Stale ev. & Target ev. & Auth ev. & Op. ev. & View ev. \\',
        r'\midrule',
    ]
    shown_conditions = {'adaptive_wccu_execution_trace','adaptive_wccu_model_certificate','adaptive_readset_occ','adaptive_policy'}
    for r in rows:
        if r['condition'] not in shown_conditions:
            continue
        lines.append(
            f"{_tex_escape(r['adversarial_kind'])} & {_tex_escape(r['condition_label'])} & "
            f"{int(r['freshness_pass'])}/{int(r['runs'])} & {int(r['unsafe'])} & {int(r['review_block'])} & "
            f"{int(r['stale_read_events'])} & {int(r['wrong_target_events'])} & {int(r['authority_events'])} & "
            f"{int(r['operation_events'])} & {int(r['view_events'])} " + r'\\'
        )
    lines.extend([r'\bottomrule',r'\end{tabular}',''])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='Make adversarial WCCU certificate/event tables.')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', required=True)
    args=parser.parse_args(argv)
    rows=summarize(args.inputs)
    write_csv(args.out_csv, rows)
    write_tex(args.out_tex, rows)
    print(json.dumps({'ok':True,'rows':len(rows),'out_csv':args.out_csv,'out_tex':args.out_tex}, indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
