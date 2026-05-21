from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _event_kinds(event: dict[str, Any]) -> list[str]:
    kinds: list[str] = []
    for err in as_list(event.get('errors')):
        k = clean(as_dict(err).get('kind'))
        if k:
            kinds.append(k)
    for warn in as_list(event.get('warnings')):
        k = clean(as_dict(warn).get('kind'))
        if k:
            kinds.append(k)
    return kinds


def build_report(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for row in as_list(payload.get('results')):
        r = as_dict(row)
        condition = clean(r.get('condition')) or 'unknown'
        acc = summary.setdefault(condition, {
            'condition': condition,
            'runs': 0,
            'failed_runs': 0,
            'wrong_target_count': 0,
            'low_target_confidence_count': 0,
            'certificate_invalid_count': 0,
            'target_events': 0,
            'target_event_scenarios': set(),
            'target_event_kinds': Counter(),
        })
        acc['runs'] += 1
        if r.get('failed'):
            acc['failed_runs'] += 1
        acc['wrong_target_count'] += int(float(r.get('wrong_target_count') or 0))
        acc['low_target_confidence_count'] += int(float(r.get('low_target_confidence_count') or 0))
        acc['certificate_invalid_count'] += int(float(r.get('certificate_invalid_count') or 0))
        for ev in as_list(r.get('wccu_events')):
            event = as_dict(ev)
            kinds = _event_kinds(event)
            is_target_event = any(k in {'wrong_target_certificate', 'low_target_confidence', 'target_not_visible', 'target_resolution_failed'} or 'target' in k for k in kinds)
            if not is_target_event:
                continue
            acc['target_events'] += 1
            acc['target_event_scenarios'].add(clean(r.get('scenario_id')))
            for k in kinds:
                acc['target_event_kinds'][k] += 1
            cert = as_dict(as_dict(event.get('certificate')).get('target_certificate'))
            cases.append({
                'scenario_id': clean(r.get('scenario_id')),
                'condition': condition,
                'repetition': r.get('repetition', ''),
                'source_result_file': clean(r.get('source_result_file')),
                'source_agent': clean(event.get('source_agent')),
                'intent_id': clean(event.get('intent_id')),
                'event_action': clean(event.get('action')),
                'target_id': clean(event.get('target_id')),
                'event_kinds': ';'.join(kinds),
                'obligation_failures': ';'.join(clean(x) for x in as_list(event.get('obligation_failures'))),
                'claimed_target_id': clean(cert.get('claimed_target_id')),
                'raw_target_string': clean(cert.get('raw_target_string')),
                'target_confidence': cert.get('confidence', ''),
                'errors_json': json.dumps(as_list(event.get('errors')), ensure_ascii=False, sort_keys=True),
                'warnings_json': json.dumps(as_list(event.get('warnings')), ensure_ascii=False, sort_keys=True),
            })
    rows: list[dict[str, Any]] = []
    for condition, acc in sorted(summary.items()):
        rows.append({
            'condition': condition,
            'runs': acc['runs'],
            'failed_runs': acc['failed_runs'],
            'wrong_target_count': acc['wrong_target_count'],
            'low_target_confidence_count': acc['low_target_confidence_count'],
            'certificate_invalid_count': acc['certificate_invalid_count'],
            'target_events': acc['target_events'],
            'target_event_scenario_count': len(acc['target_event_scenarios']),
            'target_event_kinds': json.dumps(dict(acc['target_event_kinds']), sort_keys=True),
        })
    return rows, cases


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def write_markdown(path: Path, summary: list[dict[str, Any]], cases: list[dict[str, Any]], *, max_cases: int = 20) -> None:
    ensure_dir(path.parent)
    lines = ['# Target grounding and certificate case report', '']
    lines.append('## Summary by condition')
    lines.append('')
    lines.append('| Condition | Runs | Wrong target | Low confidence | Target events | Event scenarios |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for r in summary:
        lines.append(f"| {r['condition']} | {r['runs']} | {r['wrong_target_count']} | {r['low_target_confidence_count']} | {r['target_events']} | {r['target_event_scenario_count']} |")
    lines.append('')
    lines.append('## Representative cases')
    lines.append('')
    if not cases:
        lines.append('No target-grounding certificate events found.')
    for c in cases[:max_cases]:
        lines.append(f"### {c['scenario_id']} / {c['condition']}")
        lines.append('')
        lines.append(f"- Agent: `{c['source_agent']}`")
        lines.append(f"- Runtime target: `{c['target_id']}`")
        lines.append(f"- Claimed target: `{c['claimed_target_id']}`")
        lines.append(f"- Raw target string: `{c['raw_target_string']}`")
        lines.append(f"- Event kinds: `{c['event_kinds']}`")
        lines.append(f"- Action: `{c['event_action']}`")
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Export target-grounding and certificate-mismatch case reports from result JSON files.')
    parser.add_argument('input_json')
    parser.add_argument('--out-summary-csv', default='analysis/target_grounding_summary.csv')
    parser.add_argument('--out-cases-jsonl', default='analysis/target_grounding_cases.jsonl')
    parser.add_argument('--out-md', default='analysis/target_grounding_report.md')
    parser.add_argument('--max-md-cases', type=int, default=20)
    args = parser.parse_args(argv)
    summary, cases = build_report(_load(args.input_json))
    write_csv(Path(args.out_summary_csv), summary)
    write_jsonl(Path(args.out_cases_jsonl), cases)
    write_markdown(Path(args.out_md), summary, cases, max_cases=args.max_md_cases)
    print(json.dumps({'ok': True, 'summary_rows': len(summary), 'case_rows': len(cases), 'out_summary_csv': args.out_summary_csv, 'out_cases_jsonl': args.out_cases_jsonl, 'out_md': args.out_md}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
