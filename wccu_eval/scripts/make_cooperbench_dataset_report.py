from __future__ import annotations

"""Report how CooperBench-derived WCCU datasets were constructed."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from wccu_eval.external.cooperbench_adapter import load_cooperbench_tasks
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, mean, write_json


def _load(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return load_cooperbench_tasks(p)


def _first_file_count(row: dict[str, Any]) -> int:
    return len(as_list(row.get('shared_targets') or row.get('targets') or row.get('shared_files') or row.get('files')))


def _describe(name: str, path: str | Path) -> dict[str, Any]:
    rows = _load(path)
    repos = Counter(clean(r.get('repo') or r.get('repository') or r.get('repo_name') or 'unknown') for r in rows)
    langs = Counter(clean(r.get('language') or r.get('lang') or 'unknown') for r in rows)
    ctypes = Counter(clean(r.get('expected_conflict_type') or r.get('scenario_type') or r.get('task_type') or 'unknown') for r in rows)
    source_types = Counter(clean(as_dict(r.get('metadata')).get('converter') or as_dict(r.get('metadata')).get('source_layout') or as_dict(r.get('metadata')).get('source') or 'unknown') for r in rows)
    feature_lens = []
    for r in rows:
        for key in ('agent_a_task', 'agent_b_task', 'feature_a', 'feature_b'):
            if clean(r.get(key)):
                feature_lens.append(len(clean(r.get(key))))
    return {
        'stage': name,
        'path': str(path),
        'records': len(rows),
        'with_shared_targets': sum(1 for r in rows if _first_file_count(r) > 0),
        'mean_shared_targets': mean([_first_file_count(r) for r in rows]) if rows else 0.0,
        'repos': dict(repos.most_common(10)),
        'languages': dict(langs.most_common(10)),
        'conflict_or_scenario_types': dict(ctypes.most_common(10)),
        'source_metadata': dict(source_types.most_common(10)),
        'mean_feature_chars': mean(feature_lens) if feature_lens else 0.0,
        'sample_task_ids': [clean(r.get('task_id') or r.get('id')) for r in rows[:5]],
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    keys = ['stage', 'path', 'records', 'with_shared_targets', 'mean_shared_targets', 'mean_feature_chars', 'sample_task_ids']
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), (list, dict)) else r.get(k) for k in keys})


def write_md(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    lines = ['# CooperBench-derived WCCU dataset report', '']
    lines.append('This report is generated from the actual converted/subsampled files used by the live-LLM CooperBench-derived experiments. It does not include model outputs or mock rows.')
    lines.append('')
    for r in rows:
        lines.append(f"## {r['stage']}")
        lines.append(f"- path: `{r['path']}`")
        lines.append(f"- records: {r['records']}")
        lines.append(f"- records with shared targets: {r['with_shared_targets']}")
        lines.append(f"- mean shared targets: {r['mean_shared_targets']:.2f}")
        lines.append(f"- mean feature chars: {r['mean_feature_chars']:.1f}")
        lines.append(f"- repos: `{json.dumps(r['repos'], ensure_ascii=False)}`")
        lines.append(f"- conflict/scenario types: `{json.dumps(r['conflict_or_scenario_types'], ensure_ascii=False)}`")
        lines.append(f"- sample task ids: `{json.dumps(r['sample_task_ids'], ensure_ascii=False)}`")
        lines.append('')
    p.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Describe CooperBench-derived datasets used in WCCU live-LLM experiments.')
    parser.add_argument('--converted', default='')
    parser.add_argument('--subset', default='')
    parser.add_argument('--commitment-diag', default='')
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-csv', default='')
    parser.add_argument('--out-md', default='')
    args = parser.parse_args(argv)
    stages = []
    if args.converted:
        stages.append(_describe('converted_cooperbench', args.converted))
    if args.subset:
        stages.append(_describe('conflict_preferred_subset', args.subset))
    if args.commitment_diag:
        stages.append(_describe('commitment_staleness_diagnostic', args.commitment_diag))
    payload = {'kind': 'cooperbench_wccu_dataset_report_v1', 'stages': stages}
    write_json(Path(args.out_json), payload)
    if args.out_csv:
        write_csv(stages, args.out_csv)
    if args.out_md:
        write_md(stages, args.out_md)
    print(json.dumps({'ok': True, 'stages': len(stages), 'out_json': args.out_json}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
