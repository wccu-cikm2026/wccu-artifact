from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _sum_results(data: dict[str, Any], fields: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: {f: 0.0 for f in fields} | {'runs': 0.0, 'failed': 0.0})
    for row in data.get('results') or []:
        cond = str(row.get('condition') or '')
        if not cond:
            continue
        out[cond]['runs'] += 1
        if row.get('failed'):
            out[cond]['failed'] += 1
        for f in fields:
            try:
                out[cond][f] += float(row.get(f) or 0)
            except Exception:
                pass
    return dict(out)


def _bundle_provider_counts(path: Path) -> Counter[tuple[str, str, str]]:
    data = _load_json(path)
    c: Counter[tuple[str, str, str]] = Counter()
    for row in data.get('agent_outputs') or []:
        c[(str(row.get('agent_id') or ''), str(row.get('generation_provider') or ''), str(row.get('generation_model') or ''))] += 1
    return c


def _provider_counts_str(c: Counter[tuple[str, str, str]]) -> str:
    parts = []
    for (agent, provider, model), n in sorted(c.items()):
        if agent or provider or model:
            parts.append(f'{agent}:{provider}:{model}={n}')
    return '; '.join(parts)


def _verify_replay_no_provider_calls(data: dict[str, Any]) -> tuple[bool, int]:
    actual = 0
    for row in data.get('results') or []:
        for ar in row.get('agentRuns') or []:
            if ((ar.get('llm') or {}).get('api_usage') or {}):
                actual += 1
    flag = (data.get('frozen_replay') or {}).get('provider_api_called_in_replay')
    return (flag is False and actual == 0), actual


def _read_dataset_report(tag: str, base: Path) -> dict[str, Any]:
    return _load_json(base / 'analysis' / tag / 'cooperbench_dataset_report.json')


def build_summary(tags: list[str], base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = [
        'unsafe_auto_commit_count',
        'stale_dependency_accepted_count',
        'stale_dependency_count',
        'review_burden_count',
        'wccu_intervention_count',
        'lock_conflict_count',
    ]
    for tag in tags:
        ds = _read_dataset_report(tag, base)
        converted_n = ((ds.get('converted') or {}).get('record_count') or ds.get('converted_records') or '')
        subset_n = ((ds.get('subset') or {}).get('record_count') or ds.get('subset_records') or '')
        commitment_n = ((ds.get('commitment_diag') or ds.get('commitment') or {}).get('record_count') or ds.get('commitment_records') or '')
        for workload, filename in [
            ('workspace', 'cooperbench_workspace_frozen_replay.json'),
            ('commitment', 'cooperbench_commitment_frozen_replay.json'),
        ]:
            replay_path = base / 'results' / tag / filename
            data = _load_json(replay_path)
            replay_ok, replay_api_usage_rows = _verify_replay_no_provider_calls(data)
            bundle_path = base / 'results' / tag / filename.replace('_frozen_replay.json', '_frozen_bundle.json')
            provider_counts = _bundle_provider_counts(bundle_path)
            gen_path = base / 'results' / tag / filename.replace('_frozen_replay.json', '_generation.json')
            gen = _load_json(gen_path)
            generation_failed = sum(1 for r in gen.get('results') or [] if r.get('failed'))
            condition_sums = _sum_results(data, metrics)
            for cond, vals in sorted(condition_sums.items()):
                row = {
                    'tag': tag,
                    'workload': workload,
                    'condition': cond,
                    'runs': int(vals.get('runs', 0)),
                    'failed': int(vals.get('failed', 0)),
                    'unsafe_auto_commit_count': int(vals.get('unsafe_auto_commit_count', 0)),
                    'stale_dependency_accepted_count': int(vals.get('stale_dependency_accepted_count', 0)),
                    'stale_dependency_count': int(vals.get('stale_dependency_count', 0)),
                    'review_burden_count': int(vals.get('review_burden_count', 0)),
                    'wccu_intervention_count': int(vals.get('wccu_intervention_count', 0)),
                    'lock_conflict_count': int(vals.get('lock_conflict_count', 0)),
                    'replay_provider_calls': 'no' if replay_ok else 'yes_or_unknown',
                    'replay_api_usage_rows': replay_api_usage_rows,
                    'generation_failed_rows': generation_failed,
                    'agent_provider_model_counts': _provider_counts_str(provider_counts),
                    'converted_records': converted_n,
                    'subset_records': subset_n,
                    'commitment_records': commitment_n,
                }
                rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('No rows.\n', encoding='utf-8')
        return
    # concise paper-facing table: key conditions only
    key_conds = {
        'adaptive_wccu_execution_trace',
        'adaptive_wccu_projection_trace',
        'adaptive_wccu_model_certificate',
        'adaptive_readset_occ',
        'adaptive_wccu_no_read_validation',
        'adaptive_policy',
        'uniform_snapshot_occ',
        'uniform_review_gated',
        'uniform_append_only',
    }
    selected = [r for r in rows if r['condition'] in key_conds]
    lines = ['# Provider robustness summary', '']
    lines.append('| Tag | Workload | Condition | Runs | Unsafe commits | Stale accepted | Review burden | WCCU interventions | Replay provider calls | Agent provider/model counts |')
    lines.append('|---|---|---|---:|---:|---:|---:|---:|---|---|')
    for r in selected:
        lines.append(
            f"| {r['tag']} | {r['workload']} | {r['condition']} | {r['runs']} | "
            f"{r['unsafe_auto_commit_count']} | {r['stale_dependency_accepted_count']} | "
            f"{r['review_burden_count']} | {r['wccu_intervention_count']} | {r['replay_provider_calls']} | "
            f"{r['agent_provider_model_counts']} |"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description='Summarize WCCU single-provider and mixed-provider frozen replay runs.')
    p.add_argument('--tags', nargs='+', required=True, help='Experiment tags, e.g. wccu_frozen_seed7 wccu_frozen_gemini31_seed7 wccu_frozen_mixed_openai_gemini_seed7')
    p.add_argument('--base-dir', default='.', help='Repository/run base directory containing results/ and analysis/.')
    p.add_argument('--out-csv', required=True)
    p.add_argument('--out-json', default='')
    p.add_argument('--out-md', default='')
    args = p.parse_args(argv)
    rows = build_summary(args.tags, Path(args.base_dir))
    write_csv(rows, Path(args.out_csv))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.out_md:
        write_md(rows, Path(args.out_md))


if __name__ == '__main__':
    main()
