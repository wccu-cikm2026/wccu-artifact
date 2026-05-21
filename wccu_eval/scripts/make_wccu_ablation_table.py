from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_list, clean, ensure_dir

DISPLAY_NAMES = {
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_wccu_projection_trace': 'WCCU, projection trace',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_wccu_unguided_certificate': 'WCCU, unguided model cert',
    'adaptive_wccu_oracle_dependency': 'WCCU, oracle dependency',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    'adaptive_readset_occ': 'Read-set OCC',
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_wccu_projection_trace': 'WCCU, projection trace',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_wccu_unguided_certificate': 'WCCU, unguided model cert',
    'adaptive_wccu_oracle_dependency': 'WCCU, oracle dependency',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_snapshot_occ': 'Snapshot OCC',
    'uniform_review_gated': 'Review-gated',
    'uniform_append_only': 'Append-only',
}

ORDER = [
    'adaptive_wccu_execution_trace',
    'adaptive_wccu_projection_trace',
    'adaptive_wccu_model_certificate',
    'adaptive_wccu_unguided_certificate',
    'adaptive_wccu_oracle_dependency',
    'adaptive_wccu_no_read_validation',
    'adaptive_readset_occ',
    'adaptive_wccu_execution_trace',
    'adaptive_wccu_projection_trace',
    'adaptive_wccu_model_certificate',
    'adaptive_wccu_unguided_certificate',
    'adaptive_wccu_oracle_dependency',
    'adaptive_wccu_no_read_validation',
    'adaptive_policy',
    'uniform_snapshot_occ',
    'uniform_review_gated',
    'uniform_append_only',
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _wccu_intervention(row: dict[str, Any]) -> float:
    if row.get('wccu_intervention_count') is not None:
        return float(row.get('wccu_intervention_count') or 0)
    return float(row.get('wccu_review_routed_count') or 0) + float(row.get('wccu_blocked_count') or 0)


def _freshness_pass(row: dict[str, Any]) -> float:
    return 1.0 if (
        not row.get('failed')
        and float(row.get('unsafe_auto_commit_count') or 0) <= 0
        and float(row.get('stale_dependency_accepted_count') or 0) <= 0
    ) else 0.0


def _intervention_aware_success(row: dict[str, Any]) -> float:
    """Recompute success for WCCU stale-dependency diagnostics.

    Older result files may count only review-routed WCCU writes as success.
    For stale-dependency diagnostics, review and block are both safe
    interventions as long as the stale-dependent write is not accepted.
    """
    condition = clean(row.get('condition'))
    if (condition.startswith('adaptive_wccu') or condition.startswith('adaptive_wccu')) and condition not in {'adaptive_wccu_no_read_validation', 'adaptive_wccu_no_read_validation'}:
        ok = (
            float(row.get('stale_dependency_count') or 0) >= 1
            and float(row.get('stale_dependency_accepted_count') or 0) <= 0
            and float(row.get('unsafe_auto_commit_count') or 0) <= 0
            and _wccu_intervention(row) >= 1
        )
        return 1.0 if ok else 0.0
    return 1.0 if row.get('task_success') else 0.0


def build_rows(payload: dict[str, Any], scenario_filter: str = '') -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in as_list(payload.get('results')):
        row = dict(raw)
        if scenario_filter and clean(row.get('scenario_id')) != clean(scenario_filter):
            continue
        condition = clean(row.get('condition')) or 'unknown'
        groups[condition].append(row)

    order = {c: i for i, c in enumerate(ORDER)}
    rows: list[dict[str, Any]] = []
    for condition in sorted(groups, key=lambda c: (order.get(c, 999), c)):
        group = groups[condition]
        n = len(group)
        rows.append({
            'condition': condition,
            'display_name': DISPLAY_NAMES.get(condition, condition),
            'n': n,
            'raw_task_success_rate': _mean([1.0 if r.get('task_success') else 0.0 for r in group]),
            'freshness_pass_rate': _mean([_freshness_pass(r) for r in group]),
            'success_rate': _mean([_intervention_aware_success(r) for r in group]),
            'diagnostic_pass_count': int(sum(_intervention_aware_success(r) for r in group)),
            'freshness_pass_count': int(sum(_freshness_pass(r) for r in group)),
            'safety_pass_count': int(sum(_freshness_pass(r) for r in group)),
            'stale_dependency_count': _mean([float(r.get('stale_dependency_count') or 0) for r in group]),
            'stale_dependency_accepted_count': _mean([float(r.get('stale_dependency_accepted_count') or 0) for r in group]),
            'stale_read_validation_ignored_count': _mean([float(r.get('stale_read_validation_ignored_count') or 0) for r in group]),
            'wccu_review_routed_count': _mean([float(r.get('wccu_review_routed_count') or 0) for r in group]),
            'wccu_blocked_count': _mean([float(r.get('wccu_blocked_count') or 0) for r in group]),
            'wccu_intervention_count': _mean([_wccu_intervention(r) for r in group]),
            'unsafe_auto_commit_count': _mean([float(r.get('unsafe_auto_commit_count') or 0) for r in group]),
            'review_burden_count': _mean([float(r.get('review_burden_count') or 0) for r in group]),
            'certificate_invalid_count': _mean([float(r.get('certificate_invalid_count') or 0) for r in group]),
            'low_target_confidence_count': _mean([float(r.get('low_target_confidence_count') or 0) for r in group]),
            'authority_insufficient_count': _mean([float(r.get('authority_insufficient_count') or 0) for r in group]),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return value.replace('_', r'\_')


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'Condition & Freshness & Unsafe & Stale accepted & WCCU int. & Review/block \\',
        r'\midrule',
    ]
    for row in rows:
        cond = latex_escape(row['display_name'])
        line = (
            f"{cond} & {row['freshness_pass_rate']:.2f} & "
            f"{row['unsafe_auto_commit_count']:.2f} & "
            f"{row['stale_dependency_accepted_count']:.2f} & "
            f"{row['wccu_intervention_count']:.2f} & "
            f"{row['review_burden_count']:.2f} " + r'\\'
        )
        lines.append(line)
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{WCCU stale-dependency ablation. Values are means per run. WCCU intervention counts review-routed and blocked transactions as safe interventions; freshness treats either outcome as safe when stale writes are not accepted.}',
        r'\label{tab:wccu-ablation}',
        r'\end{table}',
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Build a compact WCCU ablation table from experiment results.')
    parser.add_argument('input')
    parser.add_argument('--scenario', default='', help='Optional exact scenario_id filter. Empty means aggregate all scenarios by condition.')
    parser.add_argument('--out-csv', default='analysis/wccu_ablation_table.csv')
    parser.add_argument('--out-tex', default='analysis/wccu_ablation_table.tex')
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding='utf-8'))
    rows = build_rows(payload, scenario_filter=args.scenario)
    write_csv(Path(args.out_csv), rows)
    write_latex(Path(args.out_tex), rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex, 'scenario': args.scenario}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
