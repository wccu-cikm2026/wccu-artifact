from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, ensure_dir

LABELS = {
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_wccu_projection_trace': 'WCCU, projection trace',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_wccu_unguided_certificate': 'WCCU, unguided model cert',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    'adaptive_wccu_oracle_dependency': 'WCCU, oracle dependency',
    'adaptive_readset_occ': 'Read-set OCC',
    'adaptive_wccu_execution_trace': 'WCCU, execution trace',
    'adaptive_wccu_projection_trace': 'WCCU, projection trace',
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_wccu_unguided_certificate': 'WCCU, unguided model cert',
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
    'adaptive_wccu_no_read_validation',
    'adaptive_policy',
    'uniform_snapshot_occ',
    'uniform_review_gated',
    'uniform_append_only',
]


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _as_int(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except Exception:
        return 0


def _wccu_intervention_count(row: dict[str, Any]) -> int:
    if row.get('wccu_intervention_count') is not None:
        return _as_int(row, 'wccu_intervention_count')
    if row.get('wccu_intervention_count') is not None:
        return _as_int(row, 'wccu_intervention_count')
    return (_as_int(row, 'wccu_review_routed_count') or _as_int(row, 'wccu_review_routed_count')) + (_as_int(row, 'wccu_blocked_count') or _as_int(row, 'wccu_blocked_count'))


def _freshness_pass(row: dict[str, Any]) -> bool:
    """Safety-oriented pass for stale-dependency diagnostics.

    ``task_success`` is a narrow mechanism diagnostic that can distinguish
    review from block. For stale-dependency diagnostics, both review and block
    are safe outcomes as long as the stale-dependent write is not accepted and
    no unsafe auto-commit occurs.
    """
    return (
        not row.get('failed')
        and _as_int(row, 'unsafe_auto_commit_count') == 0
        and _as_int(row, 'stale_dependency_accepted_count') == 0
    )


def _wccu_freshness_success(row: dict[str, Any]) -> bool:
    """Whether a WCCU condition successfully enforced a stale dependency.

    This metric is stricter than ``freshness_pass``: it requires a stale
    dependency to be present, no stale accept, no unsafe auto-commit, and a WCCU
    review/block intervention. It is false for non-WCCU baselines and for the
    no-read-validation ablation by design.
    """
    condition = str(row.get('condition') or '')
    if not (condition.startswith('adaptive_wccu') or condition.startswith('adaptive_wccu')) or condition in {'adaptive_wccu_no_read_validation', 'adaptive_wccu_no_read_validation'}:
        return False
    return (
        _as_int(row, 'stale_dependency_count') > 0
        and _as_int(row, 'stale_dependency_accepted_count') == 0
        and _as_int(row, 'unsafe_auto_commit_count') == 0
        and _wccu_intervention_count(row) > 0
    )


def compute_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in payload.get('results', []):
        groups[str(r.get('condition'))].append(as_dict(r))
    rows: list[dict[str, Any]] = []
    for condition, rs in groups.items():
        n = len(rs)
        failed = sum(1 for r in rs if r.get('failed'))
        unsafe = sum(_as_int(r, 'unsafe_auto_commit_count') for r in rs)
        stale = sum(_as_int(r, 'stale_dependency_accepted_count') for r in rs)
        stale_detected = sum(_as_int(r, 'stale_dependency_count') for r in rs)
        wccu = sum(_wccu_intervention_count(r) for r in rs)
        wccu_review = sum(_as_int(r, 'wccu_review_routed_count') or _as_int(r, 'wccu_review_routed_count') for r in rs)
        wccu_block = sum(_as_int(r, 'wccu_blocked_count') or _as_int(r, 'wccu_blocked_count') for r in rs)
        ignored = sum(_as_int(r, 'stale_read_validation_ignored_count') for r in rs)
        review = sum(_as_int(r, 'review_burden_count') for r in rs)
        diagnostic = sum(1 for r in rs if r.get('task_success'))
        freshness = sum(1 for r in rs if _freshness_pass(r))
        wccu_freshness = sum(1 for r in rs if _wccu_freshness_success(r))
        rows.append({
            'condition': condition,
            'label': LABELS.get(condition, condition),
            'runs': n,
            'failed': failed,
            'diagnostic_pass': diagnostic,
            'freshness_pass': freshness,
            # Backward-compatible alias. Older papers/scripts used safety_pass;
            # for commitment-staleness tables this is equivalent to freshness_pass.
            'safety_pass': freshness,
            'wccu_freshness_success': wccu_freshness,
            'wccu_freshness_success': wccu_freshness,
            'stale_dependency_detected_count': stale_detected,
            'stale_dependency_accepted_count': stale,
            'unsafe_auto_commit_count': unsafe,
            'wccu_intervention_count': wccu,
            'wccu_intervention_count': wccu,
            'wccu_review_routed_count': wccu_review,
            'wccu_review_routed_count': wccu_review,
            'wccu_blocked_count': wccu_block,
            'wccu_blocked_count': wccu_block,
            'stale_read_validation_ignored_count': ignored,
            'review_burden_count': review,
        })
    order = {c: i for i, c in enumerate(ORDER)}
    return sorted(rows, key=lambda r: (order.get(r['condition'], 999), r['condition']))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text('', encoding='utf-8')
        return
    fields = list(rows[0].keys())
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    lines = [
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'Condition & Freshness & Stale accepted & Unsafe & WCCU int. & Review/block \\',
        r'\midrule',
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['freshness_pass']}/{r['runs']} & "
            f"{r['stale_dependency_accepted_count']} & "
            f"{r['unsafe_auto_commit_count']} & "
            f"{r['wccu_intervention_count']} & "
            f"{r['review_burden_count']} " + r'\\'
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    p.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create CooperBench-derived commitment-staleness WCCU table.')
    parser.add_argument('input_json')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', default='')
    args = parser.parse_args(argv)
    rows = compute_rows(_load(args.input_json))
    write_csv(args.out_csv, rows)
    if args.out_tex:
        write_tex(args.out_tex, rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
