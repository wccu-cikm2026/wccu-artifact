from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from wccu_eval.scripts.coordination_quality import row_quality
from wccu_eval.scripts.make_stress_table import DISPLAY_NAMES, ORDER
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir


EXTRA_LABELS = {
    'adaptive_wccu_model_certificate': 'WCCU, model cert',
    'adaptive_wccu_unguided_certificate': 'WCCU, unguided model cert',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    'adaptive_no_candidates_no_grounding': 'No candidates / no grounding',
    'adaptive_no_candidates_with_grounding': 'No candidates / grounding',
    'adaptive_candidates_no_grounding': 'Candidates / no grounding',
    'adaptive_candidates_with_grounding': 'Candidates / grounding',
    'adaptive_readset_occ': 'Read-set OCC',
}
LABELS = {**DISPLAY_NAMES, **EXTRA_LABELS}
CONDITION_ORDER = {c: i for i, c in enumerate(ORDER + [
    'adaptive_wccu_model_certificate',
    'adaptive_wccu_unguided_certificate',
    'adaptive_wccu_no_read_validation',
    'adaptive_readset_occ',
    'adaptive_policy',
    'uniform_snapshot_occ',
    'uniform_review_gated',
    'uniform_append_only',
])}
FAMILY_LABELS = {
    'adversarial_certificates': 'Adversarial',
    'commitment_staleness': 'Commitment',
    'cooperbench_derived': 'CooperBench',
    'obligation_matrix': 'Obligation',
    'randomized_stress': 'Stress',
    'target_grounding': 'Target',
    'witness_completeness': 'Witness',
    'llm_obligation_benchmark': 'LLM obligation',
}



def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _num(value: Any) -> float:
    try:
        n = float(value or 0)
        return n if math.isfinite(n) else 0.0
    except Exception:
        return 0.0


def _int(value: Any) -> int:
    return int(round(_num(value)))


def _args_from_path(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    name = str(path)
    args: dict[str, Any] = {}
    m = re.search(r'_w(\d+)_a(\d+)_p([0-9.]+)_seed(\d+)', name)
    if m:
        args.update({
            'writers': int(m.group(1)),
            'atom_count': int(m.group(2)),
            'invalidation_prob': float(m.group(3)),
            'seed': int(m.group(4)),
        })
    return args


def _scenario_family(payload: dict[str, Any], row: dict[str, Any], path: str | Path | None = None) -> str:
    kind = clean(row.get('kind'))
    sid = clean(row.get('scenario_id'))
    args = as_dict(payload.get('args'))
    payload_kind = clean(payload.get('kind'))
    if 'witness_completeness' in payload_kind:
        return 'witness_completeness'
    if 'llm_obligation_benchmark' in payload_kind or clean(row.get('llm_obligation_family')):
        return 'llm_obligation_benchmark'
    if 'adversarial_wccu' in payload_kind:
        return 'adversarial_certificates'
    if 'obligation_matrix' in payload_kind:
        return 'obligation_matrix'
    if payload_kind == 'wccu_randomized_stress_result' or 'wccu_stress' in sid or (path and 'wccu_stress' in str(path)):
        return 'randomized_stress'
    if 'commitment_stale' in sid or 'commitment' in sid:
        return 'commitment_staleness'
    if 'ambiguous' in sid or 'target' in sid:
        return 'target_grounding'
    if args.get('input') and 'cooperbench' in str(args.get('input')):
        return 'cooperbench_derived'
    return kind or 'unknown'


def _latent_dependency_count(row: dict[str, Any]) -> int:
    """Estimate how many writes depend on another concurrent mutation.

    Baselines often cannot observe the latent dependency.  We therefore use the
    maximum available signal for cross-condition ground truth, but this is still
    an update-instance count rather than a sum of verifier events.
    """
    meta = as_dict(row.get('stress_metadata'))
    signals = [
        _int(row.get('ground_truth_issue_count')),
        _int(row.get('ground_truth_problematic_count')),
        _int(row.get('stale_dependency_count')),
        _int(row.get('stale_dependency_accepted_count')),
        _int(row.get('stale_read_validation_ignored_count')),
        _int(row.get('stale_write_blocked_count')),
    ]
    if 'invalidated_dependency_count' in meta:
        signals.append(_int(meta.get('invalidated_dependency_count')))
    sid = clean(row.get('scenario_id'))
    if 'commitment' in sid and 'stale' in sid:
        signals.append(1)
    return max(signals)


def _write_total(row: dict[str, Any]) -> int:
    commit = as_dict(row.get('commit'))
    return max(_int(row.get('ground_truth_total_writes')), _int(row.get('write_intent_count')), _int(commit.get('total')), _int(row.get('agent_count')))


def _committed(row: dict[str, Any]) -> int:
    return _int(as_dict(row.get('commit')).get('committed'))


def _review_block(row: dict[str, Any]) -> int:
    return max(_int(row.get('review_burden_count')), _int(row.get('wccu_blocked_count')))


def _fresh(row: dict[str, Any]) -> bool:
    return not row.get('failed') and _int(row.get('unsafe_auto_commit_count')) == 0 and _int(row.get('stale_dependency_accepted_count')) == 0


def _row_drop(row: dict[str, Any], payload: dict[str, Any] | None = None) -> float:
    drop = row.get('witness_drop_rate')
    if drop is None and payload is not None:
        drop = as_dict(payload.get('args')).get('witness_drop_rate')
    return float(drop or 0.0)


def _setting_key(payload: dict[str, Any], row: dict[str, Any], group_by_setting: bool, path: str | Path | None = None) -> tuple[Any, ...]:
    family = _scenario_family(payload, row, path)
    condition = clean(row.get('condition'))
    args = {**_args_from_path(path), **as_dict(payload.get('args'))}

    # Never aggregate witness-completeness drop rates into one main row.  The
    # drop-rate sweep is an assumption-boundary experiment; combining drop=0
    # with intentionally degraded witnesses makes the main WCCU row look unsafe.
    if family == 'witness_completeness':
        return (family, _row_drop(row, payload), condition)

    if not group_by_setting:
        return (family, condition)

    meta = as_dict(row.get('stress_metadata'))
    writers = int(meta.get('writers') or args.get('writers') or row.get('agent_count') or 0)
    atoms = int(meta.get('atom_count') or args.get('atom_count') or 0)
    prob = float(args.get('invalidation_prob') or 0.0)
    if family == 'randomized_stress':
        return (family, writers, atoms, prob, condition)
    return (family, condition)


def _scenario_key(payload: dict[str, Any], row: dict[str, Any], path: str | Path | None = None) -> tuple[Any, ...]:
    family = _scenario_family(payload, row, path)
    args = {**_args_from_path(path), **as_dict(payload.get('args'))}
    meta = as_dict(row.get('stress_metadata'))
    return (
        family,
        int(meta.get('writers') or args.get('writers') or 0),
        float(args.get('invalidation_prob') or 0.0),
        clean(row.get('scenario_id')),
        int(row.get('repetition') or 0),
        clean(row.get('obligation_kind')),
    )


def _empty_acc(key: tuple[Any, ...]) -> dict[str, Any]:
    family = str(key[0])
    condition = str(key[-1])
    row = {
        'family': family,
        'family_label': FAMILY_LABELS.get(family, family),
        'setting': '',
        'condition': condition,
        'label': LABELS.get(condition, condition),
        'runs': 0,
        'writes': 0,
        'latent_dependencies': 0,
        'stale_accepted': 0,
        'unsafe': 0,
        'wccu_intervention': 0,
        'review_block': 0,
        'committed': 0,
        'freshness_count': 0,
        'failed_runs': 0,
        'serial_rounds_est': 0,
        'critical_path_rounds_est': 0,
        'issue_count': 0,
        'hold_required_count': 0,
        'problematic_held_count': 0,
        'false_hold_count': 0,
        'issue_accepted_count': 0,
        'safe_write_count': 0,
        'safe_auto_commit_count': 0,
    }
    if len(key) == 3 and family == 'witness_completeness':
        _, drop, _ = key
        row.update({'witness_drop_rate': float(drop), 'setting': f'drop={float(drop):.2f}'})
    if len(key) == 5 and family == 'randomized_stress':
        _, writers, atoms, prob, _ = key
        row.update({'writers': writers, 'atom_count': atoms, 'invalidation_prob': prob, 'setting': f'w={int(writers)}, p={float(prob):.2f}'})
    return row


def _add_record(acc: dict[str, Any], row: dict[str, Any], latent_override: int | None = None) -> None:
    writes = _write_total(row)
    latent = max(_latent_dependency_count(row), _int(latent_override))
    review = _review_block(row)
    q = row_quality(row, latent)
    acc['runs'] += 1
    acc['writes'] += writes
    acc['latent_dependencies'] += latent
    acc['stale_accepted'] += _int(row.get('stale_dependency_accepted_count'))
    acc['unsafe'] += _int(row.get('unsafe_auto_commit_count'))
    acc['wccu_intervention'] += _int(row.get('wccu_intervention_count'))
    acc['review_block'] += review
    acc['committed'] += _committed(row)
    acc['freshness_count'] += 1 if _fresh(row) else 0
    acc['failed_runs'] += 1 if row.get('failed') else 0
    acc['serial_rounds_est'] += max(1, _int(row.get('agent_count')) or writes)
    acc['critical_path_rounds_est'] += 1 + (1 if review > 0 else 0)
    for key in ['issue_count', 'hold_required_count', 'problematic_held_count', 'false_hold_count', 'issue_accepted_count', 'safe_write_count', 'safe_auto_commit_count']:
        acc[key] += int(q[key])


def _finalize(acc: dict[str, Any]) -> dict[str, Any]:
    runs = int(acc['runs'])
    writes = int(acc['writes'])
    latent = int(acc['latent_dependencies'])
    review = int(acc['review_block'])
    independent_writes = max(0, writes - latent)
    unnecessary_coordination = max(0, review - latent)
    auto_progress = max(0, writes - review)
    independent_auto_progress = max(0, independent_writes - unnecessary_coordination)
    acc['avg_writes_per_run'] = round(writes / runs, 3) if runs else 0.0
    acc['avg_latent_dependencies_per_run'] = round(latent / runs, 3) if runs else 0.0
    acc['dependency_density'] = round(latent / writes, 4) if writes else 0.0
    acc['freshness_pass'] = f"{int(acc['freshness_count'])}/{runs}"
    acc['auto_progress_rate'] = round(auto_progress / writes, 4) if writes else 0.0
    acc['coordination_ratio'] = round(review / writes, 4) if writes else 0.0
    acc['coordination_selectivity'] = round(latent / review, 4) if review else (1.0 if latent == 0 else 0.0)
    acc['independent_auto_progress_rate'] = round(independent_auto_progress / independent_writes, 4) if independent_writes else 1.0
    acc['unnecessary_coordination'] = unnecessary_coordination
    acc['coordination_precision'] = round(acc['problematic_held_count'] / acc['review_block'], 4) if acc['review_block'] else (1.0 if acc['issue_count'] == 0 else 0.0)
    acc['coordination_recall'] = round(acc['problematic_held_count'] / acc['hold_required_count'], 4) if acc['hold_required_count'] else 1.0
    p = float(acc['coordination_precision'])
    r = float(acc['coordination_recall'])
    acc['coordination_f1'] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    acc['unsafe_issue_accept_rate'] = round(acc['issue_accepted_count'] / acc['issue_count'], 4) if acc['issue_count'] else 0.0
    acc['safe_automatic_progress'] = round(acc['safe_auto_commit_count'] / acc['safe_write_count'], 4) if acc['safe_write_count'] else 1.0
    acc['over_coordination_rate'] = round(acc['false_hold_count'] / acc['safe_write_count'], 4) if acc['safe_write_count'] else 0.0
    acc['sp_at_zero_unsafe'] = acc['safe_automatic_progress'] if acc['issue_accepted_count'] == 0 else 0.0
    acc['review_yield'] = acc['coordination_precision']
    acc['reviews_per_true_issue'] = round(acc['review_block'] / acc['problematic_held_count'], 4) if acc['problematic_held_count'] else (0.0 if acc['review_block'] == 0 else float('inf'))
    acc['round_savings_vs_serial'] = int(acc['serial_rounds_est']) - int(acc['critical_path_rounds_est'])
    del acc['freshness_count']
    return acc


def _iter_records_from_path(path: str | Path) -> Iterable[tuple[dict[str, Any], dict[str, Any], str | Path]]:
    p = Path(path)
    if p.suffix == '.jsonl':
        payload = {'kind': '', 'args': _args_from_path(p)}
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield payload, as_dict(json.loads(line)), p
        return
    payload = _load(p)
    for row in as_list(payload.get('results')):
        yield payload, as_dict(row), p


def _record_matches(payload: dict[str, Any], row: dict[str, Any], path: str | Path | None, *, only_families: set[str] | None = None, only_witness_drop: float | None = None) -> bool:
    family = _scenario_family(payload, row, path)
    if only_families and family not in only_families:
        return False
    if only_witness_drop is not None:
        if family != 'witness_completeness':
            return False
        return abs(_row_drop(row, payload) - only_witness_drop) < 1e-9
    return True


def build_rows(
    payloads: list[tuple[dict[str, Any], str | Path | None]],
    group_by_setting: bool = False,
    only_families: set[str] | None = None,
    only_witness_drop: float | None = None,
) -> list[dict[str, Any]]:
    scenario_latent: dict[tuple[Any, ...], int] = defaultdict(int)
    filtered: list[tuple[dict[str, Any], dict[str, Any], str | Path | None]] = []
    for payload, path in payloads:
        for raw in as_list(payload.get('results')):
            row = as_dict(raw)
            if not _record_matches(payload, row, path, only_families=only_families, only_witness_drop=only_witness_drop):
                continue
            filtered.append((payload, row, path))
            skey = _scenario_key(payload, row, path)
            scenario_latent[skey] = max(scenario_latent[skey], _latent_dependency_count(row))

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for payload, row, path in filtered:
        key = _setting_key(payload, row, group_by_setting, path)
        groups.setdefault(key, _empty_acc(key))
        _add_record(groups[key], row, scenario_latent.get(_scenario_key(payload, row, path)))
    return _sort_rows([_finalize(v) for v in groups.values()])


def build_rows_from_paths(
    paths: list[str | Path],
    group_by_setting: bool = False,
    only_families: set[str] | None = None,
    only_witness_drop: float | None = None,
) -> list[dict[str, Any]]:
    records: list[tuple[dict[str, Any], dict[str, Any], str | Path]] = []
    scenario_latent: dict[tuple[Any, ...], int] = defaultdict(int)
    for path in paths:
        for payload, row, row_path in _iter_records_from_path(path):
            if not _record_matches(payload, row, row_path, only_families=only_families, only_witness_drop=only_witness_drop):
                continue
            records.append((payload, row, row_path))
            skey = _scenario_key(payload, row, row_path)
            scenario_latent[skey] = max(scenario_latent[skey], _latent_dependency_count(row))

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for payload, row, row_path in records:
        key = _setting_key(payload, row, group_by_setting, row_path)
        groups.setdefault(key, _empty_acc(key))
        _add_record(groups[key], row, scenario_latent.get(_scenario_key(payload, row, row_path)))
    return _sort_rows([_finalize(v) for v in groups.values()])


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (
        r.get('family'),
        float(r.get('witness_drop_rate', -1)),
        int(r.get('writers', -1)),
        float(r.get('invalidation_prob', -1)),
        CONDITION_ORDER.get(str(r['condition']), 999),
        r['condition'],
    ))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tex_escape(value: Any) -> str:
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def write_tex(path: Path, rows: list[dict[str, Any]], compact: bool = True) -> None:
    ensure_dir(path.parent)
    if compact:
        lines = [
            r'\begin{tabular}{lllrrrrrr}',
            r'\toprule',
            r'Family & Setting & Condition & Unsafe & Safe prog. & Coord. P & Coord. R & Coord. F1 & Overcoord. \\',
            r'\midrule',
        ]
        for r in rows:
            lines.append(
                f"{_tex_escape(r.get('family_label', r['family']))} & {_tex_escape(r.get('setting', ''))} & {_tex_escape(r['label'])} & "
                f"{float(r['unsafe_issue_accept_rate']):.2f} & {float(r['safe_automatic_progress']):.2f} & "
                f"{float(r['coordination_precision']):.2f} & {float(r['coordination_recall']):.2f} & "
                f"{float(r['coordination_f1']):.2f} & {float(r['over_coordination_rate']):.2f} " + r'\\'
            )
        lines.extend([r'\bottomrule', r'\end{tabular}', ''])
        path.write_text('\n'.join(lines), encoding='utf-8')
        return

    lines = [
        r'\begin{tabular}{lllrrrrrrrr}',
        r'\toprule',
        r'Family & Setting & Condition & Runs & Writes & Issues & Unsafe & Safe prog. & Coord. P & Coord. R & Coord. F1 & Overcoord. \\',
        r'\midrule',
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r.get('family_label', r['family']))} & {_tex_escape(r.get('setting', ''))} & {_tex_escape(r['label'])} & "
            f"{int(r['runs'])} & {int(r['writes'])} & {int(r['issue_count'])} & {float(r['unsafe_issue_accept_rate']):.2f} & "
            f"{float(r['safe_automatic_progress']):.2f} & {float(r['coordination_precision']):.2f} & "
            f"{float(r['coordination_recall']):.2f} & {float(r['coordination_f1']):.2f} & {float(r['over_coordination_rate']):.2f} " + r'\\'
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Summarize structural coordination metrics from WCCU/WCCU experiment result JSON/JSONL files. '
        'The script reports update-instance rates rather than raw event counts, so unsafe rates stay in [0, 1]. '
        'Witness-completeness drop rates are never silently aggregated.'
    )
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', required=True)
    parser.add_argument('--group-by-setting', action='store_true', help='For stress results, keep writers/probability settings separate.')
    parser.add_argument('--only-family', action='append', default=[], help='Restrict to a scenario family such as witness_completeness, obligation_matrix, adversarial_certificates, or randomized_stress. May be repeated.')
    parser.add_argument('--only-witness-drop', type=float, default=None, help='Restrict witness-completeness results to one witness drop rate, e.g. 0 or 0.5.')
    parser.add_argument('--full-tex', action='store_true', help='Write a wider table with family, setting, counts, and metrics.')
    args = parser.parse_args(argv)
    rows = build_rows_from_paths(
        args.inputs,
        group_by_setting=args.group_by_setting,
        only_families=set(args.only_family) if args.only_family else None,
        only_witness_drop=args.only_witness_drop,
    )
    write_csv(Path(args.out_csv), rows)
    write_tex(Path(args.out_tex), rows, compact=not args.full_tex)
    print(json.dumps({'ok': True, 'rows': len(rows), 'out_csv': args.out_csv, 'out_tex': args.out_tex}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
