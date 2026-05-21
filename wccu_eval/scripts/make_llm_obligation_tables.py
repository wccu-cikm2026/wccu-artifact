from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.scripts.coordination_quality import aggregate_quality
from wccu_eval.scripts.make_coordination_metrics_table import LABELS
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, mean

FAMILY_ORDER = ['freshness', 'commitment', 'authority', 'operation', 'derived_view', 'witness_gap', 'safe']
FAMILY_LABELS = {
    'freshness': 'Freshness',
    'commitment': 'Commitment',
    'authority': 'Authority',
    'operation': 'Operation',
    'derived_view': 'Derived view',
    'witness_gap': 'Witness gap',
    'safe': 'Safe',
}


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _scenario_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (clean(row.get('llm_obligation_family')), clean(row.get('scenario_id')), int(row.get('repetition') or 0))


def _issue_overrides(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], int]:
    # aggregate_quality from coordination_quality expects its default scenario_key;
    # our rows already carry explicit ground_truth_issue_count, so no override is
    # needed.  This helper is intentionally retained for future filters.
    return {}


def build_generation_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in as_list(payload.get('generations')):
        groups[clean(row.get('family'))].append(as_dict(row))
    out = []
    for family in sorted(groups, key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 999):
        rows = groups[family]
        out.append({
            'family': family,
            'family_label': FAMILY_LABELS.get(family, family),
            'generations': len(rows),
            'schema_valid_rate': mean([1 if r.get('llm_schema_valid') else 0 for r in rows]),
            'proposal_rate': mean([1 if int(r.get('llm_proposal_count') or 0) > 0 else 0 for r in rows]),
            'model_cert_dep_recall': mean([r.get('llm_model_cert_dependency_recall') or 0 for r in rows]),
            'model_cert_dep_precision': mean([r.get('llm_model_cert_dependency_precision') or 0 for r in rows]),
            'mean_proposals': mean([r.get('llm_proposal_count') or 0 for r in rows]),
        })
    return out


def build_decision_rows(payload: dict[str, Any], *, by_family: bool = False) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in as_list(payload.get('results')):
        row = as_dict(row)
        if row.get('failed'):
            continue
        key = (clean(row.get('llm_obligation_family')) if by_family else 'all', clean(row.get('condition')))
        groups[key].append(row)
    out = []
    for (family, condition), rows in groups.items():
        q = aggregate_quality(rows)
        out.append({
            'family': family,
            'family_label': FAMILY_LABELS.get(family, 'All LLM scenarios' if family == 'all' else family),
            'condition': condition,
            'label': LABELS.get(condition, condition),
            'runs': len(rows),
            **q,
            'schema_valid_rate': mean([1 if r.get('llm_schema_valid') else 0 for r in rows]),
            'proposal_rate': mean([1 if int(r.get('llm_proposal_count') or 0) > 0 else 0 for r in rows]),
            'model_cert_dep_recall': mean([r.get('llm_model_cert_dependency_recall') or 0 for r in rows]),
        })
    return sorted(out, key=lambda r: (FAMILY_ORDER.index(r['family']) if r['family'] in FAMILY_ORDER else (-1 if r['family'] == 'all' else 999), r.get('condition')))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _tex(v: Any) -> str:
    return str(v).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def write_generation_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Family & Gen. & Schema & Proposal & Cert. R & Cert. P \\', r'\midrule']
    for r in rows:
        lines.append(
            f"{_tex(r['family_label'])} & {int(r['generations'])} & "
            f"{float(r['schema_valid_rate']):.2f} & {float(r['proposal_rate']):.2f} & "
            f"{float(r['model_cert_dep_recall']):.2f} & {float(r['model_cert_dep_precision']):.2f} " + r'\\'
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_decision_tex(path: Path, rows: list[dict[str, Any]], *, include_family: bool = False) -> None:
    ensure_dir(path.parent)
    if include_family:
        lines = [r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Family & Condition & Unsafe & Safe prog. & Coord. P & Coord. R & Coord. F1 & Overcoord. \\', r'\midrule']
        for r in rows:
            lines.append(
                f"{_tex(r['family_label'])} & {_tex(r['label'])} & {float(r['unsafe_issue_accept_rate']):.2f} & "
                f"{float(r['safe_automatic_progress']):.2f} & {float(r['coordination_precision']):.2f} & "
                f"{float(r['coordination_recall']):.2f} & {float(r['coordination_f1']):.2f} & "
                f"{float(r['over_coordination_rate']):.2f} " + r'\\'
            )
    else:
        lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule', r'Condition & Unsafe & Safe prog. & Coord. P & Coord. R & Coord. F1 & Overcoord. \\', r'\midrule']
        for r in rows:
            lines.append(
                f"{_tex(r['label'])} & {float(r['unsafe_issue_accept_rate']):.2f} & "
                f"{float(r['safe_automatic_progress']):.2f} & {float(r['coordination_precision']):.2f} & "
                f"{float(r['coordination_recall']):.2f} & {float(r['coordination_f1']):.2f} & "
                f"{float(r['over_coordination_rate']):.2f} " + r'\\'
            )
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Build tables for the LLM-generated WCCU obligation benchmark.')
    p.add_argument('input')
    p.add_argument('--out-prefix', required=True, help='Prefix such as analysis/tag/llm_obligation')
    args = p.parse_args(argv)
    payload = _load(args.input)
    prefix = Path(args.out_prefix)
    gen = build_generation_rows(payload)
    overall = build_decision_rows(payload, by_family=False)
    fam = build_decision_rows(payload, by_family=True)
    write_csv(prefix.with_name(prefix.name + '_generation.csv'), gen)
    write_generation_tex(prefix.with_name(prefix.name + '_generation.tex'), gen)
    write_csv(prefix.with_name(prefix.name + '_decision_overall.csv'), overall)
    write_decision_tex(prefix.with_name(prefix.name + '_decision_overall.tex'), overall)
    write_csv(prefix.with_name(prefix.name + '_decision_by_family.csv'), fam)
    write_decision_tex(prefix.with_name(prefix.name + '_decision_by_family.tex'), fam, include_family=True)
    print(json.dumps({'ok': True, 'generation_rows': len(gen), 'overall_rows': len(overall), 'family_rows': len(fam)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
