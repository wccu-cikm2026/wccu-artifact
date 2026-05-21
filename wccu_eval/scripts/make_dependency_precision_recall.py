from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.eval.scenarios import get_scenario
from wccu_eval.utils import as_dict, as_list, clean, ensure_dir


_EXISTENCE_STATUSES = {'present', 'exists', 'existing', 'available', 'any'}


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = clean(value).lower()
    return s in {'1', 'true', 'yes', 'y', 'required', 'require', 'on'}


def _expected_deps(scenario: dict[str, Any], agent_id: str) -> dict[str, dict[str, Any]]:
    """Return oracle dependency obligations keyed by target id.

    The old analysis only checked whether the dependency target appeared.  For
    WCCU, the important question is whether the dependency is also declared as a
    freshness/validity obligation.  Scenario fixtures can mark this with
    freshness_required/no_retracted_dependencies/expected_status.
    """
    declared = as_dict(scenario.get('wccu_read_dependencies'))
    out: dict[str, dict[str, Any]] = {}
    for row in as_list(declared.get(agent_id)) + as_list(declared.get('*')):
        r = as_dict(row)
        tid = clean(r.get('target_id') or r.get('atom_id') or r.get('id'))
        if not tid:
            continue
        expected_status = clean(r.get('expected_status')).lower()
        freshness_required = _boolish(r.get('freshness_required')) or _boolish(r.get('no_retracted_dependencies'))
        # If a fixture says a dependency must remain active, treat it as a
        # critical freshness obligation even if the boolean is omitted.
        if expected_status and expected_status not in _EXISTENCE_STATUSES:
            freshness_required = True
        out[tid] = {
            'target_id': tid,
            'freshness_required': freshness_required,
            'expected_status': expected_status,
            'reason': clean(r.get('reason')),
        }
    return out


def _predicted_deps_with_obligations(intent: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cert = as_dict(intent.get('certificate'))
    cert_pre = as_dict(cert.get('preconditions'))
    cert_freshness = _boolish(cert_pre.get('freshness_required')) or _boolish(cert_pre.get('no_retracted_dependencies'))
    requires_review_if_invalid = cert_pre.get('requires_review_if_invalid')
    requires_review_if_invalid_bool = True if requires_review_if_invalid is None else _boolish(requires_review_if_invalid)
    out: dict[str, dict[str, Any]] = {}
    for dep0 in as_list(cert.get('read_dependencies')):
        dep = as_dict(dep0)
        tid = clean(dep.get('target_id') or dep.get('atom_id') or dep.get('id'))
        if not tid:
            continue
        expected_status = clean(dep.get('expected_status')).lower()
        dep_freshness = _boolish(dep.get('freshness_required')) or _boolish(dep.get('no_retracted_dependencies'))
        if expected_status and expected_status not in _EXISTENCE_STATUSES:
            dep_freshness = True
        freshness_obligation = dep_freshness or cert_freshness
        # A certificate can be freshness-critical but still ask the runtime not
        # to review invalidation.  That is an important unsafe precondition, so
        # keep it separate rather than silently treating it as a good obligation.
        out[tid] = {
            'target_id': tid,
            'freshness_obligation': freshness_obligation,
            'review_if_invalid': requires_review_if_invalid_bool,
            'expected_status': expected_status,
            'reason': clean(dep.get('reason')),
        }
    return out


def _predicted_deps(intent: dict[str, Any]) -> set[str]:
    return set(_predicted_deps_with_obligations(intent).keys())


def _stale_error_targets(row: dict[str, Any], intent_id: str) -> set[str]:
    out: set[str] = set()
    for event in as_list(row.get('wccu_events')):
        if clean(event.get('intent_id')) != clean(intent_id):
            continue
        for err in as_list(event.get('errors')):
            e = as_dict(err)
            if clean(e.get('kind')) == 'stale_read_dependency':
                tid = clean(e.get('target_id') or e.get('atom_id') or e.get('id'))
                if tid:
                    out.add(tid)
    # Some older rows only store event details under merge_decisions.
    for decision in as_list(row.get('merge_decisions')):
        for event in as_list(as_dict(decision).get('wccu_events')):
            if clean(event.get('intent_id')) and clean(event.get('intent_id')) != clean(intent_id):
                continue
            for err in as_list(event.get('errors')):
                e = as_dict(err)
                if clean(e.get('kind')) == 'stale_read_dependency':
                    tid = clean(e.get('target_id') or e.get('atom_id') or e.get('id'))
                    if tid:
                        out.add(tid)
    return out


def _ignored_stale_count(row: dict[str, Any], intent_id: str) -> int:
    count = 0
    for event in as_list(row.get('wccu_events')):
        if clean(event.get('intent_id')) != clean(intent_id):
            continue
        for warn in as_list(event.get('warnings')):
            w = as_dict(warn)
            if clean(w.get('kind')) == 'stale_read_dependency_ignored':
                try:
                    count += int(w.get('count') or 1)
                except Exception:
                    count += 1
    return count


def _iter_intents(row: dict[str, Any]):
    for run in as_list(row.get('agentRuns')):
        for intent in as_list(run.get('write_intents')):
            yield clean(run.get('agent_id')), intent


def _init_group() -> dict[str, Any]:
    return {
        'target_tp': 0,
        'target_fp': 0,
        'target_fn': 0,
        'freshness_tp': 0,
        'freshness_fp': 0,
        'freshness_fn': 0,
        'enforced_tp': 0,
        'enforced_fp': 0,
        'enforced_fn': 0,
        'evaluated_intents': 0,
        'oracle_intents': 0,
        'predicted_dep_intents': 0,
        'freshness_obligation_intents': 0,
        'enforced_dependency_intents': 0,
        'missing_dependency_count': 0,
        'missing_freshness_obligation_count': 0,
        'non_reviewing_invalid_precondition_count': 0,
        'ignored_stale_validation_count': 0,
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(_init_group)
    for row in as_list(payload.get('results')):
        scenario_id = clean(row.get('scenario_id'))
        condition = clean(row.get('condition'))
        try:
            scenario = get_scenario(scenario_id)
        except Exception:
            # External CooperBench-derived scenarios are not in the built-in registry.
            continue
        if not as_dict(scenario.get('wccu_read_dependencies')):
            continue
        g = groups[(scenario_id, condition)]
        for agent_id, intent in _iter_intents(row):
            expected_map = _expected_deps(scenario, agent_id)
            if not expected_map:
                continue
            predicted_map = _predicted_deps_with_obligations(intent)
            expected = set(expected_map.keys())
            predicted = set(predicted_map.keys())
            expected_fresh = {tid for tid, meta in expected_map.items() if meta.get('freshness_required')}
            predicted_fresh = {tid for tid, meta in predicted_map.items() if meta.get('freshness_obligation')}
            predicted_reviewable_fresh = {tid for tid, meta in predicted_map.items() if meta.get('freshness_obligation') and meta.get('review_if_invalid')}
            enforced = _stale_error_targets(row, clean(intent.get('id')))

            g['evaluated_intents'] += 1
            g['oracle_intents'] += 1
            g['predicted_dep_intents'] += 1 if predicted else 0
            g['freshness_obligation_intents'] += 1 if predicted_fresh else 0
            g['enforced_dependency_intents'] += 1 if enforced else 0

            g['target_tp'] += len(expected & predicted)
            g['target_fp'] += len(predicted - expected)
            g['target_fn'] += len(expected - predicted)

            g['freshness_tp'] += len(expected_fresh & predicted_fresh)
            g['freshness_fp'] += len(predicted_fresh - expected_fresh)
            g['freshness_fn'] += len(expected_fresh - predicted_fresh)

            g['enforced_tp'] += len(expected_fresh & enforced)
            g['enforced_fp'] += len(enforced - expected_fresh)
            g['enforced_fn'] += len(expected_fresh - enforced)

            g['missing_dependency_count'] += len(expected - predicted)
            g['missing_freshness_obligation_count'] += len((expected_fresh & predicted) - predicted_fresh)
            # The target dependency was present and freshness-critical, but the
            # certificate explicitly disabled review on invalidation.  This is a
            # separate failure mode from missing the target entirely.
            for tid in expected_fresh & predicted_fresh:
                if tid not in predicted_reviewable_fresh:
                    g['non_reviewing_invalid_precondition_count'] += 1
            g['ignored_stale_validation_count'] += _ignored_stale_count(row, clean(intent.get('id')))

    rows: list[dict[str, Any]] = []
    for (scenario_id, condition), g in sorted(groups.items()):
        target_p, target_r, target_f1 = _prf(int(g['target_tp']), int(g['target_fp']), int(g['target_fn']))
        fresh_p, fresh_r, fresh_f1 = _prf(int(g['freshness_tp']), int(g['freshness_fp']), int(g['freshness_fn']))
        enforced_p, enforced_r, enforced_f1 = _prf(int(g['enforced_tp']), int(g['enforced_fp']), int(g['enforced_fn']))
        evaluated = int(g['evaluated_intents'])
        rows.append({
            'scenario_id': scenario_id,
            'condition': condition,
            'evaluated_intents': evaluated,
            # Backward-compatible names from the original script.  These measure
            # target presence only, not WCCU obligation correctness.
            'tp': int(g['target_tp']),
            'fp': int(g['target_fp']),
            'fn': int(g['target_fn']),
            'precision': target_p,
            'recall': target_r,
            'f1': target_f1,
            'predicted_dependency_intent_rate': g['predicted_dep_intents'] / evaluated if evaluated else 0.0,
            # Explicit target-level metrics.
            'target_dependency_tp': int(g['target_tp']),
            'target_dependency_fp': int(g['target_fp']),
            'target_dependency_fn': int(g['target_fn']),
            'target_dependency_precision': target_p,
            'target_dependency_recall': target_r,
            'target_dependency_f1': target_f1,
            # Obligation-level metrics: did the certificate mark the dependency
            # as freshness-critical?
            'freshness_obligation_tp': int(g['freshness_tp']),
            'freshness_obligation_fp': int(g['freshness_fp']),
            'freshness_obligation_fn': int(g['freshness_fn']),
            'freshness_obligation_precision': fresh_p,
            'freshness_obligation_recall': fresh_r,
            'freshness_obligation_f1': fresh_f1,
            'freshness_obligation_intent_rate': g['freshness_obligation_intents'] / evaluated if evaluated else 0.0,
            # Runtime-level metrics: did verification actually enforce the
            # stale dependency when it was invalidated by the parallel group?
            'enforced_dependency_tp': int(g['enforced_tp']),
            'enforced_dependency_fp': int(g['enforced_fp']),
            'enforced_dependency_fn': int(g['enforced_fn']),
            'enforced_dependency_precision': enforced_p,
            'enforced_dependency_recall': enforced_r,
            'enforced_dependency_f1': enforced_f1,
            'enforced_dependency_intent_rate': g['enforced_dependency_intents'] / evaluated if evaluated else 0.0,
            # Failure-mode counters.
            'missing_dependency_count': int(g['missing_dependency_count']),
            'missing_freshness_obligation_count': int(g['missing_freshness_obligation_count']),
            'non_reviewing_invalid_precondition_count': int(g['non_reviewing_invalid_precondition_count']),
            'ignored_stale_validation_count': int(g['ignored_stale_validation_count']),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Measure WCCU read-dependency and obligation precision/recall against scenario-declared oracle dependencies.')
    parser.add_argument('input')
    parser.add_argument('--out-csv', default='analysis/dependency_precision_recall.csv')
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding='utf-8'))
    rows = compute_rows(payload)
    write_csv(Path(args.out_csv), rows)
    print(json.dumps({'ok': True, 'out_csv': args.out_csv, 'rows': len(rows)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
