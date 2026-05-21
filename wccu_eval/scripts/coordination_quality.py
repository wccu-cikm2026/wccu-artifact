from __future__ import annotations

"""Reusable coordination-quality metrics for context-update diagnostics.

The raw experiment logs already report binary safety signals such as stale
accepts and review counts.  These helpers turn the same logs into classifier-like
metrics: precision/recall/F1 for selective coordination, safe automatic progress,
over-coordination, and safe-progress-at-zero-unsafe.

The helpers deliberately count *unique problematic update instances* rather than
raw verifier events.  A single bad update can trigger multiple events (for
example stale-read plus wrong-target), but unsafe issue-accept rates should stay
in the [0, 1] interval for every row and aggregate.
"""

import math
from collections import defaultdict
from typing import Any, Iterable

from wccu_eval.utils import as_dict, clean


def _num(value: Any) -> float:
    try:
        n = float(value or 0)
        return n if math.isfinite(n) else 0.0
    except Exception:
        return 0.0


def _int(value: Any) -> int:
    return int(round(_num(value)))


def _ratio(num: float, den: float, *, empty: float = 0.0) -> float:
    return float(num) / float(den) if den else empty


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def write_total(row: dict[str, Any]) -> int:
    commit = as_dict(row.get('commit'))
    return max(
        _int(row.get('ground_truth_total_writes')),
        _int(row.get('write_intent_count')),
        _int(commit.get('total')),
        _int(row.get('agent_count')),
    )


def review_block_count(row: dict[str, Any]) -> int:
    # Most runners include blocked WCCU updates in review_burden_count.  The max
    # guard keeps older logs that only expose wccu_blocked_count interpretable
    # without double counting modern logs.
    return max(_int(row.get('review_burden_count')), _int(row.get('wccu_blocked_count')))


def unsafe_issue_accept_count(row: dict[str, Any]) -> int:
    # Prefer ground-truth issue acceptance when the diagnostic runner provides
    # it.  Fall back to stale-dependency accepts for older stress logs.  Use max
    # rather than sum so one update with multiple violation events is still one
    # accepted problematic update.
    return max(
        _int(row.get('ground_truth_issue_accepted_count')),
        _int(row.get('ground_truth_problematic_accepted_count')),
        _int(row.get('stale_dependency_accepted_count')),
    )


def issue_count(row: dict[str, Any], override: int | None = None) -> int:
    if override is not None:
        return _int(override)
    explicit = max(_int(row.get('ground_truth_issue_count')), _int(row.get('ground_truth_problematic_count')))
    if explicit:
        return explicit
    return max(
        _int(row.get('stale_dependency_count')),
        _int(row.get('stale_dependency_accepted_count')),
        _int(row.get('stale_read_validation_ignored_count')),
        _int(row.get('stale_write_blocked_count')),
    )


def hold_required_count(row: dict[str, Any], issues: int | None = None) -> int:
    explicit = row.get('ground_truth_hold_required_count')
    if explicit is not None:
        return _int(explicit)
    return issue_count(row, issues)


def issue_detected_count(row: dict[str, Any], issues: int | None = None) -> int:
    explicit = row.get('ground_truth_issue_detected_count')
    if explicit is not None:
        return min(issue_count(row, issues), _int(explicit))
    if bool(row.get('expected_event_observed')):
        return min(issue_count(row, issues), 1)
    return min(issue_count(row, issues), _int(row.get('wccu_intervention_count')))


def problematic_held_count(row: dict[str, Any], issues: int | None = None) -> int:
    explicit = row.get('ground_truth_problematic_held_count')
    if explicit is not None:
        return min(hold_required_count(row, issues), _int(explicit))
    hold_required = hold_required_count(row, issues)
    if hold_required <= 0:
        return 0
    accepted = unsafe_issue_accept_count(row)
    if accepted > 0:
        return max(0, hold_required - accepted)
    return min(hold_required, review_block_count(row))


def scenario_key(row: dict[str, Any], family: str | None = None) -> tuple[Any, ...]:
    return (
        clean(family or row.get('kind') or row.get('task_type')),
        clean(row.get('scenario_id')),
        int(row.get('repetition') or 0),
        clean(row.get('obligation_kind')),
    )


def infer_scenario_issue_map(rows: Iterable[dict[str, Any]], family: str | None = None) -> dict[tuple[Any, ...], int]:
    """Infer ground-truth issue counts across conditions for shared scenarios.

    Baselines often cannot observe the stale dependency and therefore log zero
    stale_dependency_count.  We recover the shared ground truth by taking the
    maximum signal over all conditions for the same scenario.
    """
    out: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = scenario_key(row, family)
        signals = [issue_count(row), unsafe_issue_accept_count(row)]
        meta = as_dict(row.get('stress_metadata'))
        if 'invalidated_dependency_count' in meta:
            signals.append(_int(meta.get('invalidated_dependency_count')))
        sid = clean(row.get('scenario_id'))
        if 'commitment' in sid and 'stale' in sid:
            signals.append(1)
        out[key] = max(out[key], *signals)
    return dict(out)


def row_quality(row: dict[str, Any], issue_override: int | None = None) -> dict[str, Any]:
    total = write_total(row)
    issue_accepted_raw = unsafe_issue_accept_count(row)
    issues = max(issue_count(row, issue_override), issue_accepted_raw)
    # Clamp accepted problematic updates to the number of problematic updates in
    # this row.  Event-level diagnostics remain available in their own columns;
    # rates in this module are update-instance rates.
    issue_accepted = min(issue_accepted_raw, issues) if issues else 0
    hold_required = max(0, hold_required_count(row, issues))
    held = review_block_count(row)
    problem_held = min(hold_required, problematic_held_count(row, issues))
    false_holds = max(0, held - problem_held)
    safe_writes = max(0, total - hold_required)
    safe_auto = max(0, safe_writes - false_holds)
    precision = _ratio(problem_held, held, empty=1.0 if problem_held == 0 and held == 0 and issues == 0 else 0.0)
    recall = _ratio(problem_held, hold_required, empty=1.0 if hold_required == 0 else 0.0)
    detection_recall = _ratio(issue_detected_count(row, issues), issues, empty=1.0 if issues == 0 else 0.0)
    unsafe_rate = _ratio(issue_accepted, issues, empty=0.0)
    safe_progress = _ratio(safe_auto, safe_writes, empty=1.0)
    overcoord = _ratio(false_holds, safe_writes, empty=0.0)
    return {
        'total_writes': total,
        'issue_count': issues,
        'hold_required_count': hold_required,
        'review_block_count': held,
        'problematic_held_count': problem_held,
        'false_hold_count': false_holds,
        'issue_accepted_count': issue_accepted,
        'safe_write_count': safe_writes,
        'safe_auto_commit_count': safe_auto,
        'coordination_precision': precision,
        'coordination_recall': recall,
        'coordination_f1': _f1(precision, recall),
        'obligation_detection_recall': detection_recall,
        'unsafe_issue_accept_rate': unsafe_rate,
        'safe_automatic_progress': safe_progress,
        'over_coordination_rate': overcoord,
        'sp_at_zero_unsafe': safe_progress if issue_accepted == 0 else 0.0,
        'reviews_per_true_issue': _ratio(held, problem_held, empty=0.0 if held == 0 else math.inf),
    }


def aggregate_quality(rows: Iterable[dict[str, Any]], issue_overrides: dict[tuple[Any, ...], int] | None = None, family: str | None = None) -> dict[str, Any]:
    totals = defaultdict(float)
    rows_list = list(rows)
    for row in rows_list:
        override = issue_overrides.get(scenario_key(row, family)) if issue_overrides else None
        q = row_quality(row, override)
        for key in ['total_writes', 'issue_count', 'hold_required_count', 'review_block_count', 'problematic_held_count', 'false_hold_count', 'issue_accepted_count', 'safe_write_count', 'safe_auto_commit_count']:
            totals[key] += q[key]
    precision = _ratio(totals['problematic_held_count'], totals['review_block_count'], empty=1.0 if totals['issue_count'] == 0 else 0.0)
    recall = _ratio(totals['problematic_held_count'], totals['hold_required_count'], empty=1.0 if totals['hold_required_count'] == 0 else 0.0)
    safe_progress = _ratio(totals['safe_auto_commit_count'], totals['safe_write_count'], empty=1.0)
    issue_accept_rate = _ratio(totals['issue_accepted_count'], totals['issue_count'], empty=0.0)
    return {
        'total_writes': int(totals['total_writes']),
        'issue_count': int(totals['issue_count']),
        'hold_required_count': int(totals['hold_required_count']),
        'review_block_count': int(totals['review_block_count']),
        'problematic_held_count': int(totals['problematic_held_count']),
        'false_hold_count': int(totals['false_hold_count']),
        'issue_accepted_count': int(totals['issue_accepted_count']),
        'safe_write_count': int(totals['safe_write_count']),
        'safe_auto_commit_count': int(totals['safe_auto_commit_count']),
        'coordination_precision': precision,
        'coordination_recall': recall,
        'coordination_f1': _f1(precision, recall),
        'unsafe_issue_accept_rate': issue_accept_rate,
        'safe_automatic_progress': safe_progress,
        'over_coordination_rate': _ratio(totals['false_hold_count'], totals['safe_write_count'], empty=0.0),
        'sp_at_zero_unsafe': safe_progress if totals['issue_accepted_count'] == 0 else 0.0,
        'reviews_per_true_issue': _ratio(totals['review_block_count'], totals['problematic_held_count'], empty=0.0 if totals['review_block_count'] == 0 else math.inf),
    }
