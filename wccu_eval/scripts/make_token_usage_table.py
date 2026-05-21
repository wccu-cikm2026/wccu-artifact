from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir

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
    'adaptive_wccu_model_certificate': 'WCCU, model certificate',
    'adaptive_wccu_unguided_certificate': 'WCCU, unguided model cert',
    'adaptive_wccu_oracle_dependency': 'WCCU, oracle dependency',
    'adaptive_wccu_no_read_validation': 'WCCU, no read validation',
    'adaptive_policy': 'Adaptive, no WCCU',
    'uniform_snapshot_occ': 'Snapshot OCC',
    'uniform_review_gated': 'Review-gated',
    'uniform_append_only': 'Append-only',
}

DEFAULT_ORDER = [
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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
        return n if math.isfinite(n) else default
    except Exception:
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _pct_delta(value: float, baseline: float) -> float:
    return ((value - baseline) / baseline * 100.0) if baseline else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(math.ceil(0.95 * len(xs))) - 1
    idx = max(0, min(idx, len(xs) - 1))
    return xs[idx]


def _iter_result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [as_dict(r) for r in as_list(payload.get('results'))]


def _usage_from_agent_run(agent_run: dict[str, Any]) -> dict[str, float | bool]:
    """Return actual provider token usage when available, falling back to estimates.

    OpenAI response usage reports total_tokens, input_tokens, output_tokens,
    reasoning_tokens, and cached_input_tokens.  We keep reasoning tokens separate
    because provider total_tokens already includes billed input/output accounting;
    adding reasoning_tokens again would double count for most summaries.
    """
    llm = as_dict(agent_run.get('llm'))
    usage = as_dict(llm.get('api_usage'))
    has_actual = bool(usage)

    input_tokens = _num(usage.get('input_tokens')) if has_actual else _num(llm.get('prompt_tokens_est'))
    output_tokens = _num(usage.get('output_tokens')) if has_actual else _num(llm.get('output_tokens_est'))
    total_tokens = _num(usage.get('total_tokens')) if has_actual else input_tokens + output_tokens
    reasoning_tokens = _num(usage.get('reasoning_tokens')) if has_actual else 0.0
    cached_input_tokens = _num(usage.get('cached_input_tokens')) if has_actual else 0.0
    non_cached_input_tokens = max(0.0, input_tokens - cached_input_tokens)

    return {
        'has_actual_usage': has_actual,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'reasoning_tokens': reasoning_tokens,
        'cached_input_tokens': cached_input_tokens,
        'non_cached_input_tokens': non_cached_input_tokens,
        'prompt_tokens_est': _num(llm.get('prompt_tokens_est')),
        'output_tokens_est': _num(llm.get('output_tokens_est')),
        'latency_ms': _num(agent_run.get('latency_ms')),
    }


def _empty_acc(condition: str) -> dict[str, Any]:
    return {
        'condition': condition,
        'display_name': DISPLAY_NAMES.get(condition, condition),
        'runs': 0,
        'failed_runs': 0,
        'diagnostic_pass_count': 0,
        'safety_pass_count': 0,
        'agent_calls': 0,
        'agent_calls_with_actual_usage': 0,
        'unsafe_auto_commit_count': 0.0,
        'stale_dependency_accepted_count': 0.0,
        'stale_dependency_detected_count': 0.0,
        'review_burden_count': 0.0,
        'wccu_intervention_count': 0.0,
        'wccu_review_routed_count': 0.0,
        'wccu_blocked_count': 0.0,
        'input_tokens_sum': 0.0,
        'output_tokens_sum': 0.0,
        'reasoning_tokens_sum': 0.0,
        'cached_input_tokens_sum': 0.0,
        'non_cached_input_tokens_sum': 0.0,
        'total_tokens_sum': 0.0,
        'prompt_tokens_est_sum': 0.0,
        'output_tokens_est_sum': 0.0,
        'elapsed_ms_values': [],
        'agent_latency_ms_values': [],
        'completed_run_token_values': [],
    }


def compute_rows(payload: dict[str, Any], baseline_condition: str = 'adaptive_policy') -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in _iter_result_rows(payload):
        condition = clean(row.get('condition'))
        if not condition:
            condition = 'unknown'
        acc = groups.setdefault(condition, _empty_acc(condition))
        acc['runs'] += 1
        failed = bool(row.get('failed'))
        if failed:
            acc['failed_runs'] += 1
        if row.get('task_success'):
            acc['diagnostic_pass_count'] += 1
        unsafe = _num(row.get('unsafe_auto_commit_count'))
        stale_accepted = _num(row.get('stale_dependency_accepted_count'))
        if not failed and unsafe <= 0 and stale_accepted <= 0:
            acc['safety_pass_count'] += 1
        acc['unsafe_auto_commit_count'] += unsafe
        acc['stale_dependency_accepted_count'] += stale_accepted
        acc['stale_dependency_detected_count'] += _num(row.get('stale_dependency_count'))
        acc['review_burden_count'] += _num(row.get('review_burden_count'))
        acc['wccu_intervention_count'] += _num(row.get('wccu_intervention_count'))
        acc['wccu_review_routed_count'] += _num(row.get('wccu_review_routed_count'))
        acc['wccu_blocked_count'] += _num(row.get('wccu_blocked_count'))
        elapsed_ms = _num(row.get('elapsed_ms'))
        if elapsed_ms:
            acc['elapsed_ms_values'].append(elapsed_ms)

        run_total_tokens = 0.0
        for agent_run in as_list(row.get('agentRuns')):
            ar = as_dict(agent_run)
            usage = _usage_from_agent_run(ar)
            acc['agent_calls'] += 1
            if usage['has_actual_usage']:
                acc['agent_calls_with_actual_usage'] += 1
            for k in [
                'input_tokens',
                'output_tokens',
                'reasoning_tokens',
                'cached_input_tokens',
                'non_cached_input_tokens',
                'total_tokens',
                'prompt_tokens_est',
                'output_tokens_est',
            ]:
                acc[f'{k}_sum'] += float(usage[k])
            run_total_tokens += float(usage['total_tokens'])
            latency_ms = float(usage['latency_ms'])
            if latency_ms:
                acc['agent_latency_ms_values'].append(latency_ms)
        if not failed and run_total_tokens:
            acc['completed_run_token_values'].append(run_total_tokens)

    order = {c: i for i, c in enumerate(DEFAULT_ORDER)}
    raw_rows = sorted(groups.values(), key=lambda r: (order.get(r['condition'], 999), r['condition']))

    # Baseline values for relative overhead and prevented-unsafe calculations.
    baseline = groups.get(baseline_condition) or (raw_rows[0] if raw_rows else None)
    baseline_mean_tokens = _safe_div(baseline['total_tokens_sum'], baseline['runs']) if baseline else 0.0
    baseline_mean_non_cached = _safe_div(baseline['non_cached_input_tokens_sum'], baseline['runs']) if baseline else 0.0
    baseline_unsafe = baseline['unsafe_auto_commit_count'] if baseline else 0.0
    baseline_stale = baseline['stale_dependency_accepted_count'] if baseline else 0.0

    out: list[dict[str, Any]] = []
    for acc in raw_rows:
        runs = float(acc['runs'] or 0)
        completed_runs = float(max(0, acc['runs'] - acc['failed_runs']))
        agent_calls = float(acc['agent_calls'] or 0)
        total_tokens = float(acc['total_tokens_sum'])
        non_cached = float(acc['non_cached_input_tokens_sum'])
        diagnostic_pass = float(acc['diagnostic_pass_count'])
        wccu_interventions = float(acc['wccu_intervention_count'])
        prevented_unsafe = max(0.0, baseline_unsafe - float(acc['unsafe_auto_commit_count'])) if acc['condition'] != baseline_condition else 0.0
        prevented_stale = max(0.0, baseline_stale - float(acc['stale_dependency_accepted_count'])) if acc['condition'] != baseline_condition else 0.0
        row = {
            'condition': acc['condition'],
            'display_name': acc['display_name'],
            'baseline_condition': baseline_condition,
            'runs': int(acc['runs']),
            'failed_runs': int(acc['failed_runs']),
            'agent_calls': int(acc['agent_calls']),
            'agent_calls_with_actual_usage': int(acc['agent_calls_with_actual_usage']),
            'actual_usage_coverage_rate': _safe_div(acc['agent_calls_with_actual_usage'], agent_calls),
            'diagnostic_pass_count': int(acc['diagnostic_pass_count']),
            'diagnostic_pass_rate': _safe_div(diagnostic_pass, runs),
            'safety_pass_count': int(acc['safety_pass_count']),
            'safety_pass_rate': _safe_div(acc['safety_pass_count'], runs),
            'unsafe_auto_commit_count': int(acc['unsafe_auto_commit_count']),
            'stale_dependency_accepted_count': int(acc['stale_dependency_accepted_count']),
            'stale_dependency_detected_count': int(acc['stale_dependency_detected_count']),
            'review_burden_count': int(acc['review_burden_count']),
            'wccu_intervention_count': int(acc['wccu_intervention_count']),
            'wccu_review_routed_count': int(acc['wccu_review_routed_count']),
            'wccu_blocked_count': int(acc['wccu_blocked_count']),
            'input_tokens_sum': int(acc['input_tokens_sum']),
            'output_tokens_sum': int(acc['output_tokens_sum']),
            'reasoning_tokens_sum': int(acc['reasoning_tokens_sum']),
            'cached_input_tokens_sum': int(acc['cached_input_tokens_sum']),
            'non_cached_input_tokens_sum': int(acc['non_cached_input_tokens_sum']),
            'total_tokens_sum': int(total_tokens),
            'prompt_tokens_est_sum': int(acc['prompt_tokens_est_sum']),
            'output_tokens_est_sum': int(acc['output_tokens_est_sum']),
            'mean_total_tokens_per_run': _safe_div(total_tokens, runs),
            'mean_total_tokens_per_completed_run': _safe_div(total_tokens, completed_runs),
            'mean_total_tokens_per_agent_call': _safe_div(total_tokens, agent_calls),
            'mean_input_tokens_per_run': _safe_div(acc['input_tokens_sum'], runs),
            'mean_output_tokens_per_run': _safe_div(acc['output_tokens_sum'], runs),
            'mean_reasoning_tokens_per_run': _safe_div(acc['reasoning_tokens_sum'], runs),
            'mean_cached_input_tokens_per_run': _safe_div(acc['cached_input_tokens_sum'], runs),
            'mean_non_cached_input_tokens_per_run': _safe_div(non_cached, runs),
            'mean_non_cached_input_tokens_per_agent_call': _safe_div(non_cached, agent_calls),
            'token_overhead_vs_baseline_pct': _pct_delta(_safe_div(total_tokens, runs), baseline_mean_tokens),
            'non_cached_input_overhead_vs_baseline_pct': _pct_delta(_safe_div(non_cached, runs), baseline_mean_non_cached),
            'tokens_per_diagnostic_pass': _safe_div(total_tokens, diagnostic_pass),
            'tokens_per_wccu_intervention': _safe_div(total_tokens, wccu_interventions),
            'unsafe_prevented_vs_baseline_count': int(prevented_unsafe),
            'stale_acceptance_prevented_vs_baseline_count': int(prevented_stale),
            'tokens_per_unsafe_prevented_vs_baseline': _safe_div(total_tokens, prevented_unsafe),
            'tokens_per_stale_acceptance_prevented_vs_baseline': _safe_div(total_tokens, prevented_stale),
            'mean_elapsed_ms_per_run': _safe_div(sum(acc['elapsed_ms_values']), len(acc['elapsed_ms_values'])),
            'median_elapsed_ms_per_run': median(acc['elapsed_ms_values']) if acc['elapsed_ms_values'] else 0.0,
            'p95_elapsed_ms_per_run': _p95(acc['elapsed_ms_values']),
            'mean_agent_latency_ms': _safe_div(sum(acc['agent_latency_ms_values']), len(acc['agent_latency_ms_values'])),
            'p95_agent_latency_ms': _p95(acc['agent_latency_ms_values']),
        }
        # Round float fields for stable CSV diffs while keeping counts integral.
        for k, v in list(row.items()):
            if isinstance(v, float):
                row[k] = round(v, 4)
        out.append(row)
    return out


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text('', encoding='utf-8')
        return
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps({'kind': 'token_usage_table_v1', 'rows': rows}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _fmt_int(v: Any) -> str:
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return str(v)


def _fmt_float(v: Any, digits: int = 1) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def write_tex(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    lines = [
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'Condition & Diagnostic & Unsafe & Review/block & Tokens/run & Overhead \\',
        r'\midrule',
    ]
    for row in rows:
        name = str(row['display_name']).replace('_', r'\_')
        diagnostic = f"{row['diagnostic_pass_count']}/{row['runs']}"
        review_or_block = int(row.get('review_burden_count') or 0) + int(row.get('wccu_blocked_count') or 0)
        overhead = _fmt_float(row.get('token_overhead_vs_baseline_pct'), 1) + r'\%'
        lines.append(
            f"{name} & {diagnostic} & {_fmt_int(row.get('unsafe_auto_commit_count'))} & {_fmt_int(review_or_block)} & {_fmt_int(row.get('mean_total_tokens_per_run'))} & {overhead} \\\\"
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    p.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create cost-aware token/latency usage tables from context substrate result JSON.')
    parser.add_argument('input_json', help='Experiment result JSON file')
    parser.add_argument('--baseline-condition', default='adaptive_policy', help='Condition used to compute token overhead and unsafe/stale-prevented ratios')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-tex', default='')
    parser.add_argument('--out-json', default='')
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input_json).read_text(encoding='utf-8'))
    rows = compute_rows(payload, baseline_condition=args.baseline_condition)
    write_csv(args.out_csv, rows)
    if args.out_tex:
        write_tex(args.out_tex, rows)
    if args.out_json:
        write_json(args.out_json, rows)
    print(json.dumps({'ok': True, 'rows': len(rows), 'baseline_condition': args.baseline_condition, 'out_csv': args.out_csv, 'out_tex': args.out_tex, 'out_json': args.out_json}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
