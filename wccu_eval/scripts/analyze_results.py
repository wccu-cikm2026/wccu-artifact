from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, ensure_dir, write_json


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return clean(row.get('scenario_id')), clean(row.get('condition'))


def _iter_intents(row: dict[str, Any]):
    for run in as_list(row.get('agentRuns')):
        for intent in as_list(run.get('write_intents')):
            yield run, intent


def summarize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in as_list(payload.get('results')):
        groups[_row_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for (scenario_id, condition), rows in sorted(groups.items()):
        n = len(rows)
        grounded = unresolved = intents = 0
        reasoning_tokens = []
        input_tokens = []
        output_tokens = []
        for row in rows:
            for _, intent in _iter_intents(row):
                intents += 1
                tg = as_dict(intent.get('target_grounding'))
                if tg.get('resolved'):
                    grounded += 1
                elif tg:
                    unresolved += 1
            for ar in as_list(row.get('agentRuns')):
                usage = as_dict(as_dict(ar.get('llm')).get('api_usage'))
                if usage.get('reasoning_tokens') is not None: reasoning_tokens.append(float(usage.get('reasoning_tokens') or 0))
                if usage.get('input_tokens') is not None: input_tokens.append(float(usage.get('input_tokens') or 0))
                if usage.get('output_tokens') is not None: output_tokens.append(float(usage.get('output_tokens') or 0))
        out.append({
            'scenario_id': scenario_id,
            'condition': condition,
            'n': n,
            'success_rate': _mean([1.0 if r.get('task_success') else 0.0 for r in rows]),
            'failed_rate': _mean([1.0 if r.get('failed') else 0.0 for r in rows]),
            'provider_error_rate': _mean([1.0 if r.get('error_type') == 'LlmProviderError' else 0.0 for r in rows]),
            'mean_elapsed_ms': _mean([float(r.get('elapsed_ms') or 0) for r in rows]),
            'mean_unsafe_auto_commit_count': _mean([float(r.get('unsafe_auto_commit_count') or 0) for r in rows]),
            'mean_review_burden_count': _mean([float(r.get('review_burden_count') or 0) for r in rows]),
            'mean_conflict_groups': _mean([float(r.get('conflict_groups') or 0) for r in rows]),
            'mean_semantic_conflict_count': _mean([float(r.get('semantic_conflict_count') or 0) for r in rows]),
            'mean_wccu_blocked_count': _mean([float(r.get('wccu_blocked_count', r.get('wccu_blocked_count')) or 0) for r in rows]),
            'mean_wccu_review_routed_count': _mean([float(r.get('wccu_review_routed_count', r.get('wccu_review_routed_count')) or 0) for r in rows]),
            'mean_wccu_intervention_count': _mean([float(r.get('wccu_intervention_count') if r.get('wccu_intervention_count') is not None else (r.get('wccu_intervention_count') if r.get('wccu_intervention_count') is not None else ((r.get('wccu_review_routed_count') or 0) + (r.get('wccu_blocked_count') or 0)))) for r in rows]),
            # Legacy aliases remain for old analysis consumers.
            'mean_wccu_blocked_count': _mean([float(r.get('wccu_blocked_count') or 0) for r in rows]),
            'mean_wccu_review_routed_count': _mean([float(r.get('wccu_review_routed_count') or 0) for r in rows]),
            'mean_wccu_intervention_count': _mean([float(r.get('wccu_intervention_count') if r.get('wccu_intervention_count') is not None else ((r.get('wccu_review_routed_count') or 0) + (r.get('wccu_blocked_count') or 0))) for r in rows]),
            'mean_stale_dependency_accepted_count': _mean([float(r.get('stale_dependency_accepted_count') or 0) for r in rows]),
            'mean_stale_read_validation_ignored_count': _mean([float(r.get('stale_read_validation_ignored_count') or 0) for r in rows]),
            'mean_authority_correction_self_dependency_tolerated_count': _mean([float(r.get('authority_correction_self_dependency_tolerated_count') or 0) for r in rows]),
            'mean_certificate_invalid_count': _mean([float(r.get('certificate_invalid_count') or 0) for r in rows]),
            'mean_stale_dependency_count': _mean([float(r.get('stale_dependency_count') or 0) for r in rows]),
            'mean_low_target_confidence_count': _mean([float(r.get('low_target_confidence_count') or 0) for r in rows]),
            'mean_authority_insufficient_count': _mean([float(r.get('authority_insufficient_count') or 0) for r in rows]),
            'mean_view_invalidation_count': _mean([float(r.get('view_invalidation_count') or 0) for r in rows]),
            'mean_wrong_target_count': _mean([float(r.get('wrong_target_count') or 0) for r in rows]),
            'mean_weaken_rule_delta_count': _mean([float(r.get('weaken_rule_delta_count') or 0) for r in rows]),
            'mean_authority_rebase_count': _mean([float(r.get('authority_rebase_count') or 0) for r in rows]),
            'mean_committed': _mean([float(as_dict(r.get('commit')).get('committed') or 0) for r in rows]),
            'mean_proposals': _mean([float(as_dict(r.get('commit')).get('proposals') or 0) for r in rows]),
            'target_grounded_rate': grounded / intents if intents else 0.0,
            'target_unresolved_rate': unresolved / intents if intents else 0.0,
            'mean_api_input_tokens': _mean(input_tokens),
            'mean_api_output_tokens': _mean(output_tokens),
            'mean_reasoning_tokens': _mean(reasoning_tokens),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_debug_files(payload: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    failures = out_dir / 'target_grounding_failures.jsonl'
    provider_errors = out_dir / 'provider_errors.jsonl'
    debug = out_dir / 'debug_runs.jsonl'
    failures.write_text('', encoding='utf-8')
    provider_errors.write_text('', encoding='utf-8')
    debug.write_text('', encoding='utf-8')
    for row in as_list(payload.get('results')):
        debug_record = {
            'scenario_id': row.get('scenario_id'),
            'condition': row.get('condition'),
            'repetition': row.get('repetition'),
            'task_success': row.get('task_success'),
            'unsafe_auto_commit_count': row.get('unsafe_auto_commit_count'),
            'review_burden_count': row.get('review_burden_count'),
            'authority_rebase_count': row.get('authority_rebase_count'),
            'merge_decisions': row.get('merge_decisions'),
            'wccu_enabled': row.get('wccu_enabled', row.get('wccu_enabled')),
            'wccu_blocked_count': row.get('wccu_blocked_count', row.get('wccu_blocked_count')),
            'wccu_review_routed_count': row.get('wccu_review_routed_count', row.get('wccu_review_routed_count')),
            'wccu_intervention_count': row.get('wccu_intervention_count', row.get('wccu_intervention_count', (row.get('wccu_review_routed_count') or 0) + (row.get('wccu_blocked_count') or 0))),
            # Legacy aliases retained for old result readers.
            'wccu_enabled': row.get('wccu_enabled'),
            'wccu_blocked_count': row.get('wccu_blocked_count'),
            'wccu_review_routed_count': row.get('wccu_review_routed_count'),
            'wccu_intervention_count': row.get('wccu_intervention_count', (row.get('wccu_review_routed_count') or 0) + (row.get('wccu_blocked_count') or 0)),
            'stale_dependency_accepted_count': row.get('stale_dependency_accepted_count'),
            'stale_read_validation_ignored_count': row.get('stale_read_validation_ignored_count'),
            'authority_correction_self_dependency_tolerated_count': row.get('authority_correction_self_dependency_tolerated_count'),
            'wccu_certificate_mode': row.get('wccu_certificate_mode', row.get('wccu_certificate_mode')),
            'wccu_certificate_mode': row.get('wccu_certificate_mode'),
            'stale_dependency_count': row.get('stale_dependency_count'),
            'certificate_invalid_count': row.get('certificate_invalid_count'),
            'wccu_events': row.get('wccu_events', row.get('wccu_events')),
            'wccu_events': row.get('wccu_events'),
            'agents': [],
            'failed': row.get('failed'),
            'error_type': row.get('error_type'),
            'error': row.get('error'),
            'provider_error': row.get('provider_error'),
        }
        if row.get('failed') or row.get('provider_error'):
            with provider_errors.open('a', encoding='utf-8') as f:
                f.write(json.dumps(debug_record, ensure_ascii=False) + '\n')
        for run, intent in _iter_intents(row):
            p = as_dict(intent.get('payload'))
            tg = as_dict(intent.get('target_grounding'))
            debug_record['agents'].append({
                'agent_id': run.get('agent_id'),
                'role': run.get('role'),
                'output': run.get('output'),
                'intent_type': intent.get('intent_type'),
                'authority': intent.get('authority'),
                'risk': intent.get('risk'),
                'payload_id': p.get('id'),
                'payload_atom_id': p.get('atom_id'),
                'payload_target_id': p.get('target_id'),
                'payload_atom_type': p.get('atom_type'),
                'canonical_text_en': p.get('canonical_text_en'),
                'target_grounding': tg,
                'certificate': as_dict(intent.get('certificate')),
                'wccu_verification': as_dict(intent.get('wccu_verification') or intent.get('wccu_verification')),
                'wccu_verification': as_dict(intent.get('wccu_verification')),
                'llm': as_dict(run.get('llm')),
            })
            if tg and not tg.get('resolved') and intent.get('intent_type') != 'append_event':
                with failures.open('a', encoding='utf-8') as f:
                    f.write(json.dumps({**debug_record, 'failed_intent': debug_record['agents'][-1]}, ensure_ascii=False) + '\n')
        with debug.open('a', encoding='utf-8') as f:
            f.write(json.dumps(debug_record, ensure_ascii=False) + '\n')


def write_distribution_files(payload: dict[str, Any], out_dir: Path) -> None:
    target_rows: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    merge_rows: dict[tuple[str, str, str], int] = defaultdict(int)
    usage_rows: dict[tuple[str, str], dict[str, Any]] = {}
    wccu_rows: dict[tuple[str, str, str], int] = defaultdict(int)
    cert_quality: dict[tuple[str, str], dict[str, Any]] = {}

    for row in as_list(payload.get('results')):
        scenario = clean(row.get('scenario_id'))
        condition = clean(row.get('condition'))
        for decision in as_list(row.get('merge_decisions')):
            merge_rows[(scenario, condition, clean(decision.get('decision')))] += 1
        for event in as_list(row.get('wccu_events') or row.get('wccu_events')):
            if event.get('valid') is False:
                for err in as_list(event.get('errors')):
                    wccu_rows[(scenario, condition, clean(as_dict(err).get('kind') or 'invalid'))] += 1
            for warn in as_list(event.get('warnings')):
                wccu_rows[(scenario, condition, clean(as_dict(warn).get('kind') or 'warning'))] += 1
        for run, intent in _iter_intents(row):
            p = as_dict(intent.get('payload'))
            tg = as_dict(intent.get('target_grounding'))
            target_rows[(
                scenario,
                condition,
                clean(run.get('agent_id')),
                clean(p.get('target_id') or p.get('id') or p.get('atom_id')),
                clean(tg.get('resolved_target_id')),
            )] += 1
            cert = as_dict(intent.get('certificate'))
            cq = cert_quality.setdefault((scenario, condition), {
                'scenario_id': scenario,
                'condition': condition,
                'intent_count': 0,
                'certificate_count': 0,
                'read_dependency_count': 0,
                'intents_with_read_dependency': 0,
                'target_confidences': [],
                'certificate_sources': defaultdict(int),
                'certificate_modes': defaultdict(int),
                'delta_types': defaultdict(int),
            })
            cq['intent_count'] += 1
            if cert:
                cq['certificate_count'] += 1
                deps = as_list(cert.get('read_dependencies'))
                cq['read_dependency_count'] += len(deps)
                if deps:
                    cq['intents_with_read_dependency'] += 1
                tc = as_dict(cert.get('target_certificate'))
                if tc.get('confidence') is not None:
                    try:
                        cq['target_confidences'].append(float(tc.get('confidence') or 0))
                    except Exception:
                        pass
                cq['certificate_sources'][clean(cert.get('source') or 'unknown')] += 1
                cq['certificate_modes'][clean(cert.get('certificate_mode') or 'unknown')] += 1
                cq['delta_types'][clean(as_dict(cert.get('delta_contract')).get('delta_type') or 'unknown')] += 1
        key = (scenario, condition)
        usage = usage_rows.setdefault(key, {'scenario_id': scenario, 'condition': condition, 'agent_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'elapsed_ms': []})
        usage['elapsed_ms'].append(float(row.get('elapsed_ms') or 0))
        for ar in as_list(row.get('agentRuns')):
            u = as_dict(as_dict(ar.get('llm')).get('api_usage'))
            usage['agent_calls'] += 1
            usage['input_tokens'] += float(u.get('input_tokens') or 0)
            usage['output_tokens'] += float(u.get('output_tokens') or 0)
            usage['reasoning_tokens'] += float(u.get('reasoning_tokens') or 0)

    target_out = [{
        'scenario_id': k[0], 'condition': k[1], 'agent_id': k[2], 'payload_target': k[3], 'resolved_target': k[4], 'count': v
    } for k, v in sorted(target_rows.items())]
    merge_out = [{
        'scenario_id': k[0], 'condition': k[1], 'merge_decision': k[2], 'count': v
    } for k, v in sorted(merge_rows.items())]
    usage_out = []
    for row in usage_rows.values():
        calls = row['agent_calls'] or 1
        usage_out.append({
            'scenario_id': row['scenario_id'],
            'condition': row['condition'],
            'agent_calls': row['agent_calls'],
            'total_input_tokens': row['input_tokens'],
            'total_output_tokens': row['output_tokens'],
            'total_reasoning_tokens': row['reasoning_tokens'],
            'mean_input_tokens_per_call': row['input_tokens'] / calls,
            'mean_output_tokens_per_call': row['output_tokens'] / calls,
            'mean_reasoning_tokens_per_call': row['reasoning_tokens'] / calls,
            'mean_elapsed_ms': _mean(row['elapsed_ms']),
        })
    wccu_out = [{'scenario_id': k[0], 'condition': k[1], 'wccu_event': k[2], 'count': v} for k, v in sorted(wccu_rows.items())]
    cert_out = []
    for row in cert_quality.values():
        intent_count = row['intent_count'] or 1
        cert_count = row['certificate_count'] or 1
        cert_out.append({
            'scenario_id': row['scenario_id'],
            'condition': row['condition'],
            'intent_count': row['intent_count'],
            'certificate_count': row['certificate_count'],
            'certificate_coverage_rate': row['certificate_count'] / intent_count,
            'read_dependency_count': row['read_dependency_count'],
            'intents_with_read_dependency': row['intents_with_read_dependency'],
            'read_dependency_intent_rate': row['intents_with_read_dependency'] / intent_count,
            'mean_target_certificate_confidence': _mean(row['target_confidences']),
            'certificate_sources': json.dumps(dict(row['certificate_sources']), sort_keys=True),
            'certificate_modes': json.dumps(dict(row['certificate_modes']), sort_keys=True),
            'delta_types': json.dumps(dict(row['delta_types']), sort_keys=True),
        })
    write_csv(out_dir / 'target_distribution.csv', target_out)
    write_csv(out_dir / 'merge_decision_distribution.csv', merge_out)
    write_csv(out_dir / 'wccu_event_distribution.csv', wccu_out)
    # Compatibility copy for older notebooks/scripts.
    write_csv(out_dir / 'wccu_event_distribution.csv', [{'scenario_id': r['scenario_id'], 'condition': r['condition'], 'wccu_event': r['wccu_event'], 'count': r['count']} for r in wccu_out])
    write_csv(out_dir / 'certificate_quality.csv', sorted(cert_out, key=lambda r: (r['scenario_id'], r['condition'])))
    write_csv(out_dir / 'llm_usage_summary.csv', sorted(usage_out, key=lambda r: (r['scenario_id'], r['condition'])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Analyze context substrate LLM experiment results.')
    parser.add_argument('input')
    parser.add_argument('--out-dir', default='analysis')
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding='utf-8'))
    out_dir = Path(args.out_dir)
    rows = summarize(payload)
    write_json(out_dir / 'summary.json', {'kind': 'context_substrate_result_analysis_v1', 'source': str(input_path), 'summary': rows})
    write_csv(out_dir / 'summary.csv', rows)
    write_debug_files(payload, out_dir)
    write_distribution_files(payload, out_dir)
    try:
        from wccu_eval.scripts.make_dependency_precision_recall import compute_rows as _dependency_pr_rows, write_csv as _write_dependency_pr_csv
        _write_dependency_pr_csv(out_dir / 'dependency_precision_recall.csv', _dependency_pr_rows(payload))
    except Exception as exc:
        (out_dir / 'dependency_precision_recall.error.txt').write_text(str(exc), encoding='utf-8')
    token_usage_table_path = ''
    try:
        from wccu_eval.scripts.make_token_usage_table import compute_rows as _token_rows, write_csv as _write_token_csv
        token_usage_table_path = str(out_dir / 'token_usage_table.csv')
        _write_token_csv(out_dir / 'token_usage_table.csv', _token_rows(payload))
    except Exception as exc:
        (out_dir / 'token_usage_table.error.txt').write_text(str(exc), encoding='utf-8')
    cooperbench_table_path = ''
    if payload.get('external_benchmark') == 'cooperbench' or any(as_dict(r).get('external_benchmark') == 'cooperbench' for r in as_list(payload.get('results'))):
        try:
            from wccu_eval.scripts.make_cooperbench_table import compute_rows as _coop_rows, write_csv as _write_coop_csv
            cooperbench_table_path = str(out_dir / 'cooperbench_table.csv')
            _write_coop_csv(out_dir / 'cooperbench_table.csv', _coop_rows(payload))
        except Exception as exc:
            (out_dir / 'cooperbench_table.error.txt').write_text(str(exc), encoding='utf-8')
    print(json.dumps({'ok': True, 'out_dir': str(out_dir), 'summary_rows': len(rows), 'debug_runs': str(out_dir / 'debug_runs.jsonl'), 'target_grounding_failures': str(out_dir / 'target_grounding_failures.jsonl'), 'provider_errors': str(out_dir / 'provider_errors.jsonl'), 'target_distribution': str(out_dir / 'target_distribution.csv'), 'merge_decision_distribution': str(out_dir / 'merge_decision_distribution.csv'), 'llm_usage_summary': str(out_dir / 'llm_usage_summary.csv'), 'wccu_event_distribution': str(out_dir / 'wccu_event_distribution.csv'), 'wccu_event_distribution': str(out_dir / 'wccu_event_distribution.csv'), 'certificate_quality': str(out_dir / 'certificate_quality.csv'), 'dependency_precision_recall': str(out_dir / 'dependency_precision_recall.csv'), 'token_usage_table': token_usage_table_path, 'cooperbench_table': cooperbench_table_path}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
