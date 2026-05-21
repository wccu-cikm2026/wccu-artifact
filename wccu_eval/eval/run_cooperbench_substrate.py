from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from wccu_eval.agents.llm_agent import LlmProviderError, run_llm_agent
from wccu_eval.env import load_dotenv
from wccu_eval.eval.run_experiment import REPO_ROOT, aggregate, selected_conditions
from wccu_eval.eval.run_llm_experiment import build_conditions, _aggregate_llm
from wccu_eval.external.cooperbench_adapter import cooperbench_tasks_to_scenarios, load_cooperbench_tasks
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import append_jsonl, clean, ensure_dir, now_iso, remove_dir, stable_hash, write_json

DEFAULT_CONDITIONS = 'adaptive_wccu_execution_trace,adaptive_wccu_projection_trace,adaptive_readset_occ,adaptive_policy,uniform_snapshot_occ,uniform_review_gated,uniform_append_only'


def _condition_ids(value: str) -> list[str]:
    value = clean(value or DEFAULT_CONDITIONS)
    if value == 'all':
        return [c.strip() for c in DEFAULT_CONDITIONS.split(',')]
    return selected_conditions(value)


def run_cooperbench_substrate(*, input: str, condition: str = DEFAULT_CONDITIONS, repetitions: int = 1, limit: int = 0, out: str = 'results/cooperbench_substrate_results.json', provider: str = '', model: str = '', temperature: float | None = None, max_output_tokens: int = 1000, timeout_seconds: int = 90, max_parse_retries: int = 1, reasoning_effort: str = '', text_verbosity: str = '', send_temperature: bool | None = None, enable_target_grounding: bool = True, enable_target_candidates: bool = True, parallel_workers: int = 1, shuffle_cells: bool = False, max_provider_retries: int = 4, retry_backoff_base: float = 1.0, retry_backoff_max: float = 20.0, error_log: str = '', fail_fast: bool = False, certificate_guidance: str = 'guided') -> dict[str, Any]:
    load_dotenv()
    provider = provider or os.environ.get('LLM_PROVIDER', 'openai')
    model = model or os.environ.get('LLM_MODEL', '')
    tasks = load_cooperbench_tasks(input)
    if limit and limit > 0:
        tasks = tasks[:limit]
    scenarios = cooperbench_tasks_to_scenarios(tasks)
    condition_ids = _condition_ids(condition)
    repetitions = max(1, int(repetitions))
    parallel_workers = max(1, int(parallel_workers or 1))
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    jsonl_path = out_path.with_suffix('.jsonl')
    error_log_path = (REPO_ROOT / error_log).resolve() if error_log and not Path(error_log).is_absolute() else (Path(error_log) if error_log else out_path.with_suffix('.errors.jsonl'))
    ensure_dir(out_path.parent)
    for p in (jsonl_path, error_log_path):
        if p.exists():
            p.unlink()
    tmp_root = REPO_ROOT / 'runs' / f"cooperbench_tmp_{stable_hash(f'{out}:{os.getpid()}:{now_iso()}')}"
    remove_dir(tmp_root)

    llm_config = {
        'provider': provider,
        'model': model,
        'temperature': temperature,
        'max_output_tokens': max_output_tokens,
        'timeout_seconds': timeout_seconds,
        'max_parse_retries': max_parse_retries,
        'max_provider_retries': max_provider_retries,
        'retry_backoff_base': retry_backoff_base,
        'retry_backoff_max': retry_backoff_max,
        'reasoning_effort': reasoning_effort,
        'text_verbosity': text_verbosity,
        'send_temperature': send_temperature,
        'enable_target_grounding': enable_target_grounding,
        'enable_target_candidates': enable_target_candidates,
        'error_log_path': str(error_log_path),
        'certificate_guidance': certificate_guidance,
    }
    conditions = build_conditions(llm_config)
    cells: list[tuple[dict[str, Any], str, int]] = []
    for sc in scenarios:
        for cond in condition_ids:
            if cond not in conditions:
                raise KeyError(f'Unknown or unsupported condition for CooperBench adapter: {cond}')
            for rep in range(repetitions):
                cells.append((sc, cond, rep))
    if shuffle_cells:
        cells = sorted(cells, key=lambda c: stable_hash(f"{c[0]['id']}:{c[1]}:{c[2]}"))

    def run_cell(cell: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
        sc, cond, rep = cell
        run_dir = tmp_root / f"{sc['id']}_{cond}_{rep}"
        root_dir = run_dir / 'context_substrate'
        remove_dir(run_dir)
        seed_context(root_dir, sc.get('seed', {}))
        try:
            row = conditions[cond](scenario=sc, root_dir=root_dir, run_dir=run_dir, repetition=rep)
            row_cfg = {}
            for ar in row.get('agentRuns', []):
                if ar.get('llm', {}).get('request_options'):
                    row_cfg = ar['llm']['request_options']
                    break
            return {**row, 'failed': False, 'repetition': rep, 'external_benchmark': 'cooperbench', 'source_task_id': sc.get('source_task_id'), 'repo': sc.get('repo'), 'language': sc.get('language'), 'llm_experiment': {
                'provider': provider,
                'model': model or row.get('agentRuns', [{}])[0].get('llm', {}).get('model', ''),
                'temperature': temperature,
                'max_output_tokens': max_output_tokens,
                'max_parse_retries': max_parse_retries,
                'reasoning_effort': reasoning_effort,
                'text_verbosity': text_verbosity,
                'send_temperature': send_temperature,
                'enable_target_grounding': row_cfg.get('enable_target_grounding', enable_target_grounding) if row_cfg else enable_target_grounding,
                'enable_target_candidates': row_cfg.get('enable_target_candidates', enable_target_candidates) if row_cfg else enable_target_candidates,
                'parallel_workers': parallel_workers,
                'max_provider_retries': max_provider_retries,
                'retry_backoff_base': retry_backoff_base,
                'retry_backoff_max': retry_backoff_max,
                'error_log': str(error_log_path),
            }}
        except Exception as exc:
            if fail_fast:
                raise
            provider_diag = exc.to_dict() if isinstance(exc, LlmProviderError) else {}
            error_row = {
                'kind': 'parallel_execution_error_v1',
                'failed': True,
                'condition': cond,
                'policy_mode': cond,
                'scenario_id': sc.get('id'),
                'external_benchmark': 'cooperbench',
                'source_task_id': sc.get('source_task_id'),
                'repo': sc.get('repo'),
                'language': sc.get('language'),
                'started_at': now_iso(),
                'elapsed_ms': 0,
                'agent_count': len(sc.get('agents') or []),
                'write_intent_count': 0,
                'conflict_groups': 0,
                'semantic_conflict_count': 0,
                'auto_merge_groups': 0,
                'lock_conflict_count': 0,
                'authority_rebase_count': 0,
                'review_burden_count': 0,
                'commit': {'committed': 0, 'proposals': 0, 'conflicts': 0, 'total': 0},
                'context_tokens': 0,
                'unsafe_auto_commit_count': 0,
                'agentRuns': [],
                'merge_decisions': [],
                'task_success': False,
                'repetition': rep,
                'error_type': type(exc).__name__,
                'error': str(exc),
                'provider_error': provider_diag,
                'error_log_path': str(error_log_path),
            }
            append_jsonl(error_log_path, {'kind': 'cooperbench_cell_error_v1', 'timestamp': now_iso(), 'scenario_id': sc.get('id'), 'source_task_id': sc.get('source_task_id'), 'condition': cond, 'repetition': rep, 'error_type': type(exc).__name__, 'error': str(exc), 'provider_error': provider_diag})
            return error_row

    results: list[dict[str, Any]] = []
    if parallel_workers == 1:
        for cell in cells:
            row = run_cell(cell)
            results.append(row)
            append_jsonl(jsonl_path, row)
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = {pool.submit(run_cell, cell): cell for cell in cells}
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                append_jsonl(jsonl_path, row)
        results.sort(key=lambda r: (str(r.get('scenario_id')), str(r.get('condition')), int(r.get('repetition') or 0)))
    aggregated = _aggregate_llm(results)
    payload = {'kind': 'context_substrate_external_eval_results_v1', 'external_benchmark': 'cooperbench', 'generated_at': now_iso(), 'args': {'input': input, 'condition': condition, 'repetitions': repetitions, 'limit': limit, 'out': out, 'provider': provider, 'model': model, 'temperature': temperature, 'max_output_tokens': max_output_tokens, 'reasoning_effort': reasoning_effort, 'text_verbosity': text_verbosity, 'send_temperature': send_temperature, 'enable_target_grounding': enable_target_grounding, 'enable_target_candidates': enable_target_candidates, 'parallel_workers': parallel_workers, 'shuffle_cells': shuffle_cells, 'max_provider_retries': max_provider_retries, 'retry_backoff_base': retry_backoff_base, 'retry_backoff_max': retry_backoff_max, 'error_log': str(error_log_path), 'fail_fast': fail_fast, 'certificate_guidance': certificate_guidance}, 'task_count': len(tasks), 'scenario_count': len(scenarios), 'results': results, 'aggregated': aggregated}
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description='Run CooperBench-derived collaborative coding tasks through the WCCU context-store harness.')
    parser.add_argument('--input', required=True, help='CooperBench-style JSON/JSONL task file. See data/cooperbench_mini_sample.jsonl.')
    parser.add_argument('--condition', default=DEFAULT_CONDITIONS)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', default='results/cooperbench_substrate_results.json')
    parser.add_argument('--provider', default=os.environ.get('LLM_PROVIDER', 'openai'))
    parser.add_argument('--model', default=os.environ.get('LLM_MODEL', ''))
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--send-temperature', action='store_true', default=None)
    parser.add_argument('--no-target-grounding', dest='enable_target_grounding', action='store_false')
    parser.add_argument('--no-target-candidates', dest='enable_target_candidates', action='store_false')
    parser.add_argument('--parallel-workers', type=int, default=1)
    parser.add_argument('--shuffle-cells', action='store_true')
    parser.add_argument('--reasoning-effort', default=os.environ.get('LLM_REASONING_EFFORT', ''))
    parser.add_argument('--text-verbosity', default=os.environ.get('LLM_TEXT_VERBOSITY', ''))
    parser.add_argument('--max-output-tokens', type=int, default=1000)
    parser.add_argument('--timeout-seconds', type=int, default=90)
    parser.add_argument('--max-parse-retries', type=int, default=1)
    parser.add_argument('--max-provider-retries', type=int, default=int(os.environ.get('LLM_MAX_PROVIDER_RETRIES', 4)))
    parser.add_argument('--retry-backoff-base', type=float, default=float(os.environ.get('LLM_RETRY_BACKOFF_BASE', 1.0)))
    parser.add_argument('--retry-backoff-max', type=float, default=float(os.environ.get('LLM_RETRY_BACKOFF_MAX', 20.0)))
    parser.add_argument('--error-log', default=os.environ.get('LLM_ERROR_LOG_PATH', ''))
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--certificate-guidance', default=os.environ.get('LLM_CERTIFICATE_GUIDANCE', 'guided'), choices=['guided', 'unguided'])
    args = parser.parse_args(argv)
    try:
        payload = run_cooperbench_substrate(**vars(args))
        print(json.dumps({'ok': True, 'out': payload['args']['out'], 'task_count': payload['task_count'], 'result_count': len(payload['results']), 'aggregate_count': len(payload['aggregated']), 'failed_count': sum(1 for r in payload['results'] if r.get('failed')), 'error_log': payload['args'].get('error_log')}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc), 'error_type': type(exc).__name__, 'provider_error': exc.to_dict() if isinstance(exc, LlmProviderError) else {}}, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
