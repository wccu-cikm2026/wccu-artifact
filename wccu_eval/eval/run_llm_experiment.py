from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from wccu_eval.agents.llm_agent import LlmProviderError, run_llm_agent
from wccu_eval.env import load_dotenv
from wccu_eval.eval.run_experiment import REPO_ROOT, aggregate, selected_conditions, selected_scenarios
from wccu_eval.eval.scenarios import get_scenario
from wccu_eval.scheduler.context_concurrency_policy import PolicyMode
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel, execute_substrate_serial
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import append_jsonl, clean, ensure_dir, mean, now_iso, remove_dir, stable_hash, write_json


def _policy_runner(policy_mode: str, condition: str, llm_config: dict[str, Any], overrides: dict[str, Any] | None = None) -> Callable[..., dict[str, Any]]:
    merged_cfg = {**llm_config, **(overrides or {})}
    def runner(**kwargs: Any) -> dict[str, Any]:
        return execute_context_policy_parallel(**kwargs, policy_mode=policy_mode, condition=condition, agent_runner=run_llm_agent, agent_runner_config=merged_cfg)
    return runner


def _serial_policy_runner(policy_mode: str, condition: str, llm_config: dict[str, Any], overrides: dict[str, Any] | None = None) -> Callable[..., dict[str, Any]]:
    merged_cfg = {**llm_config, **(overrides or {})}
    def runner(**kwargs: Any) -> dict[str, Any]:
        return execute_substrate_serial(**kwargs, policy_mode=policy_mode, condition=condition, agent_runner=run_llm_agent, agent_runner_config=merged_cfg)
    return runner


def build_conditions(llm_config: dict[str, Any]) -> dict[str, Callable[..., dict[str, Any]]]:
    """Build LLM-backed experiment conditions.

    Public experiment conditions use WCCU names.
    """
    conditions: dict[str, Callable[..., dict[str, Any]]] = {
        'adaptive_policy': _policy_runner(PolicyMode.ADAPTIVE, 'adaptive_policy', llm_config),
        'adaptive_readset_occ': _policy_runner(PolicyMode.ADAPTIVE_READSET_OCC, 'adaptive_readset_occ', llm_config),
        'adaptive_wccu': _policy_runner(PolicyMode.ADAPTIVE_WCCU, 'adaptive_wccu', llm_config),
        'adaptive_wccu_model_certificate': _policy_runner(PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE, 'adaptive_wccu_model_certificate', llm_config),
        'adaptive_wccu_oracle_dependency': _policy_runner(PolicyMode.ADAPTIVE_WCCU_ORACLE_DEPENDENCY, 'adaptive_wccu_oracle_dependency', llm_config),
        'adaptive_wccu_projection_trace': _policy_runner(PolicyMode.ADAPTIVE_WCCU_PROJECTION_TRACE, 'adaptive_wccu_projection_trace', llm_config),
        'adaptive_wccu_no_read_validation': _policy_runner(PolicyMode.ADAPTIVE_WCCU_NO_READ_VALIDATION, 'adaptive_wccu_no_read_validation', llm_config),
        'adaptive_wccu_unguided_certificate': _policy_runner(PolicyMode.ADAPTIVE_WCCU_UNGUIDED_CERTIFICATE, 'adaptive_wccu_unguided_certificate', llm_config, {'certificate_guidance': 'unguided'}),
        'adaptive_wccu_execution_trace': _policy_runner(PolicyMode.ADAPTIVE_WCCU_EXECUTION_TRACE, 'adaptive_wccu_execution_trace', llm_config),
        'serial_adaptive_policy': _serial_policy_runner(PolicyMode.ADAPTIVE, 'serial_adaptive_policy', llm_config),
        'serial_adaptive_wccu_execution_trace': _serial_policy_runner(PolicyMode.ADAPTIVE_WCCU_EXECUTION_TRACE, 'serial_adaptive_wccu_execution_trace', llm_config),
        'serial_adaptive_wccu_projection_trace': _serial_policy_runner(PolicyMode.ADAPTIVE_WCCU_PROJECTION_TRACE, 'serial_adaptive_wccu_projection_trace', llm_config),
        'substrate_parallel': _policy_runner(PolicyMode.ADAPTIVE, 'substrate_parallel', llm_config),
        'uniform_snapshot_occ': _policy_runner(PolicyMode.UNIFORM_SNAPSHOT_OCC, 'uniform_snapshot_occ', llm_config),
        'uniform_pessimistic_lock': _policy_runner(PolicyMode.UNIFORM_PESSIMISTIC_LOCK, 'uniform_pessimistic_lock', llm_config),
        'uniform_review_gated': _policy_runner(PolicyMode.UNIFORM_REVIEW_GATED, 'uniform_review_gated', llm_config),
        'uniform_append_only': _policy_runner(PolicyMode.UNIFORM_APPEND_ONLY, 'uniform_append_only', llm_config),
        'adaptive_no_review_gate': _policy_runner(PolicyMode.ADAPTIVE_NO_REVIEW_GATE, 'adaptive_no_review_gate', llm_config),
        'adaptive_no_authority_rebase': _policy_runner(PolicyMode.ADAPTIVE_NO_AUTHORITY_REBASE, 'adaptive_no_authority_rebase', llm_config),
        'adaptive_no_append_only': _policy_runner(PolicyMode.ADAPTIVE_NO_APPEND_ONLY, 'adaptive_no_append_only', llm_config),
        'adaptive_no_workspace_lock': _policy_runner(PolicyMode.ADAPTIVE_NO_WORKSPACE_LOCK, 'adaptive_no_workspace_lock', llm_config),
        'adaptive_no_semantic_conflict_detection': _policy_runner(PolicyMode.ADAPTIVE_NO_SEMANTIC_CONFLICT_DETECTION, 'adaptive_no_semantic_conflict_detection', llm_config),
        # Target-grounding ablation conditions. These all use the adaptive
        # concurrency policy, but differ in whether the LLM sees stable target
        # candidates and whether the runtime applies deterministic grounding.
        'adaptive_no_candidates_no_grounding': _policy_runner(PolicyMode.ADAPTIVE, 'adaptive_no_candidates_no_grounding', llm_config, {'enable_target_candidates': False, 'enable_target_grounding': False}),
        'adaptive_no_candidates_with_grounding': _policy_runner(PolicyMode.ADAPTIVE, 'adaptive_no_candidates_with_grounding', llm_config, {'enable_target_candidates': False, 'enable_target_grounding': True}),
        'adaptive_candidates_no_grounding': _policy_runner(PolicyMode.ADAPTIVE, 'adaptive_candidates_no_grounding', llm_config, {'enable_target_candidates': True, 'enable_target_grounding': False}),
        'adaptive_candidates_with_grounding': _policy_runner(PolicyMode.ADAPTIVE, 'adaptive_candidates_with_grounding', llm_config, {'enable_target_candidates': True, 'enable_target_grounding': True}),
    }
    return conditions


def _aggregate_llm(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = aggregate(results)
    # Attach estimated LLM prompt/output token means and failure counts for each group.
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        by_key.setdefault((row.get('scenario_id'), row.get('condition')), []).append(row)
    for row in rows:
        group = by_key.get((row['scenario_id'], row['condition']), [])
        row['mean_llm_prompt_tokens_est'] = mean([sum(int(a.get('llm', {}).get('prompt_tokens_est') or 0) for a in r.get('agentRuns', [])) for r in group])
        row['mean_llm_output_tokens_est'] = mean([sum(int(a.get('llm', {}).get('output_tokens_est') or 0) for a in r.get('agentRuns', [])) for r in group])
        row['failure_count'] = sum(1 for r in group if r.get('failed'))
        row['provider_error_count'] = sum(1 for r in group if r.get('error_type') == 'LlmProviderError')
    return rows


def run_llm_experiment(*, scenario: str = 'all', condition: str = 'all', repetitions: int = 1, out: str = 'results/llm_results.json', provider: str = '', model: str = '', temperature: float | None = None, max_output_tokens: int = 1000, timeout_seconds: int = 90, max_parse_retries: int = 1, reasoning_effort: str = '', text_verbosity: str = '', send_temperature: bool | None = None, enable_target_grounding: bool = True, enable_target_candidates: bool = True, parallel_workers: int = 1, shuffle_cells: bool = False, max_provider_retries: int = 4, retry_backoff_base: float = 1.0, retry_backoff_max: float = 20.0, error_log: str = '', fail_fast: bool = False, certificate_guidance: str = 'guided') -> dict[str, Any]:
    load_dotenv()
    provider = provider or os.environ.get('LLM_PROVIDER', 'openai')
    model = model or os.environ.get('LLM_MODEL', '')
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
        'certificate_guidance': certificate_guidance,
    }
    scenario_ids = selected_scenarios(scenario)
    condition_ids = selected_conditions(condition)
    repetitions = max(1, int(repetitions))
    parallel_workers = max(1, int(parallel_workers or 1))
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    jsonl_path = out_path.with_suffix('.jsonl')
    error_log_path = (REPO_ROOT / error_log).resolve() if error_log and not Path(error_log).is_absolute() else (Path(error_log) if error_log else out_path.with_suffix('.errors.jsonl'))
    llm_config['error_log_path'] = str(error_log_path)
    conditions = build_conditions(llm_config)
    ensure_dir(out_path.parent)
    if jsonl_path.exists():
        jsonl_path.unlink()
    if error_log_path.exists():
        error_log_path.unlink()
    tmp_root = REPO_ROOT / 'runs' / f"llm_eval_tmp_{stable_hash(f'{out}:{os.getpid()}:{now_iso()}')}"
    remove_dir(tmp_root)
    results: list[dict[str, Any]] = []

    cells: list[tuple[str, str, int]] = []
    for scenario_id in scenario_ids:
        for condition_id in condition_ids:
            if condition_id not in conditions:
                raise KeyError(f'Unknown or unsupported LLM condition: {condition_id}')
            for rep in range(repetitions):
                cells.append((scenario_id, condition_id, rep))
    if shuffle_cells:
        # Deterministic shuffle to reduce provider load/order bias while keeping reproducibility.
        cells = sorted(cells, key=lambda c: stable_hash(f'{c[0]}:{c[1]}:{c[2]}'))

    def run_cell(cell: tuple[str, str, int]) -> dict[str, Any]:
        scenario_id, condition_id, rep = cell
        runner = conditions[condition_id]
        sc = get_scenario(scenario_id)
        run_dir = tmp_root / f'{scenario_id}_{condition_id}_{rep}'
        root_dir = run_dir / 'context_substrate'
        remove_dir(run_dir)
        seed_context(root_dir, sc.get('seed', {}))
        cell_started = now_iso()
        try:
            row = runner(scenario=sc, root_dir=root_dir, run_dir=run_dir, repetition=rep)
        except Exception as exc:
            if fail_fast:
                raise
            error_type = type(exc).__name__
            provider_diag = exc.to_dict() if isinstance(exc, LlmProviderError) else {}
            error_row = {
                'kind': 'parallel_execution_error_v1',
                'failed': True,
                'condition': condition_id,
                'policy_mode': condition_id,
                'scenario_id': scenario_id,
                'started_at': cell_started,
                'completed_at': now_iso(),
                'elapsed_ms': 0,
                'base_snapshot_id': '',
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
                'stale_write_blocked_count': 0,
                'agentRuns': [],
                'merge_decisions': [],
                'task_success': False,
                'repetition': rep,
                'error_type': error_type,
                'error': str(exc),
                'provider_error': provider_diag,
                'error_log_path': str(error_log_path),
            }
            append_jsonl(error_log_path, {'kind': 'llm_experiment_cell_error_v1', 'timestamp': now_iso(), 'scenario_id': scenario_id, 'condition': condition_id, 'repetition': rep, 'error_type': error_type, 'error': str(exc), 'provider_error': provider_diag})
            return error_row
        row_cfg = {}
        for ar in row.get('agentRuns', []):
            if ar.get('llm', {}).get('request_options'):
                row_cfg = ar['llm']['request_options']
                break
        return {**row, 'failed': False, 'repetition': rep, 'llm_experiment': {
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

    if parallel_workers == 1:
        for cell in cells:
            enriched = run_cell(cell)
            results.append(enriched)
            append_jsonl(jsonl_path, enriched)
    else:
        # Parallelize independent experiment cells only. A single cell still uses
        # the scenario-defined internal multi-agent execution behavior.
        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            future_to_cell = {pool.submit(run_cell, cell): cell for cell in cells}
            for future in as_completed(future_to_cell):
                enriched = future.result()
                results.append(enriched)
                append_jsonl(jsonl_path, enriched)
        results.sort(key=lambda r: (str(r.get('scenario_id')), str(r.get('condition')), int(r.get('repetition') or 0)))
    aggregated = _aggregate_llm(results)
    payload = {'kind': 'context_substrate_llm_eval_results_v1', 'generated_at': now_iso(), 'args': {'scenario': scenario, 'condition': condition, 'repetitions': repetitions, 'out': out, 'provider': provider, 'model': model, 'temperature': temperature, 'max_output_tokens': max_output_tokens, 'reasoning_effort': reasoning_effort, 'text_verbosity': text_verbosity, 'send_temperature': send_temperature, 'enable_target_grounding': enable_target_grounding, 'enable_target_candidates': enable_target_candidates, 'parallel_workers': parallel_workers, 'shuffle_cells': shuffle_cells, 'max_provider_retries': max_provider_retries, 'retry_backoff_base': retry_backoff_base, 'retry_backoff_max': retry_backoff_max, 'error_log': str(error_log_path), 'fail_fast': fail_fast, 'certificate_guidance': certificate_guidance}, 'results': results, 'aggregated': aggregated}
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description='Run LLM-backed context substrate experiments.')
    parser.add_argument('--scenario', default='all')
    parser.add_argument('--condition', default='all')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--out', default='results/llm_results.json')
    parser.add_argument('--provider', default=os.environ.get('LLM_PROVIDER', 'openai'))
    parser.add_argument('--model', default=os.environ.get('LLM_MODEL', ''))
    parser.add_argument('--temperature', type=float, default=None, help='Optional sampling temperature. Omitted by default for OpenAI Responses reasoning models.')
    parser.add_argument('--send-temperature', action='store_true', default=None, help='Force sending temperature to the provider.')
    parser.add_argument('--no-target-grounding', dest='enable_target_grounding', action='store_false', help='Disable deterministic target grounding before merge.')
    parser.add_argument('--no-target-candidates', dest='enable_target_candidates', action='store_false', help='Do not show stable target candidates in the LLM prompt.')
    parser.add_argument('--parallel-workers', type=int, default=1, help='Run independent scenario/condition/repetition cells concurrently. Does not change within-run agent scheduling.')
    parser.add_argument('--shuffle-cells', action='store_true', help='Deterministically shuffle independent cells to reduce ordering bias.')
    parser.add_argument('--reasoning-effort', default=os.environ.get('LLM_REASONING_EFFORT', ''))
    parser.add_argument('--text-verbosity', default=os.environ.get('LLM_TEXT_VERBOSITY', ''))
    parser.add_argument('--max-output-tokens', type=int, default=1000)
    parser.add_argument('--timeout-seconds', type=int, default=90)
    parser.add_argument('--max-parse-retries', type=int, default=1)
    parser.add_argument('--max-provider-retries', type=int, default=int(os.environ.get('LLM_MAX_PROVIDER_RETRIES', 4)), help='Retry retryable provider/gateway failures such as 429 and 5xx/520 this many times per request.')
    parser.add_argument('--retry-backoff-base', type=float, default=float(os.environ.get('LLM_RETRY_BACKOFF_BASE', 1.0)))
    parser.add_argument('--retry-backoff-max', type=float, default=float(os.environ.get('LLM_RETRY_BACKOFF_MAX', 20.0)))
    parser.add_argument('--error-log', default=os.environ.get('LLM_ERROR_LOG_PATH', ''), help='JSONL path for provider/cell errors. Defaults to <out>.errors.jsonl.')
    parser.add_argument('--fail-fast', action='store_true', help='Abort the experiment on the first failed cell instead of writing an error row and continuing.')
    parser.add_argument('--certificate-guidance', default=os.environ.get('LLM_CERTIFICATE_GUIDANCE', 'guided'), choices=['guided', 'unguided'], help='How much certificate-specific guidance to include in the LLM prompt. Unguided keeps the certificate schema but removes oracle-like hints from agent tasks when available.')
    args = parser.parse_args(argv)
    try:
        payload = run_llm_experiment(**vars(args))
        print(json.dumps({'ok': True, 'out': payload['args']['out'], 'error_log': payload['args'].get('error_log'), 'result_count': len(payload['results']), 'aggregate_count': len(payload['aggregated']), 'failed_count': sum(1 for r in payload['results'] if r.get('failed'))}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc), 'error_type': type(exc).__name__, 'provider_error': exc.to_dict() if isinstance(exc, LlmProviderError) else {}, 'validation_errors': getattr(exc, 'validation_errors', []), 'raw_llm_text': getattr(exc, 'raw_llm_text', '')}, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
