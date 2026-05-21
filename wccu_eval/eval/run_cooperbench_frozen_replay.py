from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from wccu_eval.agents.frozen_agent import build_frozen_index, load_frozen_agent_bundle, make_bundle_from_generation_results, run_frozen_agent
from wccu_eval.env import load_dotenv
from wccu_eval.eval.run_experiment import REPO_ROOT, selected_conditions
from wccu_eval.eval.run_llm_experiment import _aggregate_llm
from wccu_eval.eval.run_cooperbench_substrate import DEFAULT_CONDITIONS, run_cooperbench_substrate
from wccu_eval.external.cooperbench_adapter import cooperbench_tasks_to_scenarios, load_cooperbench_tasks
from wccu_eval.scheduler.context_concurrency_policy import PolicyMode
from wccu_eval.scheduler.team_dag_executor import execute_context_policy_parallel
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import append_jsonl, clean, ensure_dir, now_iso, remove_dir, stable_hash, write_json

GENERATION_CONDITION = 'adaptive_wccu_execution_trace'

CONDITION_TO_POLICY_MODE: dict[str, str] = {
    'adaptive_policy': PolicyMode.ADAPTIVE,
    'adaptive_readset_occ': PolicyMode.ADAPTIVE_READSET_OCC,
    'adaptive_wccu': PolicyMode.ADAPTIVE_WCCU,
    'adaptive_wccu_model_certificate': PolicyMode.ADAPTIVE_WCCU_MODEL_CERTIFICATE,
    'adaptive_wccu_oracle_dependency': PolicyMode.ADAPTIVE_WCCU_ORACLE_DEPENDENCY,
    'adaptive_wccu_projection_trace': PolicyMode.ADAPTIVE_WCCU_PROJECTION_TRACE,
    'adaptive_wccu_no_read_validation': PolicyMode.ADAPTIVE_WCCU_NO_READ_VALIDATION,
    'adaptive_wccu_unguided_certificate': PolicyMode.ADAPTIVE_WCCU_UNGUIDED_CERTIFICATE,
    'adaptive_wccu_execution_trace': PolicyMode.ADAPTIVE_WCCU_EXECUTION_TRACE,
    'uniform_snapshot_occ': PolicyMode.UNIFORM_SNAPSHOT_OCC,
    'uniform_pessimistic_lock': PolicyMode.UNIFORM_PESSIMISTIC_LOCK,
    'uniform_review_gated': PolicyMode.UNIFORM_REVIEW_GATED,
    'uniform_append_only': PolicyMode.UNIFORM_APPEND_ONLY,
    'adaptive_no_review_gate': PolicyMode.ADAPTIVE_NO_REVIEW_GATE,
    'adaptive_no_authority_rebase': PolicyMode.ADAPTIVE_NO_AUTHORITY_REBASE,
    'adaptive_no_append_only': PolicyMode.ADAPTIVE_NO_APPEND_ONLY,
    'adaptive_no_workspace_lock': PolicyMode.ADAPTIVE_NO_WORKSPACE_LOCK,
    'adaptive_no_semantic_conflict_detection': PolicyMode.ADAPTIVE_NO_SEMANTIC_CONFLICT_DETECTION,
}


def _condition_ids(value: str) -> list[str]:
    value = clean(value or DEFAULT_CONDITIONS)
    if value == 'all':
        return [c.strip() for c in DEFAULT_CONDITIONS.split(',')]
    return selected_conditions(value)


def extract_frozen_bundle(*, generation_results: str, out: str, bundle_id: str = '', generation_condition: str = '') -> dict[str, Any]:
    path = Path(generation_results)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    bundle = make_bundle_from_generation_results(payload, out_path=out_path, bundle_id=bundle_id, generation_condition=generation_condition)
    return bundle


def run_cooperbench_frozen_generation(*, input: str, out: str, bundle_out: str, provider: str = '', model: str = '', temperature: float | None = None, max_output_tokens: int = 1800, timeout_seconds: int = 180, max_parse_retries: int = 1, reasoning_effort: str = '', text_verbosity: str = '', send_temperature: bool | None = None, parallel_workers: int = 1, shuffle_cells: bool = False, max_provider_retries: int = 8, retry_backoff_base: float = 1.0, retry_backoff_max: float = 30.0, error_log: str = '', fail_fast: bool = False, certificate_guidance: str = 'guided', agent_model_specs: str = '', limit: int = 0, repetitions: int = 1, generation_condition: str = GENERATION_CONDITION, bundle_id: str = '') -> dict[str, Any]:
    payload = run_cooperbench_substrate(
        input=input,
        condition=generation_condition,
        repetitions=repetitions,
        limit=limit,
        out=out,
        provider=provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        max_parse_retries=max_parse_retries,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
        send_temperature=send_temperature,
        parallel_workers=parallel_workers,
        shuffle_cells=shuffle_cells,
        max_provider_retries=max_provider_retries,
        retry_backoff_base=retry_backoff_base,
        retry_backoff_max=retry_backoff_max,
        error_log=error_log,
        fail_fast=fail_fast,
        certificate_guidance=certificate_guidance,
        agent_model_specs=agent_model_specs,
    )
    bundle = make_bundle_from_generation_results(payload, out_path=(REPO_ROOT / bundle_out if not Path(bundle_out).is_absolute() else Path(bundle_out)), bundle_id=bundle_id)
    payload['frozen_bundle'] = {'path': bundle_out, 'bundle_id': bundle.get('bundle_id'), 'agent_output_count': bundle.get('agent_output_count'), 'scenario_count': bundle.get('scenario_count')}
    write_json((REPO_ROOT / out if not Path(out).is_absolute() else Path(out)), payload)
    return {'generation': payload, 'bundle': bundle}


def run_cooperbench_frozen_replay(*, input: str, frozen_bundle: str, condition: str = DEFAULT_CONDITIONS, repetitions: int = 1, limit: int = 0, out: str = 'results/cooperbench_frozen_replay_results.json', parallel_workers: int = 1, shuffle_cells: bool = False, enable_target_grounding: bool = True, fail_fast: bool = False) -> dict[str, Any]:
    bundle_path = Path(frozen_bundle)
    if not bundle_path.is_absolute():
        bundle_path = REPO_ROOT / bundle_path
    bundle = load_frozen_agent_bundle(bundle_path)
    frozen_index = build_frozen_index(bundle)
    tasks = load_cooperbench_tasks(input)
    if limit and limit > 0:
        tasks = tasks[:limit]
    scenarios = cooperbench_tasks_to_scenarios(tasks)
    condition_ids = _condition_ids(condition)
    repetitions = max(1, int(repetitions))
    parallel_workers = max(1, int(parallel_workers or 1))
    for cond in condition_ids:
        if cond not in CONDITION_TO_POLICY_MODE:
            raise KeyError(f'Unknown frozen-replay condition: {cond}')
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    jsonl_path = out_path.with_suffix('.jsonl')
    ensure_dir(out_path.parent)
    if jsonl_path.exists():
        jsonl_path.unlink()
    tmp_root = REPO_ROOT / 'runs' / f"cooperbench_frozen_tmp_{stable_hash(f'{out}:{os.getpid()}:{now_iso()}')}"
    remove_dir(tmp_root)

    cells: list[tuple[dict[str, Any], str, int]] = []
    for sc in scenarios:
        for cond in condition_ids:
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
            row = execute_context_policy_parallel(
                scenario=sc,
                root_dir=root_dir,
                run_dir=run_dir,
                repetition=rep,
                policy_mode=CONDITION_TO_POLICY_MODE[cond],
                condition=cond,
                agent_runner=run_frozen_agent,
                agent_runner_config={
                    'frozen_index': frozen_index,
                    'frozen_repetition': rep,
                    'frozen_bundle_id': bundle.get('bundle_id'),
                    'enable_target_grounding': enable_target_grounding,
                    'enable_target_candidates': True,
                },
            )
            return {**row, 'failed': False, 'repetition': rep, 'external_benchmark': 'cooperbench', 'source_task_id': sc.get('source_task_id'), 'repo': sc.get('repo'), 'language': sc.get('language'), 'frozen_replay': {
                'enabled': True,
                'bundle_path': str(bundle_path),
                'bundle_id': bundle.get('bundle_id'),
                'generation_condition': bundle.get('generation_condition'),
                'generation_args': bundle.get('generation_args') or {},
                'provider_api_called_in_replay': False,
            }}
        except Exception as exc:
            if fail_fast:
                raise
            return {
                'kind': 'parallel_execution_error_v1',
                'failed': True,
                'condition': cond,
                'policy_mode': cond,
                'scenario_id': sc.get('id'),
                'external_benchmark': 'cooperbench',
                'source_task_id': sc.get('source_task_id'),
                'repo': sc.get('repo'),
                'language': sc.get('language'),
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
                'frozen_replay': {'enabled': True, 'bundle_path': str(bundle_path), 'bundle_id': bundle.get('bundle_id'), 'provider_api_called_in_replay': False},
            }

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
    payload = {
        'kind': 'context_substrate_cooperbench_frozen_replay_results_v1',
        'external_benchmark': 'cooperbench',
        'generated_at': now_iso(),
        'args': {
            'input': input,
            'frozen_bundle': str(bundle_path),
            'condition': condition,
            'repetitions': repetitions,
            'limit': limit,
            'out': out,
            'parallel_workers': parallel_workers,
            'shuffle_cells': shuffle_cells,
            'enable_target_grounding': enable_target_grounding,
        },
        'frozen_replay': {
            'enabled': True,
            'bundle_id': bundle.get('bundle_id'),
            'bundle_path': str(bundle_path),
            'generation_condition': bundle.get('generation_condition'),
            'agent_output_count': bundle.get('agent_output_count'),
            'provider_api_called_in_replay': False,
        },
        'task_count': len(tasks),
        'scenario_count': len(scenarios),
        'results': results,
        'aggregated': aggregated,
    }
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description='Generate and replay frozen live-LLM CooperBench-derived WCCU outputs.')
    sub = parser.add_subparsers(dest='command', required=True)

    gen = sub.add_parser('generate', help='Call the real LLM once per scenario/agent and write a frozen output bundle.')
    gen.add_argument('--input', required=True)
    gen.add_argument('--out', required=True)
    gen.add_argument('--bundle-out', required=True)
    gen.add_argument('--provider', default=os.environ.get('LLM_PROVIDER', 'openai'))
    gen.add_argument('--model', default=os.environ.get('LLM_MODEL', ''))
    gen.add_argument('--agent-model-specs', default=os.environ.get('WCCU_AGENT_MODEL_SPECS', '') or os.environ.get('LLM_AGENT_MODEL_SPECS', ''), help='Per-agent provider/model routing, e.g. coop_agent_a=openai:gpt-5.4-nano,coop_agent_b=gemini:gemini-3.1-flash-lite')
    gen.add_argument('--temperature', type=float, default=None)
    gen.add_argument('--send-temperature', action='store_true', default=None)
    gen.add_argument('--max-output-tokens', type=int, default=int(os.environ.get('WCCU_COOPER_MAX_OUTPUT_TOKENS', 1800)))
    gen.add_argument('--timeout-seconds', type=int, default=int(os.environ.get('WCCU_COOPER_TIMEOUT_SECONDS', 180)))
    gen.add_argument('--max-parse-retries', type=int, default=1)
    gen.add_argument('--max-provider-retries', type=int, default=int(os.environ.get('WCCU_COOPER_MAX_PROVIDER_RETRIES', 8)))
    gen.add_argument('--retry-backoff-base', type=float, default=float(os.environ.get('WCCU_COOPER_RETRY_BACKOFF_BASE', 1.0)))
    gen.add_argument('--retry-backoff-max', type=float, default=float(os.environ.get('WCCU_COOPER_RETRY_BACKOFF_MAX', 30.0)))
    gen.add_argument('--parallel-workers', type=int, default=int(os.environ.get('WCCU_COOPER_PARALLEL_WORKERS', 1)))
    gen.add_argument('--shuffle-cells', action='store_true')
    gen.add_argument('--reasoning-effort', default=os.environ.get('LLM_REASONING_EFFORT', ''))
    gen.add_argument('--text-verbosity', default=os.environ.get('LLM_TEXT_VERBOSITY', ''))
    gen.add_argument('--error-log', default=os.environ.get('LLM_ERROR_LOG_PATH', ''))
    gen.add_argument('--fail-fast', action='store_true')
    gen.add_argument('--certificate-guidance', default=os.environ.get('LLM_CERTIFICATE_GUIDANCE', 'guided'), choices=['guided', 'unguided'])
    gen.add_argument('--limit', type=int, default=0)
    gen.add_argument('--repetitions', type=int, default=1)
    gen.add_argument('--generation-condition', default=GENERATION_CONDITION)
    gen.add_argument('--bundle-id', default='')

    ext = sub.add_parser('extract', help='Extract a frozen bundle from an existing one-condition generation result JSON.')
    ext.add_argument('--generation-results', required=True)
    ext.add_argument('--out', required=True)
    ext.add_argument('--bundle-id', default='')
    ext.add_argument('--generation-condition', default='', help='Select one condition from a previous multi-condition live run, e.g. adaptive_wccu_execution_trace.')

    rep = sub.add_parser('replay', help='Replay a frozen bundle across commit policies without provider calls.')
    rep.add_argument('--input', required=True)
    rep.add_argument('--frozen-bundle', required=True)
    rep.add_argument('--condition', default=DEFAULT_CONDITIONS)
    rep.add_argument('--repetitions', type=int, default=1)
    rep.add_argument('--limit', type=int, default=0)
    rep.add_argument('--out', required=True)
    rep.add_argument('--parallel-workers', type=int, default=1)
    rep.add_argument('--shuffle-cells', action='store_true')
    rep.add_argument('--no-target-grounding', dest='enable_target_grounding', action='store_false')
    rep.add_argument('--fail-fast', action='store_true')

    args = parser.parse_args(argv)
    if args.command == 'generate':
        payload = run_cooperbench_frozen_generation(**vars(args) | {'command': None}) if False else run_cooperbench_frozen_generation(
            input=args.input, out=args.out, bundle_out=args.bundle_out, provider=args.provider, model=args.model,
            temperature=args.temperature, max_output_tokens=args.max_output_tokens, timeout_seconds=args.timeout_seconds,
            max_parse_retries=args.max_parse_retries, reasoning_effort=args.reasoning_effort, text_verbosity=args.text_verbosity,
            send_temperature=args.send_temperature, parallel_workers=args.parallel_workers, shuffle_cells=args.shuffle_cells,
            max_provider_retries=args.max_provider_retries, retry_backoff_base=args.retry_backoff_base,
            retry_backoff_max=args.retry_backoff_max, error_log=args.error_log, fail_fast=args.fail_fast,
            certificate_guidance=args.certificate_guidance, limit=args.limit, repetitions=args.repetitions,
            generation_condition=args.generation_condition, bundle_id=args.bundle_id, agent_model_specs=args.agent_model_specs)
        print(json.dumps({'ok': True, 'generation_out': args.out, 'bundle_out': args.bundle_out, 'scenario_count': payload['bundle'].get('scenario_count'), 'agent_output_count': payload['bundle'].get('agent_output_count'), 'failed_generation_count': payload['bundle'].get('failed_generation_count')}, indent=2))
        return 0
    if args.command == 'extract':
        bundle = extract_frozen_bundle(generation_results=args.generation_results, out=args.out, bundle_id=args.bundle_id, generation_condition=args.generation_condition)
        print(json.dumps({'ok': True, 'bundle_out': args.out, 'scenario_count': bundle.get('scenario_count'), 'agent_output_count': bundle.get('agent_output_count'), 'failed_generation_count': bundle.get('failed_generation_count')}, indent=2))
        return 0
    if args.command == 'replay':
        payload = run_cooperbench_frozen_replay(input=args.input, frozen_bundle=args.frozen_bundle, condition=args.condition, repetitions=args.repetitions, limit=args.limit, out=args.out, parallel_workers=args.parallel_workers, shuffle_cells=args.shuffle_cells, enable_target_grounding=args.enable_target_grounding, fail_fast=args.fail_fast)
        print(json.dumps({'ok': True, 'out': args.out, 'result_count': len(payload['results']), 'aggregate_count': len(payload['aggregated']), 'failed_count': sum(1 for r in payload['results'] if r.get('failed')), 'provider_api_called_in_replay': False}, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == '__main__':
    raise SystemExit(main())
