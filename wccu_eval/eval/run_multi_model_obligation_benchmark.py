from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from wccu_eval.env import load_dotenv
from wccu_eval.eval.run_experiment import REPO_ROOT
from wccu_eval.eval.run_llm_obligation_benchmark import run_llm_obligation_benchmark
from wccu_eval.utils import as_list, clean, ensure_dir, mean, now_iso, stable_hash, write_json


def _parse_model_specs(value: str, *, default_provider: str, default_model: str) -> list[dict[str, str]]:
    value = clean(value)
    if not value:
        value = clean(os.environ.get('LLM_MODELS') or default_model)
    specs: list[dict[str, str]] = []
    for raw in [x.strip() for x in value.split(',') if x.strip()]:
        if ':' in raw:
            provider, model = raw.split(':', 1)
        else:
            provider, model = default_provider, raw
        if clean(provider) and clean(model):
            specs.append({'provider': clean(provider), 'model': clean(model)})
    if not specs and default_model:
        specs.append({'provider': default_provider, 'model': default_model})
    return specs


def run_multi_model_obligation_benchmark(
    *,
    model_specs: str = '',
    families: str = 'freshness,commitment,authority,operation,derived_view,witness_gap,safe',
    limit_per_family: int = 3,
    condition: str = 'adaptive_wccu_execution_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only',
    repetitions: int = 1,
    out: str = 'results/multi_model_obligation_benchmark.json',
    temperature: float | None = None,
    max_output_tokens: int = 1200,
    timeout_seconds: int = 90,
    certificate_guidance: str = 'unguided',
    mock_llm: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    load_dotenv()
    default_provider = clean(os.environ.get('LLM_PROVIDER') or 'openai')
    default_model = clean(os.environ.get('LLM_MODEL') or '')
    if mock_llm and not clean(model_specs):
        # Offline smoke tests must be hermetic: --mock-llm should not inherit
        # LLM_PROVIDER/LLM_MODEL/LLM_MODELS from a developer's .env.  This keeps
        # mock runs from accidentally being reported with live provider names.
        specs = [{'provider': 'mock', 'model': 'mock-llm'}]
    else:
        specs = _parse_model_specs(model_specs, default_provider=default_provider, default_model=default_model)
    if not specs:
        raise RuntimeError('No models configured. Set LLM_MODEL or LLM_MODELS in .env, or pass --model-specs provider:model[,provider:model].')
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    ensure_dir(out_path.parent)
    per_model_payloads: list[dict[str, Any]] = []
    for spec in specs:
        safe_model = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in spec['model'])[:80]
        child_out = str(out_path.with_name(f'{out_path.stem}_{spec["provider"]}_{safe_model}_{stable_hash(spec["model"])[:6]}.json'))
        payload = run_llm_obligation_benchmark(
            families=families,
            limit_per_family=limit_per_family,
            condition=condition,
            repetitions=repetitions,
            out=child_out,
            provider=spec['provider'],
            model=spec['model'],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            certificate_guidance=certificate_guidance,
            mock_llm=mock_llm,
            fail_fast=fail_fast,
        )
        per_model_payloads.append({'provider': spec['provider'], 'model': spec['model'], 'out': child_out, 'payload': payload})

    rows: list[dict[str, Any]] = []
    for item in per_model_payloads:
        for row in as_list(item['payload'].get('results')):
            rows.append({**row, 'provider': item['provider'], 'model': item['model']})

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((clean(row.get('provider')), clean(row.get('model')), clean(row.get('condition'))), []).append(row)
    aggregated: list[dict[str, Any]] = []
    for (provider, model, cond), group in sorted(grouped.items()):
        issue_rows = [r for r in group if int(r.get('ground_truth_hold_required_count') or 0) > 0]
        safe_rows = [r for r in group if clean(r.get('llm_obligation_family')) == 'safe']
        aggregated.append({
            'provider': provider,
            'model': model,
            'condition': cond,
            'n': len(group),
            'generation_success_rate': mean([1 if r.get('llm_generation_success') else 0 for r in group]),
            'schema_valid_rate': mean([1 if r.get('llm_schema_valid') else 0 for r in group]),
            'proposal_rate': mean([1 if int(r.get('llm_proposal_count') or 0) > 0 else 0 for r in group]),
            'issue_accept_rate': mean([r.get('ground_truth_issue_accepted_count') or 0 for r in issue_rows]),
            'safe_automatic_progress': mean([1 if int(r.get('ground_truth_issue_accepted_count') or 0) == 0 and int(r.get('commit', {}).get('committed') or 0) > 0 else 0 for r in safe_rows]),
            'mean_review_burden': mean([r.get('review_burden_count') or 0 for r in group]),
            'mean_semantic_operation_laundering_count': mean([r.get('semantic_operation_laundering_count') or 0 for r in group]),
            'mean_authority_laundering_count': mean([r.get('authority_laundering_count') or 0 for r in group]),
            'provider_error_count': sum(1 for r in group if r.get('error_type') == 'LlmProviderError'),
        })
    payload = {
        'kind': 'multi_model_obligation_benchmark_results_v1',
        'generated_at': now_iso(),
        'args': {'model_specs': specs, 'families': families, 'limit_per_family': limit_per_family, 'condition': condition, 'repetitions': repetitions, 'out': str(out), 'certificate_guidance': certificate_guidance, 'mock_llm': mock_llm},
        'child_outputs': [{'provider': item['provider'], 'model': item['model'], 'out': item['out']} for item in per_model_payloads],
        'aggregated': aggregated,
    }
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the WCCU LLM-obligation benchmark across multiple .env-configured models.')
    parser.add_argument('--model-specs', default='', help='Comma-separated specs like openai:gpt-4.1-mini,gemini:gemini-1.5-flash. Defaults to LLM_MODELS or LLM_MODEL from .env.')
    parser.add_argument('--families', default='freshness,commitment,authority,operation,derived_view,witness_gap,safe')
    parser.add_argument('--limit-per-family', type=int, default=3)
    parser.add_argument('--condition', default='adaptive_wccu_execution_trace,adaptive_wccu_model_certificate,adaptive_readset_occ,adaptive_policy,uniform_review_gated,uniform_append_only')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--out', default='results/multi_model_obligation_benchmark.json')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-output-tokens', type=int, default=1200)
    parser.add_argument('--timeout-seconds', type=int, default=90)
    parser.add_argument('--certificate-guidance', default='unguided', choices=['guided', 'unguided', 'minimal', 'no_hints'])
    parser.add_argument('--mock-llm', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args(argv)
    payload = run_multi_model_obligation_benchmark(model_specs=args.model_specs, families=args.families, limit_per_family=args.limit_per_family, condition=args.condition, repetitions=args.repetitions, out=args.out, temperature=args.temperature, max_output_tokens=args.max_output_tokens, timeout_seconds=args.timeout_seconds, certificate_guidance=args.certificate_guidance, mock_llm=args.mock_llm, fail_fast=args.fail_fast)
    print(json.dumps({'ok': True, 'models': len(payload['args']['model_specs']), 'rows': len(payload['aggregated']), 'out': args.out}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
