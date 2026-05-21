from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, stable_hash


def load_frozen_agent_bundle(path: str | Path) -> dict[str, Any]:
    """Load a frozen LLM generation bundle.

    The bundle records live LLM-generated agent outputs once and then lets replay
    experiments reuse those exact outputs across many commit policies.  Runtime
    witnesses and projection traces are reattached during replay by the executor;
    the cached output only replaces the expensive/non-deterministic model call.
    """
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    if payload.get('kind') != 'wccu_frozen_agent_bundle_v1':
        raise ValueError(f'Unsupported frozen bundle kind: {payload.get("kind")!r}')
    return payload


def build_frozen_index(bundle: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in as_list(bundle.get('agent_outputs')):
        r = as_dict(row)
        scenario_id = clean(r.get('scenario_id'))
        repetition = int(r.get('repetition') or 0)
        agent_id = clean(r.get('agent_id'))
        if not scenario_id or not agent_id:
            continue
        index[(scenario_id, repetition, agent_id)] = r
    return index


def _strip_runtime_fields(agent_result: dict[str, Any]) -> dict[str, Any]:
    """Keep LLM-authored content but drop condition-specific runtime traces.

    The replay executor will compile a fresh projection and attach projection /
    execution witnesses under the current policy.  This avoids accidentally
    treating the generation condition's projection trace as replay evidence.
    """
    result = copy.deepcopy(agent_result)
    for key in [
        'projection_trace',
        'execution_witness',
        'read_witness',
        'witness_compile_metadata',
        'commit',
        'merge_decisions',
    ]:
        result.pop(key, None)
    return result


def run_frozen_agent(*, agent: dict[str, Any], projection: dict[str, Any], scenario: dict[str, Any], llm_config: dict[str, Any] | None = None, agent_runner_config: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    """Agent runner that replays a cached live-LLM output.

    Required config keys:
      - frozen_index: dict returned by build_frozen_index, OR
      - frozen_bundle_path: path to bundle JSON
      - frozen_repetition: repetition integer, default 0

    The returned shape mirrors run_llm_agent, but its `llm` metadata explicitly
    says that no provider API call occurred in this replay cell.
    """
    cfg = {**as_dict(llm_config), **as_dict(agent_runner_config)}
    index = cfg.get('frozen_index')
    if not isinstance(index, dict):
        bundle_path = clean(cfg.get('frozen_bundle_path'))
        if not bundle_path:
            raise KeyError('run_frozen_agent requires frozen_index or frozen_bundle_path')
        index = build_frozen_index(load_frozen_agent_bundle(bundle_path))
    scenario_id = clean(scenario.get('id'))
    repetition = int(cfg.get('frozen_repetition') or 0)
    agent_id = clean(agent.get('id') or agent.get('role') or 'agent')
    record = index.get((scenario_id, repetition, agent_id))
    if record is None:
        available = [k for k in index.keys() if k[0] == scenario_id and k[1] == repetition]
        raise KeyError(f'No frozen output for scenario={scenario_id!r} repetition={repetition} agent={agent_id!r}; available={available[:10]}')
    result = _strip_runtime_fields(as_dict(record.get('agent_result')))

    projection_id = clean(projection.get('projection_id'))
    snapshot_id = clean(projection.get('snapshot_id'))
    result['agent_id'] = agent_id
    result['role'] = clean(agent.get('role') or result.get('role') or agent_id)
    result['projection_id'] = projection_id or clean(result.get('projection_id'))
    result['snapshot_id'] = snapshot_id or clean(result.get('snapshot_id'))
    result['context_tokens'] = int(as_dict(projection.get('metrics')).get('context_tokens') or result.get('context_tokens') or 0)
    result['frozen_replay'] = {
        'enabled': True,
        'bundle_id': clean(cfg.get('frozen_bundle_id') or record.get('bundle_id')),
        'generation_condition': clean(record.get('generation_condition')),
        'generation_provider': clean(record.get('generation_provider')),
        'generation_model': clean(record.get('generation_model')),
        'generation_prompt_hash': clean(record.get('generation_prompt_hash')),
        'source_result_hash': clean(record.get('source_result_hash')),
        'repetition': repetition,
    }

    # Refresh source metadata inside write intents so audit logs refer to the
    # replay projection while preserving the original LLM-generated payload and
    # certificate fields.
    for intent in as_list(result.get('write_intents')):
        src = as_dict(intent.get('source'))
        src.update({
            'kind': 'frozen_live_llm_output',
            'agent_id': agent_id,
            'projection_id': projection_id,
            'cached_from_generation_condition': clean(record.get('generation_condition')),
        })
        intent['source'] = src
        pre = as_dict(intent.get('preconditions'))
        if snapshot_id:
            pre['base_snapshot_id'] = snapshot_id
        intent['preconditions'] = pre

    original_llm = as_dict(result.get('llm'))
    result['frozen_source_llm'] = original_llm
    result['llm'] = {
        'provider': clean(record.get('generation_provider') or original_llm.get('provider')),
        'model': clean(record.get('generation_model') or original_llm.get('model')),
        'cached_frozen_output': True,
        'provider_api_called': False,
        'prompt_hash': clean(record.get('generation_prompt_hash') or original_llm.get('prompt_hash')),
        'schema_version': clean(original_llm.get('schema_version')),
        'request_options': as_dict(original_llm.get('request_options')),
        'api_usage': {},
        'prompt_tokens_est': 0,
        'output_tokens_est': 0,
    }
    result['latency_ms'] = 0
    return result


def make_bundle_from_generation_results(payload: dict[str, Any], *, out_path: str | Path | None = None, bundle_id: str = '', generation_condition: str = '') -> dict[str, Any]:
    """Extract frozen per-agent outputs from a live generation result.

    Preferred input is a one-condition generation result.  For convenience, a
    previous multi-condition live run can also be used if generation_condition
    is supplied; only that condition's agent outputs are frozen.
    """
    all_results = as_list(payload.get('results'))
    requested_condition = clean(generation_condition)
    results = [r for r in all_results if clean(as_dict(r).get('condition')) == requested_condition] if requested_condition else all_results
    conditions = sorted({clean(as_dict(r).get('condition')) for r in results if clean(as_dict(r).get('condition'))})
    if len(conditions) != 1:
        all_conditions = sorted({clean(as_dict(r).get('condition')) for r in all_results if clean(as_dict(r).get('condition'))})
        raise ValueError(f'Frozen generation input should contain exactly one generation condition, found {conditions}. Available conditions: {all_conditions}. Pass generation_condition to select one.')
    gen_condition = conditions[0]
    bundle_id = clean(bundle_id) or f"frozen_{stable_hash(str(out_path or '') + gen_condition + str(len(results)))}"
    agent_outputs: list[dict[str, Any]] = []
    failed = []
    for row in results:
        r = as_dict(row)
        if r.get('failed'):
            failed.append({'scenario_id': r.get('scenario_id'), 'repetition': r.get('repetition'), 'error': r.get('error')})
            continue
        scenario_id = clean(r.get('scenario_id'))
        rep = int(r.get('repetition') or 0)
        for ar in as_list(r.get('agentRuns')):
            agent_result = as_dict(ar)
            agent_id = clean(agent_result.get('agent_id') or agent_result.get('id') or agent_result.get('role'))
            llm = as_dict(agent_result.get('llm'))
            source_hash = stable_hash({
                'scenario_id': scenario_id,
                'repetition': rep,
                'agent_id': agent_id,
                'output': agent_result.get('output'),
                'write_intents': agent_result.get('write_intents'),
            })
            agent_outputs.append({
                'kind': 'frozen_agent_output_v1',
                'bundle_id': bundle_id,
                'scenario_id': scenario_id,
                'source_task_id': r.get('source_task_id'),
                'repo': r.get('repo'),
                'language': r.get('language'),
                'repetition': rep,
                'agent_id': agent_id,
                'role': agent_result.get('role'),
                'generation_condition': gen_condition,
                'generation_provider': clean(llm.get('provider') or as_dict(r.get('llm_experiment')).get('provider')),
                'generation_model': clean(llm.get('model') or as_dict(r.get('llm_experiment')).get('model')),
                'generation_prompt_hash': clean(llm.get('prompt_hash')),
                'source_result_hash': source_hash,
                'write_intent_count': len(as_list(agent_result.get('write_intents'))),
                'agent_result': agent_result,
            })
    scenario_count = len({clean(x.get('scenario_id')) for x in agent_outputs})
    payload_out = {
        'kind': 'wccu_frozen_agent_bundle_v1',
        'bundle_id': bundle_id,
        'source_kind': payload.get('kind'),
        'generation_condition': gen_condition,
        'generation_args': payload.get('args') or {},
        'scenario_count': scenario_count,
        'agent_output_count': len(agent_outputs),
        'failed_generation_count': len(failed),
        'failed_generations': failed,
        'agent_outputs': agent_outputs,
    }
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            json.dump(payload_out, f, ensure_ascii=False, indent=2)
    return payload_out
