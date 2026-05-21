from __future__ import annotations

import time
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, stable_hash


def _with_base(intent: dict[str, Any], projection: dict[str, Any], agent_id: str) -> dict[str, Any]:
    return {
        'id': intent.get('id') or f"intent_{stable_hash(agent_id + ':' + projection['projection_id'] + ':' + str(intent))}",
        'actor': f'agent:{agent_id}',
        **intent,
        'preconditions': {
            'base_snapshot_id': projection.get('snapshot_id'),
            **as_dict(intent.get('preconditions')),
        },
    }


def _event_intent(agent_id: str, projection: dict[str, Any], text: str) -> dict[str, Any]:
    return _with_base({
        'intent_type': 'append_event',
        'payload': {
            'atom_type': 'event',
            'title': f'{agent_id} completed',
            'canonical_text_en': text,
            'structured': {'projection_id': projection.get('projection_id')},
        },
    }, projection, agent_id)


def _atom_intent(agent_id: str, projection: dict[str, Any], payload: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return _with_base({'intent_type': 'upsert_atom', 'payload': payload, **(extra or {})}, projection, agent_id)


def run_deterministic_agent(*, agent: dict[str, Any], projection: dict[str, Any], scenario: dict[str, Any], **_: Any) -> dict[str, Any]:
    agent_id = clean(agent.get('id') or agent.get('role') or 'agent')
    role = clean(agent.get('role') or agent_id)
    spec = as_dict(as_dict(scenario.get('agent_outputs')).get(agent_id) or as_dict(scenario.get('agent_outputs')).get(role))
    artificial_ms = int(spec.get('latency_ms', agent.get('latency_ms', scenario.get('default_latency_ms', 5))))
    if artificial_ms > 0:
        time.sleep(artificial_ms / 1000)
    text = clean(spec.get('text') or f"{agent_id} produced {len(as_list(spec.get('intents'))) or 1} write intent(s).")
    write_intents = [_with_base(intent, projection, agent_id) for intent in as_list(spec.get('intents'))]
    if not write_intents:
        write_intents.append(_event_intent(agent_id, projection, text))
    if spec.get('add_summary_atom'):
        write_intents.append(_atom_intent(agent_id, projection, {
            'id': f"atom_summary_{agent_id}_{stable_hash(text, 8)}",
            'atom_type': 'agent_summary',
            'title': f'{agent_id} summary',
            'canonical_text_en': text,
            'tags': [role, 'summary'],
        }))
    return {
        'agent_id': agent_id,
        'role': role,
        'projection_id': projection.get('projection_id'),
        'snapshot_id': projection.get('snapshot_id'),
        'output': text,
        'write_intents': write_intents,
        'latency_ms': artificial_ms,
        'context_tokens': projection.get('metrics', {}).get('context_tokens', 0),
    }
