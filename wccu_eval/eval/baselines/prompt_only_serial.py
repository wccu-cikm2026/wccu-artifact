from __future__ import annotations

import time
from typing import Any

from wccu_eval.scheduler.team_dag_executor import approximate_prompt_only_tokens
from wccu_eval.utils import as_list


def run_prompt_only_serial(*, scenario: dict[str, Any], **_: Any) -> dict[str, Any]:
    started = time.time()
    elapsed = 0
    unsafe = 0
    target_writes: dict[str, dict[str, Any]] = {}
    for agent in as_list(scenario.get('agents')):
        spec = scenario.get('agent_outputs', {}).get(agent.get('id'), {})
        latency = int(spec.get('latency_ms', scenario.get('default_latency_ms', 5)))
        elapsed += latency
        for intent in as_list(spec.get('intents')):
            payload = intent.get('payload') or {}
            target = payload.get('id') or payload.get('atom_id') or intent.get('id')
            if target and target in target_writes and target_writes[target].get('payload') != payload:
                unsafe += 1
            if target:
                target_writes[target] = intent
    time.sleep(min(elapsed, 25) / 1000)
    return {
        'kind': 'baseline_result_v1',
        'condition': 'prompt_only_serial',
        'scenario_id': scenario['id'],
        'elapsed_ms': int((time.time() - started) * 1000) + elapsed,
        'conflict_groups': 0,
        'unsafe_auto_commit_count': unsafe,
        'stale_write_blocked_count': 0,
        'context_tokens': approximate_prompt_only_tokens(scenario),
        'task_success': False if scenario.get('expected', {}).get('conflicts') else unsafe == 0,
        'commit': {'committed': len(target_writes), 'proposals': 0, 'conflicts': 0, 'total': len(target_writes)},
    }
