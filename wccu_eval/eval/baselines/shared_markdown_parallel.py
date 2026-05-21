from __future__ import annotations

import time
from typing import Any

from wccu_eval.utils import as_list, estimate_tokens


def run_shared_markdown_parallel(*, scenario: dict[str, Any], **_: Any) -> dict[str, Any]:
    started = time.time()
    latencies = [int(scenario.get('agent_outputs', {}).get(a.get('id'), {}).get('latency_ms', scenario.get('default_latency_ms', 5))) for a in as_list(scenario.get('agents'))]
    time.sleep(min(max(latencies or [0]), 25) / 1000)
    writes_by_target: dict[str, dict[str, Any]] = {}
    lost_updates = 0
    for agent in as_list(scenario.get('agents')):
        spec = scenario.get('agent_outputs', {}).get(agent.get('id'), {})
        for intent in as_list(spec.get('intents')):
            payload = intent.get('payload') or {}
            target = payload.get('id') or payload.get('atom_id') or intent.get('id')
            if not target:
                continue
            if target in writes_by_target:
                lost_updates += 1
            writes_by_target[target] = {'agent': agent.get('id'), 'intent': intent}
    return {
        'kind': 'baseline_result_v1',
        'condition': 'shared_markdown_parallel',
        'scenario_id': scenario['id'],
        'elapsed_ms': int((time.time() - started) * 1000) + max(latencies or [0]),
        'conflict_groups': 0,
        'unsafe_auto_commit_count': lost_updates,
        'lost_update_count': lost_updates,
        'stale_write_blocked_count': 0,
        'context_tokens': len(as_list(scenario.get('agents'))) * estimate_tokens(str(scenario.get('seed', {})) + '\n' + str(scenario.get('agent_outputs', {}))),
        'task_success': False if scenario.get('expected', {}).get('conflicts') else True,
        'commit': {'committed': len(writes_by_target), 'proposals': 0, 'conflicts': 0, 'total': len(writes_by_target)},
    }
