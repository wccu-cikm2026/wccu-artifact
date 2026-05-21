from __future__ import annotations

import time
from typing import Any

from wccu_eval.utils import as_list, estimate_tokens


def run_vector_memory_parallel(*, scenario: dict[str, Any], **_: Any) -> dict[str, Any]:
    started = time.time()
    latencies = [int(scenario.get('agent_outputs', {}).get(a.get('id'), {}).get('latency_ms', scenario.get('default_latency_ms', 5))) for a in as_list(scenario.get('agents'))]
    time.sleep(min(max(latencies or [0]), 25) / 1000)
    seen: dict[str, str] = {}
    missed = 0
    for agent in as_list(scenario.get('agents')):
        for intent in as_list(scenario.get('agent_outputs', {}).get(agent.get('id'), {}).get('intents')):
            payload = intent.get('payload') or {}
            target = payload.get('id') or payload.get('atom_id') or intent.get('id')
            text = payload.get('canonical_text_en') or payload.get('text_original') or ''
            if target and target in seen and seen[target] != text:
                missed += 1
            if target:
                seen[target] = text
    return {
        'kind': 'baseline_result_v1',
        'condition': 'vector_memory_parallel',
        'scenario_id': scenario['id'],
        'elapsed_ms': int((time.time() - started) * 1000) + max(latencies or [0]),
        'conflict_groups': 0,
        'unsafe_auto_commit_count': missed,
        'stale_write_blocked_count': 0,
        'context_tokens': len(as_list(scenario.get('agents'))) * max(40, int(estimate_tokens(str(scenario.get('seed', {}))) * 0.45 + 0.999)),
        'task_success': False if scenario.get('expected', {}).get('conflicts') else True,
        'commit': {'committed': len(seen), 'proposals': 0, 'conflicts': 0, 'total': len(seen)},
    }
