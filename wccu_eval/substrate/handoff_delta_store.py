from __future__ import annotations

from pathlib import Path
from typing import Any

from wccu_eval.utils import append_jsonl, as_dict, clean, now_iso, stable_hash


def build_handoff_delta(*, from_agent: str, to_agent: str, handoff_type: str, snapshot_id: str = '', projection_id: str = '', delta: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = as_dict(delta)
    return {
        'kind': 'handoff_delta_v1',
        'id': f"handoff_{stable_hash(f'{from_agent}:{to_agent}:{handoff_type}:{snapshot_id}:{projection_id}:{payload}:{now_iso()}')}",
        'created_at': now_iso(),
        'from_agent': clean(from_agent),
        'to_agent': clean(to_agent),
        'handoff_type': clean(handoff_type),
        'snapshot_id': clean(snapshot_id),
        'projection_id': clean(projection_id),
        'delta': payload,
    }


def append_handoff_delta(run_dir: str | Path, delta: dict[str, Any]) -> None:
    append_jsonl(Path(run_dir) / 'handoff_deltas.jsonl', delta)
