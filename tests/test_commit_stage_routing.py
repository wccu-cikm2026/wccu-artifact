from __future__ import annotations

import tempfile
from pathlib import Path

from wccu_eval.substrate.context_commit_lanes import classify_context_commit_lane
from wccu_eval.substrate.context_substrate_store import commit_context_write_intents_batch, seed_context, paths_for
from wccu_eval.utils import read_jsonl


def test_requested_review_overrides_schema_commit_mode_none_at_lane_stage() -> None:
    lane = classify_context_commit_lane({
        'intent_type': 'patch_atom',
        'commit_mode': 'none',
        'requested_commit_mode': 'review_required',
        'payload': {'id': 'atom_a', 'atom_id': 'atom_a', 'atom_type': 'memory', 'canonical_text_en': 'patch'},
    })
    assert lane['commit_mode'] == 'review_required'


def test_review_routed_intent_is_not_written_to_operations_log() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'ctx'
        seed_context(root, {'atoms': [{'id': 'atom_a', 'atom_type': 'memory', 'canonical_text_en': 'old'}]})
        result = commit_context_write_intents_batch(root, [{
            'intent_type': 'patch_atom',
            'commit_mode': 'none',
            'requested_commit_mode': 'review_required',
            'payload': {'id': 'atom_a', 'atom_id': 'atom_a', 'atom_type': 'memory', 'canonical_text_en': 'new'},
            'policy': {'commit_mode': 'review_required', 'conflict_reason': 'test_review'},
        }])
        paths = paths_for(root)
        assert result['committed'] == 0
        assert result['proposals'] == 1
        assert read_jsonl(paths['operations']) == []
        proposals = read_jsonl(paths['proposals'])
        assert len(proposals) == 1
        assert proposals[0]['status'] == 'proposal'
