from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wccu_eval.eval.run_experiment import REPO_ROOT, run_experiment
from wccu_eval.utils import clean, write_json

ABSTRACT_ABLATION_SCENARIOS = [
    'high_risk_rule_change',
    'user_correction_rebase',
    'append_only_evidence_log',
    'workspace_compatible_patch_contention',
    'conflict_detection',
    'low_risk_memory_merge',
]

ABSTRACT_ABLATION_CONDITIONS = [
    'adaptive_policy',
    'adaptive_no_review_gate',
    'adaptive_no_authority_rebase',
    'adaptive_no_append_only',
    'adaptive_no_workspace_lock',
    'adaptive_no_semantic_conflict_detection',
    'uniform_snapshot_occ',
    'uniform_review_gated',
]


def _row(aggregated: list[dict[str, Any]], scenario: str, condition: str) -> dict[str, Any] | None:
    return next((r for r in aggregated if r.get('scenario_id') == scenario and r.get('condition') == condition), None)


def _n(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _claim(name: str, passed: bool, evidence: dict[str, Any], abstract_use: str) -> dict[str, Any]:
    return {'name': name, 'passed': bool(passed), 'evidence': evidence, 'abstract_use': abstract_use}


def build_abstract_claim_summary(payload: dict[str, Any]) -> dict[str, Any]:
    agg = payload.get('aggregated') or []
    adaptive_high = _row(agg, 'high_risk_rule_change', 'adaptive_policy')
    no_review_high = _row(agg, 'high_risk_rule_change', 'adaptive_no_review_gate')
    adaptive_correction = _row(agg, 'user_correction_rebase', 'adaptive_policy')
    no_authority_correction = _row(agg, 'user_correction_rebase', 'adaptive_no_authority_rebase')
    adaptive_append = _row(agg, 'append_only_evidence_log', 'adaptive_policy')
    no_append = _row(agg, 'append_only_evidence_log', 'adaptive_no_append_only')
    adaptive_workspace = _row(agg, 'workspace_compatible_patch_contention', 'adaptive_policy')
    no_workspace = _row(agg, 'workspace_compatible_patch_contention', 'adaptive_no_workspace_lock')
    adaptive_conflict = _row(agg, 'conflict_detection', 'adaptive_policy')
    no_semantic = _row(agg, 'conflict_detection', 'adaptive_no_semantic_conflict_detection')
    adaptive_low = _row(agg, 'low_risk_memory_merge', 'adaptive_policy')
    review_low = _row(agg, 'low_risk_memory_merge', 'uniform_review_gated')

    claims = [
        _claim(
            'review_gate_prevents_high_risk_auto_commit',
            bool(adaptive_high and no_review_high and _n(adaptive_high.get('mean_unsafe_auto_commit_count')) == 0 and _n(no_review_high.get('mean_unsafe_auto_commit_count')) > _n(adaptive_high.get('mean_unsafe_auto_commit_count')) and _n(adaptive_high.get('mean_proposals')) > 0),
            {'adaptive_policy': adaptive_high, 'adaptive_no_review_gate': no_review_high},
            'Removing the review gate turns high-risk policy changes into unsafe automatic commits.',
        ),
        _claim(
            'authority_rebase_commits_user_correction_and_reviews_stale_agent_write',
            bool(adaptive_correction and no_authority_correction and _n(adaptive_correction.get('mean_authority_rebase_count')) > _n(no_authority_correction.get('mean_authority_rebase_count')) and _n(adaptive_correction.get('mean_unsafe_auto_commit_count')) == 0),
            {'adaptive_policy': adaptive_correction, 'adaptive_no_authority_rebase': no_authority_correction},
            'Authority-aware rebase lets user corrections dominate stale agent writes without serializing all work.',
        ),
        _claim(
            'append_only_lane_avoids_unnecessary_review_burden',
            bool(adaptive_append and no_append and _n(adaptive_append.get('mean_review_burden_count')) == 0 and _n(no_append.get('mean_review_burden_count')) > _n(adaptive_append.get('mean_review_burden_count'))),
            {'adaptive_policy': adaptive_append, 'adaptive_no_append_only': no_append},
            'Append-only evidence can remain parallel and review-free, while conservative fallback increases review burden.',
        ),
        _claim(
            'workspace_lock_prevents_compatible_looking_patch_auto_merge',
            bool(adaptive_workspace and no_workspace and _n(adaptive_workspace.get('mean_lock_conflict_count')) > 0 and _n(no_workspace.get('mean_unsafe_auto_commit_count')) > _n(adaptive_workspace.get('mean_unsafe_auto_commit_count'))),
            {'adaptive_policy': adaptive_workspace, 'adaptive_no_workspace_lock': no_workspace},
            'Workspace patches require stronger isolation than compatible-looking memory updates.',
        ),
        _claim(
            'semantic_conflict_detection_blocks_incompatible_context_updates',
            bool(adaptive_conflict and no_semantic and _n(adaptive_conflict.get('mean_proposals')) > 0 and _n(no_semantic.get('mean_unsafe_auto_commit_count')) > _n(adaptive_conflict.get('mean_unsafe_auto_commit_count'))),
            {'adaptive_policy': adaptive_conflict, 'adaptive_no_semantic_conflict_detection': no_semantic},
            'Semantic conflict detection is necessary because same-target writes can be syntactically valid but behaviorally incompatible.',
        ),
        _claim(
            'adaptive_is_not_uniform_review_gating',
            bool(adaptive_low and review_low and _n(adaptive_low.get('mean_review_burden_count')) < _n(review_low.get('mean_review_burden_count')) and _n(adaptive_low.get('mean_committed')) > 0),
            {'adaptive_policy': adaptive_low, 'uniform_review_gated': review_low},
            'The adaptive substrate is not merely conservative review gating; it auto-merges low-risk compatible context writes.',
        ),
    ]
    passed = sum(1 for c in claims if c['passed'])
    return {
        'kind': 'abstract_ablation_claim_summary_v1',
        'passed': passed,
        'total': len(claims),
        'all_passed': passed == len(claims),
        'claims': claims,
        'abstract_sentence_candidate': 'In targeted ablations, removing review gates, authority-aware rebase, append-only lanes, workspace locks, or semantic conflict detection selectively increased unsafe commits or review burden, while the full adaptive substrate preserved low-risk automatic merges.',
    }


def run_abstract_ablation(*, repetitions: int = 5, out: str = 'results/abstract_ablation_results.json', scenario: str | None = None, condition: str | None = None) -> dict[str, Any]:
    scenario = clean(scenario) or ','.join(ABSTRACT_ABLATION_SCENARIOS)
    condition = clean(condition) or ','.join(ABSTRACT_ABLATION_CONDITIONS)
    out = clean(out) or 'results/abstract_ablation_results.json'
    payload = run_experiment(scenario=scenario, condition=condition, repetitions=max(1, int(repetitions)), out=out)
    claim_summary = build_abstract_claim_summary(payload)
    enriched = {**payload, 'kind': 'context_substrate_abstract_ablation_results_v1', 'claim_summary': claim_summary}
    out_path = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    write_json(out_path, enriched)
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the minimal abstract ablation suite.')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--out', default='results/abstract_ablation_results.json')
    parser.add_argument('--scenario', default='')
    parser.add_argument('--condition', default='')
    args = parser.parse_args(argv)
    payload = run_abstract_ablation(repetitions=args.repetitions, out=args.out, scenario=args.scenario, condition=args.condition)
    print(json.dumps({
        'ok': payload['claim_summary']['all_passed'],
        'out': payload['args']['out'],
        'result_count': len(payload['results']),
        'aggregate_count': len(payload['aggregated']),
        'claims_passed': payload['claim_summary']['passed'],
        'claims_total': payload['claim_summary']['total'],
    }, indent=2))
    return 0 if payload['claim_summary']['all_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
