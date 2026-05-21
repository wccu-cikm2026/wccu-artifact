from wccu_eval.agents.llm_agent import build_llm_agent_prompt, call_llm_provider
from wccu_eval.eval.scenarios import get_scenario
from wccu_eval.scheduler.target_grounder import resolve_intent_target


def test_target_grounder_resolves_backup_url_alias_to_atom_pref():
    scenario = get_scenario('user_correction_rebase')
    intent = {
        'intent_type': 'patch_atom',
        'payload': {
            'id': 'ctx_000000:memory:backup_url_preference',
            'atom_id': 'memory',
            'atom_type': 'memory',
            'canonical_text_en': 'The user prefers backup_url in API responses.',
        },
    }
    grounded = resolve_intent_target(intent, scenario)
    assert grounded['payload']['atom_id'] == 'atom_pref'
    assert grounded['target_grounding']['resolved'] is True


def test_llm_prompt_includes_stable_target_candidates():
    scenario = get_scenario('high_risk_rule_change')
    prompt = build_llm_agent_prompt(agent={'id': 'builder', 'role': 'builder'}, projection={'prompt': '', 'projection_id': 'p', 'snapshot_id': 's'}, scenario=scenario)
    assert '[STABLE TARGET CANDIDATES]' in prompt
    assert 'atom_permission_policy' in prompt


def test_mock_provider_records_no_api_temperature_requirements():
    result = call_llm_provider(provider='mock', model='fixture', prompt='{}', scenario=get_scenario('high_risk_rule_change'), agent={'id': 'builder'})
    assert result['endpoint'] == 'mock'
