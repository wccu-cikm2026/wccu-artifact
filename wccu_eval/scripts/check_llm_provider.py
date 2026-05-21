from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wccu_eval.agents.llm_agent import LlmProviderError, run_llm_agent
from wccu_eval.env import load_dotenv
from wccu_eval.eval.scenarios import get_scenario
from wccu_eval.substrate.context_projection_compiler import compile_projection
from wccu_eval.substrate.context_substrate_store import seed_context
from wccu_eval.utils import clean, remove_dir

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description='Check whether an LLM provider returns valid write-intent JSON.')
    parser.add_argument('--provider', default=os.environ.get('LLM_PROVIDER', 'openai'))
    parser.add_argument('--model', default=os.environ.get('LLM_MODEL', ''))
    parser.add_argument('--scenario', default='high_risk_rule_change')
    parser.add_argument('--agent', default='')
    parser.add_argument('--temperature', type=float, default=None, help='Optional sampling temperature. Omitted by default for OpenAI Responses reasoning models.')
    parser.add_argument('--send-temperature', action='store_true', default=None)
    parser.add_argument('--reasoning-effort', default=os.environ.get('LLM_REASONING_EFFORT', ''))
    parser.add_argument('--text-verbosity', default=os.environ.get('LLM_TEXT_VERBOSITY', ''))
    parser.add_argument('--max-output-tokens', type=int, default=1000)
    parser.add_argument('--timeout-seconds', type=int, default=90)
    parser.add_argument('--max-provider-retries', type=int, default=int(os.environ.get('LLM_MAX_PROVIDER_RETRIES', 4)))
    parser.add_argument('--retry-backoff-base', type=float, default=float(os.environ.get('LLM_RETRY_BACKOFF_BASE', 1.0)))
    parser.add_argument('--retry-backoff-max', type=float, default=float(os.environ.get('LLM_RETRY_BACKOFF_MAX', 20.0)))
    parser.add_argument('--error-log', default=os.environ.get('LLM_ERROR_LOG_PATH', 'results/llm_provider_check.errors.jsonl'))
    args = parser.parse_args(argv)
    try:
        scenario = get_scenario(args.scenario)
        agent = next((a for a in scenario['agents'] if a.get('id') == args.agent or a.get('role') == args.agent), scenario['agents'][0])
        run_dir = REPO_ROOT / 'runs' / 'llm_provider_check'
        root_dir = run_dir / 'context_substrate'
        remove_dir(run_dir)
        seed_context(root_dir, scenario.get('seed', {}))
        projection = compile_projection(root_dir, role=agent.get('role') or agent.get('id'), task_type=scenario.get('task_type', 'general_task'), goal=scenario.get('goal', ''), budget_tokens=scenario.get('budget_tokens', 1200))
        result = run_llm_agent(agent=agent, projection=projection, scenario=scenario, llm_config={'provider': args.provider, 'model': clean(args.model), 'temperature': args.temperature, 'max_output_tokens': args.max_output_tokens, 'timeout_seconds': args.timeout_seconds, 'max_parse_retries': 1, 'send_temperature': args.send_temperature, 'reasoning_effort': args.reasoning_effort, 'text_verbosity': args.text_verbosity, 'max_provider_retries': args.max_provider_retries, 'retry_backoff_base': args.retry_backoff_base, 'retry_backoff_max': args.retry_backoff_max, 'error_log_path': str((REPO_ROOT / args.error_log).resolve() if args.error_log and not Path(args.error_log).is_absolute() else args.error_log)})
        print(json.dumps({'ok': True, 'provider': args.provider, 'model': clean(args.model) or result['llm']['model'], 'scenario_id': scenario['id'], 'agent_id': result['agent_id'], 'endpoint': result['llm']['endpoint'], 'provider_http': result['llm'].get('provider_http', {}), 'error_log': result['llm'].get('error_log_path', ''), 'write_intent_count': len(result['write_intents']), 'output': result['output'], 'first_intent': result['write_intents'][0]}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc), 'error_type': type(exc).__name__, 'provider_error': exc.to_dict() if isinstance(exc, LlmProviderError) else {}, 'validation_errors': getattr(exc, 'validation_errors', []), 'raw_llm_text': getattr(exc, 'raw_llm_text', '')}, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
