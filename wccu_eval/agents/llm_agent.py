from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from wccu_eval.agents.llm_output_schema import LLM_WRITE_INTENT_JSON_SCHEMA, OUTPUT_SCHEMA_VERSION, gemini_compatible_schema, normalize_llm_output, parse_json_object_from_text, schema_instruction_text
from wccu_eval.env import load_dotenv
from wccu_eval.scheduler.target_grounder import target_candidates_for_scenario
from wccu_eval.utils import append_jsonl, as_dict, as_list, clean, estimate_tokens, now_iso, stable_hash

DEFAULT_TIMEOUT_SECONDS = 90


def parse_agent_model_specs(value: Any) -> dict[str, dict[str, str]]:
    """Parse per-agent model routing specs.

    Accepted forms:
      - "coop_agent_a=openai:gpt-5.4-nano,coop_agent_b=gemini:gemini-3.1-flash-lite"
      - "coop_agent_a:openai:gpt-5.4-nano;coop_agent_b:gemini:gemini-3.1-flash-lite"
      - {"coop_agent_a": {"provider": "openai", "model": "..."}, ...}

    Keys are matched against agent id first, then role, then "*" / "default".
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        out: dict[str, dict[str, str]] = {}
        for key, raw in value.items():
            k = clean(key)
            if not k:
                continue
            if isinstance(raw, dict):
                provider = clean(raw.get('provider'))
                model = clean(raw.get('model'))
            else:
                token = clean(raw)
                if ':' not in token:
                    continue
                provider, model = [clean(x) for x in token.split(':', 1)]
            if provider and model:
                out[k] = {'provider': provider, 'model': model}
        return out
    spec = clean(value)
    if not spec:
        return {}
    out: dict[str, dict[str, str]] = {}
    for raw_entry in spec.replace(';', ',').split(','):
        entry = clean(raw_entry)
        if not entry:
            continue
        if '=' in entry:
            key, rhs = entry.split('=', 1)
        else:
            parts = entry.split(':', 2)
            if len(parts) != 3:
                continue
            key, rhs = parts[0], f'{parts[1]}:{parts[2]}'
        key = clean(key)
        if ':' not in rhs:
            continue
        provider, model = [clean(x) for x in rhs.split(':', 1)]
        if key and provider and model:
            out[key] = {'provider': provider, 'model': model}
    return out


def agent_model_override(agent: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str]:
    specs = parse_agent_model_specs(
        cfg.get('agent_model_specs')
        or cfg.get('agent_models')
        or os.environ.get('WCCU_AGENT_MODEL_SPECS')
        or os.environ.get('LLM_AGENT_MODEL_SPECS')
    )
    agent_id = clean(agent.get('id') or agent.get('role') or 'agent')
    role = clean(agent.get('role') or agent_id)
    for key in (agent_id, role, '*', 'default'):
        if key in specs:
            return specs[key]
    return {}


def default_llm_agent_task(*, scenario: dict[str, Any], agent: dict[str, Any]) -> str:
    agent_id = clean(agent.get('id') or agent.get('role') or 'agent')
    role = clean(agent.get('role') or agent_id)
    scenario_id = clean(scenario.get('id'))
    specific = as_dict(scenario.get('llm_agent_tasks')).get(agent_id) or as_dict(scenario.get('llm_agent_tasks')).get(role)
    if specific:
        return clean(specific)
    mapping = {
        'snapshot_fanout': {
            'researcher': 'Write one upsert_atom memory summary about which context projection facts matter for this task. Use a new id starting with atom_llm_researcher_summary.',
            'test_planner': 'Write one upsert_atom memory summary describing tests needed for projections, merges, and stale write rejection. Use a new id starting with atom_llm_test_plan.',
            'risk_reviewer': 'Write one upsert_atom review_finding about unsafe commits and review gates. Use a new id starting with atom_llm_risk_review.',
            'synthesizer': 'Write one upsert_atom memory synthesis of the independent findings. Use a new id starting with atom_llm_synthesis.',
        },
        'append_only_evidence_log': {
            'evidence_a': 'Append one evidence_event to stream id evidence_stream. Title it Observation A. Do not patch or replace state.',
            'evidence_b': 'Append one evidence_event to stream id evidence_stream. Title it Observation B. Do not patch or replace state.',
            'evidence_c': 'Append one evidence_event to stream id evidence_stream. Title it Observation C. Do not patch or replace state.',
        },
        'conflict_detection': {
            'agent_a': 'Patch atom_api_policy. Your policy must require API responses to include backup_url for recoverability.',
            'agent_b': 'Patch atom_api_policy. Your policy must prohibit API responses from including backup_url for privacy.',
        },
        'high_risk_rule_change': {'builder': 'Patch atom_permission_policy with a high-risk permission_policy change that relaxes deployment approval for speed. Set risk to high.'},
        'user_correction_rebase': {
            'old_snapshot_agent': 'Patch atom_pref from the older agent perspective by strengthening the backup_url preference.',
            'correction_agent': 'Retract atom_pref as a user correction. Set authority to user and include a clear reason.',
        },
        'user_correction_ambiguous_targets': {
            'old_snapshot_agent': 'From the older snapshot, patch the existing memory about backup_url in API responses by strengthening that preference. Do not update callback_url, fallback_url, debug_url, or backup email preferences.',
            'correction_agent': 'Retract the existing memory about backup_url in API responses as a user correction. Set authority to user. Do not retract callback_url, fallback_url, debug_url, or backup email preferences.',
        },
        'workspace_patch_contention': {
            'builder_a': 'Patch atom_file_plan for file_path src/api.ts so getStatus returns ok.',
            'builder_b': 'Patch atom_file_plan for file_path src/api.ts so getStatus returns a health object.',
            'reviewer': 'Upsert atom_review_test_required as a review_finding saying src/api.ts changes require unit test coverage.',
        },
        'low_risk_memory_merge': {
            'agent_a': 'Patch atom_arch with the same canonical_text_en as the existing architecture note, adding a confirmed-style tag only if useful.',
            'agent_b': 'Patch atom_arch with the same canonical_text_en as the existing architecture note, adding a verified-style tag only if useful.',
        },
        'workspace_compatible_patch_contention': {
            'builder_a': 'Patch atom_file_plan_compatible for file_path src/cache.ts so getCacheStatus returns ready.',
            'builder_b': 'Patch atom_file_plan_compatible for file_path src/cache.ts with the same getCacheStatus returns ready behavior.',
        },
        'wccu_stale_dependency_cross_target': {
            'policy_agent': 'Patch atom_api_response_policy based on the active memory atom_pref_backup_url. In your certificate.read_dependencies, list atom_pref_backup_url and require freshness because the policy write is unsafe if that memory is retracted.',
            'correction_agent': 'Retract atom_pref_backup_url as a user correction. Set authority to user. The correction invalidates any policy write that read the old backup_url preference.',
        },
        'wccu_stale_dependency_lowrisk_cross_target': {
            'format_agent': 'Patch atom_response_format_hint based on the active memory atom_pref_backup_url. Keep the write low risk. In certificate.read_dependencies, list atom_pref_backup_url and require freshness because this different-target write is unsafe if that memory is retracted. In certificate.authority_certificate set actor_authority=agent and required_authority=agent; do not require builder, reviewer, user, or system authority for this low-risk memory hint patch.',
            'correction_agent': 'Retract atom_pref_backup_url as a user correction. Set authority to user. The correction invalidates any response-format hint that read the old backup_url preference. In certificate.authority_certificate set actor_authority=user and required_authority=user.',
        },
        'mini_coding_team': {
            'builder_a': 'Patch atom_file_plan for file_path src/api.ts so getStatus returns ok.',
            'builder_b': 'Patch atom_file_plan for file_path src/api.ts so getStatus returns a health object.',
            'reviewer': 'Upsert atom_review_test_required as a review_finding saying src/api.ts changes require unit test coverage.',
        },
    }
    return clean(mapping.get(scenario_id, {}).get(agent_id) or mapping.get(scenario_id, {}).get(role)) or f'Act as {role}. Produce the smallest set of context write intents needed for the scenario goal.'


def build_llm_agent_prompt(*, agent: dict[str, Any], projection: dict[str, Any], scenario: dict[str, Any], include_target_candidates: bool = True, certificate_guidance: str = 'guided') -> str:
    agent_id = clean(agent.get('id') or agent.get('role') or 'agent')
    role = clean(agent.get('role') or agent_id)
    task = default_llm_agent_task(scenario=scenario, agent=agent)
    certificate_guidance = clean(certificate_guidance or 'guided').lower()
    if certificate_guidance in {'unguided', 'minimal', 'no_hints'}:
        unguided = as_dict(scenario.get('llm_agent_tasks_unguided')).get(agent_id) or as_dict(scenario.get('llm_agent_tasks_unguided')).get(role)
        if unguided:
            task = clean(unguided)
    target_candidates = target_candidates_for_scenario(scenario) if include_target_candidates else []
    target_block = json.dumps(target_candidates, ensure_ascii=False, indent=2) if target_candidates else '[]'
    if certificate_guidance in {'unguided', 'minimal', 'no_hints'}:
        certificate_rules = [
            '- Include a witness-carrying context update (WCCU) certificate for every write intent because the output schema requires one.',
            '- Fill certificate fields from your own understanding of the compiled projection and task. Do not rely on hidden oracle dependencies.',
            '- Do not assume which dependency will be tested; only list dependencies whose content you actually used to create the write.',
            '- In certificate.target_certificate, explain why the write grounds to the claimed target id and include a confidence score from 0 to 1.',
            '- In certificate.delta_contract, describe the semantic kind of mutation using a compact delta_type.',
            '- In certificate.authority_certificate, set the required authority implied by the task, not a stricter authority than necessary.',
        ]
    else:
        certificate_rules = [
            '- Include a witness-carrying context update (WCCU) certificate for every write intent.',
            '- Treat the certificate as a set of checkable obligations, not a proof the runtime blindly trusts.',
            '- O-READ: in certificate.read_dependencies, list any atom/view whose content you relied on to create the write; leave it empty only for pure append events or writes that do not depend on existing context.',
            '- O-TARGET: in certificate.target_certificate, explain why the write grounds to the claimed target id and include a confidence score from 0 to 1.',
            '- O-DELTA: in certificate.delta_contract, name how the write changes context, such as append_evidence, strengthen_rule, weaken_rule, retract_memory, patch_workspace, or patch_memory.',
            '- O-FRESH: in certificate.preconditions, state whether freshness and non-retracted dependencies are required. Use requires_review_if_invalid=true for ordinary state writes.',
            '- O-AUTH: in certificate.authority_certificate, do not invent a stricter required_authority than the task requires. For low-risk memory or task-state patches by an ordinary agent, use actor_authority=agent and required_authority=agent. Use required_authority=reviewer only for high-risk, weakening, deployment, permission, or policy-relaxing writes. Use required_authority=user for user corrections/retractions.',
            '- O-VIEW: when a write retracts, weakens, or invalidates derived context, list affected view ids when known.',
        ]
    return '\n'.join([
        'You are an agent in a controlled multi-agent context substrate evaluation harness.',
        'Your job is not to answer the user directly. Your job is to propose typed context write intents.',
        '',
        f"[SCENARIO] {scenario.get('id')}",
        f"Goal: {scenario.get('goal','')}",
        f"Task type: {scenario.get('task_type','general_task')}",
        f'Agent id: {agent_id}',
        f'Role: {role}',
        '',
        '[AGENT TASK]',
        task,
        '',
        '[COMPILED CONTEXT PROJECTION]',
        projection.get('prompt', ''),
        '',
        '[STABLE TARGET CANDIDATES]',
        target_block,
        '',
        '[RULES]',
        '- Do not decide whether a write is safe to commit; the runtime will do that.',
        '- Prefer exactly one write intent unless the task clearly requires more.',
        '- For patch_atom and retract_atom, select payload.id and payload.atom_id from STABLE TARGET CANDIDATES whenever the write modifies existing context.',
        '- If STABLE TARGET CANDIDATES is empty, use the most specific stable-looking id from the task/projection and include descriptive title/text so runtime grounding can resolve it.',
        '- Do not invent a new id for an existing target when a candidate exists. Use the exact target_id value, such as atom_pref or atom_permission_policy.',
        *certificate_rules,
        '- For retract_atom, include payload.id or payload.atom_id and a reason.',
        '- For append_event, use the stream id as payload.id when a stream is specified.',
        '- Use canonical English text in payload.canonical_text_en.',
        '',
        '[OUTPUT FORMAT]',
        schema_instruction_text(),
    ])



class LlmProviderError(RuntimeError):
    """Provider transport/API failure with diagnostic metadata."""

    def __init__(self, message: str, *, provider: str = '', endpoint: str = '', status_code: int | None = None, response_text: str = '', retryable: bool = False, attempts: int = 1, request_id: str = '', error_log_path: str = '') -> None:
        super().__init__(message)
        self.provider = provider
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_text = response_text
        self.retryable = retryable
        self.attempts = attempts
        self.request_id = request_id
        self.error_log_path = error_log_path
        self.raw_llm_text = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'provider': self.provider,
            'endpoint': self.endpoint,
            'status_code': self.status_code,
            'retryable': self.retryable,
            'attempts': self.attempts,
            'request_id': self.request_id,
            'error_log_path': self.error_log_path,
            'message': str(self),
            'response_text_preview': self.response_text[:1000],
        }


_RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def _is_retryable_http_status(status_code: int | None) -> bool:
    if status_code is None:
        return True
    return status_code in _RETRYABLE_HTTP_STATUS or status_code >= 500


def _summarize_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Record useful request metadata without storing the full prompt or API key."""
    summary: dict[str, Any] = {
        'model': body.get('model'),
        'max_output_tokens': body.get('max_output_tokens') or body.get('max_tokens'),
        'has_temperature': 'temperature' in body,
        'has_text_schema': bool(as_dict(body.get('text')).get('format')),
        'reasoning': body.get('reasoning'),
    }
    # Prompt text can be long/private, so only log a stable hash and rough size.
    prompt_text = ''
    if 'input' in body:
        prompt_text = json.dumps(body.get('input'), ensure_ascii=False)
    elif 'messages' in body:
        prompt_text = json.dumps(body.get('messages'), ensure_ascii=False)
    elif 'contents' in body:
        prompt_text = json.dumps(body.get('contents'), ensure_ascii=False)
    if prompt_text:
        summary['prompt_hash'] = stable_hash(prompt_text)
        summary['prompt_chars'] = len(prompt_text)
        summary['prompt_tokens_est'] = estimate_tokens(prompt_text)
    return {k: v for k, v in summary.items() if v not in (None, '', {})}


def _log_provider_error(*, error_log_path: str = '', row: dict[str, Any]) -> None:
    if not error_log_path:
        return
    try:
        append_jsonl(error_log_path, row)
    except Exception:
        # Diagnostics must never mask the original provider failure.
        pass


def _post_json(url: str, *, headers: dict[str, str] | None = None, body: dict[str, Any], timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, provider: str = '', endpoint: str = '', request_metadata: dict[str, Any] | None = None, max_provider_retries: int = 4, retry_backoff_base: float = 1.0, retry_backoff_max: float = 20.0, error_log_path: str = '') -> dict[str, Any]:
    data = json.dumps(body).encode('utf-8')
    safe_headers = {'content-type': 'application/json', **(headers or {})}
    attempts_allowed = max(1, int(max_provider_retries or 0) + 1)
    last_error: LlmProviderError | None = None
    for attempt_index in range(attempts_allowed):
        attempt = attempt_index + 1
        started = time.time()
        req = urllib.request.Request(url, data=data, headers=safe_headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                text = resp.read().decode('utf-8')
                payload = json.loads(text or '{}')
                if isinstance(payload, dict):
                    payload['_pcse_http'] = {
                        'attempts': attempt,
                        'status_code': getattr(resp, 'status', None) or getattr(resp, 'code', None),
                        'endpoint': endpoint,
                    }
                return payload
        except urllib.error.HTTPError as e:
            text = e.read().decode('utf-8', errors='replace')
            status_code = int(getattr(e, 'code', 0) or 0)
            retryable = _is_retryable_http_status(status_code)
            request_id = ''
            try:
                request_id = clean(e.headers.get('x-request-id') or e.headers.get('request-id') or e.headers.get('cf-ray'))
            except Exception:
                request_id = ''
            message = f'LLM provider HTTP {status_code}: {text[:500]}'
            last_error = LlmProviderError(message, provider=provider, endpoint=endpoint, status_code=status_code, response_text=text, retryable=retryable, attempts=attempt, request_id=request_id, error_log_path=error_log_path)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            status_code = None
            retryable = True
            text = str(e)
            last_error = LlmProviderError(f'LLM provider transport/decode error: {text[:500]}', provider=provider, endpoint=endpoint, status_code=status_code, response_text=text, retryable=retryable, attempts=attempt, error_log_path=error_log_path)
        assert last_error is not None
        elapsed_ms = int((time.time() - started) * 1000)
        should_retry = bool(last_error.retryable and attempt < attempts_allowed)
        log_row = {
            'kind': 'llm_provider_error_v1',
            'timestamp': now_iso(),
            'provider': provider,
            'endpoint': endpoint,
            'url_host': url.split('/')[2] if '://' in url else '',
            'status_code': last_error.status_code,
            'retryable': last_error.retryable,
            'attempt': attempt,
            'attempts_allowed': attempts_allowed,
            'will_retry': should_retry,
            'elapsed_ms': elapsed_ms,
            'request_id': last_error.request_id,
            'request': _summarize_request_body(body),
            'metadata': request_metadata or {},
            'error': str(last_error),
            'response_text_preview': last_error.response_text[:2000],
        }
        _log_provider_error(error_log_path=error_log_path, row=log_row)
        if not should_retry:
            raise last_error
        sleep_seconds = min(float(retry_backoff_max), float(retry_backoff_base) * (2 ** attempt_index))
        time.sleep(max(0.0, sleep_seconds))
    raise last_error or LlmProviderError('LLM provider failed without an error object', provider=provider, endpoint=endpoint, error_log_path=error_log_path)

def _extract_openai_responses_text(payload: dict[str, Any]) -> str:
    """Extract model text from the OpenAI Responses API payload.

    Some Responses API payloads include the same text both in a convenience
    field and nested output items. Earlier versions of this helper could collect
    the same output_text node twice, producing two JSON objects separated by a
    newline and causing `json.loads(...): Extra data`. We now prefer
    output_text when available, otherwise collect each text node once.
    """
    if clean(payload.get('output_text')):
        return clean(payload.get('output_text'))
    texts: list[str] = []
    seen: set[str] = set()

    def add_text(value: Any) -> None:
        text = clean(value)
        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    def visit(value: Any) -> None:
        if value is None or isinstance(value, str):
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            value_type = clean(value.get('type'))
            if value_type in {'output_text', 'text'}:
                add_text(value.get('text'))
                # Do not recurse into the same output text node, or the nested
                # `text` string can be discovered again through generic traversal.
                return
            for item in value.values():
                visit(item)

    visit(payload.get('output'))
    return clean('\n'.join(texts))


def _mock_provider_output(*, scenario: dict[str, Any], agent: dict[str, Any]) -> dict[str, Any]:
    agent_id = clean(agent.get('id') or agent.get('role') or 'agent')
    role = clean(agent.get('role') or agent_id)
    spec = as_dict(as_dict(scenario.get('agent_outputs')).get(agent_id) or as_dict(scenario.get('agent_outputs')).get(role))
    return {
        'output': clean(spec.get('text') or f'{agent_id} mock LLM output'),
        'write_intents': as_list(spec.get('intents')) or [{
            'intent_type': 'append_event',
            'payload': {'id': f'mock_stream_{agent_id}', 'atom_type': 'event', 'title': f'{agent_id} completed', 'canonical_text_en': clean(spec.get('text') or f'{agent_id} completed')},
        }],
    }


def _env_bool(name: str, default: bool = False) -> bool:
    value = clean(os.environ.get(name)).lower()
    if not value:
        return default
    return value in {'1', 'true', 'yes', 'on'}


def _default_send_temperature(provider: str, model: str) -> bool:
    # OpenAI reasoning-family models often reject non-default sampling knobs.
    # For reproducible evals we omit temperature by default on OpenAI Responses.
    if provider == 'openai':
        return False
    return True


def _usage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    usage = as_dict(payload.get('usage'))
    output_details = as_dict(usage.get('output_tokens_details'))
    input_details = as_dict(usage.get('input_tokens_details'))
    return {
        'input_tokens': usage.get('input_tokens'),
        'output_tokens': usage.get('output_tokens'),
        'total_tokens': usage.get('total_tokens'),
        'reasoning_tokens': output_details.get('reasoning_tokens'),
        'cached_input_tokens': input_details.get('cached_tokens'),
    }


def call_llm_provider(*, provider: str = '', model: str = '', prompt: str = '', scenario: dict[str, Any] | None = None, agent: dict[str, Any] | None = None, api_key: str = '', base_url: str = '', temperature: float | None = None, max_output_tokens: int = 1000, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, strict_schema: bool = True, send_temperature: bool | None = None, reasoning_effort: str = '', text_verbosity: str = '', max_provider_retries: int = 4, retry_backoff_base: float = 1.0, retry_backoff_max: float = 20.0, error_log_path: str = '') -> dict[str, Any]:
    load_dotenv()
    provider = clean(provider or os.environ.get('LLM_PROVIDER') or 'openai').lower()
    request_metadata = {
        'scenario_id': as_dict(scenario).get('id'),
        'agent_id': as_dict(agent).get('id') or as_dict(agent).get('role'),
        'prompt_hash': stable_hash(prompt),
    }
    if provider in {'mock', 'fixture'}:
        return {'text': json.dumps(_mock_provider_output(scenario=scenario or {}, agent=agent or {}), ensure_ascii=False), 'raw': {'provider': provider}, 'endpoint': 'mock'}
    if provider in {'openai', 'openai-compatible'}:
        key = api_key or os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_COMPATIBLE_API_KEY')
        url = base_url or (os.environ.get('OPENAI_COMPATIBLE_BASE_URL') if provider == 'openai-compatible' else os.environ.get('OPENAI_BASE_URL')) or 'https://api.openai.com/v1'
        selected_model = model or os.environ.get('LLM_MODEL') or os.environ.get('OPENAI_MODEL') or 'gpt-4.1-mini'
        if not key:
            raise RuntimeError('Missing OPENAI_API_KEY or OPENAI_COMPATIBLE_API_KEY')
        if provider == 'openai':
            body = {
                'model': selected_model,
                'input': [{'role': 'system', 'content': 'You produce valid JSON for an evaluation harness. Never include Markdown fences.'}, {'role': 'user', 'content': prompt}],
                'max_output_tokens': max_output_tokens,
            }
            if send_temperature is None:
                send_temperature = _env_bool('OPENAI_SEND_TEMPERATURE', _default_send_temperature(provider, selected_model))
            if send_temperature and temperature is not None:
                body['temperature'] = temperature
            reasoning_effort = clean(reasoning_effort or os.environ.get('OPENAI_REASONING_EFFORT') or os.environ.get('LLM_REASONING_EFFORT'))
            if reasoning_effort:
                body['reasoning'] = {'effort': reasoning_effort}
            text_verbosity = clean(text_verbosity or os.environ.get('OPENAI_TEXT_VERBOSITY') or os.environ.get('LLM_TEXT_VERBOSITY'))
            if strict_schema:
                body['text'] = {'format': {'type': 'json_schema', 'name': 'context_write_intent_output', 'schema': LLM_WRITE_INTENT_JSON_SCHEMA, 'strict': True}}
            if text_verbosity:
                body.setdefault('text', {})['verbosity'] = text_verbosity
            payload = _post_json(url.rstrip('/') + '/responses', headers={'authorization': f'Bearer {key}'}, body=body, timeout_seconds=timeout_seconds, provider=provider, endpoint='responses', request_metadata=request_metadata, max_provider_retries=max_provider_retries, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max, error_log_path=error_log_path)
            text = _extract_openai_responses_text(payload)
            if not text:
                raise RuntimeError('OpenAI Responses API returned no output text')
            return {'text': text, 'raw': payload, 'endpoint': 'responses', 'request_options': {'sent_temperature': 'temperature' in body, 'temperature': body.get('temperature'), 'reasoning_effort': reasoning_effort, 'text_verbosity': text_verbosity}, 'usage': _usage_summary(payload), 'http': payload.get('_pcse_http', {})}
        body = {
            'model': selected_model,
            'max_tokens': max_output_tokens,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': 'You produce valid JSON for an evaluation harness. Never include Markdown fences.'}, {'role': 'user', 'content': prompt}],
        }
        if send_temperature is None:
            send_temperature = _env_bool('OPENAI_COMPATIBLE_SEND_TEMPERATURE', True)
        if send_temperature and temperature is not None:
            body['temperature'] = temperature
        payload = _post_json(url.rstrip('/') + '/chat/completions', headers={'authorization': f'Bearer {key}'}, body=body, timeout_seconds=timeout_seconds, provider=provider, endpoint='chat_completions', request_metadata=request_metadata, max_provider_retries=max_provider_retries, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max, error_log_path=error_log_path)
        text = clean((payload.get('choices') or [{}])[0].get('message', {}).get('content'))
        if not text:
            raise RuntimeError('OpenAI-compatible chat endpoint returned no message content')
        return {'text': text, 'raw': payload, 'endpoint': 'chat_completions', 'request_options': {'sent_temperature': bool(send_temperature), 'temperature': temperature}, 'usage': _usage_summary(payload), 'http': payload.get('_pcse_http', {})}
    if provider == 'gemini':
        key = api_key or os.environ.get('GEMINI_API_KEY')
        if not key:
            raise RuntimeError('Missing GEMINI_API_KEY')
        selected_model = model or os.environ.get('LLM_MODEL') or os.environ.get('GEMINI_MODEL') or 'gemini-1.5-flash'
        base = base_url or os.environ.get('GEMINI_BASE_URL') or 'https://generativelanguage.googleapis.com/v1beta'
        endpoint = f"{base.rstrip('/')}/models/{selected_model}:generateContent?key={key}"
        generation_config = {'temperature': temperature, 'maxOutputTokens': max_output_tokens, 'responseMimeType': 'application/json'}
        gemini_schema_mode = clean(os.environ.get('GEMINI_RESPONSE_SCHEMA_MODE') or os.environ.get('LLM_GEMINI_RESPONSE_SCHEMA_MODE') or 'sanitize').lower()
        if strict_schema and gemini_schema_mode not in {'none', 'off', 'false', '0'}:
            if gemini_schema_mode in {'raw', 'openai', 'strict'}:
                generation_config['responseSchema'] = LLM_WRITE_INTENT_JSON_SCHEMA
                sent_schema_mode = 'raw_openai_strict'
            else:
                generation_config['responseSchema'] = gemini_compatible_schema(LLM_WRITE_INTENT_JSON_SCHEMA)
                sent_schema_mode = 'gemini_sanitized'
        else:
            sent_schema_mode = 'none'
        payload = _post_json(endpoint, body={'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': generation_config}, timeout_seconds=timeout_seconds, provider=provider, endpoint='generateContent', request_metadata=request_metadata, max_provider_retries=max_provider_retries, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max, error_log_path=error_log_path)
        text = clean('\n'.join(part.get('text', '') for part in (payload.get('candidates') or [{}])[0].get('content', {}).get('parts', [])))
        if not text:
            raise RuntimeError('Gemini API returned no candidate text')
        return {'text': text, 'raw': payload, 'endpoint': 'generateContent', 'request_options': {'response_schema_mode': sent_schema_mode, 'strict_schema': bool(strict_schema)}, 'http': payload.get('_pcse_http', {})}
    raise RuntimeError(f'Unsupported LLM provider: {provider}')


def run_llm_agent(*, agent: dict[str, Any], projection: dict[str, Any], scenario: dict[str, Any], llm_config: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    load_dotenv()
    cfg = llm_config or {}
    agent_id = clean(agent.get('id') or agent.get('role') or 'agent')
    role = clean(agent.get('role') or agent_id)
    model_override = agent_model_override(agent, cfg)
    provider = clean(model_override.get('provider') or cfg.get('provider') or os.environ.get('LLM_PROVIDER') or 'openai')
    model = clean(model_override.get('model') or cfg.get('model') or os.environ.get('LLM_MODEL') or (os.environ.get('GEMINI_MODEL') if provider == 'gemini' else os.environ.get('OPENAI_MODEL')) or '')
    temperature = cfg.get('temperature')
    if temperature is None or clean(temperature).lower() in {'', 'none', 'omit'}:
        temperature = None
    else:
        temperature = float(temperature)
    max_output_tokens = int(cfg.get('max_output_tokens') or cfg.get('maxOutputTokens') or 1000)
    timeout_seconds = int(cfg.get('timeout_seconds') or cfg.get('timeoutMs') or DEFAULT_TIMEOUT_SECONDS)
    max_parse_retries = max(0, int(cfg.get('max_parse_retries') if cfg.get('max_parse_retries') is not None else cfg.get('maxParseRetries', 1)))
    max_provider_retries = max(0, int(cfg.get('max_provider_retries') if cfg.get('max_provider_retries') is not None else cfg.get('maxProviderRetries', os.environ.get('LLM_MAX_PROVIDER_RETRIES', 4))))
    retry_backoff_base = float(cfg.get('retry_backoff_base') if cfg.get('retry_backoff_base') is not None else os.environ.get('LLM_RETRY_BACKOFF_BASE', 1.0))
    retry_backoff_max = float(cfg.get('retry_backoff_max') if cfg.get('retry_backoff_max') is not None else os.environ.get('LLM_RETRY_BACKOFF_MAX', 20.0))
    error_log_path = clean(cfg.get('error_log_path') or os.environ.get('LLM_ERROR_LOG_PATH'))
    include_target_candidates = bool(cfg.get('enable_target_candidates', True))
    certificate_guidance = clean(cfg.get('certificate_guidance') or os.environ.get('LLM_CERTIFICATE_GUIDANCE') or 'guided')
    task = default_llm_agent_task(scenario=scenario, agent=agent)
    if certificate_guidance in {'unguided', 'minimal', 'no_hints'}:
        unguided_task = as_dict(scenario.get('llm_agent_tasks_unguided')).get(agent_id) or as_dict(scenario.get('llm_agent_tasks_unguided')).get(role)
        if unguided_task:
            task = clean(unguided_task)
    prompt = build_llm_agent_prompt(agent=agent, projection=projection, scenario=scenario, include_target_candidates=include_target_candidates, certificate_guidance=certificate_guidance)
    started = time.time()
    last_error: Exception | None = None
    last_raw_text = ''
    for attempt in range(max_parse_retries + 1):
        repair_suffix = '' if attempt == 0 else f"\n\n[REPAIR REQUIRED]\nThe previous response could not be parsed or validated: {last_error}\nReturn a corrected JSON object only.\n{last_raw_text[:4000]}"
        provider_call_cfg = {k: v for k, v in cfg.items() if k not in {'provider', 'model', 'agent_model_specs', 'agent_models', 'prompt', 'scenario', 'agent', 'temperature', 'max_output_tokens', 'maxOutputTokens', 'timeout_seconds', 'timeoutMs', 'max_parse_retries', 'maxParseRetries', 'max_provider_retries', 'maxProviderRetries', 'retry_backoff_base', 'retry_backoff_max', 'error_log_path', 'enable_target_grounding', 'enable_target_candidates', 'certificate_guidance', 'witness_compiler_enabled', 'witness_attach_to_all_intents', 'witness_source_label'}}
        provider_result = call_llm_provider(**provider_call_cfg, provider=provider, model=model, prompt=prompt + repair_suffix, scenario=scenario, agent=agent, temperature=temperature, max_output_tokens=max_output_tokens, timeout_seconds=timeout_seconds, max_provider_retries=max_provider_retries, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max, error_log_path=error_log_path)
        last_raw_text = provider_result['text']
        try:
            parsed = parse_json_object_from_text(provider_result['text'])
            normalized = normalize_llm_output(parsed, agent_id=agent_id, projection_id=projection.get('projection_id', ''), snapshot_id=projection.get('snapshot_id', ''), provider=provider, model=model)
            return {
                'agent_id': agent_id,
                'role': role,
                'projection_id': projection.get('projection_id'),
                'snapshot_id': projection.get('snapshot_id'),
                'output': normalized['output'],
                'agent_task': task,
                'certificate_guidance': certificate_guidance,
                'write_intents': normalized['write_intents'],
                'latency_ms': int((time.time() - started) * 1000),
                'context_tokens': projection.get('metrics', {}).get('context_tokens', 0),
                'llm': {'provider': provider, 'model': model, 'endpoint': provider_result.get('endpoint'), 'temperature': temperature, 'max_output_tokens': max_output_tokens, 'prompt_tokens_est': estimate_tokens(prompt), 'output_tokens_est': estimate_tokens(provider_result['text']), 'prompt_hash': stable_hash(prompt), 'schema_version': OUTPUT_SCHEMA_VERSION, 'parse_attempts': attempt + 1, 'request_options': {**provider_result.get('request_options', {}), 'enable_target_candidates': include_target_candidates, 'enable_target_grounding': cfg.get('enable_target_grounding', True), 'certificate_guidance': certificate_guidance, 'mixed_provider_routing': bool(model_override)}, 'api_usage': provider_result.get('usage', {}), 'provider_http': provider_result.get('http', {}), 'error_log_path': error_log_path},
            }
        except Exception as exc:
            last_error = exc
            if attempt == max_parse_retries:
                # Surface raw model text in CLI errors. This is especially useful
                # for structured-output debugging, where providers may return a
                # valid JSON object plus extra text or duplicate JSON blocks.
                if hasattr(exc, 'raw_llm_text'):
                    exc.raw_llm_text = last_raw_text
                    raise
                from wccu_eval.agents.llm_output_schema import LlmOutputValidationError
                raise LlmOutputValidationError(str(exc), raw_llm_text=last_raw_text) from exc
    raise last_error or RuntimeError('LLM agent failed without an error')
