from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from wccu_eval.env import load_dotenv
from wccu_eval.utils import clean

PLACEHOLDER_MARKERS = (
    'your-key',
    'sk-your-key',
    'your-gemini-key',
    'changeme',
    'replace-me',
    'placeholder',
)


def _redact(value: str) -> str:
    value = clean(value)
    if not value:
        return ''
    if len(value) <= 8:
        return '***'
    return f'{value[:4]}...{value[-4:]}'


def _provider_key(provider: str) -> str:
    p = clean(provider).lower()
    if p == 'openai':
        return 'OPENAI_API_KEY'
    if p == 'gemini':
        return 'GEMINI_API_KEY'
    if p in {'openai-compatible', 'openai_compatible', 'compatible'}:
        return 'OPENAI_COMPATIBLE_API_KEY'
    return ''


def validate_real_llm_config(*, provider: str = '', model: str = '', require_key: bool = True) -> dict[str, Any]:
    load_dotenv()
    provider = clean(provider or os.environ.get('LLM_PROVIDER') or 'openai').lower()
    model = clean(model or os.environ.get('LLM_MODEL') or '')
    errors: list[str] = []
    warnings: list[str] = []

    if provider in {'mock', 'deterministic'}:
        errors.append(f'LLM_PROVIDER={provider!r} is not a real provider. Use openai, gemini, or openai-compatible for paper runs.')
    if not model:
        errors.append('LLM_MODEL is empty. Set an explicit model id for real-LLM paper runs.')
    if 'mock' in model.lower():
        errors.append(f'LLM_MODEL={model!r} looks like a mock model.')

    key_name = _provider_key(provider)
    key_value = clean(os.environ.get(key_name) or '') if key_name else ''
    if not key_name:
        errors.append(f'Unsupported LLM_PROVIDER={provider!r}. Expected openai, gemini, or openai-compatible.')
    elif require_key:
        if not key_value:
            errors.append(f'{key_name} is not set.')
        elif any(marker in key_value.lower() for marker in PLACEHOLDER_MARKERS):
            errors.append(f'{key_name} still looks like a placeholder value.')

    if provider == 'openai-compatible' and not clean(os.environ.get('OPENAI_COMPATIBLE_BASE_URL') or ''):
        warnings.append('OPENAI_COMPATIBLE_BASE_URL is empty; set it if you are using a non-OpenAI endpoint.')

    payload = {
        'ok': not errors,
        'provider': provider,
        'model': model,
        'api_key_env': key_name,
        'api_key_present': bool(key_value),
        'api_key_redacted': _redact(key_value),
        'errors': errors,
        'warnings': warnings,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate that a paper run is configured for a real LLM provider, not mock/deterministic mode.')
    parser.add_argument('--provider', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--no-require-key', dest='require_key', action='store_false')
    args = parser.parse_args(argv)
    payload = validate_real_llm_config(provider=args.provider, model=args.model, require_key=args.require_key)
    print(json.dumps(payload, indent=2))
    return 0 if payload['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
