from __future__ import annotations

import json
import re
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, stable_hash
from wccu_eval.scheduler.wccu import minimal_certificate, normalize_certificate

OUTPUT_SCHEMA_VERSION = 'write_intent_schema_v2_openai_strict'
ALLOWED_INTENT_TYPES = {'append_event', 'upsert_atom', 'patch_atom', 'retract_atom', 'assert_link', 'upsert_link', 'patch_link', 'retract_link'}
ALLOWED_RISK_LEVELS = {'low', 'medium', 'high', 'critical'}
ALLOWED_AUTHORITIES = {'agent', 'researcher', 'builder', 'reviewer', 'user', 'system'}
ALLOWED_COMMIT_MODES = {'auto', 'review_required', 'proposal', 'none'}

# OpenAI Structured Outputs strict mode requires every object in the schema to set
# additionalProperties:false and every declared field to be listed in required.
# Optional payload values are represented as empty strings or empty arrays, then
# normalized by the local validator below. This keeps the provider-facing schema
# strict while preserving the runtime's more permissive internal normalization.
PAYLOAD_PROPERTIES: dict[str, Any] = {
    'id': {'type': 'string'},
    'target_id': {'type': 'string'},
    'atom_id': {'type': 'string'},
    'stream_id': {'type': 'string'},
    'atom_type': {'type': 'string'},
    'title': {'type': 'string'},
    'canonical_text_en': {'type': 'string'},
    'text_original': {'type': 'string'},
    'reason': {'type': 'string'},
    'file_path': {'type': 'string'},
    'tags': {'type': 'array', 'items': {'type': 'string'}},
    'risk': {'type': 'string'},
}


READ_DEPENDENCY_PROPERTIES: dict[str, Any] = {
    'target_id': {'type': 'string'},
    'view_id': {'type': 'string'},
    'snapshot_id': {'type': 'string'},
    'expected_status': {'type': 'string'},
    'expected_text_hash': {'type': 'string'},
    'freshness_required': {'type': 'boolean'},
    'reason': {'type': 'string'},
}

TARGET_CERTIFICATE_PROPERTIES: dict[str, Any] = {
    'claimed_target_id': {'type': 'string'},
    'raw_target': {'type': 'string'},
    'grounding_rationale': {'type': 'string'},
    'confidence': {'type': 'number'},
}

DELTA_CONTRACT_PROPERTIES: dict[str, Any] = {
    'delta_type': {'type': 'string'},
    'semantic_direction': {'type': 'string'},
    'affected_view_ids': {'type': 'array', 'items': {'type': 'string'}},
    'invalidates_views': {'type': 'boolean'},
    'summary': {'type': 'string'},
}

AUTHORITY_CERTIFICATE_PROPERTIES: dict[str, Any] = {
    'actor_authority': {'type': 'string'},
    'required_authority': {'type': 'string'},
    'authority_rationale': {'type': 'string'},
}

WCCU_PRECONDITION_PROPERTIES: dict[str, Any] = {
    'base_snapshot_id': {'type': 'string'},
    'freshness_required': {'type': 'boolean'},
    'no_retracted_dependencies': {'type': 'boolean'},
    'min_target_confidence': {'type': 'number'},
    'requires_review_if_invalid': {'type': 'boolean'},
}

CERTIFICATE_PROPERTIES: dict[str, Any] = {
    'schema_version': {'type': 'string'},
    'certificate_id': {'type': 'string'},
    'certificate_mode': {'type': 'string'},
    'read_dependencies': {
        'type': 'array',
        'items': {
            'type': 'object',
            'additionalProperties': False,
            'required': list(READ_DEPENDENCY_PROPERTIES.keys()),
            'properties': READ_DEPENDENCY_PROPERTIES,
        },
    },
    'target_certificate': {
        'type': 'object',
        'additionalProperties': False,
        'required': list(TARGET_CERTIFICATE_PROPERTIES.keys()),
        'properties': TARGET_CERTIFICATE_PROPERTIES,
    },
    'delta_contract': {
        'type': 'object',
        'additionalProperties': False,
        'required': list(DELTA_CONTRACT_PROPERTIES.keys()),
        'properties': DELTA_CONTRACT_PROPERTIES,
    },
    'authority_certificate': {
        'type': 'object',
        'additionalProperties': False,
        'required': list(AUTHORITY_CERTIFICATE_PROPERTIES.keys()),
        'properties': AUTHORITY_CERTIFICATE_PROPERTIES,
    },
    'preconditions': {
        'type': 'object',
        'additionalProperties': False,
        'required': list(WCCU_PRECONDITION_PROPERTIES.keys()),
        'properties': WCCU_PRECONDITION_PROPERTIES,
    },
}

INTENT_PROPERTIES: dict[str, Any] = {
    'intent_type': {'type': 'string', 'enum': sorted(ALLOWED_INTENT_TYPES)},
    'risk': {'type': 'string', 'enum': sorted(ALLOWED_RISK_LEVELS)},
    'authority': {'type': 'string', 'enum': sorted(ALLOWED_AUTHORITIES)},
    'commit_mode': {'type': 'string', 'enum': sorted(ALLOWED_COMMIT_MODES)},
    'payload': {
        'type': 'object',
        'additionalProperties': False,
        'required': list(PAYLOAD_PROPERTIES.keys()),
        'properties': PAYLOAD_PROPERTIES,
    },
    'certificate': {
        'type': 'object',
        'additionalProperties': False,
        'required': list(CERTIFICATE_PROPERTIES.keys()),
        'properties': CERTIFICATE_PROPERTIES,
    },
}

LLM_WRITE_INTENT_JSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['output', 'write_intents'],
    'properties': {
        'output': {'type': 'string'},
        'write_intents': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': list(INTENT_PROPERTIES.keys()),
                'properties': INTENT_PROPERTIES,
            },
        },
    },
}


def _schema_object_errors(schema: Any, path: str = '$') -> list[str]:
    """Return provider-facing JSON Schema strictness errors.

    This is intentionally small and targets the OpenAI Structured Outputs subset
    used by this package: all object schemas must have additionalProperties:false
    and all defined properties must be required.
    """
    errors: list[str] = []
    if isinstance(schema, dict):
        if schema.get('type') == 'object':
            if schema.get('additionalProperties') is not False:
                errors.append(f'{path}: object schema must set additionalProperties=false')
            props = as_dict(schema.get('properties'))
            required = set(as_list(schema.get('required')))
            missing = [key for key in props if key not in required]
            if missing:
                errors.append(f'{path}: object schema properties missing from required: {missing}')
        for key, value in schema.items():
            if key == 'properties' and isinstance(value, dict):
                for prop_key, prop_schema in value.items():
                    errors.extend(_schema_object_errors(prop_schema, f'{path}.properties.{prop_key}'))
            elif key == 'items':
                errors.extend(_schema_object_errors(value, f'{path}.items'))
            elif key in {'$defs', 'definitions'} and isinstance(value, dict):
                for def_key, def_schema in value.items():
                    errors.extend(_schema_object_errors(def_schema, f'{path}.{key}.{def_key}'))
            elif isinstance(value, (dict, list)):
                errors.extend(_schema_object_errors(value, f'{path}.{key}'))
    elif isinstance(schema, list):
        for idx, item in enumerate(schema):
            errors.extend(_schema_object_errors(item, f'{path}[{idx}]'))
    return errors


def assert_openai_strict_schema(schema: dict[str, Any] = LLM_WRITE_INTENT_JSON_SCHEMA) -> None:
    errors = _schema_object_errors(schema)
    if errors:
        raise AssertionError('Invalid strict JSON schema: ' + '; '.join(errors))


assert_openai_strict_schema()


def schema_instruction_text() -> str:
    return '\n'.join([
        'Return exactly one JSON object and no Markdown fences.',
        'The JSON object must match this shape:',
        json.dumps(LLM_WRITE_INTENT_JSON_SCHEMA, ensure_ascii=False, indent=2),
        'Do not add commentary outside the JSON object.',
        'Do not choose the concurrency policy. The runtime will select it.',
        'For each write intent, include a compact witness-carrying context update (WCCU) certificate.',
        'The certificate should identify read_dependencies, target_certificate, delta_contract, authority_certificate, and preconditions.',
        'If there are no read dependencies, use an empty read_dependencies array. Use confidence between 0 and 1.',
        'All schema fields are required for provider strictness. Use an empty string or empty tags array for fields that do not apply.',
        'For commit_mode, use "none" unless the task explicitly asks for auto, review_required, or proposal.',
        'Use payload.target_id for the stable substrate object id. For atom writes, set payload.id, payload.atom_id, and payload.target_id to the same stable id when possible.',
        'Use the exact atom id, stream id, and file path requested by the task when the task specifies one.',
    ])


def _normalize_intent_type(value: Any) -> str:
    v = clean(value).lower()
    return {'assert_atom': 'upsert_atom', 'delete_atom': 'retract_atom', 'link': 'upsert_link'}.get(v, v)


def _normalize_risk(value: Any) -> str:
    v = clean(value or 'low').lower()
    return v if v in ALLOWED_RISK_LEVELS else 'low'


def _normalize_authority(value: Any) -> str:
    v = clean(value or 'agent').lower()
    return v if v in ALLOWED_AUTHORITIES else 'agent'


def _normalize_payload(payload: Any) -> dict[str, Any]:
    p = as_dict(payload)
    atom_type = clean(p.get('atom_type') or p.get('type') or 'memory').lower()
    target_id = clean(p.get('target_id') or p.get('id') or p.get('atom_id') or p.get('stream_id') or p.get('file_path'))
    title = clean(p.get('title') or p.get('canonical_text_en') or p.get('reason') or atom_type)[:160] or atom_type
    normalized = {
        **p,
        **({'id': target_id, 'target_id': target_id} if target_id else {}),
        'atom_type': atom_type,
        'title': title,
        'canonical_text_en': clean(p.get('canonical_text_en') or p.get('text') or p.get('summary') or ''),
        'text_original': clean(p.get('text_original') or ''),
        'reason': clean(p.get('reason') or ''),
        'file_path': clean(p.get('file_path') or ''),
        'tags': list(dict.fromkeys([clean(x) for x in as_list(p.get('tags')) if clean(x)])),
    }
    for key in PAYLOAD_PROPERTIES:
        if key == 'tags':
            normalized[key] = as_list(normalized.get(key))
        else:
            normalized[key] = clean(normalized.get(key))
    return normalized


def validate_llm_output_shape(value: Any) -> dict[str, Any]:
    errors: list[str] = []
    row = as_dict(value)
    if not clean(row.get('output')):
        errors.append('output must be a non-empty string')
    intents = as_list(row.get('write_intents'))
    if not intents:
        errors.append('write_intents must contain at least one item')
    if len(intents) > 8:
        errors.append('write_intents must contain no more than eight items')
    for idx, raw in enumerate(intents):
        intent = as_dict(raw)
        intent_type = _normalize_intent_type(intent.get('intent_type'))
        if intent_type not in ALLOWED_INTENT_TYPES:
            errors.append(f'write_intents[{idx}].intent_type is not allowed: {intent.get("intent_type")}')
        payload = _normalize_payload(intent.get('payload'))
        if not payload.get('atom_type'):
            errors.append(f'write_intents[{idx}].payload.atom_type is required')
        if not payload.get('title'):
            errors.append(f'write_intents[{idx}].payload.title is required')
        needs_target = intent_type in {'upsert_atom', 'patch_atom', 'retract_atom', 'upsert_link', 'patch_link', 'retract_link'}
        if needs_target and not clean(payload.get('target_id') or payload.get('id') or payload.get('atom_id') or payload.get('link_id')):
            errors.append(f'write_intents[{idx}] requires payload.target_id, payload.id, payload.atom_id, or payload.link_id')
        if intent_type == 'append_event' and not clean(payload.get('id') or payload.get('stream_id')):
            errors.append(f'write_intents[{idx}] append_event should include payload.id or payload.stream_id')
    return {'ok': not errors, 'errors': errors}


class LlmOutputValidationError(ValueError):
    def __init__(self, message: str, *, validation_errors: list[str] | None = None, normalized: Any = None, raw_llm_text: str = ''):
        super().__init__(message)
        self.validation_errors = validation_errors or []
        self.normalized = normalized
        self.raw_llm_text = raw_llm_text


def normalize_llm_output(value: Any, *, agent_id: str = 'agent', projection_id: str = '', snapshot_id: str = '', provider: str = '', model: str = '') -> dict[str, Any]:
    row = as_dict(value)
    normalized = {
        'output': clean(row.get('output')),
        'write_intents': [],
    }
    for index, raw in enumerate(as_list(row.get('write_intents'))):
        intent = as_dict(raw)
        intent_type = _normalize_intent_type(intent.get('intent_type'))
        payload = _normalize_payload(intent.get('payload'))
        risk = _normalize_risk(intent.get('risk') or payload.get('risk'))
        authority = _normalize_authority(intent.get('authority'))
        id_seed = f'{agent_id}:{projection_id}:{index}:{intent_type}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}'
        row_intent = {
            'id': clean(intent.get('id')) or f'intent_{stable_hash(id_seed)}',
            'intent_type': intent_type,
            'risk': risk,
            'authority': authority,
            'actor': 'user' if authority == 'user' else f'agent:{agent_id}',
            'payload': payload,
            'preconditions': {'base_snapshot_id': snapshot_id, **as_dict(intent.get('preconditions'))},
            'source': {'kind': 'llm_agent_output', 'schema_version': OUTPUT_SCHEMA_VERSION, 'agent_id': agent_id, 'projection_id': projection_id, 'provider': provider, 'model': model, **as_dict(intent.get('source'))},
        }
        # Backward compatibility: intent-only outputs are upgraded to a compact
        # proof-carrying context transaction certificate. Strict provider-facing
        # schemas ask the model to supply it, but deterministic fixtures and
        # older logs can omit it safely.
        row_intent['certificate'] = normalize_certificate({**row_intent, 'certificate': as_dict(intent.get('certificate'))}) if intent.get('certificate') else minimal_certificate(row_intent)
        commit_mode = clean(intent.get('commit_mode')).lower()
        if commit_mode and commit_mode != 'none':
            row_intent['commit_mode'] = commit_mode
        normalized['write_intents'].append(row_intent)
    validation = validate_llm_output_shape(normalized)
    if not validation['ok']:
        raise LlmOutputValidationError('Invalid LLM structured output: ' + '; '.join(validation['errors']), validation_errors=validation['errors'], normalized=normalized)
    return normalized


def parse_json_object_from_text(text: Any = '') -> dict[str, Any]:
    raw = clean(text)
    if not raw:
        raise ValueError('LLM returned empty text')
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw, re.I)
    candidate = fenced.group(1).strip() if fenced else raw

    def raw_decode_first_object(value: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        idx = value.find('{')
        if idx < 0:
            raise ValueError('Could not find a JSON object in LLM output')
        obj, end = decoder.raw_decode(value[idx:])
        if not isinstance(obj, dict):
            raise ValueError('Parsed JSON value is not an object')
        trailing = value[idx + end:].strip()
        # Allow duplicated or trailing JSON/text, but only after successfully
        # parsing the first object. This handles small/nano model responses that
        # sometimes emit the same JSON twice, and also protects against OpenAI
        # output extraction accidentally concatenating duplicate output_text nodes.
        return obj

    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError('Parsed JSON value is not an object')
        return parsed
    except json.JSONDecodeError:
        return raw_decode_first_object(candidate)
