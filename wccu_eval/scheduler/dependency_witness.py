from __future__ import annotations

"""Runtime read-witness compilation for WCCU experiments.

The key distinction from model-supplied certificates is that this module treats
read provenance as a harness/runtime artifact.  The compiler consumes projected
atoms plus explicit reads logged by deterministic fixtures, tools, retrievers,
workspace/file readers, or handoff readers, and emits compact witness objects
that can be attached to proposed context mutations before WCCU verification.

The implementation is intentionally lightweight, but it gives experiments a
single place to stress the assumption that read witnesses are complete.  The
``degrade_*`` knobs are deterministic and support missing-read and overbroad-read
ablations without changing the verifier.
"""

import random
from dataclasses import dataclass
from typing import Any

from wccu_eval.utils import as_dict, as_list, clean, stable_hash

READ_SOURCE_FIELDS = (
    'read_atoms',
    'read_dependencies',
    'read_set',
    'retrieval_reads',
    'tool_reads',
    'file_reads',
    'workspace_reads',
    'handoff_reads',
    'materialized_view_reads',
)


@dataclass(frozen=True)
class WitnessCompileConfig:
    enabled: bool = True
    attach_to_all_intents: bool = True
    drop_rate: float = 0.0
    fake_dependency_count: int = 0
    source_label: str = 'runtime_read_set_witness'
    seed: str = ''

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None = None) -> 'WitnessCompileConfig':
        value = as_dict(value)
        return cls(
            enabled=bool(value.get('witness_compiler_enabled', value.get('enabled', True))),
            attach_to_all_intents=bool(value.get('witness_attach_to_all_intents', value.get('attach_to_all_intents', True))),
            drop_rate=max(0.0, min(1.0, float(value.get('witness_drop_rate', value.get('drop_rate', 0.0)) or 0.0))),
            fake_dependency_count=max(0, int(value.get('witness_fake_dependency_count', value.get('fake_dependency_count', 0)) or 0)),
            source_label=clean(value.get('witness_source_label') or value.get('source_label') or 'runtime_read_set_witness'),
            seed=clean(value.get('witness_seed') or value.get('seed') or ''),
        )


def _projection_atom_index(projection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        clean(as_dict(atom).get('id') or as_dict(atom).get('target_id')): as_dict(atom)
        for atom in as_list(projection.get('atoms'))
        if clean(as_dict(atom).get('id') or as_dict(atom).get('target_id'))
    }


def _target_id_from_read(row: Any) -> str:
    if isinstance(row, str):
        return clean(row)
    row = as_dict(row)
    return clean(row.get('target_id') or row.get('atom_id') or row.get('id') or row.get('view_target_id') or row.get('file_path'))


def _normalize_read(row: Any, *, atom_index: dict[str, dict[str, Any]], snapshot_id: str, projection_id: str, source: str) -> dict[str, Any] | None:
    tid = _target_id_from_read(row)
    if not tid:
        return None
    raw = {'target_id': tid} if isinstance(row, str) else as_dict(row)
    atom = atom_index.get(tid) or as_dict(raw.get('atom'))
    return {
        'target_id': tid,
        'snapshot_id': clean(raw.get('snapshot_id') or snapshot_id),
        'view_id': clean(raw.get('view_id') or projection_id),
        'expected_status': clean(raw.get('expected_status') or atom.get('status') or 'active'),
        'freshness_required': bool(raw.get('freshness_required', True)),
        'reason': clean(raw.get('reason') or f'{source} runtime read'),
        'source': source,
    }


def _collect_declared_reads(*, agent: dict[str, Any], result: dict[str, Any], scenario: dict[str, Any]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    agent_id = clean(result.get('agent_id') or agent.get('id') or agent.get('role'))
    role = clean(result.get('role') or agent.get('role') or agent_id)

    # Agent result or deterministic fixture can log explicit runtime reads.
    for container_name, container in [
        ('agent_result', result),
        ('agent_spec', as_dict(as_dict(scenario.get('agent_outputs')).get(agent_id) or as_dict(scenario.get('agent_outputs')).get(role))),
        ('agent_config', agent),
    ]:
        container = as_dict(container)
        for field in READ_SOURCE_FIELDS:
            for row in as_list(container.get(field)):
                rows.append((field if container_name == 'agent_result' else f'{container_name}.{field}', row))
        for row in as_list(as_dict(container.get('execution_witness')).get('read_dependencies')):
            rows.append((f'{container_name}.execution_witness', row))
        for row in as_list(as_dict(container.get('runtime_witness')).get('read_dependencies')):
            rows.append((f'{container_name}.runtime_witness', row))

    # Existing scenario-level WCCU dependency fixture becomes a harness-visible
    # read witness only when the experiment explicitly enables the compiler.  It
    # is useful for deterministic completeness ablations and is reported as a
    # fixture/runtime source rather than a model certificate.
    declared = as_dict(scenario.get('wccu_read_dependencies'))
    for row in as_list(declared.get(agent_id)) + as_list(declared.get(role)) + as_list(declared.get('*')):
        rows.append(('scenario.wccu_read_dependencies', row))
    return rows


def compile_dependency_witness(*, agent: dict[str, Any], projection: dict[str, Any], result: dict[str, Any], scenario: dict[str, Any], config: WitnessCompileConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if isinstance(config, WitnessCompileConfig) else WitnessCompileConfig.from_mapping(as_dict(config))
    if not cfg.enabled:
        return {'enabled': False, 'read_atoms': [], 'read_dependencies': [], 'source_counts': {}, 'dropped_count': 0, 'fake_dependency_count': 0}

    atom_index = _projection_atom_index(projection)
    snapshot_id = clean(projection.get('snapshot_id'))
    projection_id = clean(projection.get('projection_id'))
    deps_by_id: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    for source, row in _collect_declared_reads(agent=agent, result=result, scenario=scenario):
        dep = _normalize_read(row, atom_index=atom_index, snapshot_id=snapshot_id, projection_id=projection_id, source=source)
        if not dep:
            continue
        deps_by_id[dep['target_id']] = dep
        source_counts[source] = source_counts.get(source, 0) + 1

    deps = list(deps_by_id.values())
    rng = random.Random(cfg.seed or stable_hash({'agent': agent.get('id'), 'projection': projection_id, 'deps': [d['target_id'] for d in deps]}))
    kept: list[dict[str, Any]] = []
    dropped = 0
    for dep in deps:
        if cfg.drop_rate > 0 and rng.random() < cfg.drop_rate:
            dropped += 1
            continue
        kept.append(dep)

    for idx in range(cfg.fake_dependency_count):
        fake_id = f'fake_dep_{stable_hash(f"{projection_id}:{idx}:{cfg.seed}", 8)}'
        kept.append({
            'target_id': fake_id,
            'snapshot_id': snapshot_id,
            'view_id': projection_id,
            'expected_status': 'active',
            'freshness_required': True,
            'reason': f'{cfg.source_label} injected overbroad dependency',
            'source': 'synthetic_overbroad_witness',
        })

    return {
        'enabled': True,
        'witness_id': f"witness_{stable_hash({'projection': projection_id, 'agent': agent.get('id'), 'reads': [d['target_id'] for d in kept]})}",
        'snapshot_id': snapshot_id,
        'projection_id': projection_id,
        'read_atoms': [d['target_id'] for d in kept],
        'read_dependencies': kept,
        'source_counts': source_counts,
        'declared_count': len(deps),
        'kept_count': len(kept),
        'dropped_count': dropped,
        'fake_dependency_count': cfg.fake_dependency_count,
        'drop_rate': cfg.drop_rate,
        'source_label': cfg.source_label,
    }


def attach_dependency_witness_to_result(*, agent: dict[str, Any], projection: dict[str, Any], result: dict[str, Any], scenario: dict[str, Any], config: WitnessCompileConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if isinstance(config, WitnessCompileConfig) else WitnessCompileConfig.from_mapping(as_dict(config))
    witness = compile_dependency_witness(agent=agent, projection=projection, result=result, scenario=scenario, config=cfg)
    if not witness.get('enabled'):
        return result
    intents = []
    for intent in as_list(result.get('write_intents')):
        intent = as_dict(intent)
        if cfg.attach_to_all_intents or not (intent.get('execution_witness') or intent.get('read_witness')):
            intent = {
                **intent,
                'execution_witness': as_dict(intent.get('execution_witness')) or witness,
                'read_witness': as_dict(intent.get('read_witness')) or witness,
                'witness_compile_metadata': {
                    'witness_id': witness.get('witness_id'),
                    'declared_count': witness.get('declared_count'),
                    'kept_count': witness.get('kept_count'),
                    'dropped_count': witness.get('dropped_count'),
                    'fake_dependency_count': witness.get('fake_dependency_count'),
                    'source_counts': witness.get('source_counts'),
                },
            }
        intents.append(intent)
    return {
        **result,
        'write_intents': intents,
        'execution_witness': witness,
        'read_witness': witness,
        'witness_compile_metadata': {
            'witness_id': witness.get('witness_id'),
            'declared_count': witness.get('declared_count'),
            'kept_count': witness.get('kept_count'),
            'dropped_count': witness.get('dropped_count'),
            'fake_dependency_count': witness.get('fake_dependency_count'),
            'source_counts': witness.get('source_counts'),
        },
    }
