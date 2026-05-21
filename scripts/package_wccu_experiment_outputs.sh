#!/usr/bin/env bash
# Package experiment outputs into a safe zip to share for analysis.
# This script intentionally excludes .env, .git, API keys, caches, and unrelated output directories.
# Usage:
#   ./scripts/package_wccu_experiment_outputs.sh --tag wccu_full_20260519_120000
#   ./scripts/package_wccu_experiment_outputs.sh --tag "$EXP_TAG" --out /tmp/wccu_outputs.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TAG="${EXP_TAG:-}"
OUT=""
INCLUDE_SOURCE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)
      TAG="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --include-source)
      INCLUDE_SOURCE="1"
      shift
      ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "ERROR: provide --tag or set EXP_TAG." >&2
  exit 2
fi

if [ -z "$OUT" ]; then
  OUT="${TAG}_outputs.zip"
fi

python - "$TAG" "$OUT" "$INCLUDE_SOURCE" <<'PY'
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TAG = sys.argv[1]
OUT = Path(sys.argv[2])
INCLUDE_SOURCE = sys.argv[3] == '1'
ROOT = Path.cwd()

SECRET_NAME_FRAGMENTS = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL')
NEVER_INCLUDE_PARTS = {
    '.git', '.env', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
    'node_modules', '.DS_Store',
}
NEVER_INCLUDE_SUFFIXES = {'.pyc', '.pyo'}

for d in [ROOT / 'results' / TAG, ROOT / 'analysis' / TAG, ROOT / 'data' / TAG, ROOT / 'logs']:
    d.mkdir(parents=True, exist_ok=True)

# Write a small sanitized environment manifest. Values of secrets are not stored.
manifest = {
    'exp_tag': TAG,
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version,
    'platform': platform.platform(),
    'cwd': str(ROOT),
    'git_commit': None,
    'known_llm_config_keys_present': {
        key: bool(os.environ.get(key))
        for key in ['LLM_PROVIDER', 'LLM_MODEL', 'LLM_MODELS', 'WCCU_ENV_FILE']
    },
}
try:
    manifest['git_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    manifest['git_commit'] = None

manifest_path = ROOT / 'results' / TAG / 'run_environment_sanitized.json'
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

paths: list[Path] = []
for rel in [Path('results') / TAG, Path('analysis') / TAG, Path('data') / TAG]:
    if (ROOT / rel).exists():
        paths.append(ROOT / rel)

log_dir = ROOT / 'logs'
if log_dir.exists():
    for p in sorted(log_dir.glob(f'{TAG}_*.log')):
        paths.append(p)
    # Include provider check log if it was written in a tag-specific run directory only via error log.
    for p in sorted(log_dir.glob('check_llm_provider.log')):
        paths.append(p)

for rel in ['README.md', 'README_REPRODUCE.md', '.env.example', 'pyproject.toml']:
    p = ROOT / rel
    if p.exists():
        paths.append(p)

for rel in ['scripts/prepare_cooperbench_data.sh',
            'scripts/lib/load_wccu_env.sh',
            'scripts/run_wccu_primary_real_llm_experiments.sh',
            'scripts/run_wccu_true_frozen_replay.sh',
            'scripts/run_wccu_cooperbench_derived.sh',
            'scripts/run_wccu_gemini_only_frozen_replay.sh',
            'scripts/run_wccu_mixed_provider_frozen_replay.sh',
            'scripts/run_wccu_provider_robustness_suite.sh',
            'scripts/run_wccu_main_llm_obligation.sh', 'scripts/run_wccu_certificate_guidance_ablation.sh',
            'scripts/run_wccu_multi_model.sh', 'scripts/run_wccu_shared_context.sh',
            'scripts/run_wccu_offline_sanity.sh', 'scripts/run_wccu_llm_smoke.sh',
            'scripts/run_wccu_stress_and_target_ablation.sh', 'scripts/package_wccu_experiment_outputs.sh']:
    p = ROOT / rel
    if p.exists():
        paths.append(p)

if INCLUDE_SOURCE:
    for rel in ['wccu_eval', 'tests']:
        p = ROOT / rel
        if p.exists():
            paths.append(p)

included: list[str] = []
skipped_secret_like: list[str] = []

def safe_to_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & NEVER_INCLUDE_PARTS:
        return False
    if path.suffix in NEVER_INCLUDE_SUFFIXES:
        return False
    name_upper = path.name.upper()
    if any(fragment in name_upper for fragment in SECRET_NAME_FRAGMENTS):
        # Allow source files whose names mention token usage, but avoid likely secret material.
        if path.suffix.lower() not in {'.py', '.md', '.tex', '.csv', '.json', '.jsonl', '.txt', '.sh', '.toml'}:
            skipped_secret_like.append(str(rel))
            return False
    if path.name == '.env':
        return False
    return True

def add_path(zf: zipfile.ZipFile, path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in sorted(path.rglob('*')):
            if child.is_file() and safe_to_include(child):
                arc = child.relative_to(ROOT)
                zf.write(child, arcname=str(arc))
                included.append(str(arc))
    elif path.is_file() and safe_to_include(path):
        arc = path.relative_to(ROOT)
        zf.write(path, arcname=str(arc))
        included.append(str(arc))

OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in paths:
        add_path(zf, p)
    summary = {
        'exp_tag': TAG,
        'file_count': len(included),
        'include_source': INCLUDE_SOURCE,
        'excluded_by_design': ['.env', '.git/', 'API keys', 'virtualenvs', 'caches'],
        'top_level_expected': [f'results/{TAG}', f'analysis/{TAG}', f'data/{TAG}', 'logs/*.log'],
    }
    zf.writestr('PACKAGE_MANIFEST.json', json.dumps(summary, indent=2, ensure_ascii=False))

print(json.dumps({
    'ok': True,
    'out': str(OUT),
    'file_count': len(included),
    'include_source': INCLUDE_SOURCE,
}, indent=2, ensure_ascii=False))
PY
