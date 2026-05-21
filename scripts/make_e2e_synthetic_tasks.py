#!/usr/bin/env python3
"""
Generate synthetic end-to-end patch/test tasks for WCCU evaluation.

Output:
  data/e2e_patch_tasks.jsonl
  patches/e2e_synth/*.patch

Each JSONL row is compatible with the e2e patch/test runner:
  - base_files creates a temporary mini-repository
  - patches contains prepared multi-agent patch proposals
  - test_commands runs pytest after the policy decides which patches are auto-applied

Usage:
  python scripts/make_e2e_synthetic_tasks.py --count 20
  python scripts/make_e2e_synthetic_tasks.py --count 50 --out data/e2e_patch_tasks_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def unified_patch(path: str, before: str, after: str) -> str:
    """Build a git-apply-compatible unified patch for one file."""
    import difflib

    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    from_file = f"a/{path}" if before else "/dev/null"
    to_file = f"b/{path}" if after else "/dev/null"
    diff = list(difflib.unified_diff(before_lines, after_lines, fromfile=from_file, tofile=to_file, lineterm=""))
    normalized: List[str] = []
    for line in diff:
        if not line.endswith("\n"):
            line += "\n"
        normalized.append(line)
    header = f"diff --git a/{path} b/{path}\n"
    if not before and after:
        header += "new file mode 100644\n"
    if before and not after:
        header += "deleted file mode 100644\n"
    return header + "".join(normalized)


def multi_file_patch(changes: Iterable[Tuple[str, str, str]]) -> str:
    return "\n".join(unified_patch(path, before, after) for path, before, after in changes)


def write_patch(patch_dir: Path, name: str, patch_text: str) -> str:
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / name
    patch_path.write_text(patch_text, encoding="utf-8")
    return str(patch_path)


def common_base_files(task_id: str) -> Dict[str, str]:
    runner = '\nimport importlib.util\nimport pathlib\nimport sys\n\nROOT = pathlib.Path(__file__).resolve().parents[1]\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\nfailures = []\nfor path in sorted(pathlib.Path(__file__).resolve().parent.glob("test_*.py")):\n    spec = importlib.util.spec_from_file_location(path.stem, path)\n    module = importlib.util.module_from_spec(spec)\n    try:\n        assert spec.loader is not None\n        spec.loader.exec_module(module)\n        for name, value in sorted(vars(module).items()):\n            if name.startswith("test_") and callable(value):\n                value()\n    except Exception as exc:\n        failures.append(f"{path.name}: {type(exc).__name__}: {exc}")\n\nif failures:\n    print("FAILURES:")\n    for failure in failures:\n        print(f"- {failure}")\n    raise SystemExit(1)\nprint("OK")\n'
    return {"pkg/__init__.py": "", "tests/__init__.py": "", "tests/run_all.py": runner.lstrip(), "README.md": f"# {task_id}\n"}


def independent_task(idx: int, patch_dir: Path) -> dict:
    task_id = f"toy_independent_{idx:03d}"
    math_path = f"pkg/math_utils_{idx}.py"
    text_path = f"pkg/text_utils_{idx}.py"
    test_base = f"tests/test_base_{idx}.py"
    math_before = "def add(a, b):\n    return a + b\n"
    text_before = "def shout(s):\n    return s.upper()\n"
    test_before = (
        f"from pkg.math_utils_{idx} import add\n"
        f"from pkg.text_utils_{idx} import shout\n\n"
        "def test_base_math():\n    assert add(1, 2) == 3\n\n"
        "def test_base_text():\n    assert shout('hi') == 'HI'\n"
    )
    math_after = math_before + "\n\ndef double(x):\n    return x * 2\n"
    test_a_path = f"tests/test_feature_a_{idx}.py"
    test_a_after = f"from pkg.math_utils_{idx} import double\n\ndef test_double():\n    assert double(4) == 8\n"
    text_after = text_before + "\n\ndef slugify(s):\n    return s.lower().replace(' ', '-')\n"
    test_b_path = f"tests/test_feature_b_{idx}.py"
    test_b_after = f"from pkg.text_utils_{idx} import slugify\n\ndef test_slugify():\n    assert slugify('Hello World') == 'hello-world'\n"
    patch_a = multi_file_patch([(math_path, math_before, math_after), (test_a_path, "", test_a_after)])
    patch_b = multi_file_patch([(text_path, text_before, text_after), (test_b_path, "", test_b_after)])
    patch_a_file = write_patch(patch_dir, f"{task_id}_agent_a.patch", patch_a)
    patch_b_file = write_patch(patch_dir, f"{task_id}_agent_b.patch", patch_b)
    base_files = common_base_files(task_id)
    base_files.update({math_path: math_before, text_path: text_before, test_base: test_before})
    return {
        "task_id": task_id,
        "task_type": "independent",
        "base_files": base_files,
        "patches": [
            {"agent_id": "agent_a", "patch_file": patch_a_file, "workspace_targets": [math_path, test_a_path], "description": "Add independent math feature.", "llm_task": "Add a double(x) helper to the math utility module and add a test for double(4) == 8."},
            {"agent_id": "agent_b", "patch_file": patch_b_file, "workspace_targets": [text_path, test_b_path], "description": "Add independent text feature.", "llm_task": "Add a slugify(s) helper to the text utility module and add a test for slugify(\"Hello World\") == \"hello-world\"."},
        ],
        "test_commands": ["python tests/run_all.py"],
        "scenario": {"kind": "independent", "workspace_targets": [math_path, text_path], "expected_relation": "independent patches should auto-progress"},
    }


def shared_file_lock_task(idx: int, patch_dir: Path) -> dict:
    task_id = f"toy_shared_lock_{idx:03d}"
    parser_path = f"pkg/parser_{idx}.py"
    test_base = f"tests/test_parser_base_{idx}.py"
    parser_before = "MODE = 'base'\n\ndef parse_value(x):\n    return str(x)\n"
    test_before = f"from pkg.parser_{idx} import parse_value\n\ndef test_parse_base():\n    assert parse_value(7) == '7'\n"
    parser_after_a = "MODE = 'date'\n\ndef parse_value(x):\n    return str(x)\n\ndef parse_date(x):\n    return f'date:{x}'\n"
    test_a_path = f"tests/test_parser_date_{idx}.py"
    test_a_after = f"from pkg.parser_{idx} import parse_date\n\ndef test_parse_date():\n    assert parse_date('2026-05-18') == 'date:2026-05-18'\n"
    parser_after_b = "MODE = 'duration'\n\ndef parse_value(x):\n    return str(x)\n\ndef parse_duration(x):\n    return f'duration:{x}'\n"
    test_b_path = f"tests/test_parser_duration_{idx}.py"
    test_b_after = f"from pkg.parser_{idx} import parse_duration\n\ndef test_parse_duration():\n    assert parse_duration('3h') == 'duration:3h'\n"
    patch_a = multi_file_patch([(parser_path, parser_before, parser_after_a), (test_a_path, "", test_a_after)])
    patch_b = multi_file_patch([(parser_path, parser_before, parser_after_b), (test_b_path, "", test_b_after)])
    patch_a_file = write_patch(patch_dir, f"{task_id}_agent_a.patch", patch_a)
    patch_b_file = write_patch(patch_dir, f"{task_id}_agent_b.patch", patch_b)
    base_files = common_base_files(task_id)
    base_files.update({parser_path: parser_before, test_base: test_before})
    return {
        "task_id": task_id,
        "task_type": "shared_file_lock",
        "base_files": base_files,
        "patches": [
            {"agent_id": "agent_a", "patch_file": patch_a_file, "workspace_targets": [parser_path], "description": "Patch parser into date mode.", "llm_task": "Modify the parser file to add parse_date(x) returning f\"date:{x}\" and add a test for parse_date(\"2026-05-18\")."},
            {"agent_id": "agent_b", "patch_file": patch_b_file, "workspace_targets": [parser_path], "description": "Patch parser into duration mode.", "llm_task": "Modify the parser file to add parse_duration(x) returning f\"duration:{x}\" and add a test for parse_duration(\"3h\")."},
        ],
        "test_commands": ["python tests/run_all.py"],
        "scenario": {"kind": "workspace_lock", "lock_scope": parser_path, "workspace_targets": [parser_path], "expected_relation": "same-file lock contention should trigger lock-scoped review"},
    }


def commitment_staleness_task(idx: int, patch_dir: Path) -> dict:
    task_id = f"toy_commitment_{idx:03d}"
    api_path = f"pkg/api_{idx}.py"
    client_path = f"pkg/client_{idx}.py"
    test_base = f"tests/test_client_base_{idx}.py"
    api_before = "def normalize_user(raw):\n    return {'id': int(raw['id']), 'name': raw['name']}\n"
    # The base client is deliberately compatible with both the original dict
    # contract and the revised object contract.  This lets the revising patch
    # make safe downstream progress when the stale dependent patch is held.
    client_before = (
        f"from pkg.api_{idx} import normalize_user\n\n"
        "def _get(user, key):\n"
        "    if isinstance(user, dict):\n"
        "        return user[key]\n"
        "    return getattr(user, key)\n\n"
        "def render_user(raw):\n"
        "    user = normalize_user(raw)\n"
        "    return f'{_get(user, \"id\")}:{_get(user, \"name\")}'\n"
    )
    test_before = f"from pkg.client_{idx} import render_user\n\ndef test_render_user_base_contract():\n    assert render_user({{'id': '7', 'name': 'Ada'}}) == '7:Ada'\n"
    # Agent A depends on the old dict contract and adds a test that exposes the
    # stale assumption if Agent B's contract revision is also applied.
    client_after_a = client_before + "\n\ndef render_user_name(raw):\n    user = normalize_user(raw)\n    return user['name']\n"
    test_a_path = f"tests/test_client_feature_a_{idx}.py"
    test_a_after = f"from pkg.client_{idx} import render_user_name\n\ndef test_render_user_name_old_contract():\n    assert render_user_name({{'id': '7', 'name': 'Ada'}}) == 'Ada'\n"
    # Agent B revises the API commitment.  The base client still passes, but the
    # stale dependent helper from Agent A fails because it indexes the User
    # object as if it were a dict.  This makes unsafe auto-apply visible as a
    # real test failure rather than only as a substrate-level stale accept.
    api_after_b = (
        "class User:\n"
        "    def __init__(self, id, name):\n"
        "        self.id = id\n"
        "        self.name = name\n\n"
        "def normalize_user(raw):\n"
        "    return User(int(raw['id']), raw['name'])\n"
    )
    patch_a = multi_file_patch([(client_path, client_before, client_after_a), (test_a_path, "", test_a_after)])
    patch_b = multi_file_patch([(api_path, api_before, api_after_b)])
    patch_a_file = write_patch(patch_dir, f"{task_id}_agent_a.patch", patch_a)
    patch_b_file = write_patch(patch_dir, f"{task_id}_agent_b.patch", patch_b)
    commitment_atom = f"commitment:agent_b:normalize_user_contract:{idx}"
    base_files = common_base_files(task_id)
    base_files.update({api_path: api_before, client_path: client_before, test_base: test_before})
    return {
        "task_id": task_id,
        "task_type": "commitment_staleness",
        "base_files": base_files,
        "patches": [
            {"agent_id": "agent_a", "patch_file": patch_a_file, "workspace_targets": [client_path, test_a_path], "depends_on": [commitment_atom], "description": "Add client helper that assumes normalize_user returns a dict.", "llm_task": "Add render_user_name(raw) to the client module. It should call normalize_user(raw) and then use dictionary indexing user[\"name\"]. Add a test asserting render_user_name({\"id\": \"7\", \"name\": \"Ada\"}) == \"Ada\"."},
            {"agent_id": "agent_b", "patch_file": patch_b_file, "workspace_targets": [api_path], "revises": [commitment_atom], "description": "Revise normalize_user to return a User object.", "llm_task": "Revise the API module so normalize_user(raw) returns a User object with id and name attributes instead of a dict. Preserve the existing base client tests; do not edit the stale helper that another agent may add."},
        ],
        "test_commands": ["python tests/run_all.py"],
        "scenario": {"kind": "commitment_staleness", "commitment_atom_id": commitment_atom, "dependent_agent_id": "agent_a", "revising_agent_id": "agent_b", "workspace_targets": [api_path, client_path], "expected_relation": "agent_a patch depends on a commitment revised by agent_b"},
    }

def target_ambiguity_task(idx: int, patch_dir: Path) -> dict:
    task_id = f"toy_target_ambiguity_{idx:03d}"
    config_a = f"pkg/config_primary_{idx}.py"
    config_b = f"pkg/config_backup_{idx}.py"
    test_base = f"tests/test_config_base_{idx}.py"
    primary_before = "TIMEOUT = 10\nRETRIES = 2\n"
    backup_before = "TIMEOUT = 30\nRETRIES = 5\n"
    # The base test checks imports/types rather than exact values, because the
    # prepared patches are intended to update the configuration values.
    test_before = (
        f"from pkg.config_primary_{idx} import TIMEOUT as PRIMARY_TIMEOUT\n"
        f"from pkg.config_backup_{idx} import TIMEOUT as BACKUP_TIMEOUT\n\n"
        "def test_base_config_imports():\n"
        "    assert isinstance(PRIMARY_TIMEOUT, int)\n"
        "    assert isinstance(BACKUP_TIMEOUT, int)\n"
    )
    primary_after = "TIMEOUT = 15\nRETRIES = 2\n"
    backup_after = "TIMEOUT = 30\nRETRIES = 6\n"
    test_a_path = f"tests/test_config_primary_{idx}.py"
    test_a_after = f"from pkg.config_primary_{idx} import TIMEOUT\n\ndef test_primary_timeout_update():\n    assert TIMEOUT == 15\n"
    test_b_path = f"tests/test_config_backup_{idx}.py"
    test_b_after = f"from pkg.config_backup_{idx} import RETRIES\n\ndef test_backup_retry_update():\n    assert RETRIES == 6\n"
    patch_a = multi_file_patch([(config_a, primary_before, primary_after), (test_a_path, "", test_a_after)])
    patch_b = multi_file_patch([(config_b, backup_before, backup_after), (test_b_path, "", test_b_after)])
    patch_a_file = write_patch(patch_dir, f"{task_id}_agent_a.patch", patch_a)
    patch_b_file = write_patch(patch_dir, f"{task_id}_agent_b.patch", patch_b)
    base_files = common_base_files(task_id)
    base_files.update({config_a: primary_before, config_b: backup_before, test_base: test_before})
    return {
        "task_id": task_id,
        "task_type": "target_ambiguity",
        "base_files": base_files,
        "patches": [
            {"agent_id": "agent_a", "patch_file": patch_a_file, "workspace_targets": [config_a, test_a_path], "raw_target": "the config timeout", "target_candidates": [config_a, config_b], "description": "Ambiguous correction intended for primary config timeout.", "llm_task": "Update the primary config TIMEOUT value to 15 and add a test that imports TIMEOUT from the primary config module and checks it equals 15."},
            {"agent_id": "agent_b", "patch_file": patch_b_file, "workspace_targets": [config_b, test_b_path], "raw_target": "the config retry setting", "target_candidates": [config_a, config_b], "description": "Ambiguous correction intended for backup config retries.", "llm_task": "Update the backup config RETRIES value to 6 and add a test that imports RETRIES from the backup config module and checks it equals 6."},
        ],
        "test_commands": ["python tests/run_all.py"],
        "scenario": {"kind": "target_ambiguity", "target_candidates": [config_a, config_b], "workspace_targets": [config_a, config_b], "expected_relation": "stable target candidates should reduce ambiguous free-form target errors"},
    }

def task_plan(count: int) -> List[str]:
    if count <= 20:
        plan = ["independent"] * 5 + ["shared_file_lock"] * 5 + ["commitment_staleness"] * 7 + ["target_ambiguity"] * 3
    else:
        plan = ["independent"] * 10 + ["shared_file_lock"] * 10 + ["commitment_staleness"] * 20 + ["target_ambiguity"] * 10
    if count <= len(plan):
        return plan[:count]
    base = plan[:]
    while len(plan) < count:
        plan.extend(base)
    return plan[:count]


def generate_tasks(count: int, patch_dir: Path, seed: int) -> List[dict]:
    random.seed(seed)
    plan = task_plan(count)
    counters = {"independent": 0, "shared_file_lock": 0, "commitment_staleness": 0, "target_ambiguity": 0}
    tasks: List[dict] = []
    for kind in plan:
        counters[kind] += 1
        idx = counters[kind]
        if kind == "independent":
            task = independent_task(idx, patch_dir)
        elif kind == "shared_file_lock":
            task = shared_file_lock_task(idx, patch_dir)
        elif kind == "commitment_staleness":
            task = commitment_staleness_task(idx, patch_dir)
        elif kind == "target_ambiguity":
            task = target_ambiguity_task(idx, patch_dir)
        else:
            raise ValueError(f"Unknown task kind: {kind}")
        tasks.append(task)
    random.shuffle(tasks)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/e2e_patch_tasks.jsonl"))
    parser.add_argument("--patch-dir", type=Path, default=Path("patches/e2e_synth"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.patch_dir.mkdir(parents=True, exist_ok=True)
    for old in args.patch_dir.glob("*.patch"):
        old.unlink()
    tasks = generate_tasks(args.count, args.patch_dir, args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    counts: Dict[str, int] = {}
    for task in tasks:
        counts[task["task_type"]] = counts.get(task["task_type"], 0) + 1
    print(f"Wrote {len(tasks)} tasks to {args.out}")
    print(f"Wrote patches to {args.patch_dir}")
    print("Task counts:")
    for kind, n in sorted(counts.items()):
        print(f"  {kind}: {n}")


if __name__ == "__main__":
    main()
