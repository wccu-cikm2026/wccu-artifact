from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_JSONL_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clean(value: Any = '') -> str:
    return str(value if value is not None else '').strip()


def stable_hash(value: Any = '', length: int = 12) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(value.encode('utf-8')).hexdigest()[:length]


def slugify(value: Any = '') -> str:
    s = re.sub(r'[^a-z0-9._-]+', '_', clean(value).lower())
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'item'


def read_json(path: str | Path, fallback: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return fallback


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def append_jsonl(path: str | Path, row: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    # Several LLM experiment workers may append diagnostics at the same time.
    # Keep each JSONL row atomic within this process.
    with _JSONL_LOCK:
        with p.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def read_jsonl(path: str | Path, limit: int = 0) -> list[Any]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        return rows[-limit:] if limit and limit > 0 else rows
    except Exception:
        return []


def remove_dir(path: str | Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def estimate_tokens(text: Any = '') -> int:
    return math.ceil(len(str(text or '')) / 4)


def mean(values: Iterable[Any]) -> float:
    nums = []
    for v in values:
        try:
            n = float(v)
            if math.isfinite(n):
                nums.append(n)
        except Exception:
            pass
    return sum(nums) / len(nums) if nums else 0.0


def deep_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
