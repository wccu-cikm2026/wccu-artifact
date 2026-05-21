from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

_ENV_LINE_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def repo_root() -> Path:
    """Return the project root for the source checkout."""
    return Path(__file__).resolve().parents[1]


def _strip_inline_comment(value: str) -> str:
    """Strip unquoted inline comments from dotenv values."""
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in value:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and quote == '"':
            escaped = True
            out.append(ch)
            continue
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
            continue
        if ch in {'"', "'"}:
            quote = ch
            out.append(ch)
            continue
        if ch == '#':
            break
        out.append(ch)
    return ''.join(out).strip()


def parse_dotenv_text(text: str) -> dict[str, str]:
    """Parse a small, dependency-free subset of .env syntax.

    Supported forms:
      KEY=value
      export KEY=value
      KEY="quoted value"
      KEY='quoted value'

    Blank lines and lines starting with # are ignored. Existing environment
    variables are not expanded intentionally; experiments should remain explicit.
    """
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        value = _strip_inline_comment(value)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
            if raw_line.strip().split('=', 1)[1].lstrip().startswith('"'):
                value = value.encode('utf-8').decode('unicode_escape')
        parsed[key] = value
    return parsed


def _candidate_env_paths(start: Path | None = None) -> Iterable[Path]:
    env_file = os.environ.get('PCSE_ENV_FILE')
    if env_file:
        yield Path(env_file).expanduser()
        return

    start_path = (start or Path.cwd()).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    for parent in [start_path, *start_path.parents]:
        yield parent / '.env'

    root_env = repo_root() / '.env'
    yield root_env


def find_dotenv(start: str | Path | None = None) -> Path | None:
    """Find the first .env file from cwd upward, then the repo root.

    Set PCSE_ENV_FILE=/path/to/file.env to force a specific file.
    """
    seen: set[Path] = set()
    for path in _candidate_env_paths(Path(start) if start is not None else None):
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists() and path.is_file():
            return path
    return None


def load_dotenv(path: str | Path | None = None, *, override: bool = False, verbose: bool = False) -> dict[str, str]:
    """Load dotenv values into os.environ.

    This intentionally avoids a hard dependency on python-dotenv. If python-dotenv
    is installed, the same .env file will still work, but this fallback keeps the
    experiment harness self-contained.
    """
    dotenv_path = Path(path).expanduser() if path else find_dotenv()
    if not dotenv_path or not dotenv_path.exists():
        if verbose:
            print(f'[pcse] no .env found (cwd={Path.cwd()})')
        return {}
    values = parse_dotenv_text(dotenv_path.read_text(encoding='utf-8'))
    loaded: dict[str, str] = {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    if verbose:
        print(f'[pcse] loaded {len(loaded)} values from {dotenv_path}')
    return loaded


def redact_secret(value: str, *, visible: int = 4) -> str:
    if not value:
        return ''
    if len(value) <= visible * 2:
        return '*' * len(value)
    return f'{value[:visible]}...{value[-visible:]}'
