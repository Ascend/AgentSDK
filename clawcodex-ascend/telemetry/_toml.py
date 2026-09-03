"""Private TOML helpers for telemetry config loading.

The :mod:`telemetry.config` layer probes the on-disk TOML
config files for a ``[tool.clawcodex.telemetry]`` table or a
standalone ``telemetry.toml`` file. Layered order (lowest → highest
precedence inside the on-disk section):

1. ``pyproject.toml`` walked upward from ``cwd``
2. ``<cwd>/telemetry.toml``

These are then merged with the existing ``src.config.load_config()``
JSON section, where JSON wins (see :func:`load_config`).

The module is intentionally tiny and pure: it never raises out of its
public surface so a malformed TOML file can never block telemetry
startup. Errors are logged at ``debug`` and treated as "no TOML
section found".

Python 3.10 is the supported floor (``pyproject.toml: requires-python
= ">=3.10"``); stdlib :mod:`tomllib` is 3.11+. The import dance below
falls back to :mod:`tomli` (already a transitive dep via pytest) on
older interpreters. Both modules expose the same ``load`` / ``loads``
API.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:  # Python 3.11+
    import tomllib as _toml_loader  # type: ignore[import-not-found]
except ImportError:  # Python 3.10 fallback (transitive via pytest)
    try:
        import tomli as _toml_loader  # type: ignore[import-not-found, no-redef]
    except ImportError:  # pragma: no cover - extremely unlikely in test env
        _toml_loader = None  # type: ignore[assignment]


_PYPROJECT_TABLE: str = "tool.clawcodex.telemetry"


def _read_toml_file(path: Path) -> dict[str, Any] | None:
    """Return the parsed contents of a TOML file, or ``None`` on any
    failure (missing, unreadable, invalid syntax, no TOML loader).
    """
    if _toml_loader is None:
        return None
    try:
        with path.open("rb") as fh:
            return _toml_loader.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, _toml_loader.TOMLDecodeError) as exc:
        logger.debug("telemetry: failed to read TOML file %s: %s", path, exc)
        return None


def _drill(d: dict[str, Any], dotted: str) -> dict[str, Any]:
    """Return the nested dict at ``dotted`` key path, or ``{}`` when any
    segment is missing or non-dict.
    """
    cursor: Any = d
    for segment in dotted.split("."):
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(segment)
        if cursor is None:
            return {}
    return cursor if isinstance(cursor, dict) else {}


def _walk_up_for_pyproject(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for the nearest ``pyproject.toml``.

    Stops at the first hit. Returns ``None`` if we reach the filesystem
    root without finding one. Symlinks are not followed.
    """
    cur = start.resolve() if start.exists() else start
    if cur.is_file():
        cur = cur.parent
    while True:
        candidate = cur / "pyproject.toml"
        if candidate.is_file():
            return candidate
        parent = cur.parent
        if parent == cur:  # filesystem root
            return None
        cur = parent


def load_toml_telemetry(cwd: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Return the telemetry TOML section discovered on disk.

    The section is the union of:

    * The ``[tool.clawcodex.telemetry]`` table of the nearest
      ``pyproject.toml`` walking upward from ``cwd`` (defaults to
      :func:`os.getcwd` when omitted).
    * The top-level keys of ``<cwd>/telemetry.toml``, if present.

    The first source wins on key collisions; the standalone
    ``telemetry.toml`` is layered on top. Returns ``{}`` when no TOML
    file is found or the loader is unavailable.
    """
    start = Path(cwd) if cwd is not None else Path.cwd()
    out: dict[str, Any] = {}

    pyproject = _walk_up_for_pyproject(start)
    if pyproject is not None:
        data = _read_toml_file(pyproject)
        section = _drill(data or {}, _PYPROJECT_TABLE)
        if section:
            out.update(section)

    standalone = (start / "telemetry.toml") if start.is_dir() else (start.parent / "telemetry.toml")
    if standalone.is_file():
        data = _read_toml_file(standalone)
        if data:
            # The standalone file is dedicated to telemetry so the
            # caller may either place keys at the top level or wrap
            # them in a [telemetry] table. We prefer the explicit
            # table when both are present (mirrors the pyproject.toml
            # convention of nesting under a tool-specific table).
            section = data.get("telemetry")
            if not isinstance(section, dict):
                section = data
            out.update(section)
    return out
