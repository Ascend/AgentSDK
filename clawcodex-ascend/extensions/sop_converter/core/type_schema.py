# ruff: noqa
# pylint: skip-file
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Resolve Python type hints to rich JSON Schema for pos convert tool specs.

When a type hint refers to a Pydantic ``BaseModel`` or ``@dataclass`` defined
under *source_dir*, emit structured properties plus a minimal ``examples`` entry
instead of a bare ``{"type": "string"}`` / ``{"type": "object"}``.
"""

# pylint: disable=too-many-lines,too-many-nested-blocks
from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import logging
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from types import NoneType
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)

_PRIMITIVE_JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "null", "array", "object"})

# Keep in sync with SourceCodeParser exclusions: type indexes must not scan
# SDK test/example trees (avoids DeprecationWarning noise and false hits).
_INDEX_EXCLUDE_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        "dist",
        "build",
        "test",
        "tests",
        "unit_tests",
        "example",
        "examples",
        ".clawcodex",
        ".egg-info",
    }
)


def _iter_indexable_py_files(root: Path):
    """Yield project/SDK source ``.py`` files, skipping tests and caches."""
    for path in root.rglob("*.py"):
        if path.name.startswith("_"):
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if any(part.startswith(".") for part in parts):
            continue
        if any(part in _INDEX_EXCLUDE_DIR_NAMES for part in parts[:-1]):
            continue
        if any(
            part.startswith("test_") or part.startswith("example_") or part.endswith("_test") or part.endswith("_tests")
            for part in parts[:-1]
        ):
            continue
        yield path


# Built-in/stdlib type names that are not pydantic models — no warning when they degrade.
_BUILTIN_TYPE_NAMES = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "NoneType",
        "None",
        "list",
        "dict",
        "set",
        "frozenset",
        "tuple",
        "List",
        "Dict",
        "Set",
        "FrozenSet",
        "Tuple",
        "Sequence",
        "Iterable",
        "Iterator",
        "Mapping",
        "MutableMapping",
        "Any",
        "Optional",
        "Union",
        "Callable",
        "Awaitable",
        "Coroutine",
        "Type",
        "Event",
        "Queue",
        "Lock",
        "Future",
        "Task",
        "Protocol",
        "Generic",
        "datetime",
        "date",
        "time",
        "timedelta",
        "Decimal",
        "UUID",
        "Path",
        "URL",
        "AsyncGenerator",
        "Generator",
        "AsyncIterator",
        "ClassVar",
    }
)


def _looks_like_structured_type(type_hint: str) -> bool:
    """True when *type_hint* looks like a class name (not a builtin/primitive).

    Used to decide whether to emit a warning when schema extraction fails.
    Examples:
      "AgentCreateParams" → True  (warning if degraded)
      "str"               → False (no warning)
      "Dict[str, Any]"    → False (container, not a class)
    """
    root = type_hint.strip().split("[")[0].split(".")[-1].strip("'\"")
    if not root or not root[0].isupper():
        return False
    return root not in _BUILTIN_TYPE_NAMES


def _has_top_level_union(hint: str) -> bool:
    """True when ``|`` separates union members outside of brackets."""
    depth = 0
    for ch in hint:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(depth - 1, 0)
        elif ch == "|" and depth == 0:
            return True
    return False


def _split_union(type_hint: str) -> list[str]:
    """Split ``A | B | None`` / ``Union[A, B]`` into non-None member hints."""
    cleaned = type_hint.strip()
    if not cleaned:
        return []

    if cleaned.startswith("Union[") and cleaned.endswith("]"):
        inner = cleaned[len("Union[") : -1]
        parts = [p.strip() for p in _split_top_level_commas(inner)]
    elif _has_top_level_union(cleaned):
        parts = []
        current: list[str] = []
        depth = 0
        for ch in cleaned:
            if ch in "[({":
                depth += 1
            elif ch in "])}":
                depth = max(depth - 1, 0)
            if ch == "|" and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(ch)
        parts.append("".join(current).strip())
    else:
        parts = [cleaned]

    out: list[str] = []
    for part in parts:
        if part in ("None", "NoneType"):
            continue
        if part.startswith("Optional[") and part.endswith("]"):
            out.extend(_split_union(part[len("Optional[") : -1]))
        else:
            out.append(part)
    return out


def _type_root(type_hint: str) -> str:
    cleaned = type_hint.strip()
    if cleaned.startswith(("Optional[", "Union[")):
        parts = _split_union(cleaned)
        return parts[0].split("[", 1)[0] if parts else cleaned
    return cleaned.split("[", 1)[0]


def _extract_container_inner_type(type_hint: str) -> str:
    """Extract the inner type from a container type hint, handling nested brackets.

    Examples:
        "list[str]" → "str"
        "List[dict[str, Any]]" → "dict[str, Any]"
        "Sequence[Optional[int]]" → "Optional[int]"
    """
    cleaned = type_hint.strip()
    start = cleaned.find("[")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                return cleaned[start + 1 : i].strip()
    return ""


def _extract_dict_value_type(type_hint: str) -> str:
    """Extract the value type from a Dict/Mapping type hint, handling nested brackets.

    Examples:
        "Dict[str, int]" → "int"
        "dict[str, dict[str, Any]]" → "dict[str, Any]"
        "Mapping[str, List[SomeModel]]" → "List[SomeModel]"
    """
    cleaned = type_hint.strip()
    start = cleaned.find("[")
    if start == -1:
        return ""

    depth = 0
    comma_pos = -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
        elif cleaned[i] == "," and depth == 1:
            comma_pos = i
            break

    if comma_pos == -1:
        return ""

    # Extract value type after the comma
    value_part = cleaned[comma_pos + 1 :]
    # Find the closing bracket at depth 1
    depth = 0
    for i in range(0, len(value_part)):
        if value_part[i] == "[":
            depth += 1
        elif value_part[i] == "]":
            depth -= 1
            if depth == 0:
                return value_part[:i].strip()
    return value_part.strip().rstrip("]")


@lru_cache(maxsize=32)
def _build_class_index(source_dir: str) -> dict[str, str]:
    """Map class name → dotted module path under *source_dir*."""
    root = Path(source_dir).resolve()
    index: dict[str, str] = {}
    if not root.is_dir():
        return index

    for path in _iter_indexable_py_files(root):
        rel = path.relative_to(root)
        module = ".".join(rel.with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                index.setdefault(node.name, module)
    return index


# Pydantic base-class names, used for AST pre-filtering (so Enum/ABC/dataclass
# types are never sent to the subprocess probe).
_PYDANTIC_BASE_NAMES = frozenset({"BaseModel", "RootModel", "GenericModel"})


@lru_cache(maxsize=32)
def _build_pydantic_class_index(source_dir: str) -> dict[str, set[str]]:
    """Map class name → set of dotted module paths, for all pydantic BaseModel subclasses.

    Uses a two-pass transitive-closure algorithm to handle indirect inheritance
    across modules:
    Pass 1: scan all ``.py`` files and collect the base-class names of every ClassDef.
    Pass 2: start from classes that directly inherit BaseModel/RootModel/GenericModel,
    then iteratively propagate until a fixed point is reached.

    Used to pre-filter probe targets so non-pydantic types never reach the subprocess.
    """
    root = Path(source_dir).resolve()
    if not root.is_dir():
        return {}

    # Pass 1: collect every class and its base-class names
    # class_name -> set[module_path]
    all_classes: dict[str, set[str]] = {}
    # (module_path, class_name) -> set[base_class_name]
    class_bases: dict[tuple[str, str], set[str]] = {}

    for path in _iter_indexable_py_files(root):
        rel = path.relative_to(root)
        module = ".".join(rel.with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                all_classes.setdefault(node.name, set()).add(module)
                bases = set()
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.add(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.add(base.attr)
                class_bases[(module, node.name)] = bases

    # Pass 2: transitive closure
    # Seed set: classes that directly inherit BaseModel/RootModel/GenericModel
    pydantic_classes: set[tuple[str, str]] = set()
    for (module, class_name), bases in class_bases.items():
        if any(base in _PYDANTIC_BASE_NAMES for base in bases):
            pydantic_classes.add((module, class_name))

    # Iteratively propagate until a fixed point
    changed = True
    while changed:
        changed = False
        for (module, class_name), bases in class_bases.items():
            if (module, class_name) in pydantic_classes:
                continue
            for base_name in bases:
                if base_name in all_classes:
                    for base_module in all_classes[base_name]:
                        if (base_module, base_name) in pydantic_classes:
                            pydantic_classes.add((module, class_name))
                            changed = True
                            break
                    if (module, class_name) in pydantic_classes:
                        break

    # Build the final index
    index: dict[str, set[str]] = {}
    for module, class_name in pydantic_classes:
        index.setdefault(class_name, set()).add(module)

    return index


def _import_type(source_dir: str, type_name: str) -> type[Any] | None:
    return _import_resolved_type(source_dir, type_name, module_path=None)


def _resolve_type_import(
    source_dir: str,
    type_hint: str,
    module_path: str | None = None,
) -> tuple[str, str] | None:
    """Resolve a type hint to ``(module_path, class_name)`` via module import index.

    Resolution priority:
    1. module_path context (if given, resolve via ModuleImportIndex) — trust the
       alias-resolution result; do not override with a same-named pydantic class,
       because the wrapper-enforced type must match the schema
    2. pydantic index (when no module_path, used to disambiguate same-named classes)
    3. general class index (fallback)
    """
    from .import_alias_resolver import ModuleImportIndex

    root = _type_root(type_hint)
    if not root:
        return None
    if module_path:
        try:
            resolved = ModuleImportIndex(source_dir).resolve_import_path(module_path, root)
            if resolved:
                return resolved
        except Exception:  # nosec
            pass
    # Without a module_path context, use the pydantic index to disambiguate same-named classes
    pydantic_idx = _build_pydantic_class_index(source_dir)
    pydantic_modules = pydantic_idx.get(root) or set()
    if pydantic_modules:
        return next(iter(pydantic_modules)), root
    # Fall back to the general class index
    index = _build_class_index(source_dir)
    indexed = index.get(root)
    if indexed:
        return indexed, root
    return None


def _import_resolved_type(
    source_dir: str,
    type_hint: str,
    module_path: str | None = None,
) -> type[Any] | None:
    resolved = _resolve_type_import(source_dir, type_hint, module_path)
    if not resolved:
        return None
    mp, class_name = resolved
    root = str(Path(source_dir).resolve())
    from .path_resolver import infer_extra_sys_path_entries

    # Insert order: root first, then extras (extras end up searched first).
    # Module-level sys.path.insert(dirname(__file__)) still wins for CWD-style
    # demos; sibling src/ remains on the path so bare imports resolve.
    inserted_paths: list[str] = []
    for path in (root, *infer_extra_sys_path_entries(source_dir, mp)):
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted_paths.append(path)
    try:
        module = importlib.import_module(mp)
        obj = getattr(module, class_name, None)
        return obj if isinstance(obj, type) else None
    # SystemExit — SDK demos often call sys.exit() on ImportError
    except (Exception, SystemExit) as exc:  # pragma: no cover - import failures are expected
        logger.debug("Could not import %s from %s: %s", class_name, mp, exc)
        sys.modules.pop(mp, None)
        return None
    finally:
        for path in inserted_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Subprocess probe: import SDK module in isolated process with timeout
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = r"""\
import sys, json, importlib, signal, time

def main():
    req = json.loads(sys.stdin.read())
    source_dir = req["source_dir"]
    module_path = req["module_path"]
    class_name = req["class_name"]
    skip_direct = req.get("skip_direct_import", False)

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    for p in req.get("extra_sys_path") or []:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    # Direct import with timeout.
    # On Linux/WSL we use signal.alarm() so a blocking import can be
    # interrupted and we return kind=None; the parent then uses in-process
    # AST / batch-probe fallbacks.
    # On Windows there is no SIGALRM, so we rely on the parent process's
    # communicate(timeout=...) to kill the whole subprocess; the import
    # itself is still attempted (no skip) so Windows users get schemas too.
    schema = None
    direct_ok = False
    direct_time = 0.0
    if not skip_direct:
        try:
            t0 = time.monotonic()
            if hasattr(signal, "SIGALRM"):
                def _alarm(signum, frame):
                    raise TimeoutError("import timed out")
                old = signal.signal(signal.SIGALRM, _alarm)
                signal.alarm(15)
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name, None)
                if isinstance(cls, type):
                    try:
                        from pydantic import BaseModel
                        if issubclass(cls, BaseModel):
                            schema = cls.model_json_schema(mode="validation")
                            direct_ok = True
                    except ImportError:
                        pass
            except TimeoutError:
                pass
            except (Exception, SystemExit):
                sys.modules.pop(module_path, None)
            finally:
                direct_time = time.monotonic() - t0
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)
        except (Exception, SystemExit):
            pass

    if schema is not None:
        print(json.dumps({"kind": "pydantic", "schema": schema, "method": "direct",
                          "direct_time": round(direct_time, 3)}))
        return

    # Import failed or skipped: return None so the parent can fall back to
    # batch probe / in-process AST (pydantic / pure / dataclass).
    print(json.dumps({"kind": None, "direct_failed": not direct_ok,
                      "direct_time": round(direct_time, 3)}))

main()
"""

# Per-source_dir circuit breaker: when the subprocess times out on direct
# import, subsequent probes for the same source_dir are skipped entirely.
# This prevents N × timeout delay.
#
# Scope: process-local state for ONE ``sop convert`` run.  The CLI entrypoint
# (``_handle_convert_from_source`` in clawcodex_ext/cli/sop_cmd/commands.py)
# calls ``reset_schema_probe_runtime_state()`` before each conversion so a
# previous conversion's circuit breaker cannot poison the next one.  In a
# long-lived REPL session, however, a tripped breaker persists until the
# process exits — by design, since re-probing a genuinely blocking SDK would
# just hang again.
_BLOCKED_SOURCE_DIRS: set[str] = set()

# Per-source_dir flag: when direct import fails OR is too slow for one type,
# it will almost certainly be slow for all types in the same SDK (same import
# chain).  Skipping direct import for subsequent types saves the import cost
# (up to 6s per type); the parent still has batch / in-process AST fallbacks.
_DIRECT_IMPORT_BLOCKED: set[str] = set()

# Threshold: if direct import takes longer than this even on success, mark
# the SDK to skip direct import for subsequent types.  This handles SDKs
# whose import is slow but not truly hanging (e.g., 3-5s due to heavy
# initialization).  Set high enough to tolerate SDKs with heavy module-level
# initialization (e.g., 10-20s for connection pools, model registries).
# Only truly pathological SDKs (>60s per import) will trigger the circuit
# breaker.
_DIRECT_IMPORT_SLOW_THRESHOLD = 60.0

# Pre-filled cache from batch probe.  When non-None for a key, individual
# _probe_type_in_subprocess calls return immediately without spawning.
_BATCH_CACHE: dict[tuple[str, str, str], dict[str, Any] | None] = {}

# Warning deduplication: track type names that already emitted a degradation
# warning, so the same type does not warn repeatedly across many tool params.
_WARNED_TYPES: set[str] = set()

# Python executable of the current SDK's bundle venv.  Set by
# preload_schemas_for_source_dir() for later use in _probe_type_in_subprocess(),
# ensuring the subprocess can import the SDK's third-party dependencies.
# ``sop convert`` creates the venv before probing, so this value is ready
# by probe time.
_CURRENT_VENV_PYTHON: str | None = None


def reset_schema_probe_runtime_state() -> None:
    """Reset process-local schema probe caches and circuit breakers."""

    global _CURRENT_VENV_PYTHON

    _BLOCKED_SOURCE_DIRS.clear()
    _DIRECT_IMPORT_BLOCKED.clear()
    _BATCH_CACHE.clear()
    _WARNED_TYPES.clear()
    _CURRENT_VENV_PYTHON = None
    _probe_type_in_subprocess.cache_clear()


def _reap_probe_proc(proc: subprocess.Popen | None) -> None:
    """Kill and reap a still-running probe subprocess so it cannot zombify.

    No-op when the process was never created or already exited.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait()
    except OSError:
        pass


@lru_cache(maxsize=256)
def _probe_type_in_subprocess(
    source_dir: str,
    module_path: str,
    class_name: str,
    timeout: float = 60.0,
    venv_python: str | None = None,
) -> dict[str, Any] | None:
    """Extract a pydantic schema in an isolated subprocess (direct import only), with timeout.

    The subprocess only does direct import + signal alarm / parent communicate
    timeout.  Returns None on failure; the caller ``pydantic_schema_for_type``
    then falls back to the batch cache / in-process AST (pydantic / pure / dataclass).

    Circuit-breaker mechanism:
    - _DIRECT_IMPORT_BLOCKED: after direct import fails or is too slow, later
      probes skip the subprocess import
    - _BLOCKED_SOURCE_DIRS: after a whole subprocess times out, all further
      single-type probes are skipped

    Batch warm-up: preload_schemas_for_source_dir() can fill _BATCH_CACHE in
    one pass; later calls hit the cache directly without spawning a subprocess.

    venv_python: python executable of the SDK bundle venv.  When given, the
    subprocess is launched with that python so it can import the SDK's
    third-party dependencies (e.g. jsonschema_path, pysbd).
    """
    root = str(Path(source_dir).resolve())

    # Check batch cache first (pre-filled by preload_schemas_for_source_dir).
    cache_key = (root, module_path, class_name)
    if cache_key in _BATCH_CACHE:
        return _BATCH_CACHE[cache_key]

    if root in _BLOCKED_SOURCE_DIRS:
        return None

    skip_direct = root in _DIRECT_IMPORT_BLOCKED
    from .path_resolver import infer_extra_sys_path_entries

    request = json.dumps(
        {
            "source_dir": root,
            "module_path": module_path,
            "class_name": class_name,
            "skip_direct_import": skip_direct,
            "extra_sys_path": infer_extra_sys_path_entries(root, module_path),
        }
    )
    proc: subprocess.Popen | None = None
    try:
        python_exe = venv_python or _CURRENT_VENV_PYTHON or sys.executable
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            [python_exe, "-c", _PROBE_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, _ = proc.communicate(input=request, timeout=timeout)
        if proc.returncode != 0:
            return None
        lines = stdout.strip().splitlines()
        if not lines:
            return None
        result = json.loads(lines[-1])

        # Mark SDK to skip direct import for subsequent probes when:
        # 1. Direct import failed (timeout/error), OR
        # 2. Direct import succeeded but was slow (> threshold)
        direct_time = result.get("direct_time", 0.0)
        direct_failed = result.get("direct_failed", False)
        if direct_failed and root not in _DIRECT_IMPORT_BLOCKED:
            _DIRECT_IMPORT_BLOCKED.add(root)
            logger.info(
                "Direct import failed for %s, subsequent probes will skip direct import",
                root,
            )
        elif not direct_failed and direct_time > _DIRECT_IMPORT_SLOW_THRESHOLD and root not in _DIRECT_IMPORT_BLOCKED:
            _DIRECT_IMPORT_BLOCKED.add(root)
            logger.info(
                "Direct import slow for %s (%.1fs > %.1fs threshold), subsequent probes will skip direct import",
                root,
                direct_time,
                _DIRECT_IMPORT_SLOW_THRESHOLD,
            )

        return result if result.get("kind") else None
    except subprocess.TimeoutExpired:
        # Reap the still-running child before tripping the circuit breaker.
        _reap_probe_proc(proc)
        # Direct-import subprocess timed out — circuit breaker for this SDK.
        _BLOCKED_SOURCE_DIRS.add(root)
        logger.warning(
            "Probe %s.%s timed out after %ss (direct import), circuit breaker tripped for %s",
            module_path,
            class_name,
            timeout,
            root,
        )
        return None
    except KeyboardInterrupt:
        # Reap before propagating so the probe child cannot zombify.
        _reap_probe_proc(proc)
        raise
    except Exception as exc:
        # Reap on every error path (BrokenPipeError, JSONDecodeError, OSError, ...).
        _reap_probe_proc(proc)
        logger.debug("Probe %s.%s failed: %s", module_path, class_name, exc)
        return None


# ---------------------------------------------------------------------------
# Batch probe: one subprocess per SDK, import once, extract all schemas
# ---------------------------------------------------------------------------


def _extract_nested_types(type_hint: str) -> list[str]:
    """Extract all class-like type names from a (possibly generic) type hint.

    Examples:
      "AgentConfig"              → ["AgentConfig"]
      "Dict[str, Operator]"      → ["Operator"]  (str excluded as builtin)
      "List[AgentConfig]"        → ["AgentConfig"]
      "Optional[AgentConfig]"    → ["AgentConfig"]
      "str"                      → []
      "Dict[str, Any]"           → []
    """
    # Strip the outermost type and recurse into bracket contents.
    # We extract all Capitalized identifiers that aren't builtins.
    results: list[str] = []
    # Find all substrings inside brackets
    depth = 0
    start = 0
    for i, ch in enumerate(type_hint):
        if ch == "[":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                inner = type_hint[start:i]
                # Split by comma at depth 0
                for part in _split_top_level_commas(inner):
                    part = part.strip()
                    root = part.split("[")[0].split(".")[-1].strip("'\"")
                    if root and root[0].isupper() and root not in _BUILTIN_TYPE_NAMES:
                        results.append(part)
                    # Recurse into nested generics
                    if "[" in part:
                        results.extend(_extract_nested_types(part))
    return results


def _split_top_level_commas(s: str) -> list[str]:
    """Split string by commas at bracket depth 0."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def collect_probe_targets(
    source_dir: str,
    type_hints: list[tuple[str, str | None]],
) -> list[tuple[str, str]]:
    """Resolve a list of (type_hint, module_path) to (module_path, class_name) pairs.

    Recursively extracts nested types from generic containers (e.g.,
    ``Dict[str, Operator]`` → ``Operator``) so they are included in the
    batch probe.

    Args:
        source_dir: SDK source root.
        type_hints: List of (type_hint, module_path) tuples from operation params.

    Returns:
        Deduplicated list of (module_path, class_name) pairs suitable for
        preload_schemas_for_source_dir().
    """
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    # Pre-filter: only probe pydantic types directly inheriting BaseModel;
    # skip Enum/ABC/dataclass and aliases.
    pydantic_index = _build_pydantic_class_index(source_dir)

    def _try_resolve(hint: str, module_path: str | None) -> None:
        if not hint:
            return
        # Try resolving the hint itself (e.g., "AgentConfig")
        resolved = _resolve_type_import(source_dir, hint, module_path)
        if resolved:
            mp, cn = resolved
            # Pre-filter pydantic: skip non-BaseModel subclasses
            if cn in pydantic_index and mp in pydantic_index[cn]:
                if (mp, cn) not in seen:
                    seen.add((mp, cn))
                    result.append((mp, cn))
        # Extract nested types from generics (e.g., Dict[str, Operator] → Operator)
        for nested in _extract_nested_types(hint):
            resolved = _resolve_type_import(source_dir, nested, module_path)
            if resolved:
                mp, cn = resolved
                if cn in pydantic_index and mp in pydantic_index[cn]:
                    if (mp, cn) not in seen:
                        seen.add((mp, cn))
                        result.append((mp, cn))

    for type_hint, module_path in type_hints:
        _try_resolve(type_hint, module_path)
    return result
