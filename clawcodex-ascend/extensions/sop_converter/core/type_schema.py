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


# --- appended from PR #754 ---


_BATCH_PROBE_SCRIPT = r'''\
import sys, json, importlib, ast, signal, time
from pathlib import Path

def _extract_via_ast(source_dir, module_path, class_name):
    """Phase 2a: AST compilation + model_json_schema (for ModuleNotFoundError cases).

    Filters the AST to only include necessary imports and the target class,
    then compiles and calls model_json_schema(). This avoids importing modules
    with missing dependencies.
    """
    root = Path(source_dir)
    parts = module_path.split(".")
    source_file = root.joinpath(*parts).with_suffix(".py")
    if not source_file.is_file():
        return None
    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    filtered = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            filtered.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == class_name:
            filtered.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            filtered.append(node)
        elif isinstance(node, ast.Assign) and isinstance(
            node.value,
            (ast.Constant, ast.List, ast.Dict, ast.Set, ast.Tuple, ast.Name, ast.Attribute),
        ):
            filtered.append(node)

    module_ast = ast.Module(body=filtered, type_ignores=[])
    code = compile(module_ast, str(source_file), "exec")
    namespace = {"__name__": "extracted_module"}
    try:
        exec(code, namespace)
    except Exception:
        return None

    cls = namespace.get(class_name)
    if not isinstance(cls, type):
        return None
    try:
        from pydantic import BaseModel
        if issubclass(cls, BaseModel):
            return cls.model_json_schema(mode="validation")
    except Exception:
        pass
    return None



def _hint_to_json_type(hint):
    """Convert a type hint string to a basic JSON Schema type definition."""
    hint = hint.strip()

    if hint.startswith("Union[") or hint.startswith("Optional["):
        inner = hint[6:-1] if hint.startswith("Union") else hint[9:-1]
        parts = _split_top_level_commas(inner)
        has_none = any(p.strip() == "None" for p in parts)
        non_none_parts = [p for p in parts if p.strip() != "None"]
        if len(non_none_parts) == 1:
            base = _hint_to_json_type(non_none_parts[0].strip())
            if has_none:
                base["nullable"] = True
            return base
        return {"anyOf": [_hint_to_json_type(p.strip()) for p in non_none_parts]}

    if hint.startswith("Literal["):
        inner = hint[8:-1]
        parts = _split_top_level_commas(inner)
        enum_values = []
        for part in parts:
            part = part.strip()
            if part.startswith('"') and part.endswith('"'):
                enum_values.append(part[1:-1])
            elif part.startswith("'") and part.endswith("'"):
                enum_values.append(part[1:-1])
            elif part in ("True", "False"):
                enum_values.append(part == "True")
            elif part == "None":
                continue
            else:
                try:
                    enum_values.append(int(part))
                except ValueError:
                    try:
                        enum_values.append(float(part))
                    except ValueError:
                        enum_values.append(part)
        if enum_values:
            first_val = enum_values[0]
            if isinstance(first_val, str):
                return {"type": "string", "enum": enum_values}
            elif isinstance(first_val, bool):
                return {"type": "boolean", "enum": enum_values}
            elif isinstance(first_val, int):
                return {"type": "integer", "enum": enum_values}
            elif isinstance(first_val, float):
                return {"type": "number", "enum": enum_values}
        return {"type": "string", "enum": enum_values} if enum_values else {"type": "object"}

    if hint.startswith("list[") or hint.startswith("List["):
        inner = hint[5:-1] if hint.startswith("list") else hint[6:-1]
        return {"type": "array", "items": _hint_to_json_type(inner.strip())}

    if hint.startswith("dict[") or hint.startswith("Dict["):
        inner = hint[5:-1] if hint.startswith("dict") else hint[6:-1]
        parts = _split_top_level_commas(inner)
        if len(parts) >= 2:
            value_type = _hint_to_json_type(parts[1].strip())
            return {"type": "object", "additionalProperties": value_type}
        return {"type": "object"}

    basic_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "bytes": "string",
        "Any": "object",
    }
    if hint in basic_mapping:
        return {"type": basic_mapping[hint]}

    return {"type": "object"}


def _extract_via_pure_ast(source_dir, module_path, class_name):
    """Phase 2b: Pure AST field extraction (for model_json_schema failures like InstanceOf).

    Does NOT call model_json_schema(). Instead, parses the ClassDef AST directly
    to extract field names and type annotations, building a JSON Schema manually.
    This handles cases where model_json_schema() fails due to non-serializable
    field types like InstanceOf or Callable.
    """
    root = Path(source_dir)
    parts = module_path.split(".")
    source_file = root.joinpath(*parts).with_suffix(".py")
    if not source_file.is_file():
        return None
    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_node = node
            break
    if class_node is None:
        return None

    properties = {}
    required = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            field_name = stmt.target.id
            if stmt.annotation:
                hint = ast.unparse(stmt.annotation)
            else:
                hint = "string"
            properties[field_name] = _hint_to_json_type(hint)
            if stmt.value is None and field_name not in required:
                required.append(field_name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    field_name = target.id
                    properties.setdefault(field_name, {"type": "string"})

    if not properties:
        return None

    schema = {
        "type": "object",
        "title": class_name,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def main():
    req = json.loads(sys.stdin.read())
    source_dir = req["source_dir"]
    targets = req["targets"]

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    results = {}
    imported_modules = {}
    direct_blocked = False
    total = len(targets)

    for idx, target in enumerate(targets, 1):
        mp = target["module_path"]
        cn = target["class_name"]
        key = f"{mp}::{cn}"
        schema = None
        method = None
        fail_reason = ""

        for p in target.get("extra_sys_path") or []:
            if p and p not in sys.path:
                sys.path.insert(0, p)

        if idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}] extracting schema for {cn} from {mp}...", file=sys.stderr, flush=True)

        # Phase 1: direct import + model_json_schema
        if not direct_blocked:
            try:
                t0 = time.monotonic()
                if hasattr(signal, "SIGALRM"):
                    def _alarm(signum, frame):
                        raise TimeoutError("import timed out")
                    old = signal.signal(signal.SIGALRM, _alarm)
                    signal.alarm(60)
                try:
                    if mp not in imported_modules:
                        imported_modules[mp] = importlib.import_module(mp)
                    mod = imported_modules[mp]
                    if mod is not None:
                        cls = getattr(mod, cn, None)
                        if isinstance(cls, type):
                            try:
                                from pydantic import BaseModel
                                if issubclass(cls, BaseModel):
                                    schema = cls.model_json_schema(mode="validation")
                                    method = "direct"
                                else:
                                    fail_reason = f"not a BaseModel subclass (got {type(cls).__name__})"
                            except ImportError:
                                fail_reason = "pydantic not installed"
                            except Exception as exc:
                                fail_reason = f"model_json_schema failed: {type(exc).__name__}: {exc}"
                        else:
                            fail_reason = f"attribute '{cn}' not found on module"
                except TimeoutError:
                    imported_modules[mp] = None
                    direct_blocked = True
                    fail_reason = "import timed out (>60s)"
                except (Exception, SystemExit) as exc:
                    imported_modules[mp] = None
                    sys.modules.pop(mp, None)
                    fail_reason = f"import failed: {type(exc).__name__}: {exc}"
                finally:
                    dt = time.monotonic() - t0
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old)
                    if dt > 60.0 and not direct_blocked:
                        direct_blocked = True
            except (Exception, SystemExit):
                pass

        if schema is not None:
            results[key] = {"kind": "pydantic", "schema": schema, "method": method}
            continue

        # Phase 2b: Pure AST extraction (for model_json_schema failures like InstanceOf)
        # This handles cases where model_json_schema() fails due to non-serializable
        # field types like InstanceOf or Callable.
        if fail_reason and "model_json_schema" in fail_reason:
            ast_schema = _extract_via_pure_ast(source_dir, mp, cn)
            if ast_schema:
                results[key] = {"kind": "pydantic", "schema": ast_schema, "method": "ast_pure"}
                print(f"[ast-fallback] {cn} from {mp}: extracted via pure AST (model_json_schema failed)", file=sys.stderr, flush=True)
                continue

        # Phase 2a: AST compilation + model_json_schema (for import failures)
        # This handles cases where direct import fails due to missing dependencies.
        if fail_reason and ("ModuleNotFoundError" in fail_reason or "import failed" in fail_reason):
            ast_schema = _extract_via_ast(source_dir, mp, cn)
            if ast_schema:
                results[key] = {"kind": "pydantic", "schema": ast_schema, "method": "ast_compiled"}
                print(f"[ast-fallback] {cn} from {mp}: extracted via compiled AST (import failed)", file=sys.stderr, flush=True)
                continue

        # Phase 2c: Pure AST extraction fallback for import failures
        # If AST compilation also fails due to missing imports, try pure AST extraction.
        if fail_reason and ("ModuleNotFoundError" in fail_reason or "import failed" in fail_reason):
            ast_schema = _extract_via_pure_ast(source_dir, mp, cn)
            if ast_schema:
                results[key] = {"kind": "pydantic", "schema": ast_schema, "method": "ast_pure"}
                print(f"[ast-fallback] {cn} from {mp}: extracted via pure AST (import failed, compiled AST also failed)", file=sys.stderr, flush=True)
                continue

        # All phases failed
        if not fail_reason:
            if direct_blocked:
                fail_reason = "skipped (direct import blocked for this SDK)"
            else:
                fail_reason = "unknown"
        print(f"[degraded] {cn} from {mp}: {fail_reason} (will use basic JSON type)", file=sys.stderr, flush=True)
        results[key] = {"kind": None, "fail_reason": fail_reason}

    print(json.dumps({"results": results, "direct_blocked": direct_blocked}))

main()
'''


def preload_schemas_for_source_dir(
    source_dir: str,
    targets: list[tuple[str, str]],
    *,
    timeout: float = 600.0,
    venv_python: str | None = None,
) -> None:
    """Batch-probe all pydantic types for one SDK in a single subprocess.

    Imports the SDK once and extracts schemas for all (module_path, class_name)
    pairs, filling the ``_probe_type_in_subprocess`` lru_cache so that
    subsequent individual calls are cache hits (no subprocess spawn).

    Args:
        source_dir: SDK source root.
        targets: List of (module_path, class_name) pairs to probe.
        timeout: Total timeout for the batch (default 600s = 10min).
        venv_python: Python executable from the SDK bundle venv. When given, the
            batch subprocess is launched with it so that third-party SDK
            dependencies (jsonschema_path, pysbd, ...) can be imported.
    """
    root = str(Path(source_dir).resolve())
    if root in _BLOCKED_SOURCE_DIRS:
        print(f"   Input schema generation: skipped (circuit breaker active for {Path(root).name})")
        return

    # Deduplicate
    unique_targets = list(dict.fromkeys(targets))
    if not unique_targets:
        return

    # Remember the venv python for later _probe_type_in_subprocess single probes.
    global _CURRENT_VENV_PYTHON
    if venv_python:
        _CURRENT_VENV_PYTHON = venv_python

    python_exe = venv_python or sys.executable

    print(
        f"   Generating input schemas for tool parameters: "
        f"probing {len(unique_targets)} pydantic types (timeout={timeout}s)..."
    )

    from .path_resolver import infer_extra_sys_path_entries

    request = json.dumps(
        {
            "source_dir": root,
            "targets": [
                {
                    "module_path": mp,
                    "class_name": cn,
                    "extra_sys_path": infer_extra_sys_path_entries(root, mp),
                }
                for mp, cn in unique_targets
            ],
        }
    )
    try:
        t0 = time.monotonic()
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            [python_exe, "-c", _BATCH_PROBE_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Close stdin after writing the request so the child starts processing.
        proc.stdin.write(request)
        proc.stdin.close()

        # Stream child stderr progress lines to the user.
        import threading

        stderr_lines: list[str] = []

        def _read_stderr() -> None:
            for line in iter(proc.stderr.readline, ""):
                stderr_lines.append(line)
                # Forward progress lines (newline-terminated; strip before reprinting).
                msg = line.rstrip()
                if msg:
                    print(f"   {msg}")

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Read stdout (final result JSON).
        stdout = proc.stdout.read()
        proc.wait(timeout=timeout)
        stderr_thread.join(timeout=2.0)

        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            print(f"   Input schema generation: subprocess exited with code {proc.returncode} ({elapsed:.1f}s)")
            return
        lines = stdout.strip().splitlines()
        if not lines:
            print(f"   Input schema generation: no output ({elapsed:.1f}s)")
            return
        batch_result = json.loads(lines[-1])
        results = batch_result.get("results", {})

        # Fill _BATCH_CACHE so subsequent individual calls are cache hits.
        direct_count = 0
        failed_count = 0
        for mp, cn in unique_targets:
            key = f"{mp}::{cn}"
            entry = results.get(key)
            if entry and entry.get("kind"):
                _BATCH_CACHE[(root, mp, cn)] = entry
                direct_count += 1
            else:
                # Cache negative result too, so we don't retry.
                _BATCH_CACHE[(root, mp, cn)] = None
                failed_count += 1

        if batch_result.get("direct_blocked"):
            _DIRECT_IMPORT_BLOCKED.add(root)

        print(
            f"   Input schema generation done in {elapsed:.1f}s "
            f"({direct_count} succeeded, {failed_count} degraded to basic JSON types)"
        )
        if direct_count > 0:
            logger.info(
                "Batch probe: %d succeeded, %d failed (%.1fs)",
                direct_count,
                failed_count,
                elapsed,
            )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        _BLOCKED_SOURCE_DIRS.add(root)
        print(f"   Input schema generation: TIMEOUT after {timeout}s, remaining types will degrade to basic JSON types")
        logger.warning(
            "Batch probe for %s timed out after %ss, circuit breaker tripped",
            root,
            timeout,
        )
    except (json.JSONDecodeError, OSError) as exc:
        print(f"   Input schema generation: failed ({exc})")
        logger.debug("Batch probe for %s failed: %s", root, exc)


def _is_pydantic_model(cls: type[Any]) -> bool:
    try:
        from pydantic import BaseModel
    except ImportError:
        return False
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def _is_dataclass_type(cls: type[Any]) -> bool:
    return isinstance(cls, type) and dataclasses.is_dataclass(cls)


# ---------------------------------------------------------------------------
# Public helpers for wrapper generation (deserialization)
# ---------------------------------------------------------------------------


def import_type(source_dir: str, type_name: str) -> type[Any] | None:
    """Public wrapper around :func:`_import_type`."""
    return _import_type(source_dir, type_name)


def is_pydantic_model(cls: type[Any]) -> bool:
    """Public wrapper around :func:`_is_pydantic_model`."""
    return _is_pydantic_model(cls)


def is_dataclass_type(cls: type[Any]) -> bool:
    """Public wrapper around :func:`_is_dataclass_type`."""
    return _is_dataclass_type(cls)


def type_root(type_hint: str) -> str:
    """Public wrapper around :func:`_type_root`."""
    return _type_root(type_hint)


def split_union(type_hint: str) -> list[str]:
    """Public wrapper around :func:`_split_union`."""
    return _split_union(type_hint)


def get_type_module_path(source_dir: str, type_name: str) -> str | None:
    """Return the dotted module path where *type_name* is defined under *source_dir*."""
    index = _build_class_index(source_dir)
    return index.get(type_name)


def _class_node_from_ast(source_dir: str, type_name: str) -> ast.ClassDef | None:
    """Return the ``ClassDef`` AST node for *type_name* under *source_dir*, or None."""
    module_path = get_type_module_path(source_dir, type_name)
    if not module_path:
        return None
    root = Path(source_dir).resolve()
    rel = Path(*module_path.split(".")).with_suffix(".py")
    path = root / rel
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == type_name:
            return node
    return None


def get_model_class_info(
    source_dir: str, type_name: str, module_path: str | None = None
) -> tuple[str, str, str] | None:
    """Return ``(module_path, class_name, kind)`` when *type_name* is a model class.

    *kind* is ``"pydantic"`` for ``BaseModel`` subclasses or ``"dataclass"`` for
    decorated dataclasses.  When *module_path* is given, resolve through that
    module's import aliases (same as wrapper coercion).
    """
    resolved = _resolve_type_import(source_dir, type_name, module_path)
    if not resolved:
        return None
    module_path, class_name = resolved

    # Try runtime probe in isolated subprocess (accurate, with timeout).
    probe = _probe_type_in_subprocess(source_dir, module_path, class_name)
    if probe and probe.get("kind") == "pydantic":
        return module_path, class_name, "pydantic"

    # Fallback to AST inspection (handles both pydantic and dataclass).
    class_node = _class_node_from_module(source_dir, module_path, class_name)
    if class_node is None:
        return None

    root = Path(source_dir).resolve()
    rel = Path(*module_path.split(".")).with_suffix(".py")
    path = root / rel
    module_classes: dict[str, ast.ClassDef] = {}
    if path.is_file():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_classes = _module_class_index(tree)
        except (OSError, SyntaxError, UnicodeDecodeError):
            module_classes = {}

    if _ast_is_pydantic_class(class_node, module_classes):
        return module_path, class_name, "pydantic"

    if _has_dataclass_decorator(class_node):
        return module_path, class_name, "dataclass"

    return None


def _class_node_from_module(source_dir: str, module_path: str, class_name: str) -> ast.ClassDef | None:
    """Return the ``ClassDef`` AST node for *class_name* in *module_path*."""
    root = Path(source_dir).resolve()
    rel = Path(*module_path.split(".")).with_suffix(".py")
    path = root / rel
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _annotation_to_hint(annotation: Any) -> str:
    """Best-effort stringify of a runtime type annotation for recursive schema lookup."""
    if isinstance(annotation, str):
        return annotation
    if annotation is None or annotation is NoneType:
        return "None"

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", None)
    if origin is not None and args:
        origin_name = getattr(origin, "__name__", str(origin))
        arg_hints = ", ".join(_annotation_to_hint(arg) for arg in args)
        return f"{origin_name}[{arg_hints}]"

    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation)


def _has_dataclass_decorator(class_node: ast.ClassDef) -> bool:
    for dec in class_node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "dataclass":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
            return True
    return False


def _dataclass_schema_from_cls(
    cls: type[Any],
    source_dir: str,
    *,
    module_path: str | None = None,
    _visited: set[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Build JSON Schema from a runtime dataclass type."""
    if not _is_dataclass_type(cls):
        return None

    try:
        hints = get_type_hints(cls)
    except Exception as exc:  # pragma: no cover - forward refs / import issues
        logger.debug("get_type_hints failed for %s: %s", cls.__name__, exc)
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        hint = hints.get(field.name, field.type)
        hint_str = _annotation_to_hint(hint) if hint is not None else "Any"
        properties[field.name] = param_to_json_schema_property(
            type_hint=hint_str,
            source_dir=source_dir,
            fallback_json_type="string",
            module_path=module_path,
            _visited=_visited,
        )
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)

    if not properties:
        return None

    schema: dict[str, Any] = {
        "type": "object",
        "title": cls.__name__,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    schema["examples"] = [_minimal_value_for_schema(schema)]
    return schema


def _minimal_value_for_schema(schema: dict[str, Any]) -> Any:
    """Build a minimal JSON value from a JSON Schema fragment."""
    if "$ref" in schema:
        return {}
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            if option.get("type") != "null":
                return _minimal_value_for_schema(option)
        return None
    if "oneOf" in schema:
        for option in schema["oneOf"]:
            if option.get("type") != "null":
                return _minimal_value_for_schema(option)
        return None

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        example: dict[str, Any] = {}
        for key, sub in props.items():
            if key in required or len(required) <= 3:
                example[key] = _minimal_value_for_schema(sub)
        return example
    if schema_type == "array":
        items = schema.get("items") or {"type": "string"}
        return [_minimal_value_for_schema(items)]
    if schema_type == "string":
        if "enum" in schema and schema["enum"]:
            return schema["enum"][0]
        return ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return {}


def _ast_is_pydantic_class(
    class_node: ast.ClassDef,
    module_classes: dict[str, ast.ClassDef],
    *,
    pydantic_index: dict[str, set[str]] | None = None,
    current_module: str | None = None,
) -> bool:
    """True when *class_node* is a Pydantic ``BaseModel`` subclass (direct or indirect).

    Args:
        class_node: The AST ClassDef node to check.
        module_classes: Dict of class name -> ClassDef node in the same module.
        pydantic_index: Optional index from _build_pydantic_class_index for cross-module
            indirect inheritance detection. If provided and the class is found in the index,
            returns True immediately.
        current_module: Optional module path for cross-module lookup in pydantic_index.
    """
    class_name = class_node.name
    if pydantic_index and current_module and class_name in pydantic_index:
        if current_module in pydantic_index[class_name]:
            return True

    for base in class_node.bases:
        base_name = getattr(base, "id", None) or getattr(base, "attr", None)
        if base_name == "BaseModel":
            return True
        if isinstance(base, ast.Name) and base.id in module_classes:
            if _ast_is_pydantic_class(
                module_classes[base.id],
                module_classes,
                pydantic_index=pydantic_index,
                current_module=current_module,
            ):
                return True
    return False


def _module_class_index(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _ast_model_schema(
    source_dir: str, type_name: str, _visited: set[tuple[str, str]] | None = None
) -> dict[str, Any] | None:
    """Fallback: build object schema from ``class Name(BaseModel):`` AST."""
    index = _build_class_index(source_dir)
    import_module = index.get(type_name)
    if not import_module:
        return None
    return _ast_model_schema_for_module(source_dir, import_module, type_name, _visited=_visited)


def _ast_collect_pydantic_properties(
    class_node: ast.ClassDef,
    module_classes: dict[str, ast.ClassDef],
    *,
    source_dir: str,
    module_path: str | None,
    _visited: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Collect JSON Schema properties from a Pydantic class and its local bases."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id in module_classes:
            parent_props, parent_req = _ast_collect_pydantic_properties(
                module_classes[base.id],
                module_classes,
                source_dir=source_dir,
                module_path=module_path,
                _visited=_visited,
            )
            properties.update(parent_props)
            for name in parent_req:
                if name not in required:
                    required.append(name)

    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            field_name = stmt.target.id
            hint = ast.unparse(stmt.annotation) if stmt.annotation else "string"
            properties[field_name] = param_to_json_schema_property(
                type_hint=hint,
                source_dir=source_dir,
                fallback_json_type="string",
                module_path=module_path,
                _visited=_visited,
            )
            if stmt.value is None:
                if field_name not in required:
                    required.append(field_name)
            elif field_name in required:
                # Overriding a parent-required field with a default makes it optional.
                required.remove(field_name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    field_name = target.id
                    properties.setdefault(field_name, {"type": "string"})

    return properties, required


def _hint_to_json_type_recursive(
    hint: str,
    *,
    source_dir: str,
    module_path: str | None = None,
    _visited: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Convert a type hint string to a JSON Schema type definition, with recursive resolution.

    Unlike _hint_to_json_type, this function will recursively resolve nested structured types
    by calling pydantic_schema_for_type for Capitalized identifiers that aren't builtins.
    """
    hint = hint.strip()

    if hint.startswith("Union[") or hint.startswith("Optional["):
        inner = hint[6:-1] if hint.startswith("Union") else hint[9:-1]
        parts = _split_top_level_commas(inner)
        has_none = any(p.strip() == "None" for p in parts)
        non_none_parts = [p for p in parts if p.strip() != "None"]
        if len(non_none_parts) == 1:
            base = _hint_to_json_type_recursive(
                non_none_parts[0].strip(),
                source_dir=source_dir,
                module_path=module_path,
                _visited=_visited,
            )
            if has_none:
                base["nullable"] = True
            return base
        return {
            "anyOf": [
                _hint_to_json_type_recursive(
                    p.strip(), source_dir=source_dir, module_path=module_path, _visited=_visited
                )
                for p in non_none_parts
            ]
        }

    if hint.startswith("list[") or hint.startswith("List["):
        inner = hint[5:-1] if hint.startswith("list") else hint[6:-1]
        return {
            "type": "array",
            "items": _hint_to_json_type_recursive(
                inner.strip(), source_dir=source_dir, module_path=module_path, _visited=_visited
            ),
        }

    if hint.startswith("dict[") or hint.startswith("Dict["):
        inner = hint[5:-1] if hint.startswith("dict") else hint[6:-1]
        parts = _split_top_level_commas(inner)
        if len(parts) >= 2:
            value_type = _hint_to_json_type_recursive(
                parts[1].strip(), source_dir=source_dir, module_path=module_path, _visited=_visited
            )
            return {"type": "object", "additionalProperties": value_type}
        return {"type": "object"}

    if hint.startswith("Literal["):
        # Literal["a", "b"] → {"type": "string", "enum": ["a", "b"]};
        # mirrors the Literal branch inside _BATCH_PROBE_SCRIPT.
        inner = hint[len("Literal[") : -1]
        parts = _split_top_level_commas(inner)
        enum_values = []
        for part in parts:
            part = part.strip()
            if part.startswith('"') and part.endswith('"'):
                enum_values.append(part[1:-1])
            elif part.startswith("'") and part.endswith("'"):
                enum_values.append(part[1:-1])
            elif part in ("True", "False"):
                enum_values.append(part == "True")
            elif part == "None":
                continue
            else:
                try:
                    enum_values.append(int(part))
                except ValueError:
                    try:
                        enum_values.append(float(part))
                    except ValueError:
                        enum_values.append(part)
        if enum_values:
            first_val = enum_values[0]
            if isinstance(first_val, str):
                return {"type": "string", "enum": enum_values}
            elif isinstance(first_val, bool):
                return {"type": "boolean", "enum": enum_values}
            elif isinstance(first_val, int):
                return {"type": "integer", "enum": enum_values}
            elif isinstance(first_val, float):
                return {"type": "number", "enum": enum_values}
        return {"type": "string", "enum": enum_values} if enum_values else {"type": "object"}

    basic_mapping: dict[str, str] = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "bytes": "string",
        "Any": "object",
        "None": "null",
    }
    if hint in basic_mapping:
        return {"type": basic_mapping[hint]}

    root_name = hint.split("[")[0].split(".")[-1].strip("'\"")
    if (
        root_name
        and root_name[0].isupper()
        and root_name not in {"Union", "Optional", "List", "Dict", "Set", "Tuple", "Any"}
    ):
        nested_schema = pydantic_schema_for_type(source_dir, hint, module_path=module_path, _visited=_visited)
        if nested_schema:
            return nested_schema

    return {"type": "object"}
