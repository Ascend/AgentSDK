#!/usr/bin/env python3

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

# ruff: noqa
"""Tool Registry Bridge — bridges SourceOperation → AgentToolSpec → Tool Registry.

Converts parsed source operations into executable Agent tools with bash-callable
wrapper scripts.  Every operation — class method or standalone function — is
uniformly handled via ``call_type="bash"`` so that import / instantiation logic
lives inside an isolated subprocess rather than the main server process.

Key design decisions
--------------------
* **Unified bash call_type**: all operations use ``call_type="bash"``, even
  standalone functions.  This avoids the ``_PYTHON_FUNCTION_REGISTRY`` problem.
* **Wrapper scripts live alongside the persisted ``AgentToolSpec`` JSON files**
  — inside the active bundle directory when converting through the bundle flow,
  or under the global fallback ``~/.clawcodex/agent-tools/scripts/`` otherwise
  (never in the source directory).
* **Name normalization**: original tool names like ``LLM.invoke`` or
  ``Utils.load_config`` are converted to kebab-case (``llm-invoke``,
  ``utils-load-config``).  The returned name map allows the caller to update
  ``SkillSpec.allowed_tools`` so agent markdown can reference the registered
  kebab-case names directly.

Usage::

    from extensions.sop_converter.tool_registry_bridge import register_component_tools

    name_map = register_component_tools(components, str(source_dir), persist=True)
    # name_map: {"LLM.invoke": "llm-invoke", ...}
    # Update SkillSpec.allowed_tools with kebab-case names before writing markdown.
"""

# pylint: disable=too-many-lines,too-many-nested-blocks
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from ..adapters import DEFAULTS

from ..core.path_resolver import (
    format_extra_sys_path_inserts,
    infer_extra_sys_path_entries,
    resolve_source_file,
)
from ..core.search_tags import generate_search_tags
from ..core.source_parser import SourceComponent, SourceOperation, ParamSpec
from ..core.sdk_parser import SdkMethod, SdkParam
from ..core.sdk_serialization import (
    WRAPPER_COERCION_HELPERS,
    WRAPPER_SERIALIZATION_HELPERS,
)
from ..core.tool_dependencies import (
    _PRIMITIVE_TYPES,
    ToolOperationDeps,
    build_tool_dependency_index,
    enrich_input_schema_with_dependencies,
    extract_type_roots,
    to_kebab_tool_name,
)

# L1 / L2 helpers — lifecycle catalog hook + tool-dependencies.yaml generation.
from ..core.heuristics.lifecycle import (
    infer_lifecycle_kind,
    inject_resource_ref_schema,
    invoke_lifecycle_id_param,
    lifecycle_fallback_payload,
    lifecycle_metadata_payload,
)
from ..core.bundle_resources import ResourceBinding, load_resource_bindings
from ..core.dependency import (
    ToolDependencyGraph,
    write_tool_dependencies,
)

from ..core.import_alias_resolver import ModuleImportIndex
from ..core.type_schema import (
    collect_probe_targets,
    get_model_class_info,
    param_to_json_schema_property,
    preload_schemas_for_source_dir,
    split_union,
    type_root,
)

logger = logging.getLogger(__name__)


def _stable_resource_handle(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    return ""


def resolve_catalog_handle_from_args(
    args: dict[str, Any],
    catalog_fallback: dict[str, Any],
) -> str:
    """Resolve an invoke handle without making an SDK parameter name primary."""
    candidates = [
        "resource_ref",
        str(catalog_fallback.get("handle_field") or ""),
        str(catalog_fallback.get("id_arg") or ""),
        "agent_id",
        "resource_id",
        "id",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        handle = _stable_resource_handle(args.get(candidate))
        if handle:
            return handle
    return ""


# Backward-compatible test/private helper name kept for older imports.
_infer_extra_sys_path_entries = infer_extra_sys_path_entries


def _bridge_progress_enabled() -> bool:
    """Show convert progress on interactive CLI, not under unittest/pytest."""
    if os.environ.get("CLAWCODEX_SOP_QUIET", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("CLAWCODEX_SOP_PROGRESS", "").strip().lower() in {"1", "true", "yes"}:
        return True
    argv = " ".join(sys.argv).lower()
    if "unittest" in argv or "pytest" in argv:
        return False
    return sys.stderr.isatty()


def _bridge_progress(message: str, *, end: str = "\n") -> None:
    if not _bridge_progress_enabled():
        return
    print(message, end=end, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Legacy global script dir (used when no bundle_dir is supplied).
# Backward-compatible aliases — tests patch these on the package-root module.
# The scripts directory itself is created lazily on first use by the
# wrapper-script writers (idempotent mkdir(parents=True, exist_ok=True)),
# never as an import-time side effect.
TOOL_DIR = DEFAULTS.tool_authoring.TOOL_DIR
SCRIPTS_DIR = TOOL_DIR / "scripts"


# ---------------------------------------------------------------------------
# Type hint → JSON Schema
# ---------------------------------------------------------------------------


def _resource_type_from_hint(
    *,
    resolver: ModuleImportIndex | None,
    module_path: str,
    type_hint: str | None,
) -> str:
    """Return the normalized resource token used by dependency metadata."""
    if not type_hint:
        return ""
    if resolver and module_path:
        try:
            resolved = resolver.resolve_type_identity(module_path, type_hint)
            if resolved:
                return resolved
        except Exception:  # nosec
            pass
    roots = sorted(extract_type_roots(type_hint))
    return roots[0] if roots else ""


def _resource_type_hint_tokens(type_hint: str | None) -> set[str]:
    """Return lifecycle-comparison tokens visible in a raw type hint."""
    if not type_hint:
        return set()
    tokens: set[str] = set()
    for root in extract_type_roots(type_hint):
        if root and root.rsplit("_", 1)[-1] not in _PRIMITIVE_TYPES:
            tokens.add(root)
            if "_" in root:
                tokens.add(root.rsplit("_", 1)[-1])
    for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", type_hint):
        if not name or not name[0].isupper():
            continue
        token = re.sub(r"[^a-z0-9]+", "", name.lower())
        if token and token not in _PRIMITIVE_TYPES:
            tokens.add(token)
    return tokens


def _resource_type_tokens_for_op(op: SourceOperation) -> set[str]:
    tokens: set[str] = set()
    tokens.update(_resource_type_hint_tokens(op.return_type))
    for param in op.parameters:
        if param.name.startswith("*"):
            continue
        tokens.update(_resource_type_hint_tokens(param.type_hint))
    return tokens


def _first_resource_type_for_op(
    op: SourceOperation,
    *,
    resolver: ModuleImportIndex | None,
    module_path: str,
    prefer_return: bool,
) -> str:
    hints: list[str | None] = []
    if prefer_return:
        hints.append(op.return_type)
    hints.extend(param.type_hint for param in op.parameters if param.required and not param.name.startswith("*"))
    if not prefer_return:
        hints.append(op.return_type)
    for hint in hints:
        token = _resource_type_from_hint(
            resolver=resolver,
            module_path=module_path,
            type_hint=hint,
        )
        if token and token.rsplit("_", 1)[-1] not in _PRIMITIVE_TYPES:
            return token
    return ""


_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "None": "null",
    "NoneType": "null",
    "Path": "string",
    "UUID": "string",
    "UUID4": "string",
    "RCConfig": "object",
    "AdapterBundle": "object",
    "Stage": "string",
    "StageResult": "object",
}

# Python identifiers from ast.unparse() that should become JSON literals.
_LITERAL_DEFAULT_MAP: dict[str, Any] = {
    "True": True,
    "False": False,
    "None": None,
}


def _strip_optional_union(type_hint: str) -> str:
    """Reduce ``Optional[X]``, ``Union[X, None]``, and ``X | None`` to ``X``."""
    cleaned = type_hint.strip()

    for prefix in ("Optional[", "Union["):
        if cleaned.startswith(prefix) and cleaned.endswith("]"):
            inner = cleaned[len(prefix) : -1]
            parts = [p.strip() for p in inner.split(",")]
            for part in parts:
                if part not in ("None", "NoneType"):
                    return _strip_optional_union(part)

    if "|" in cleaned:
        parts = [p.strip() for p in cleaned.split("|")]
        for part in parts:
            if part not in ("None", "NoneType"):
                return _strip_optional_union(part)

    return cleaned


def _type_hint_to_json_type(type_hint: str | None) -> str:
    """Map a Python type-hint string to a JSON Schema ``type`` value.

    >>> _type_hint_to_json_type("str")
    'string'
    >>> _type_hint_to_json_type("Optional[int]")
    'integer'
    >>> _type_hint_to_json_type("dict[str, TeamAgentSpec] | None")
    'object'
    >>> _type_hint_to_json_type("Iterable[str | Path] | None")
    'array'
    >>> _type_hint_to_json_type(None)
    'string'
    """
    if not type_hint:
        return "string"

    cleaned = _strip_optional_union(type_hint.strip())

    # Handle List[...], Sequence[...], Iterable[...] → array
    if cleaned.startswith(
        (
            "List[",
            "list[",
            "Sequence[",
            "sequence[",
            "Iterable[",
            "iterable[",
            "Set[",
            "set[",
            "FrozenSet[",
            "frozenset[",
        )
    ):
        return "array"

    # Handle Dict[...], Mapping[...] → object
    if cleaned.startswith(
        (
            "Dict[",
            "dict[",
            "Mapping[",
            "mapping[",
            "MutableMapping[",
        )
    ):
        return "object"

    # SDK / pathlib types (bare name or generic alias root)
    root = cleaned.split("[", 1)[0]
    if root in _TYPE_MAP:
        return _TYPE_MAP[root]

    # Direct lookup
    return _TYPE_MAP.get(cleaned, "string")


def _is_loose_mapping_inputs_type_hint(type_hint: str | None) -> bool:
    """True when ``inputs`` is typed as an open mapping (not a concrete model/str).

    Applies generically to any SDK method whose ``inputs`` parameter is annotated
    as ``Any``, ``object``, ``dict``, ``Mapping``, etc.  Explicit ``str`` or
    Pydantic model types are excluded.
    """
    if not type_hint:
        return False

    cleaned = _strip_optional_union(type_hint.strip())
    if cleaned in ("Any", "object", "dict", "Dict", "Mapping", "mapping", "MutableMapping"):
        return True
    return cleaned.startswith(("Dict[", "dict[", "Mapping[", "mapping[", "MutableMapping["))


def _is_dict_type_hint(type_hint: str | None) -> bool:
    """True when a parameter is explicitly typed as a mapping (not Any/object)."""
    if not type_hint:
        return False

    cleaned = _strip_optional_union(type_hint.strip())
    if cleaned == "dict":
        return True
    return cleaned.startswith(("Dict[", "dict[", "Mapping[", "mapping[", "MutableMapping["))


def _normalize_schema_default(default: Any, *, json_type: str) -> Any:
    """Coerce ast-unparsed Python literal defaults into JSON Schema values."""
    if isinstance(default, str) and default in _LITERAL_DEFAULT_MAP:
        return _LITERAL_DEFAULT_MAP[default]
    if json_type == "boolean" and isinstance(default, str):
        lowered = default.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    return default


def _adjust_pipeline_execute_stage_schema(
    op: SourceOperation,
    properties: dict[str, Any],
    required: list[str],
) -> None:
    """Relax workflow executor schemas without assuming a specific SDK.

    A workflow executor is identified by its context-shaped parameters rather
    than by a module path or a vendor-specific type name. The same rule is
    used by wrapper generation below so schema and execution stay aligned.
    """
    if not _is_workflow_execute_operation(op):
        return

    for key in ("config", "adapters", "run_id", "context"):
        if key in properties:
            properties[key]["description"] = properties[key].get("description") or (
                "Optional SDK runtime configuration"
                if key == "config"
                else (
                    "Optional SDK runtime adapters"
                    if key == "adapters"
                    else (
                        "Optional workflow context mapping"
                        if key == "context"
                        else "Optional; defaults to run_dir directory name"
                    )
                )
            )
            if key in required:
                required.remove(key)

    for key in ("stage", "stage_id", "step", "step_id"):
        if key in properties:
            properties[key]["description"] = "Workflow stage/step identifier (name or numeric value)"
    for key in ("run_dir", "project_dir", "workspace", "work_dir"):
        if key in properties:
            properties[key]["description"] = "Workflow run workspace directory"


def _is_workflow_execute_operation(op: SourceOperation) -> bool:
    """Return whether *op* has the conventional workflow execution shape.

    This deliberately uses semantic parameter names instead of a package,
    enum, or class name. SDKs can therefore expose ``run_stage(stage_id,
    project_dir)`` or ``execute_step(stage, workspace)`` without an adapter.
    """
    names = {p.name.lstrip("*").lower() for p in op.parameters}
    stage_names = {"stage", "stage_id", "step", "step_id", "phase", "phase_id"}
    workspace_names = {"run_dir", "project_dir", "workspace", "work_dir"}
    return bool(names & stage_names) and bool(names & workspace_names)


# ---------------------------------------------------------------------------
# Name conversion
# ---------------------------------------------------------------------------


def _to_kebab_case(name: str) -> str:
    """Convert dot.separated / snake_case → kebab-case.

    Only dots, double-underscores, and single underscores act as word
    separators.  CamelCase within a segment is preserved as one word:
    ``"VideoProcessor"`` → ``"videoprocessor"`` (not ``"video-processor"``).

    >>> _to_kebab_case("VideoProcessor.transcode")
    'videoprocessor-transcode'
    >>> _to_kebab_case("video_ops.transcode")
    'video-ops-transcode'
    >>> _to_kebab_case("utils__load_config")
    'utils-load-config'
    >>> _to_kebab_case("LLM.invoke")
    'llm-invoke'
    >>> _to_kebab_case("foundation.LLM.invoke")
    'foundation-llm-invoke'
    """
    # Replace dots and double-underscores with hyphens
    s = name.replace(".", "-").replace("__", "-")
    # Replace single underscores with hyphens
    s = s.replace("_", "-")
    # Collapse multiple consecutive hyphens
    s = re.sub(r"-+", "-", s)
    # Strip leading/trailing hyphens and lowercase
    return s.strip("-").lower()


# ---------------------------------------------------------------------------
# Module path resolution
# ---------------------------------------------------------------------------


def _resolve_module_path(
    component: SourceComponent,
    source_dir: str,
    file_stem: str,
) -> str:
    """Infer the Python import path for a specific source file within a component.

    ``SourceComponent.file_path`` is relative to ``source_dir.parent``
    (see :class:`SourceCodeParser._walk_module`).  Since the wrapper script
    injects *source_dir* into ``sys.path``, the import path must be relative
    to *source_dir*.

    Example::

        source_dir   = "/mnt/d/projects/AutoResearchClaw"
        component.file_path = "researchclaw/literature"
        file_stem    = "llm"
        → "core.foundation.llm"

    Returns:
        Dotted Python module path suitable for ``importlib.import_module()``.
    """
    source_dir_path = Path(source_dir).resolve()
    source_dir_name = source_dir_path.name
    comp_rel = Path(component.file_path)

    # Strip the source_dir name prefix from file_path (it's relative to parent)
    try:
        module_dir = comp_rel.relative_to(source_dir_name)
    except ValueError:
        # If file_path doesn't start with source_dir_name, use it as-is
        # (e.g. when source_dir itself is the repo root)
        module_dir = comp_rel

    parts = list(module_dir.parts) if module_dir.parts != (".",) else []
    parts.append(file_stem)
    return ".".join(parts)


def _script_name_for_class(module_path: str, class_name: str) -> str:
    """Build a unique script filename for a class within a module.

    Uses a short hash of the full module path to avoid collisions
    between identically-named classes from different projects/packages.
    """
    hash_hex = hashlib.sha256(module_path.encode()).hexdigest()[:8]
    return f"{class_name}_{hash_hex}.py"


def _script_name_for_functions(module_path: str, file_stem: str) -> str:
    """Build a unique script filename for standalone functions in a module."""
    hash_hex = hashlib.sha256(module_path.encode()).hexdigest()[:8]
    return f"{file_stem}_fn_{hash_hex}.py"


def _module_path_needs_importlib(module_path: str) -> bool:
    """Return True if any segment of *module_path* is not a valid Python identifier.

    Python ``from X import Y`` requires every dot-separated segment of X to be a
    valid identifier (``[a-zA-Z_][a-zA-Z0-9_]*``).  Some SDK source directories
    contain hyphens (e.g. ``agent-perf-analyzer``, ``gitcode-issue-reply``).
    For those we must use ``importlib.import_module()`` instead.
    """
    _ident_re = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    return not all(_ident_re.match(segment) for segment in module_path.split("."))


# ---------------------------------------------------------------------------
# Class constructor parameter handling
# ---------------------------------------------------------------------------


def _skip_variadic_params(params: list[ParamSpec]) -> list[ParamSpec]:
    return [p for p in params if not p.name.startswith("*")]


def _has_signature_default(param: ParamSpec) -> bool:
    """True when the generated stub should use ``name=...`` syntax."""
    return param.default is not None or not param.required


def _sort_params_for_python_signature(params: list[ParamSpec]) -> list[ParamSpec]:
    """Required positional params first, then params with defaults (Python syntax rule)."""
    positional = [p for p in params if not _has_signature_default(p)]
    defaulted = [p for p in params if _has_signature_default(p)]
    return positional + defaulted


def _merge_init_and_method_params(
    init_params: list[ParamSpec],
    method_params: list[ParamSpec],
) -> list[ParamSpec]:
    """Merge ``__init__`` params with method params; method wins on name clash."""
    method_skip = _skip_variadic_params(method_params)
    init_skip = _skip_variadic_params(init_params)
    method_names = {p.name for p in method_skip}

    by_name: dict[str, ParamSpec] = {}
    order: list[str] = []

    for param in init_skip:
        if param.name in method_names:
            continue
        by_name[param.name] = param
        order.append(param.name)

    for param in method_skip:
        if param.name not in by_name:
            order.append(param.name)
        by_name[param.name] = param

    merged = [by_name[name] for name in order]
    return _sort_params_for_python_signature(merged)


def _param_signature_parts(params: list[ParamSpec]) -> list[str]:
    parts: list[str] = []
    for param in params:
        if param.name.startswith("*"):
            continue
        if param.default is not None:
            parts.append(f"{param.name}={param.default}")
        elif not param.required:
            parts.append(f"{param.name}=None")
        else:
            parts.append(param.name)
    return parts


def _generate_get_instance_helper(init_params: list[ParamSpec] | None) -> str:
    """Generate ``_get_instance`` (and optional ``_resolve_init_kwargs``) helper."""
    callable_init = _skip_variadic_params(init_params or [])
    if not callable_init:
        return (
            "def _get_instance(class_name, module_name):\n"
            '    """Lazily create and cache a class instance."""\n'
            "    if class_name not in _instances:\n"
            "        module = importlib.import_module(module_name)\n"
            "        cls = getattr(module, class_name)\n"
            "        _instances[class_name] = cls()\n"
            "    return _instances[class_name]\n"
        )

    resolver_lines: list[str] = ["    kwargs = dict(provided)"]
    for param in callable_init:
        if param.default is not None:
            resolver_lines.append(f'    kwargs.setdefault("{param.name}", {param.default})')
        else:
            # Treat explicit ``None`` the same as "not provided" so that
            # the auto-resolution path (module-level factory function) gets a
            # chance to supply the value.  Otherwise a property-accessor stub
            # that defaults ``card=None`` would skip resolution and crash the
            # SDK constructor with ``NoneType … has no attribute …``.
            resolver_lines.append(f'    if kwargs.get("{param.name}") is None:')
            resolver_lines.append(f'        _fn = getattr(module, "{param.name}", None)')
            resolver_lines.append("        if callable(_fn):")
            resolver_lines.append("            try:")
            resolver_lines.append(f'                kwargs["{param.name}"] = _fn()')
            resolver_lines.append("            except TypeError:")
            resolver_lines.append('                _team = os.environ.get("OPENJIUWEN_TEAM_NAME", "team")')
            resolver_lines.append("                try:")
            resolver_lines.append(f'                    kwargs["{param.name}"] = _fn(team_name=_team)')
            resolver_lines.append("                except TypeError:")
            resolver_lines.append("                    pass")

    required = [p.name for p in callable_init if p.required and p.default is None]
    if required:
        missing_check = " or ".join(f'kwargs.get("{name}") is None' for name in required)
        resolver_lines.append(f"    if {missing_check}:")
        resolver_lines.append(f"        _missing = [n for n in {required!r} if kwargs.get(n) is None]")
        resolver_lines.append('        raise TypeError("Missing constructor argument(s): " + ", ".join(_missing))')
    resolver_lines.append("    return kwargs")

    resolver_body = "\n".join(resolver_lines)
    return (
        "def _resolve_init_kwargs(module, **provided):\n"
        f"{resolver_body}\n\n"
        "def _get_instance(class_name, module_name, **init_kwargs):\n"
        '    """Lazily create and cache a class instance keyed by constructor args."""\n'
        "    cache_key = (class_name, json.dumps(_to_jsonable(init_kwargs), sort_keys=True, ensure_ascii=False))\n"
        "    if cache_key not in _instances:\n"
        "        module = importlib.import_module(module_name)\n"
        "        cls = getattr(module, class_name)\n"
        "        resolved = _resolve_init_kwargs(module, **init_kwargs)\n"
        "        _instances[cache_key] = cls(**resolved)\n"
        "    return _instances[cache_key]\n"
    )


# ---------------------------------------------------------------------------
# Wrapper script SDK symbol imports
# ---------------------------------------------------------------------------

_BUILTIN_DEFAULT_NAMES = frozenset({"True", "False", "None"})


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _resolve_relative_import(module_name: str, level: int, module: str | None) -> str:
    parts = module_name.split(".")
    if level > len(parts):
        base: list[str] = []
    else:
        base = parts[: len(parts) - level]
    if module:
        return ".".join([*base, *module.split(".")])
    return ".".join(base)


def _parse_import_map(source_file: Path, module_name: str) -> dict[str, str]:
    """Map local symbol names to importable modules from a source file."""
    if not source_file.is_file():
        return {}

    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}

    mapping: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                mapping[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            resolved_module = (
                _resolve_relative_import(module_name, node.level, node.module) if node.level else (node.module or "")
            )
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                mapping[local] = resolved_module
    return mapping


def _identifiers_in_default(default: str) -> set[str]:
    """Extract root names referenced by a default-value expression."""
    cleaned = default.strip()
    if not cleaned or cleaned in _BUILTIN_DEFAULT_NAMES:
        return set()

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return set(re.findall(r"(?<![.\w])([A-Za-z_][A-Za-z0-9_]*)(?=\.[A-Za-z_])", cleaned))

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in _BUILTIN_DEFAULT_NAMES:
                names.add(node.id)
        elif isinstance(node, ast.Attribute):
            root: ast.AST = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id not in _BUILTIN_DEFAULT_NAMES:
                names.add(root.id)
    return names


def _collect_runtime_symbols(
    ops: list[SourceOperation],
    init_params: list[ParamSpec] | None,
) -> set[str]:
    """Collect symbols that must exist at wrapper import time (default values)."""
    symbols: set[str] = set()

    def _scan_params(params: list[ParamSpec]) -> None:
        for param in params:
            if param.default is not None:
                symbols.update(_identifiers_in_default(str(param.default)))

    if init_params:
        _scan_params(_skip_variadic_params(init_params))

    for op in ops:
        _scan_params(op.parameters)

    return symbols


# ---------------------------------------------------------------------------
# Module working directory resolution
# ---------------------------------------------------------------------------

# Project markers for determining the effective CWD of a wrapped module.
# Walk up from source_file.parent; stop at the innermost directory containing
# any of these markers.  This handles both flat SDKs (JiuwenAgent: markers at
# SDK root == _SOURCE_DIR) and nested SDKs (data_generation_platform: markers
# at the subproject root, not the monorepo _SOURCE_DIR).
_PROJECT_MARKER_FILES: frozenset[str] = frozenset(
    {
        "config.json",
        "config.yaml",
        "config.yml",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
    }
)
_PROJECT_MARKER_DIRS: frozenset[str] = frozenset({"backend"})


def _resolve_module_working_dir(source_dir: str, module_name: str) -> str:
    """Return the best CWD for a wrapped module.

    Walks up from the module's source file directory looking for common
    project markers.  Returns the innermost match so that nested SDK apps
    resolve to their subproject root while flat SDKs resolve to *source_dir*.

    Flatten SDK (e.g. JiuwenAgent):
        ``pyproject.toml`` at the SDK root → returns *source_dir*.
    Nested SDK (e.g. data_generation_platform under mindsdk-referenceapps):
        ``config.json`` at the subproject root → returns the subproject dir.
    """
    source_file = resolve_source_file(source_dir, module_name)
    if not source_file.is_file():
        return source_dir

    root = Path(source_dir).resolve()
    current = source_file.parent.resolve()

    for _ in range(20):  # safety limit — walk at most 20 levels up
        for marker in _PROJECT_MARKER_FILES:
            if (current / marker).is_file():
                return str(current)
        for marker in _PROJECT_MARKER_DIRS:
            marker_dir = current / marker
            if marker_dir.is_dir() and any(marker_dir.rglob("*.py")):
                return str(current)
        if current == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    return source_dir


def _format_wrapper_imports(
    symbols: set[str],
    import_map: dict[str, str],
    module_name: str,
) -> str:
    """Render import lines for wrapper scripts.

    Uses ``from ... import ...`` for modules with valid Python identifiers,
    and ``importlib.import_module()`` attribute access for modules whose
    path segments contain hyphens or other non-identifier characters.
    """
    if not symbols:
        return ""

    by_module: dict[str, list[str]] = {}
    for symbol in sorted(symbols):
        resolved_module = import_map.get(symbol) or module_name
        by_module.setdefault(resolved_module, []).append(symbol)

    lines: list[str] = []
    for resolved_module in sorted(by_module):
        names = sorted(by_module[resolved_module])
        if _module_path_needs_importlib(resolved_module):
            mod_alias = f"_mod_{hashlib.sha256(resolved_module.encode()).hexdigest()[:8]}"
            lines.append(f'{mod_alias} = importlib.import_module("{resolved_module}")')
            for name in names:
                lines.append(f"{name} = {mod_alias}.{name}")
        else:
            lines.append(f"from {resolved_module} import {', '.join(names)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI handler detection (subprocess bridge — mirrors CLI mode)
# ---------------------------------------------------------------------------

_CLI_EXCLUDED_HANDLER_NAMES = frozenset(
    {
        "main",
        "build_parser",
    }
)


def _is_namespace_args_param(param: ParamSpec) -> bool:
    if param.name != "args":
        return False
    if not param.type_hint:
        return False
    return "Namespace" in param.type_hint


def _is_cli_handler_op(op: SourceOperation, dispatch_map: dict[str, str]) -> bool:
    """True when *op* should run via CLI subprocess instead of importlib."""
    if op.class_name is not None:
        return False
    if op.name in _CLI_EXCLUDED_HANDLER_NAMES:
        return False
    if op.name not in dispatch_map:
        return False
    if op.name.startswith("cmd_"):
        return True
    return any(_is_namespace_args_param(p) for p in op.parameters)


def _source_file_uses_argparse(source_file: Path) -> bool:
    """True when *source_file* contains ``argparse.add_argument`` calls."""
    if not source_file.is_file():
        return False
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                return True
    return False


def _is_cli_main_op(
    op: SourceOperation,
    source_dir: str,
    module_name: str,
) -> bool:
    """True when *op* is a CLI main entry point — a parameterless standalone
    function whose source file uses argparse to read from ``sys.argv``.

    These functions cannot be called via importlib because their input comes
    from ``sys.argv``, not from Python parameters.  They need subprocess mode
    so that CLI arguments are passed on the real command line.
    """
    if op.class_name is not None:
        return False
    if op.parameters:
        return False
    if op.name in _CLI_EXCLUDED_HANDLER_NAMES:
        return False
    source_file = resolve_source_file(source_dir, module_name)
    return _source_file_uses_argparse(source_file)


def _extract_command_literal(test: ast.AST) -> str | None:
    """Extract subcommand from ``command == "foo"`` in ``main()`` dispatch."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    if not isinstance(test.ops[0], (ast.Eq, ast.Is)):
        return None
    left, right = test.left, test.comparators[0]
    if isinstance(left, ast.Name) and left.id == "command":
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            return right.value
    return None


def _extract_cmd_handler_from_body(body: list[ast.stmt]) -> str | None:
    for stmt in body:
        if not isinstance(stmt, ast.Return) or stmt.value is None:
            continue
        value = stmt.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            name = value.func.id
            if name.startswith("cmd_"):
                return name
    return None


def _walk_command_dispatch(node: ast.stmt, dispatch: dict[str, str]) -> None:
    if not isinstance(node, ast.If):
        return
    subcommand = _extract_command_literal(node.test)
    handler = _extract_cmd_handler_from_body(node.body)
    if subcommand and handler:
        dispatch[handler] = subcommand
    if not node.orelse:
        return
    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        _walk_command_dispatch(node.orelse[0], dispatch)
        return
    for child in node.orelse:
        _walk_command_dispatch(child, dispatch)


def _parse_cli_dispatch_map(source_file: Path) -> dict[str, str]:
    """Map ``cmd_*`` handler names to CLI subcommand strings from ``main()``."""
    if not source_file.is_file():
        return {}
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        logger.debug("Cannot parse CLI dispatch from %s: %s", source_file, exc)
        return {}

    dispatch: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for stmt in node.body:
                _walk_command_dispatch(stmt, dispatch)
    return dispatch


def _resolve_cli_argv_prefix(
    source_dir: str,
    source_file: Path,
    *,
    cli_prefix_override: str | None = None,
) -> list[str]:
    """CLI argv prefix for subprocess dispatch (reuses CLI discovery)."""
    from extensions.sop_converter.workflow_mode.bridge.cli_discovery import (
        discover_cli_prefix,
        split_cli_prefix,
    )

    project_name = Path(source_dir).name
    discovered = discover_cli_prefix(
        Path(source_dir),
        project_name,
        override=cli_prefix_override,
    )
    if discovered:
        prefix = split_cli_prefix(discovered)
        if prefix:
            return prefix

    return [sys.executable, str(source_file.resolve())]


_CLI_SUBPROCESS_OPTIONAL_PARAMS = "__stdin_config: dict | None = None, __env: dict | None = None"


_CLI_SUBPROCESS_STDIN_ENV_BODY = """
    # Merge session/runtime secrets for nested CLI subprocesses.
    _bridge_stdin = __stdin_config if __stdin_config is not None else globals().get("_bridge_stdin_config")
    _bridge_env_extra = __env if __env is not None else globals().get("_bridge_subprocess_env")
    _stdin_payload = dict(_bridge_stdin or {})
    if _interactive_input_queue:
        _stdin_payload.setdefault("llm_api_key", _interactive_input_queue[0])
    _stdin_input = _json.dumps(_stdin_payload) if _stdin_payload else None
    _run_env = {**os.environ, **(_bridge_env_extra or {})}
    # When no stdin payload is available use DEVNULL so input() / sys.stdin.read()
    # fail fast (EOFError) instead of blocking on inherited stdin for 300 s.
    _stdin_kwarg = {'input': _stdin_input} if _stdin_input else {'stdin': subprocess.DEVNULL}
"""


def _generate_cli_handler_stub(op: SourceOperation, *, subcommand: str) -> str:
    """Generate a subprocess-based stub for a CLI ``cmd_*`` handler."""
    docstring = op.description.replace('"', '\\"') if op.description else op.name
    return (
        f"def {op.name}(args: str, {_CLI_SUBPROCESS_OPTIONAL_PARAMS}) -> dict:\n"
        f'    """{docstring}"""\n'
        "    import shlex\n"
        "    import subprocess\n"
        "    import json as _json\n"
        "    tail = shlex.split(args) if args else []\n"
        f"    argv = [*CLI_PREFIX, {subcommand!r}, *tail]\n"
        + _CLI_SUBPROCESS_STDIN_ENV_BODY
        + "    proc = subprocess.run(\n"
        "        argv,\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        cwd=_MODULE_DIR,\n"
        "        env=_run_env,\n"
        "        **_stdin_kwarg,\n"
        "    )\n"
        "    result = {\n"
        '        "returncode": proc.returncode,\n'
        '        "stdout": proc.stdout,\n'
        '        "stderr": proc.stderr,\n'
        "    }\n"
        "    if proc.returncode != 0:\n"
        "        err = proc.stderr.strip() or proc.stdout.strip()\n"
        "        if err:\n"
        '            result["error"] = err\n'
        "    return result"
    )


def _generate_cli_main_stub(op: SourceOperation) -> str:
    """Generate a subprocess-based stub for a CLI main entry point.

    Unlike ``_generate_cli_handler_stub`` (which dispatches through a
    ``cmd_*`` subcommand), this stub runs the source file directly, passing
    all arguments through to ``sys.argv``.
    """
    docstring = op.description.replace('"', '\\"') if op.description else op.name
    return (
        f"def {op.name}(args: str, {_CLI_SUBPROCESS_OPTIONAL_PARAMS}) -> dict:\n"
        f'    """{docstring}"""\n'
        "    import shlex\n"
        "    import subprocess\n"
        "    import json as _json\n"
        "    tail = shlex.split(args) if args else []\n"
        "    argv = [sys.executable, str(_SOURCE_FILE), *tail]\n"
        + _CLI_SUBPROCESS_STDIN_ENV_BODY
        + "    proc = subprocess.run(\n"
        "        argv,\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        cwd=_MODULE_DIR,\n"
        "        env=_run_env,\n"
        "        **_stdin_kwarg,\n"
        "    )\n"
        "    result = {\n"
        '        "returncode": proc.returncode,\n'
        '        "stdout": proc.stdout,\n'
        '        "stderr": proc.stderr,\n'
        "    }\n"
        "    if proc.returncode != 0:\n"
        "        err = proc.stderr.strip() or proc.stdout.strip()\n"
        "        if err:\n"
        '            result["error"] = err\n'
        "    return result"
    )


# ---------------------------------------------------------------------------
# Interactive input handling for wrapper scripts
# ---------------------------------------------------------------------------


def _generate_interactive_input_preamble(ops: list[SourceOperation]) -> str:
    """Generate preamble code that monkey-patches input(), getpass.getpass(),
    sys.stdin.read() and sys.stdin.readline() to read from a pre-provided
    __interactive_inputs list, with env var fallback.

    This allows tools that use interactive input (like getpass.getpass() for API keys)
    to work in non-TTY subprocess environments like Agent tool calls.

    Always generated so that subprocess-launching stubs can forward
    ``_interactive_input_queue`` to nested subprocesses even when the
    wrapped entrypoint does not itself contain input() calls directly.
    """
    return """
# Interactive input handling for non-TTY environments
# This monkey-patches input(), getpass.getpass(), sys.stdin.read() and
# sys.stdin.readline() to read from __interactive_inputs
import builtins
import getpass as _getpass_module
import sys as _sys_module

_interactive_input_queue = []
_interactive_input_index = 0


def _set_interactive_inputs(inputs: list) -> None:
    global _interactive_input_queue, _interactive_input_index
    _interactive_input_queue = list(inputs) if inputs else []
    _interactive_input_index = 0


def _interactive_input(prompt: str = "") -> str:
    global _interactive_input_index
    if _interactive_input_index < len(_interactive_input_queue):
        value = _interactive_input_queue[_interactive_input_index]
        _interactive_input_index += 1
        return str(value)
    env_name = "".join(c.upper() if c.isalpha() else "_" for c in prompt.strip()).strip("_")
    if env_name and env_name in os.environ:
        return os.environ[env_name]
    raise RuntimeError(
        f"Interactive input required but no __interactive_inputs provided. "
        f"Prompt: '{prompt}'\\n"
        f"Provide __interactive_inputs array in tool call parameters or set "
        f"environment variable {env_name}."
    )


def _interactive_getpass(prompt: str = "Password: ") -> str:
    return _interactive_input(prompt)


def _interactive_stdin_read(size: int = -1) -> str:
    global _interactive_input_index
    if _interactive_input_index < len(_interactive_input_queue):
        value = str(_interactive_input_queue[_interactive_input_index])
        _interactive_input_index += 1
        if size >= 0:
            return value[:size]
        return value
    raise RuntimeError(
        "sys.stdin.read() called but no __interactive_inputs provided. "
        "Provide __interactive_inputs array in tool call parameters or set "
        "the relevant environment variable."
    )


def _interactive_stdin_readline(size: int = -1) -> str:
    global _interactive_input_index
    if _interactive_input_index < len(_interactive_input_queue):
        value = str(_interactive_input_queue[_interactive_input_index])
        _interactive_input_index += 1
        if size >= 0:
            return value[:size] + "\\n"
        return value + "\\n"
    raise RuntimeError(
        "sys.stdin.readline() called but no __interactive_inputs provided. "
        "Provide __interactive_inputs array in tool call parameters or set "
        "the relevant environment variable."
    )


builtins.input = _interactive_input
_getpass_module.getpass = _interactive_getpass
_sys_module.stdin.read = _interactive_stdin_read
_sys_module.stdin.readline = _interactive_stdin_readline
"""


# ---------------------------------------------------------------------------
# Wrapper script generation
# ---------------------------------------------------------------------------

_WRAPPER_SCRIPT_TEMPLATE = r'''#!/usr/bin/env python3
"""Auto-generated wrapper for {header_label} - created by pos convert."""

from __future__ import annotations

import os
import sys
import json
import traceback
import importlib
import asyncio
import dataclasses
from pathlib import Path
{serialization_helpers}
{coercion_helpers}


def _is_wsl_runtime():
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        release = os.uname().release.lower()
    except AttributeError:
        return False
    return "microsoft" in release or "wsl" in release


def _normalize_bootstrap_path(value):
    if not value:
        return ""
    path = os.path.expanduser(os.fspath(value))
    if os.name == "nt" and path.startswith("/mnt/") and len(path) >= 6:
        drive = path[5]
        if drive.isalpha() and (len(path) == 6 or path[6] == "/"):
            rest = path[7:] if len(path) > 6 else ""
            path = drive.upper() + ":\\" + rest.replace("/", "\\")
    elif _is_wsl_runtime() and len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        rest = path[2:].lstrip("\\/").replace("\\", "/")
        path = "/mnt/" + path[0].lower() + (("/" + rest) if rest else "")
    return str(Path(path).expanduser().resolve())


_REPO_ROOT = _normalize_bootstrap_path(r"{repo_root}")
_BUNDLE_DIR = _normalize_bootstrap_path(r"{bundle_dir}")
_BUNDLE_VENV_PYTHON = _normalize_bootstrap_path(r"{bundle_venv_python}")
_SDK_REQUIREMENTS = {sdk_requirements_repr}


def _seed_converter_repo_root():
    if _REPO_ROOT and _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def _ensure_bundle_venv_and_reexec():
    if not _BUNDLE_DIR or not _SDK_REQUIREMENTS:
        return
    _seed_converter_repo_root()
    from extensions.sop_converter.bundle_venv import (
        bundle_venv_python,
        ensure_bundle_venv,
        ensure_bundle_venv_and_reexec,
        is_venv_ready,
    )
    from extensions.sop_converter.sdk_dependency_resolver import SdkDependencySpec

    bundle_dir = _normalize_bootstrap_path(_BUNDLE_DIR)
    try:
        current = Path(sys.executable).resolve()
        target = bundle_venv_python(bundle_dir).resolve()
    except OSError:
        current = Path(sys.executable)
        target = bundle_venv_python(bundle_dir)
    deps = SdkDependencySpec(
        requirements=tuple(_SDK_REQUIREMENTS),
        source="manifest",
        raw_path="",
    )
    ready = is_venv_ready(bundle_dir, tuple(_SDK_REQUIREMENTS))
    if current == target:
        if not ready:
            print(
                "[bundle-venv] Completing setup: installing %d SDK dependencies..."
                % len(_SDK_REQUIREMENTS),
                file=sys.stderr,
            )
            ensure_bundle_venv(bundle_dir, deps)
        return

    if not ready:
        print(
            "[bundle-venv] First-run setup: creating venv and installing %d SDK dependencies..."
            % len(_SDK_REQUIREMENTS),
            file=sys.stderr,
        )
    ensure_bundle_venv_and_reexec(
        bundle_dir,
        deps,
        argv=sys.argv,
        script_file=__file__,
    )


# Runtime dependency setup is deliberately opt-in. Tool execution must never
# create, replace, or install into a virtual environment merely because a
# generated wrapper was invoked.
#
# CLAWCODEX_ENABLE_BUNDLE_VENV_REEXEC=1 only makes this wrapper *call*
# ensure_bundle_venv_and_reexec. A real os.execv into the bundle python happens
# only when the wrapper runs as a standalone process (not under in-process
# SDK dispatch). Agent/REPL in-process calls short-circuit to soft
# site-packages activation; see ensure_bundle_venv_and_reexec docstring.
if os.environ.get("CLAWCODEX_ENABLE_BUNDLE_VENV_REEXEC") == "1":
    _ensure_bundle_venv_and_reexec()

{extra_sys_path_inserts}
_SOURCE_DIR = _normalize_bootstrap_path(r"{source_dir}")
_SOURCE_FILE = Path(_normalize_bootstrap_path(r"{source_file}"))
sys.path.insert(0, _SOURCE_DIR)
if _REPO_ROOT and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MODULE_DIR = _normalize_bootstrap_path(r"{module_dir}")
if os.path.isdir(_MODULE_DIR):
    os.chdir(_MODULE_DIR)
{extra_imports}{model_imports}
{cli_prefix}

_instances = {{}}

{interactive_input_preamble}


def _run_async_iter(make_gen):
    """Drain an async iterator/generator into a JSON-serializable list."""

    async def _collect():
        result = []
        async for item in make_gen():
            result.append(item)
        return result

    return asyncio.run(_collect())


def _agent_not_found_text(text):
    lowered = str(text or "").lower()
    if not lowered:
        return False
    markers = (
        "not found",
        "not exist",
        "does not exist",
        "unknown agent",
        "unknown resource",
        "missing agent",
        "missing resource",
        "resource missing",
        "agent not",
        "resource not",
    )
    subjects = (
        "agent", "resource", "config", "session", "team",
        "handle", "identifier", "resource_id", "agent_id", "id",
    )
    return any(marker in lowered for marker in markers) and (
        any(subject in lowered for subject in subjects)
    )


def _should_catalog_fallback(value):
    if value is None:
        return False
    if isinstance(value, Exception):
        return _agent_not_found_text(value)
    if isinstance(value, dict):
        code = str(value.get("error_code") or value.get("code") or "").lower()
        if code in {{
            "agent_not_found",
            "agent_missing",
            "missing_agent",
            "unknown_agent",
            "resource_not_found",
            "resource_missing",
            "missing_resource",
            "unknown_resource",
        }}:
            return True
        if code == "not_found" and any(k in value for k in ("agent_id", "resource_id", "agent", "resource", "id")):
            return True
        return _agent_not_found_text(value.get("error") or value.get("message") or value)
    return _agent_not_found_text(value)


def _stable_resource_handle_from_args(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text
    return ""


def _try_catalog_fallback(catalog_fallback, args, original_error=None):
    if not catalog_fallback:
        return None
    resource_type = str(catalog_fallback.get("resource_type") or "")
    # Resolve handle using the in-script helper (do not call the package-root
    # function — this body is emitted into a standalone wrapper process).
    agent_id = ""
    _candidates = [
        "resource_ref",
        str(catalog_fallback.get("handle_field") or ""),
        str(catalog_fallback.get("id_arg") or ""),
        "agent_id",
        "resource_id",
        "id",
    ]
    _seen = set()
    for _candidate in _candidates:
        if not _candidate or _candidate in _seen:
            continue
        _seen.add(_candidate)
        agent_id = _stable_resource_handle_from_args(args.get(_candidate))
        if agent_id:
            break
    if not agent_id and not resource_type:
        return None
    query_arg = str(catalog_fallback.get("query_arg") or "query")
    query_value = args.get(query_arg)
    if query_value is None and query_arg != "query":
        query_value = args.get("query")
    inputs = None
    query = ""
    if query_arg == "inputs" or isinstance(query_value, (dict, list)):
        inputs = query_value
    elif query_value is not None:
        query = str(query_value)
    bundle_path = (
        catalog_fallback.get("_bundle_path")
        or os.environ.get("CLAWCODEX_BUNDLE_PATH", "").strip()
        or None
    )
    recovered = None
    try:
        from extensions.sop_converter.resource_handlers import get_resource_handler

        handler = get_resource_handler(resource_type)
        if handler is not None and handler.resource_type != "agent":
            from extensions.sop_converter.resource_catalog import get_resource_record

            record = get_resource_record(
                str(agent_id),
                resource_type=resource_type,
                bundle_path=bundle_path,
            )
            recovered = handler.invoke(record, query=query, inputs=inputs)
        elif handler is not None or not resource_type:
            from extensions.sop_converter.runtime.composite_tools.scripts.invoke_existing_agent_wrapper import (
                invoke_existing_agent,
            )

            recovered = invoke_existing_agent(
                agent_id=str(agent_id) if agent_id else "",
                query=query,
                inputs=inputs,
                bundle_path=bundle_path,
                resource_type=resource_type,
            )
        else:
            from extensions.sop_converter.resource_handlers import (
                require_resource_handler,
            )

            require_resource_handler(resource_type)
    except Exception as exc:
        error_code = getattr(exc, "error_code", "catalog_fallback_failed")
        recovered = {{
            "error": f"catalog_fallback_failed: {{exc}}",
            "error_code": str(error_code),
            "agent_id": str(agent_id) if agent_id else "",
        }}
    if isinstance(recovered, dict):
        recovered.setdefault("catalog_fallback_attempted", True)
        recovered.setdefault("catalog_fallback_reason", "agent_not_found")
        source_tool = catalog_fallback.get("source_tool")
        if source_tool:
            recovered.setdefault("source_tool", source_tool)
        if original_error is not None:
            recovered.setdefault("original_error", str(original_error))
    return recovered


def _augment_create_payload(payload, *, persisted, agent_id="", resource_type="", catalog_path="", catalog_reason="", error_code="", error=""):
    if not isinstance(payload, dict):
        payload = {{"sdk_output": payload}}
    payload["created_persisted"] = bool(persisted)
    payload["callable_by_agent_id"] = bool(persisted and agent_id)
    payload["callable_by_resource_ref"] = bool(persisted and agent_id)
    payload["agent_id_call_contract"] = "catalog_persisted" if persisted and agent_id else "not_persisted"
    payload["resource_ref_call_contract"] = "catalog_persisted" if persisted and agent_id else "not_persisted"
    if agent_id and not payload.get("agent_id"):
        payload["agent_id"] = str(agent_id)
    if agent_id:
        payload["resource_ref"] = str(agent_id)
    if resource_type:
        payload["resource_type"] = str(resource_type)
    if catalog_path:
        payload["catalog_path"] = str(catalog_path)
    if catalog_reason:
        payload["catalog_reason"] = str(catalog_reason)
    if error_code:
        payload["error_code"] = error_code
    if error:
        payload["error"] = error
    return payload

{body}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python {script_name} <method> '<json_args>'", file=sys.stderr)
        sys.exit(1)
    method_name = sys.argv[1]

    # optional agent-catalog hooks.  Created via
    # ``--catalog-metadata '<json>'`` on create-kind tools and
    # ``--catalog-fallback '<json>'`` on invoke-kind tools.
    catalog_meta = None
    catalog_fallback = None
    idx = 3
    while idx < len(sys.argv):
        flag = sys.argv[idx]
        if flag not in {{"--catalog-metadata", "--catalog-fallback"}}:
            idx += 1
            continue
        if idx + 1 >= len(sys.argv):
            print(json.dumps({{"error": f"{{flag}} requires a JSON payload"}}, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)
        try:
            payload = json.loads(sys.argv[idx + 1])
        except json.JSONDecodeError as exc:
            print(json.dumps({{"error": f"invalid {{flag}} JSON: {{exc}}"}}, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)
        if flag == "--catalog-metadata":
            catalog_meta = payload
        else:
            catalog_fallback = payload
        idx += 2

    try:
        args = json.loads(sys.argv[2])
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON args: {{exc}}", file=sys.stderr)
        sys.exit(1)

    catalog_args = dict(args)
    resource_ref = args.pop("resource_ref", None)
    args.pop("resource_type", None)
    if catalog_fallback and resource_ref:
        recovered = _try_catalog_fallback(catalog_fallback, catalog_args)
        if recovered is not None:
            print(_dumps_sdk_result(recovered))
            sys.exit(0)

    interactive_inputs = args.pop("__interactive_inputs", None)
    if interactive_inputs is not None and callable(globals().get("_set_interactive_inputs")):
        _set_interactive_inputs(interactive_inputs)

    fn = globals().get(method_name)
    if fn is None:
        print(f"Unknown method: {{method_name}}", file=sys.stderr)
        sys.exit(1)
    original_error = None
    try:
        result = fn(**args)
    except SystemExit as exc:
        original_error = f"SDK exited with code {{exc.code}}: {{exc}}"
        if catalog_fallback and _should_catalog_fallback(original_error):
            result = _try_catalog_fallback(catalog_fallback, catalog_args, original_error=original_error)
            if result is None:
                print(json.dumps({{"error": original_error}}, ensure_ascii=False), file=sys.stderr)
                sys.exit(1)
        else:
            print(json.dumps({{"error": original_error}}, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        original_error = exc
        if catalog_fallback and _should_catalog_fallback(exc):
            result = _try_catalog_fallback(catalog_fallback, catalog_args, original_error=exc)
            if result is None:
                print(json.dumps({{"error": str(exc)}}, ensure_ascii=False), file=sys.stderr)
                sys.exit(1)
        else:
            print(json.dumps({{"error": str(exc)}}, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)

    if catalog_fallback and _should_catalog_fallback(result):
        recovered = _try_catalog_fallback(catalog_fallback, catalog_args, original_error=result)
        if recovered is not None:
            result = recovered

    serialized = _dumps_sdk_result(result)

    if catalog_meta is not None:
        # Catalog write: pull a stable resource handle from the return value
        # and merge it with the static metadata emitted by the
        # tool_registry_bridge.  This makes the create-to-invoke workflow
        # recoverable across wrapper subprocess boundaries.
        try:
            from extensions.sop_converter.core.agent_catalog import AgentCatalogEntry
            from extensions.sop_converter.resource_catalog import (
                build_resource_record_from_create,
                context_from_env,
                write_record,
            )

            def _stable_resource_handle(_value):
                if _value is None:
                    return ""
                if isinstance(_value, (str, int, float, bool)):
                    return str(_value).strip()
                return ""

            def _extract_resource_handle(_payload, _meta):
                _handle_field = str(_meta.get("handle_field") or "").strip()
                _candidates = []
                if _handle_field:
                    _candidates.append(_handle_field)
                _candidates.extend([
                    "agent_id", "resource_id", "id", "handle", "key",
                    "name", "slug", "uri", "url",
                ])
                _seen = set()
                _ordered = []
                for _candidate in _candidates:
                    if _candidate and _candidate not in _seen:
                        _ordered.append(_candidate)
                        _seen.add(_candidate)
                if isinstance(_payload, dict):
                    for _candidate in _ordered:
                        _handle = _stable_resource_handle(_payload.get(_candidate))
                        if _handle:
                            if _handle_field and _candidate != _handle_field:
                                _meta["handle_field"] = _candidate
                            return _handle
                    # Factory wrappers commonly return a serialised object
                    # whose stable handle belongs to ``agent_config``. Keep
                    # this traversal narrow so unrelated nested payloads do
                    # not become resource handles.
                    for _nested_key in ("agent_config", "config", "dsl", "payload"):
                        _nested = _payload.get(_nested_key)
                        if isinstance(_nested, dict):
                            _handle = _extract_resource_handle(_nested, _meta)
                            if _handle:
                                return _handle
                    # Already a mapping — do not re-_to_jsonable (a fresh dict
                    # is never identity-equal to the input, which used to
                    # recurse forever on empty dicts).
                    return _stable_resource_handle(
                        _meta.get("agent_id") or _meta.get("resource_id")
                    )
                # Non-dict SDK objects may serialize into a handle-bearing dict.
                _jsonable = _to_jsonable(_payload)
                if isinstance(_jsonable, dict):
                    return _extract_resource_handle(_jsonable, _meta)
                return _stable_resource_handle(_meta.get("agent_id") or _meta.get("resource_id"))

            _catalog_snapshot = _serialize_factory_result(result)
            _agent_id = _extract_resource_handle(_catalog_snapshot, catalog_meta)
            # A factory may return an opaque runtime object whose serialized
            # representation omits its identity. For create-LLM-agent style
            # APIs, the stable handle is explicitly supplied in the persisted
            # JSON configuration, so use that as the deterministic fallback.
            if not _agent_id:
                _agent_id = _extract_resource_handle(
                    args.get("agent_config") or args.get("config"),
                    catalog_meta,
                )

            if _agent_id:
                _jsonable_result = _to_jsonable(_catalog_snapshot)
                _runtime_type = (
                    _jsonable_result.get("_runtime_type", {{}})
                    if isinstance(_jsonable_result, dict)
                    else {{}}
                )
                _runtime_invoker = (
                    _jsonable_result.get("_runtime_invoker", {{}})
                    if isinstance(_jsonable_result, dict)
                    else {{}}
                )
                _agent_config = args.get("agent_config") or args.get("config")
                if not isinstance(_agent_config, dict) and isinstance(_jsonable_result, dict):
                    _agent_config = (
                        _jsonable_result.get("agent_config")
                        or _jsonable_result.get("config")
                    )
                _model_spec = _agent_config.get("model", {{}}) if isinstance(_agent_config, dict) else {{}}
                _model_info = _model_spec.get("model_info", {{}}) if isinstance(_model_spec, dict) else {{}}
                _catalog_model = (
                    catalog_meta.get("model")
                    or (_model_info.get("model") if isinstance(_model_info, dict) else "")
                    or (_model_spec.get("model") if isinstance(_model_spec, dict) else "")
                    or ""
                )
                _catalog_provider = (
                    catalog_meta.get("provider")
                    or (_model_spec.get("model_provider") if isinstance(_model_spec, dict) else "")
                    or (_model_spec.get("provider") if isinstance(_model_spec, dict) else "")
                    or ""
                )
                _metadata_keys = {{
                    "sdk_source_dir", "model", "provider", "class_name",
                    "module_name", "query_arg", "invoke_method",
                    "schema_version", "sdk_version", "_bundle_path",
                }}
                # Only persist constructor kwargs — method params (e.g. ``query``
                # on ``build_agent``) must not leak into the re-materialization path.
                _init_param_allowlist = catalog_meta.get("init_param_names")
                if _init_param_allowlist is not None:
                    _init_kwargs = {{k: v for k, v in args.items() if k in _init_param_allowlist}}
                else:
                    _init_kwargs = {{k: v for k, v in args.items() if k not in {{"agent_id", "id"}}}}
                _entry = AgentCatalogEntry(
                    agent_id=str(_agent_id),
                    sdk_source_dir=str(catalog_meta.get("sdk_source_dir") or _SOURCE_DIR),
                    dsl=_jsonable_result if isinstance(_jsonable_result, dict) else {{"value": _jsonable_result}},
                    model=str(_catalog_model),
                    provider=str(_catalog_provider),
                    class_name=str(catalog_meta.get("class_name") or _runtime_type.get("class_name") or ""),
                    module_name=str(catalog_meta.get("module_name") or _runtime_type.get("module") or ""),
                    init_kwargs=_init_kwargs,
                    query_arg=str(_runtime_invoker.get("input_param") or catalog_meta.get("query_arg") or "query"),
                    invoke_method=str(_runtime_invoker.get("method") or catalog_meta.get("invoke_method") or "invoke"),
                    schema_version=int(catalog_meta.get("schema_version") or 1),
                    sdk_version=str(catalog_meta.get("sdk_version") or ""),
                    metadata={{k: v for k, v in catalog_meta.items() if k not in _metadata_keys}},
                    # §8 type-contract fields: record so invoke-kind tools
                    # can look up this entry by resource_type without knowing agent_id.
                    resource_type=str(catalog_meta.get("resource_type") or ""),
                    handle_field=str(catalog_meta.get("handle_field") or "agent_id"),
                )
                _bundle_path = catalog_meta.get("_bundle_path")
                _resource_catalog_path = ""
                _resource_catalog_error = ""
                _resource_catalog_error_code = ""
                _written_layers = []
                _catalog_paths = {{}}
                _catalog_reason = "f56_resource_catalog"
                try:
                    _bundle_id = str(catalog_meta.get("bundle_id") or "") or (
                        os.path.basename(_bundle_path) if _bundle_path else ""
                    )
                    _ctx = context_from_env(
                        bundle_path=_bundle_path,
                        bundle_id=_bundle_id,
                    )
                    _write = write_record(
                        build_resource_record_from_create(
                            resource_id=str(_agent_id),
                            resource_type=str(catalog_meta.get("resource_type") or "agent"),
                            handle_field=str(catalog_meta.get("handle_field") or "agent_id"),
                            bundle_id=_bundle_id or None,
                            source_tool=str(catalog_meta.get("source_tool") or ""),
                            snapshot=_jsonable_result if isinstance(_jsonable_result, dict) else {{"value": _jsonable_result}},
                            init_kwargs=_init_kwargs,
                            model=str(_catalog_model),
                            provider=str(_catalog_provider),
                            class_name=str(catalog_meta.get("class_name") or _runtime_type.get("class_name") or ""),
                            module_name=str(catalog_meta.get("module_name") or _runtime_type.get("module") or ""),
                            invoke_method=str(_runtime_invoker.get("method") or catalog_meta.get("invoke_method") or "invoke"),
                            query_arg=str(_runtime_invoker.get("input_param") or catalog_meta.get("query_arg") or "query"),
                            sdk_source_dir=str(catalog_meta.get("sdk_source_dir") or _SOURCE_DIR),
                            sdk_version=str(catalog_meta.get("sdk_version") or ""),
                            metadata={{k: v for k, v in catalog_meta.items() if k not in _metadata_keys}},
                            env_refs=list((_entry.metadata or {{}}).get("env_vars") or []),
                        ),
                        _ctx,
                    )
                    if not _write.written_layers:
                        # write_record already failed (disk full, permission
                        # denied, ...); surface it as an I/O failure so upstream
                        # can tell retryable filesystem errors from
                        # configuration errors.
                        raise OSError(_write.error or "resource_catalog_write_failed")
                    _written_layers = list(_write.written_layers)
                    _catalog_paths = dict(_write.catalog_paths)
                    _resource_catalog_path = _write.resource_catalog_path or next(
                        iter(_catalog_paths.values()), ""
                    )
                    if _write.error:
                        _resource_catalog_error = str(_write.error)
                except OSError as _resource_exc:
                    # Retryable I/O failures (permission denied, disk full):
                    # keep errno so upstream can tell them apart, and log the
                    # stack for diagnosis.
                    _resource_catalog_error_code = "resource_catalog_io_failed"
                    _resource_catalog_error = (
                        "resource_catalog_io_failed[{{_resource_exc.__class__.__name__}}]: {{_resource_exc}}"
                    )
                    if _resource_exc.errno:
                        _resource_catalog_error = (
                            "{{_resource_catalog_error}} (errno {{_resource_exc.errno}})"
                        )
                    traceback.print_exc()
                except (ValueError, TypeError, KeyError) as _resource_exc:
                    # Configuration/validation failure: the catalog metadata or
                    # bundle context is malformed; retrying with the same input
                    # will not help, so keep it distinct from I/O errors.
                    _resource_catalog_error_code = "resource_catalog_config_failed"
                    _resource_catalog_error = (
                        "resource_catalog_config_failed[{{_resource_exc.__class__.__name__}}]: {{_resource_exc}}"
                    )
                    traceback.print_exc()
                except Exception as _resource_exc:  # noqa: BLE001 - last resort, never fail silently
                    # Unknown failure: keep the exact exception type in the
                    # message and log the stack.  KeyboardInterrupt and
                    # SystemExit derive from BaseException and are deliberately
                    # NOT caught here, so Ctrl-C / explicit exits propagate.
                    _resource_catalog_error_code = "resource_catalog_write_failed"
                    _resource_catalog_error = (
                        "resource_catalog_write_failed[{{_resource_exc.__class__.__name__}}]: {{_resource_exc}}"
                    )
                    traceback.print_exc()
                if not _written_layers:
                    _payload = _augment_create_payload(
                        _jsonable_result,
                        persisted=False,
                        agent_id=str(_agent_id),
                        resource_type=str(catalog_meta.get("resource_type") or ""),
                        catalog_path=str(_resource_catalog_path or ""),
                        catalog_reason=_catalog_reason,
                        error_code=_resource_catalog_error_code or "resource_catalog_write_failed",
                        error=_resource_catalog_error or "resource_catalog_write_failed",
                    )
                    _payload["resource_catalog_error"] = _resource_catalog_error or "resource_catalog_write_failed"
                    _payload["resource_catalog_error_code"] = _resource_catalog_error_code or "resource_catalog_write_failed"
                    print(_dumps_sdk_result(_payload), file=sys.stderr)
                    sys.exit(1)
                _payload = _augment_create_payload(
                    _jsonable_result,
                    persisted=True,
                    agent_id=str(_agent_id),
                    resource_type=str(catalog_meta.get("resource_type") or ""),
                    catalog_path=str(_resource_catalog_path or ""),
                    catalog_reason=_catalog_reason,
                )
                _payload["written_layers"] = _written_layers
                _payload["catalog_paths"] = _catalog_paths
                if _resource_catalog_path:
                    _payload["resource_catalog_path"] = _resource_catalog_path
                    _payload["resource_catalog_reason"] = "f56_resource_catalog"
                if _resource_catalog_error:
                    _payload["resource_catalog_error"] = _resource_catalog_error
                    if _resource_catalog_error_code:
                        _payload["resource_catalog_error_code"] = _resource_catalog_error_code
                serialized = _dumps_sdk_result(_payload)
            else:
                _payload = _augment_create_payload(
                    _to_jsonable(result),
                    persisted=False,
                    resource_type=str(catalog_meta.get("resource_type") or ""),
                    error_code="resource_handle_missing",
                    error="create result did not include a stable resource handle; not persisted to catalog",
                )
                # A lifecycle create is not successful unless the returned
                # resource can be recovered by a later invocation.
                # Do not let the Agent summarize this opaque in-memory object
                # as a usable, persistent Agent.
                print(_dumps_sdk_result(_payload), file=sys.stderr)
                sys.exit(1)
        except Exception as exc:
            # Last-resort guard: log the stack so unexpected catalog-write
            # failures are not silent.  KeyboardInterrupt / SystemExit are
            # BaseException subclasses and deliberately propagate.
            traceback.print_exc()
            _payload = _augment_create_payload(
                _to_jsonable(result),
                persisted=False,
                resource_type=str(catalog_meta.get("resource_type") or ""),
                error_code="catalog_write_failed",
                error=f"catalog_write_failed: {{exc}}",
            )
            print(_dumps_sdk_result(_payload), file=sys.stderr)
            sys.exit(1)

    print(serialized)
'''


def _generate_pipeline_execute_stage_stub(op: SourceOperation, *, module_name: str) -> str:
    """Generate a vendor-neutral workflow executor wrapper."""
    params = [p for p in op.parameters if not p.name.startswith("*")]
    signature: list[str] = []
    for param in params:
        if param.required or param.default is None:
            signature.append(param.name)
        else:
            signature.append(f"{param.name}={param.default!r}")
    params_str = ", ".join(signature)
    return_type = f" -> {op.return_type}" if op.return_type else ""
    docstring = op.description.replace('"', '\\"') if op.description else op.name
    values = ", ".join(f"{param.name!r}: {param.name}" for param in params)
    return (
        f"def {op.name}({params_str}){return_type}:\n"
        f'    """{docstring}"""\n'
        "    from pathlib import Path\n"
        "    import asyncio\n"
        "    import inspect\n"
        f'    module = importlib.import_module({module_name!r})\n'
        f'    function = getattr(module, {op.name!r})\n'
        f"    values = {{{values}}}\n"
        "    parameters = inspect.signature(function).parameters\n"
        "    for name, value in list(values.items()):\n"
        "        if name in {'run_dir', 'project_dir', 'workspace', 'work_dir', 'stage_dir'} and isinstance(value, str):\n"
        "            values[name] = Path(value)\n"
        "    for name in {'stage', 'stage_id', 'step', 'step_id', 'phase', 'phase_id'}:\n"
        "        value = values.get(name)\n"
        "        annotation = parameters.get(name).annotation if name in parameters else None\n"
        "        members = getattr(annotation, '__members__', None)\n"
        "        if isinstance(value, str) and members:\n"
        "            key = value.upper().replace('-', '_').replace(' ', '_')\n"
        "            values[name] = members.get(key, value)\n"
        "    selected = {name: value for name, value in values.items() if name in parameters}\n"
        "    _original_argv = sys.argv\n"
        "    sys.argv = [sys.argv[0]]\n"
        "    try:\n"
        "        result = function(**selected)\n"
        "        return asyncio.run(result) if inspect.isawaitable(result) else result\n"
        "    finally:\n"
        "        sys.argv = _original_argv"
    )


def _model_coerce_expression(
    param_name: str,
    type_name: str,
    source_dir: str,
    module_path: str | None = None,
) -> tuple[str | None, tuple[str, str] | None]:
    """Build ``param if isinstance(param, Model) else Model(**param)`` for a model type.

    Returns ``(expression, (module_path, class_name))`` when *type_name* resolves to a
    Pydantic BaseModel or dataclass under *source_dir*.  Otherwise returns
    ``(None, None)``.

    If *module_path* is provided, the type is first resolved through that module's
    import aliases so that names like ``ReActAgentConfig`` map to the class actually
    imported by the wrapped function rather than whichever class happens to be
    indexed first by name.
    """
    info: tuple[str, str] | None = None
    if module_path:
        try:
            info = ModuleImportIndex(source_dir).resolve_import_path(module_path, type_name)
        except Exception:
            info = None
    if info is None:
        resolved = get_model_class_info(source_dir, type_name, module_path=module_path)
        if resolved is None:
            return None, None
        module_path, class_name, _kind = resolved
        info = module_path, class_name

    class_name = info[1]
    expr = f"_coerce_sdk_type({class_name}, {param_name})"
    return expr, info


def _coerce_param_expression(
    param_name: str,
    type_hint: str | None,
    source_dir: str,
    module_path: str | None = None,
) -> tuple[str | None, set[tuple[str, str]]]:
    """Return an expression that coerces JSON-decoded *param_name* to its SDK type.

    The generated wrapper scripts receive all arguments as plain JSON values.  When a
    parameter is annotated with a Pydantic BaseModel or dataclass, the SDK method
    expects an instance, not a dict.  This helper produces a runtime expression that
    converts ``dict -> Model`` (and ``list[dict] -> list[Model]``) while leaving
    already-correct instances untouched.

    Returns
    -------
    expression : str | None
        A Python expression to use in place of *param_name*, or None when no coercion
        is needed or possible.
    imports : set[tuple[str, str]]
        Set of ``(module_path, class_name)`` imports that must be added to the wrapper
        script for the expression to compile.
    """
    if param_name == "inputs" and _is_loose_mapping_inputs_type_hint(type_hint):
        return f"_normalize_mapping_inputs({param_name})", set()

    if not type_hint:
        return None, set()

    cleaned = type_hint.strip()
    if not cleaned or cleaned in ("Any", "object"):
        return None, set()

    # Check if type is optional by looking for None in the original hint.
    has_none = "None" in cleaned or "NoneType" in cleaned

    # Strip Optional / Union wrappers, ignoring None.
    all_parts = split_union(cleaned)
    union_parts = [p for p in all_parts if p not in ("None", "NoneType")]

    # Optional[T] with a single non-None member -> treat as T, then guard with None.
    is_optional = has_none and len(union_parts) >= 1
    inner_hint = union_parts[0] if len(union_parts) == 1 else cleaned

    # Container types: list[Model], List[Model], Sequence[Model], Iterable[Model].
    container_match = re.match(
        r"^(?:list|List|Sequence|sequence|Iterable|iterable|Set|set|FrozenSet|frozenset)\[(.+)]$",
        inner_hint.strip(),
    )
    if container_match:
        element_hint = container_match.group(1).strip()
        element_expr, element_imports = _coerce_param_expression(
            "__item", element_hint, source_dir, module_path=module_path
        )
        if element_expr:
            item_expr = element_expr.replace("__item", "__item")
            # Use a list-comprehension guard: coerce each dict element.
            if is_optional:
                expr = f"[{item_expr} for __item in {param_name}] if {param_name} is not None else None"
            else:
                expr = f"[{item_expr} for __item in ({param_name} or [])]"
            return expr, element_imports
        return None, set()

    if _is_dict_type_hint(inner_hint):
        expr = f"_coerce_mapping_value({param_name})"
        if is_optional:
            expr = f"None if {param_name} is None else ({expr})"
        return expr, set()

    # Plain model type.
    type_name = type_root(inner_hint)
    if not type_name or type_name in (
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "tuple",
        "set",
        "Dict",
        "Mapping",
        "Any",
        "Path",
        "object",
        "None",
        "NoneType",
    ):
        return None, set()

    expr, import_info = _model_coerce_expression(param_name, type_name, source_dir, module_path=module_path)
    if expr is None:
        return None, set()

    if is_optional:
        expr = f"None if {param_name} is None else ({expr})"

    imports = {import_info} if import_info else set()
    return expr, imports


def _build_coerced_kwargs(
    params: list[ParamSpec],
    source_dir: str,
    module_path: str | None = None,
) -> tuple[str, set[tuple[str, str]]]:
    """Build ``name=value,`` kwargs with Pydantic/dataclass coercion.

    Returns the kwargs block and any model imports required by coercion
    expressions.
    """
    lines: list[str] = []
    imports: set[tuple[str, str]] = set()
    for p in params:
        if p.name.startswith("*"):
            continue
        expr, param_imports = _coerce_param_expression(p.name, p.type_hint, source_dir, module_path=module_path)
        if expr:
            lines.append(f"        {p.name}={expr},\n")
            imports.update(param_imports)
        else:
            lines.append(f"        {p.name}={p.name},\n")
    return "".join(lines), imports


def _build_coerced_pass_list(
    params: list[ParamSpec],
    source_dir: str,
    module_path: str | None = None,
) -> tuple[str, set[tuple[str, str]]]:
    """Build ``name=value`` pass-through for _get_instance init args.

    Class ``__init__`` parameters are exposed on the stub signature; this
    produces the keyword arguments passed to ``_get_instance`` with coercion.
    """
    parts: list[str] = []
    imports: set[tuple[str, str]] = set()
    for p in _skip_variadic_params(params):
        expr, param_imports = _coerce_param_expression(p.name, p.type_hint, source_dir, module_path=module_path)
        if expr:
            parts.append(f"{p.name}={expr}")
            imports.update(param_imports)
        else:
            parts.append(f"{p.name}={p.name}")
    return ", ".join(parts), imports


def _generate_method_stub(
    op: SourceOperation,
    *,
    is_class_method: bool,
    module_name: str,
    init_params: list[ParamSpec] | None = None,
    source_dir: str = "",
) -> tuple[str, set[tuple[str, str]]]:
    """Generate a method stub for a single SourceOperation.

    Args:
        op: The parsed source operation.
        is_class_method: True if this is a class method (needs _get_instance).
        module_name: Dotted Python module path.
        init_params: Required ``__init__`` parameters for the owning class.
        source_dir: Absolute source root used to resolve Pydantic/dataclass
            parameter types for runtime coercion.

    Returns:
        (body_lines, required_model_imports)
    """
    imports: set[tuple[str, str]] = set()

    argv_guard = "    _original_argv = sys.argv\n    sys.argv = [sys.argv[0]]\n    try:\n"
    argv_restore = "    finally:\n        sys.argv = _original_argv"

    if op.is_property:
        return_type = f" -> {op.return_type}" if op.return_type else ""
        docstring = op.description.replace('"', '\\"') if op.description else op.name
        if is_class_method:
            init_pass, init_imports = _build_coerced_pass_list(init_params or [], source_dir, module_path=module_name)
            imports.update(init_imports)
            init_kw_names = [p.name for p in _skip_variadic_params(init_params or [])]
            if init_kw_names:
                params_str = ", ".join(f"{name}=None" for name in init_kw_names)
                get_instance_call = (
                    f'_get_instance("{op.class_name}", "{module_name}"' + (f", {init_pass}" if init_pass else "") + ")"
                )
            else:
                params_str = ""
                get_instance_call = f'_get_instance("{op.class_name}", "{module_name}")'
            inner_call = f"{get_instance_call}.{op.name}"
            return (
                f"def {op.name}({params_str}){return_type}:\n"
                f'    """{docstring}"""\n'
                f"{argv_guard}"
                f"        return {inner_call}\n"
                f"{argv_restore}"
            ), imports
        inner_call = f'getattr(importlib.import_module("{module_name}"), "{op.name}")'
        return (
            f"def {op.name}(){return_type}:\n"
            f'    """{docstring}"""\n'
            f"{argv_guard}"
            f"        return {inner_call}\n"
            f"{argv_restore}"
        ), imports

    effective_params = (
        _merge_init_and_method_params(init_params or [], op.parameters)
        if is_class_method
        else _sort_params_for_python_signature(op.parameters)
    )
    param_parts = _param_signature_parts(effective_params)
    params_str = ", ".join(param_parts)

    return_type = f" -> {op.return_type}" if op.return_type else ""

    docstring = op.description.replace('"', '\\"') if op.description else op.name

    call_kwargs, call_imports = _build_coerced_kwargs(op.parameters, source_dir, module_path=module_name)
    imports.update(call_imports)

    if is_class_method:
        init_pass, init_imports = _build_coerced_pass_list(init_params or [], source_dir, module_path=module_name)
        imports.update(init_imports)
        get_instance_call = f'_get_instance("{op.class_name}", "{module_name}"'
        if init_pass:
            get_instance_call += f", {init_pass}"
        get_instance_call += ")"
        inner_call = f"{get_instance_call}.{op.name}(\n{call_kwargs}    )"
    elif op.is_factory:
        factory_params, factory_imports = _build_coerced_pass_list(op.parameters, source_dir, module_path=module_name)
        imports.update(factory_imports)
        get_instance_call = f'_get_instance("{op.name}", "{module_name}"'
        if factory_params:
            get_instance_call += f", {factory_params}"
        get_instance_call += ")"
        inner_call = get_instance_call
    else:
        inner_call = f"module.{op.name}(\n{call_kwargs}    )"

    if op.is_async_generator:
        if is_class_method or op.is_factory:
            body_lines = (
                f"def {op.name}({params_str}){return_type}:\n"
                f'    """{docstring}"""\n'
                f"{argv_guard}"
                f"        return _run_async_iter(lambda: {inner_call})\n"
                f"{argv_restore}"
            )
        else:
            body_lines = (
                f"def {op.name}({params_str}){return_type}:\n"
                f'    """{docstring}"""\n'
                f'    module = importlib.import_module("{module_name}")\n'
                f"{argv_guard}"
                f"        return _run_async_iter(lambda: {inner_call})\n"
                f"{argv_restore}"
            )
        return body_lines, imports

    async_prefix = "asyncio.run(" if op.is_async else ""
    async_suffix = ")" if op.is_async else ""

    if is_class_method:
        body_lines = (
            f"def {op.name}({params_str}){return_type}:\n"
            f'    """{docstring}"""\n'
            f"{argv_guard}"
            f"        return {async_prefix}{inner_call}{async_suffix}\n"
            f"{argv_restore}"
        )
    elif op.is_factory:
        body_lines = (
            f"def {op.name}({params_str}){return_type}:\n"
            f'    """{docstring}"""\n'
            f"{argv_guard}"
            f"        instance = {async_prefix}{inner_call}{async_suffix}\n"
            f"        return _serialize_factory_result(instance)\n"
            f"{argv_restore}"
        )
    else:
        body_lines = (
            f"def {op.name}({params_str}){return_type}:\n"
            f'    """{docstring}"""\n'
            f'    module = importlib.import_module("{module_name}")\n'
            f"{argv_guard}"
            f"        return {async_prefix}{inner_call}{async_suffix}\n"
            f"{argv_restore}"
        )
    return body_lines, imports


def _generate_wrapper_script(
    ops: list[SourceOperation],
    *,
    class_name: str | None,
    module_name: str,
    file_stem: str,
    source_dir: str,
    scripts_dir: Path | None = None,
    init_params: list[ParamSpec] | None = None,
    cli_dispatch_map: dict[str, str] | None = None,
    cli_prefix_override: str | None = None,
    repo_root: str = "",
    bundle_dir: str | Path | None = None,
    bundle_venv_python: str | None = None,
    sdk_requirements: tuple[str, ...] = (),
) -> Path:
    """Generate a wrapper script for a group of related operations.

    All operations sharing the same *class_name* (or *file_stem* for standalone
    functions) are written into one script so that the class is instantiated
    only once per process.

    Args:
        ops: Operations to include in this script.
        class_name: The owning class name, or None for standalone functions.
        module_name: Dotted Python import path for the module.
        file_stem: The source file stem (used when class_name is None).
        source_dir: Absolute path to the source root (injected into sys.path).

    Returns:
        Path to the generated script file.
    """
    # Determine script filename
    if class_name:
        script_name = _script_name_for_class(module_name, class_name)
        header_label = f"{class_name} ({module_name})"
    else:
        script_name = _script_name_for_functions(module_name, file_stem)
        header_label = f"{file_stem} functions ({module_name})"

    script_path = (scripts_dir or SCRIPTS_DIR) / script_name
    script_path.parent.mkdir(parents=True, exist_ok=True)

    source_file = resolve_source_file(source_dir, module_name)
    if cli_dispatch_map is not None:
        dispatch_map = cli_dispatch_map
    elif file_stem == "cli":
        dispatch_map = _parse_cli_dispatch_map(source_file)
    else:
        dispatch_map = {}
    use_cli_subprocess = class_name is None and any(_is_cli_handler_op(op, dispatch_map) for op in ops)
    cli_prefix_line = ""
    if use_cli_subprocess:
        prefix = _resolve_cli_argv_prefix(
            source_dir,
            source_file,
            cli_prefix_override=cli_prefix_override,
        )
        cli_prefix_line = f"CLI_PREFIX: list[str] = {prefix!r}"

    # Build body: helper(s) + method stubs
    body_parts: list[str] = []
    model_imports: set[tuple[str, str]] = set()

    has_factory_ops = any(op.is_factory for op in ops)
    if class_name or has_factory_ops:
        if has_factory_ops and not class_name:
            factory_op = next(op for op in ops if op.is_factory)
            factory_params = _skip_variadic_params(factory_op.parameters)
            body_parts.append(_generate_get_instance_helper(factory_params))
        else:
            body_parts.append(_generate_get_instance_helper(init_params))

    # Sort operations by name for deterministic output
    for op in sorted(ops, key=lambda o: o.name):
        body_parts.append("")
        subcommand = dispatch_map.get(op.name)
        if subcommand and _is_cli_handler_op(op, dispatch_map):
            body_parts.append(_generate_cli_handler_stub(op, subcommand=subcommand))
        elif _is_cli_main_op(op, source_dir, module_name):
            body_parts.append(_generate_cli_main_stub(op))
        elif _is_workflow_execute_operation(op) and class_name is None:
            body_parts.append(_generate_pipeline_execute_stage_stub(op, module_name=module_name))
        else:
            stub_body, stub_imports = _generate_method_stub(
                op,
                is_class_method=class_name is not None,
                module_name=module_name,
                init_params=init_params if class_name is not None else None,
                source_dir=source_dir,
            )
            body_parts.append(stub_body)
            model_imports.update(stub_imports)

    runtime_symbols = _collect_runtime_symbols(ops, init_params)
    import_map = _parse_import_map(source_file, module_name)
    extra_imports = _format_wrapper_imports(runtime_symbols, import_map, module_name)

    # Pydantic/dataclass imports must load after sys.path is seeded.
    # Some SDK directories contain hyphens (e.g. "agent-perf-analyzer"),
    # which are invalid in Python ``from X import Y`` statements.
    # For those we generate ``importlib.import_module()`` attribute access.
    if model_imports:
        _import_lines: list[str] = []
        for module_path, model_class in sorted(model_imports):
            if _module_path_needs_importlib(module_path):
                _import_lines.append(f'{model_class} = importlib.import_module("{module_path}").{model_class}')
            else:
                _import_lines.append(f"from {module_path} import {model_class}")
        model_import_block = "\n".join(_import_lines)
        model_import_block = f"\n{model_import_block}\n"
    else:
        model_import_block = ""

    if extra_imports:
        extra_imports = f"\n{extra_imports}\n"

    extra_sys_path_entries = infer_extra_sys_path_entries(source_dir, module_name)
    extra_sys_path_inserts = format_extra_sys_path_inserts(
        extra_sys_path_entries,
        normalizer="_normalize_bootstrap_path",
    )
    module_dir = _resolve_module_working_dir(source_dir, module_name)

    body_text = "\n".join(body_parts)
    interactive_input_preamble = _generate_interactive_input_preamble(ops)

    content = _WRAPPER_SCRIPT_TEMPLATE.format(
        header_label=header_label,
        source_dir=source_dir,
        bundle_dir=str(Path(bundle_dir).resolve()) if bundle_dir is not None else "",
        bundle_venv_python=bundle_venv_python or "",
        sdk_requirements_repr=repr(tuple(sdk_requirements)),
        module_dir=module_dir,
        source_file=str(source_file.resolve()),
        repo_root=repo_root,
        cli_prefix=cli_prefix_line,
        extra_sys_path_inserts=extra_sys_path_inserts,
        extra_imports=extra_imports,
        model_imports=model_import_block,
        serialization_helpers=WRAPPER_SERIALIZATION_HELPERS,
        coercion_helpers=WRAPPER_COERCION_HELPERS,
        body=body_text,
        script_name=script_name,
        interactive_input_preamble=interactive_input_preamble,
    )

    script_path.write_text(content, encoding="utf-8")
    try:
        compile(content, str(script_path), "exec")
    except SyntaxError as exc:
        # Save failing content for debugging before deleting the .py file
        failed_path = script_path.with_suffix(script_path.suffix + ".failed")
        failed_path.write_text(content, encoding="utf-8")
        logger.warning("Saved failing wrapper content to %s", failed_path)
        script_path.unlink(missing_ok=True)
        raise RuntimeError(f"Generated wrapper failed syntax check: {script_path}: {exc}") from exc
    logger.info("Generated wrapper script: %s (%d methods)", script_path, len(ops))
    return script_path


# ---------------------------------------------------------------------------
# Bridge params — json_args injection
# ---------------------------------------------------------------------------


def _enrich_bridge_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *params* with ``json_args`` injected.

    The generated wrapper scripts accept a single JSON blob as their second
    argument (``sys.argv[2]``).  This helper serialises the tool-call input
    dict into that blob so the ``call_impl`` template placeholder
    ``{json_args}`` is always satisfied, even for parameterless functions
    where *params* is empty.
    """
    return {**params, "json_args": json.dumps(params, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# Spec generation
# ---------------------------------------------------------------------------


def _annotate_env_reference_schema(value: Any, *, property_name: str = "") -> None:
    """Document the explicit, safe environment-reference contract in schemas."""
    if isinstance(value, dict):
        if property_name.lower() in {"api_key", "apikey", "access_token"}:
            note = (
                "Use a literal value or an explicit environment reference such as "
                "env:DEEPSEEK_API_KEY. Environment references are resolved only "
                "at tool runtime and are never returned in plaintext."
            )
            existing = str(value.get("description") or "").strip()
            value["description"] = f"{existing} {note}".strip()
        for key, child in value.items():
            _annotate_env_reference_schema(child, property_name=str(key))
    elif isinstance(value, list):
        for child in value:
            _annotate_env_reference_schema(child, property_name=property_name)


def operation_to_spec(
    op: SourceOperation,
    *,
    source_dir: str,
    script_path: str,
    comp_name: str = "",
    bundle_id: str | None = None,
    init_params: list[ParamSpec] | None = None,
    cli_subcommand: str | None = None,
    cli_main: bool = False,
    tool_deps: ToolOperationDeps | None = None,
    module_path: str | None = None,
) -> Any:
    """Convert a single SourceOperation into an AgentToolSpec.

    Args:
        op: The parsed source operation.
        source_dir: Absolute path to the source root (unused here but kept
            for API consistency — the path is embedded in the wrapper script).
        script_path: Absolute path to the generated wrapper script.
        comp_name: SourceComponent name for the primary naming prefix
            (e.g. ``"agent_teams"``).  Falls back to *file_stem* or
            *class_name* when empty.

    Returns:
        A validated ``AgentToolSpec`` ready for persistence.
    """
    # Compute kebab-case tool name.
    # All names are namespace-qualified to guarantee global uniqueness:
    #   Class method  → {comp}.{class}.{method}  (or {class}.{method} when
    #                    comp_name is unavailable — legacy SDK path).
    #   Standalone fn → {comp}.{method}
    #   Fallback      → {file_stem}.{method} or bare {method}
    if op.class_name:
        if comp_name:
            raw_name = f"{comp_name}.{op.class_name}.{op.name}"
        else:
            raw_name = f"{op.class_name}.{op.name}"
    elif comp_name:
        raw_name = f"{comp_name}.{op.name}"
    elif op.file_stem:
        raw_name = f"{op.file_stem}.{op.name}"
    else:
        raw_name = op.name

    tool_name = _to_kebab_case(raw_name)

    # Build JSON Schema properties (merge class ``__init__`` params for class methods).
    if cli_subcommand:
        properties = {
            "args": {
                "type": "string",
                "description": (
                    f"CLI arguments after the '{cli_subcommand}' subcommand "
                    f"(flags and positional args, excluding '{cli_subcommand}' itself)."
                ),
            }
        }
        required = ["args"]
    elif cli_main:
        properties = {
            "args": {
                "type": "string",
                "description": (
                    "CLI arguments passed to the entry point.  Include all flags "
                    "and positional arguments exactly as they would appear on "
                    "the command line (e.g. '--project myapp --generate-questions')."
                ),
            }
        }
        required = ["args"]
    else:
        # property ops (e.g. DeepAgent.loop_coordinator) carry no method
        # params themselves but MUST expose the owning class's __init__ params
        # (e.g. ``card``) so the agent knows to supply them at call time.
        schema_params = (
            _merge_init_and_method_params(init_params or [], op.parameters) if op.class_name else op.parameters
        )
        properties = {}
        required = []

        for param in schema_params:
            # Skip *args and **kwargs
            if param.name.startswith("*"):
                continue

            json_type = _type_hint_to_json_type(param.type_hint)
            prop = param_to_json_schema_property(
                type_hint=param.type_hint,
                description=param.description,
                source_dir=source_dir,
                fallback_json_type=json_type,
                module_path=module_path,
            )
            if param.default is not None:
                prop["default"] = _normalize_schema_default(
                    param.default,
                    json_type=prop.get("type", json_type),
                )

            properties[param.name] = prop

            if param.required and param.default is None:
                required.append(param.name)

        _adjust_pipeline_execute_stage_schema(op, properties, required)

    # ponytail: always include __interactive_inputs for CLI entrypoints
    # (cli_main=True / cli_subcommand not None) because they run as subprocesses
    # and the shallow AST detector can miss input() calls delegated to helpers.
    if op.requires_interactive_input or cli_main or cli_subcommand:
        properties["__interactive_inputs"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of values for interactive input prompts (input()/getpass.getpass()). "
                f"Detected prompts: {op.interactive_prompts}. "
                "Provide values in the order they appear in the code. "
                "Alternatively, set environment variables matching the prompt text in uppercase."
            ),
        }

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required

    _annotate_env_reference_schema(input_schema)
    input_schema = enrich_input_schema_with_dependencies(input_schema, tool_deps)

    # Build call_impl template — uses absolute script path, which is baked into
    # the spec at generation time.  If the bundle is later moved to another
    # machine / absolute path, persisted specs will point at a script that no
    # longer exists — re-run convert (or rewrite the path) after migration.
    # {json_args} is a special placeholder resolved by _enrich_bridge_params()
    # at call time (NOT by the generic execute_bash).  Single-quoted so the
    # shell treats the JSON blob as one token even when it is "{}".
    call_impl = f"python3 \"{script_path}\" {op.name} '{{json_args}}'"

    # Build aliases so resolve_agent_tools() can find the tool under
    # alternative names.
    #
    # Every alias is component-scoped when a component name is
    # available.  Class methods ONLY get the fully-qualified
    # {comp}.{class}.{method} alias — the bare {comp}.{method} form
    # is reserved for standalone functions, avoiding collisions when
    # the same method name appears as both a class method and a
    # standalone function within the same component.
    alias_list: list[str] = []
    if comp_name:
        if op.class_name:
            # Class method: only the fully-qualified alias is unique.
            alias_list.append(f"{comp_name}.{op.class_name}.{op.name}")
        else:
            # Standalone function: {comp}.{method} is unambiguous.
            alias_list.append(f"{comp_name}.{op.name}")
    else:
        # Legacy SDK path: no component to scope against.
        alias_list.append(raw_name)

    # ------------------------------------------------------------------
    # Short-name suffix aliases (strip the project-root namespace).
    #
    # When *comp_name* is multi-segment (e.g. "openjiuwen.agent_teams"),
    # the first segment is the project-level namespace (derived from the
    # top-level directory under *source_dir*).  Runtimes built *within*
    # that project typically reference tools without this prefix
    # (e.g. "agent_teams.get_session_id") — they operate at module scope.
    #
    # We generate an extra set of aliases that drop the first segment,
    # so resolve_agent_tools() can find the tool by its short name
    # through the normal exact-match path:
    #     "agent_teams.get_session_id" → exact match → alias hit ✅
    #
    # Only the first segment is stripped — deeper projects can extend
    # this pattern if needed, but one level covers the common case
    # without introducing ambiguous short aliases.
    if comp_name and "." in comp_name:
        _short_comp = comp_name.split(".", 1)[1]  # strip first segment
        if op.class_name:
            alias_list.append(f"{_short_comp}.{op.class_name}.{op.name}")
        else:
            alias_list.append(f"{_short_comp}.{op.name}")

    aliases = tuple(alias_list)
    tags = generate_search_tags(op, comp_name=comp_name)

    # Docstring only — dependency edges live in input_schema.x-sop-dependencies
    # and ORCHESTRATION_ROUTES.md, not in ToolSearch-visible description/tags.
    description = op.description or op.name

    return DEFAULTS.tool_authoring.create_spec(
        name=tool_name,
        description=description,
        input_schema=input_schema,
        call_type="bash",
        call_impl=call_impl,
        tags=tags,
        aliases=aliases,
        source="sop-converter",
        bundle_id=bundle_id,
        stateful_wrapper=True,
    )


# ---------------------------------------------------------------------------
# Bulk registration
# ---------------------------------------------------------------------------


def _binding_for_operation(
    bindings: list[ResourceBinding],
    comp: SourceComponent,
    op: SourceOperation,
    *,
    role: str,
) -> ResourceBinding | None:
    references = [
        op.name,
        _to_kebab_case(op.name),
        f"{comp.name}.{op.name}",
        _to_kebab_case(f"{comp.name}.{op.name}"),
    ]
    if op.class_name:
        references.extend(
            [
                f"{op.class_name}.{op.name}",
                f"{comp.name}.{op.class_name}.{op.name}",
                _to_kebab_case(f"{comp.name}.{op.class_name}.{op.name}"),
            ]
        )
    for binding in bindings:
        matches = binding.matches_create if role == "create" else binding.matches_invoke
        if matches(*references):
            return binding
    return None


def _operation_tool_name(comp: SourceComponent, op: SourceOperation) -> str:
    if op.class_name:
        raw_name = f"{comp.name}.{op.class_name}.{op.name}" if comp.name else f"{op.class_name}.{op.name}"
    elif comp.name:
        raw_name = f"{comp.name}.{op.name}"
    elif op.file_stem:
        raw_name = f"{op.file_stem}.{op.name}"
    else:
        raw_name = op.name
    return _to_kebab_case(raw_name)


def register_component_tools(
    components: list[SourceComponent],
    source_dir: str,
    *,
    persist: bool = True,
    overwrite: bool = True,
    bundle_dir: str | Path | None = None,
    bundle_id: str | None = None,
    cli_prefix_override: str | None = None,
    repo_root: str = "",
    bundle_venv_python: str | None = None,
    sdk_requirements: tuple[str, ...] = (),
) -> dict[str, str]:
    """Bulk-register all operations from a list of SourceComponents as Tools.

    For each unique class (or source file for standalone functions), one
    wrapper script is generated.  Every ``SourceOperation`` becomes an
    ``AgentToolSpec``, which is optionally persisted to disk and validated.

    Args:
        components: Parsed source components from ``SourceCodeParser.parse()``.
        source_dir: Absolute path to the source root directory.
        persist: If True, persist each ``AgentToolSpec`` to
            ``~/.clawcodex/agent-tools/<name>.json``.
        overwrite: If True, overwrite existing specs and scripts with the
            same name (idempotent on repeated ``sop convert`` runs).

    Returns:
        A mapping from original tool names (``"LLM.invoke"``) to kebab-case
        tool names (``"llm-invoke"``).  Use this to update
        ``SkillSpec.allowed_tools`` before writing agent markdown.
    """
    source_dir_abs = str(Path(source_dir).resolve())
    bundle_path = Path(bundle_dir).resolve() if bundle_dir is not None else None
    effective_bundle_id = bundle_id or (bundle_path.name if bundle_path else None)
    resource_bindings = load_resource_bindings(bundle_path)
    tool_dir = DEFAULTS.tool_authoring.bundle_tool_dir(bundle_path) if bundle_path is not None else None
    scripts_dir = DEFAULTS.tool_authoring.scripts_dir_for(tool_dir) if tool_dir is not None else SCRIPTS_DIR
    effective_repo_root = repo_root
    if not effective_repo_root:
        here = Path(__file__).resolve()
        # runtime/tool_registry_bridge.py → walk up to the converter repo root
        # (directory that contains ``extensions/sop_converter``).
        effective_repo_root = str(here.parents[3])
        for parent in here.parents:
            if (parent / "extensions" / "sop_converter").is_dir():
                effective_repo_root = str(parent)
                break

    # ── Phase 1: group operations by (class_name, module_path) ──
    # Each group → one wrapper script.
    # Module path is per-file (file_stem), not per-component, because a
    # single component can contain operations from multiple .py files.

    from collections import defaultdict

    # Key: (class_name_or_None, module_path_for_file)
    # Value: list of SourceOperation
    groups: dict[tuple[str | None, str], list[SourceOperation]] = defaultdict(list)
    # Per-operation module path cache
    op_module_map: dict[int, str] = {}

    # Maps (class_name, module_path) → init params for wrapper generation
    group_init_params: dict[tuple[str | None, str], list[ParamSpec]] = {}
    # Per-module ``cmd_*`` → CLI subcommand (from ``main()`` dispatch table)
    cli_dispatch_by_module: dict[str, dict[str, str]] = {}

    for comp in components:
        for op in comp.operations:
            module_path = _resolve_module_path(comp, source_dir_abs, op.file_stem or "unknown")
            key = (op.class_name, module_path)
            groups[key].append(op)
            op_module_map[id(op)] = module_path
            if op.class_name and op.class_name in comp.class_init_params:
                group_init_params.setdefault(key, comp.class_init_params[op.class_name])
            if op.class_name is None and (op.file_stem or "") == "cli":
                cli_dispatch_by_module.setdefault(
                    module_path,
                    _parse_cli_dispatch_map(resolve_source_file(source_dir_abs, module_path)),
                )

    # ── Phase 1.5: batch preload pydantic schemas ──
    # Collect all structured type hints from operation params and init params,
    # then probe them in a single subprocess (one import per SDK) to fill
    # the _BATCH_CACHE.  This avoids N separate subprocess spawns during
    # Phase 2 (wrapper generation) and Phase 3 (spec creation).
    _collect_type_hints: list[tuple[str, str | None]] = []
    for comp in components:
        for op in comp.operations:
            module_path = op_module_map[id(op)]
            for param in op.parameters:
                if param.type_hint and param.name and not param.name.startswith("*"):
                    _collect_type_hints.append((param.type_hint, module_path))
            if op.class_name and op.class_name in comp.class_init_params:
                for param in comp.class_init_params[op.class_name]:
                    if param.type_hint and param.name and not param.name.startswith("*"):
                        _collect_type_hints.append((param.type_hint, module_path))
    if _collect_type_hints:
        probe_targets = collect_probe_targets(source_dir_abs, _collect_type_hints)
        if probe_targets:
            logger.info(
                "Batch preloading schemas for %d types in %s",
                len(probe_targets),
                source_dir_abs,
            )
            # Ensure the bundle venv is created with SDK dependencies installed
            # so the schema-probe subprocess can import third-party deps (jsonschema_path, pysbd, etc.).
            effective_venv_python = bundle_venv_python
            if bundle_path is not None and sdk_requirements:
                from extensions.sop_converter.bundle_venv import (
                    ensure_bundle_venv,
                    is_venv_ready,
                )
                from extensions.sop_converter.sdk_dependency_resolver import SdkDependencySpec

                try:
                    deps = SdkDependencySpec(
                        requirements=tuple(sdk_requirements),
                        source="manifest",
                        raw_path="",
                    )
                    ready = is_venv_ready(bundle_path, tuple(sdk_requirements))
                    if not ready:
                        _bridge_progress(
                            f"   Ensuring bundle venv (installing {len(sdk_requirements)} SDK deps for schema probe)..."
                        )
                    effective_venv_python = str(ensure_bundle_venv(bundle_path, deps))
                except Exception as exc:
                    logger.warning("Failed to ensure bundle venv for schema probe: %s", exc)
                    effective_venv_python = None
            preload_schemas_for_source_dir(
                source_dir_abs,
                probe_targets,
                venv_python=effective_venv_python,
            )

    # ── Phase 2: generate wrapper scripts ──

    # Maps (class_name, module_path) → script absolute path
    script_paths: dict[tuple[str | None, str], str] = {}
    skipped_groups: list[tuple[str | None, str]] = []
    total_groups = len(groups)

    _bridge_progress(f"   Generating wrappers: 0/{total_groups}...", end="")
    for group_idx, ((class_name, module_path), ops) in enumerate(groups.items(), 1):
        first_op = ops[0]
        file_stem = first_op.file_stem or "functions"

        try:
            script_path = _generate_wrapper_script(
                ops,
                class_name=class_name,
                module_name=module_path,
                file_stem=file_stem,
                source_dir=source_dir_abs,
                scripts_dir=scripts_dir,
                init_params=group_init_params.get((class_name, module_path)),
                cli_dispatch_map=cli_dispatch_by_module.get(module_path),
                cli_prefix_override=cli_prefix_override,
                repo_root=effective_repo_root,
                bundle_dir=bundle_path,
                bundle_venv_python=bundle_venv_python,
                sdk_requirements=sdk_requirements,
            )
            script_paths[(class_name, module_path)] = str(script_path.resolve())
        # ponytail: SystemExit — bad SDK modules must not abort convert
        except (Exception, SystemExit):
            logger.warning(
                "Failed to generate wrapper for class=%s module=%s, skipping %d ops",
                class_name,
                module_path,
                len(ops),
                exc_info=True,
            )
            skipped_groups.append((class_name, module_path))

    _bridge_progress(f"\r   Generating wrappers: {total_groups}/{total_groups} done")

    # ── Phase 3: create AgentToolSpec for each operation ──

    name_map: dict[str, str] = {}
    dependency_index = build_tool_dependency_index(components, source_dir=source_dir_abs)
    binding_roles: dict[int, tuple[str, ResourceBinding]] = {}
    for binding_comp in components:
        for binding_op in binding_comp.operations:
            create_binding = _binding_for_operation(
                resource_bindings,
                binding_comp,
                binding_op,
                role="create",
            )
            invoke_binding = _binding_for_operation(
                resource_bindings,
                binding_comp,
                binding_op,
                role="invoke",
            )
            if create_binding and invoke_binding:
                raise ValueError(f"resource sidecar maps {binding_op.name!r} as both create and invoke")
            if create_binding:
                binding_roles[id(binding_op)] = ("create", create_binding)
            elif invoke_binding:
                binding_roles[id(binding_op)] = ("invoke", invoke_binding)
    # §8 type-contract: pre-compute the set of resource types produced by
    # create-kind ops so invoke-kind ops can be classified via type matching
    # even when their parameter names don't end with ``_id``.
    from ..core.heuristics.lifecycle import derive_resource_type, infer_lifecycle_kind as _ilk

    _type_resolver = ModuleImportIndex(source_dir_abs)
    _op_resource_types: dict[int, str] = {}
    _create_tool_names_by_type: dict[str, str] = {}
    _canonical_create_types: set[str] = set()
    _known_create_types: set[str] = set()
    for _comp in components:
        for _op in _comp.operations:
            _binding_role = binding_roles.get(id(_op))
            if _binding_role and _binding_role[0] == "invoke":
                _op_resource_types[id(_op)] = _binding_role[1].normalized_resource_type
                continue
            if (_binding_role and _binding_role[0] == "create") or _ilk(_op) == "create":
                _module_path = op_module_map.get(id(_op), "")
                _rt = (
                    _binding_role[1].normalized_resource_type
                    if _binding_role
                    else _first_resource_type_for_op(
                        resolver=_type_resolver,
                        module_path=_module_path,
                        op=_op,
                        prefer_return=True,
                    )
                )
                if not _rt:
                    _raw_rt = derive_resource_type(_op)
                    _rt = (
                        _resource_type_from_hint(
                            resolver=_type_resolver,
                            module_path=_module_path,
                            type_hint=_raw_rt,
                        )
                        or _raw_rt
                    )
                if _rt:
                    _op_resource_types[id(_op)] = _rt
                    _canonical_create_types.add(_rt)
                    _known_create_types.add(_rt)
                    _known_create_types.update(_resource_type_tokens_for_op(_op))
                    _create_tool_names_by_type.setdefault(
                        _rt,
                        _operation_tool_name(_comp, _op),
                    )
    for _comp in components:
        for _op in _comp.operations:
            _module_path = op_module_map.get(id(_op), "")
            _hints = [_op.return_type, *(param.type_hint for param in _op.parameters)]
            for _hint in _hints:
                _resolved = _resource_type_from_hint(
                    resolver=_type_resolver,
                    module_path=_module_path,
                    type_hint=_hint,
                )
                if _resolved and _resolved in _canonical_create_types:
                    _known_create_types.update(_resource_type_hint_tokens(_hint))
    _known_create_types_frozen = frozenset(_known_create_types)
    for _comp in components:
        for _op in _comp.operations:
            if id(_op) in _op_resource_types:
                continue
            _module_path = op_module_map.get(id(_op), "")
            _rt = _first_resource_type_for_op(
                op=_op,
                resolver=_type_resolver,
                module_path=_module_path,
                prefer_return=False,
            )
            if _rt:
                _op_resource_types[id(_op)] = _rt

    specs: list[Any] = []
    total_ops = sum(len(comp.operations) for comp in components)
    spec_idx = 0
    _bridge_progress(f"   Building tool specs: 0/{total_ops}...", end="")

    for comp in components:
        for op in comp.operations:
            spec_idx += 1
            if spec_idx % 50 == 0:
                _bridge_progress(
                    f"\r   Building tool specs: {spec_idx}/{total_ops}...",
                    end="",
                )
            module_path = op_module_map[id(op)]
            key = (op.class_name, module_path)
            if key not in script_paths:
                continue
            script_path = script_paths[key]

            try:
                init_params = comp.class_init_params.get(op.class_name, []) if op.class_name else None
                dispatch_map = cli_dispatch_by_module.get(module_path, {})
                cli_subcommand = dispatch_map.get(op.name) if _is_cli_handler_op(op, dispatch_map) else None
                is_cli_main = _is_cli_main_op(op, source_dir_abs, module_path)
                tool_deps = dependency_index.get(to_kebab_tool_name(comp.name, op))
                spec = operation_to_spec(
                    op,
                    source_dir=source_dir_abs,
                    script_path=script_path,
                    comp_name=comp.name,
                    bundle_id=effective_bundle_id,
                    init_params=init_params,
                    cli_subcommand=cli_subcommand,
                    cli_main=is_cli_main,
                    tool_deps=tool_deps,
                    module_path=module_path,
                )
                # L1: create-kind tools get a ``--catalog-metadata`` payload so
                # the wrapper subprocess can persist the resulting ``agent_id`` to
                # the bundle-local AgentCatalog.  This makes the create→invoke
                # workflow recoverable across independent wrapper processes.
                binding_role = binding_roles.get(id(op))
                lifecycle_kind = (
                    binding_role[0]
                    if binding_role
                    else infer_lifecycle_kind(
                        op,
                        known_create_types=_known_create_types_frozen,
                    )
                )
                lifecycle_extra = {}
                if init_params:
                    lifecycle_extra["init_param_names"] = [p.name for p in _skip_variadic_params(init_params)]
                op_resource_type = _op_resource_types.get(id(op), "")
                if op_resource_type:
                    lifecycle_extra["resource_type"] = op_resource_type
                if binding_role:
                    lifecycle_extra["handle_field"] = binding_role[1].handle_field

                if lifecycle_kind == "invoke" and op_resource_type:
                    consume_param = invoke_lifecycle_id_param(op)
                    if (
                        not consume_param
                        and binding_role
                        and binding_role[1].handle_field in spec.input_schema.get("properties", {})
                    ):
                        consume_param = binding_role[1].handle_field
                    spec = DEFAULTS.tool_authoring.create_spec(
                        **{
                            **spec.__dict__,
                            "input_schema": inject_resource_ref_schema(
                                spec.input_schema,
                                resource_type=op_resource_type,
                                create_tool_name=_create_tool_names_by_type.get(
                                    op_resource_type,
                                    "",
                                ),
                                consume_param=consume_param,
                            ),
                        }
                    )

                DEFAULTS.tool_authoring.validate_spec(spec)

                if lifecycle_kind == "create":
                    catalog_meta = lifecycle_metadata_payload(
                        op,
                        source_dir=source_dir_abs,
                        bundle_id=effective_bundle_id,
                        module_name=module_path,
                        class_name=op.class_name,
                        tool_name=spec.name,
                        known_create_types=_known_create_types_frozen,
                        lifecycle_kind_override="create" if binding_role else None,
                        extra_metadata=lifecycle_extra,
                    )
                    if catalog_meta and isinstance(spec.call_impl, str):
                        if bundle_path is not None:
                            catalog_meta["_bundle_path"] = str(bundle_path)
                        catalog_json = json.dumps(catalog_meta, ensure_ascii=False)
                        enriched_call_impl = f"{spec.call_impl} --catalog-metadata {shlex.quote(catalog_json)}"
                        agent_compatible_type = not op_resource_type or op_resource_type.endswith(
                            ("agent", "agentconfig")
                        )
                        create_required = (
                            ["agent_id", "created_persisted", "resource_catalog_path"]
                            if agent_compatible_type
                            else [
                                "resource_ref",
                                "resource_type",
                                "created_persisted",
                                "resource_catalog_path",
                            ]
                        )
                        spec = DEFAULTS.tool_authoring.create_spec(
                            **{
                                **spec.__dict__,
                                "call_impl": enriched_call_impl,
                                "output_schema": {
                                    "type": "object",
                                    "properties": {
                                        "resource_ref": {"type": "string"},
                                        "resource_type": {"type": "string"},
                                        "agent_id": {"type": "string"},
                                        "created_persisted": {"const": True},
                                        "resource_catalog_path": {"type": "string"},
                                        "resource_catalog_reason": {"type": "string"},
                                        "callable_by_agent_id": {"type": "boolean"},
                                        "callable_by_resource_ref": {"type": "boolean"},
                                    },
                                    "required": create_required,
                                },
                            }
                        )
                        DEFAULTS.tool_authoring.validate_spec(spec)
                elif lifecycle_kind == "invoke":
                    fallback_meta = lifecycle_fallback_payload(
                        op,
                        source_dir=source_dir_abs,
                        bundle_id=effective_bundle_id,
                        module_name=module_path,
                        class_name=op.class_name,
                        tool_name=spec.name,
                        known_create_types=_known_create_types_frozen,
                        lifecycle_kind_override="invoke" if binding_role else None,
                        extra_metadata=lifecycle_extra,
                    )
                    if fallback_meta and isinstance(spec.call_impl, str):
                        if bundle_path is not None:
                            fallback_meta["_bundle_path"] = str(bundle_path)
                        fallback_json = json.dumps(fallback_meta, ensure_ascii=False)
                        enriched_call_impl = f"{spec.call_impl} --catalog-fallback {shlex.quote(fallback_json)}"
                        spec = DEFAULTS.tool_authoring.create_spec(**{**spec.__dict__, "call_impl": enriched_call_impl})
                        DEFAULTS.tool_authoring.validate_spec(spec)

                specs.append(spec)

                # Build name mapping (original → kebab-case).
                # Primary name: {comp_name}.{op_name} — matches grouper convention
                # and is used as the tool's registered name.
                grouper_name = f"{comp.name}.{op.name}"
                name_map[grouper_name] = spec.name

                # Also register fallback names for robust lookup
                if op.class_name:
                    class_name_raw = f"{op.class_name}.{op.name}"
                    name_map[class_name_raw] = spec.name
                if op.file_stem:
                    file_stem_raw = f"{op.file_stem}.{op.name}"
                    if file_stem_raw != grouper_name:
                        name_map[file_stem_raw] = spec.name
                # Class-qualified form: {comp}.{class}.{op} — used by
                # COMPONENT_GROUP / KEYWORD_MATCH strategies after the
                # skill_grouper fix that includes class_name in allowed_tools.
                if op.class_name:
                    comp_class_name = f"{comp.name}.{op.class_name}.{op.name}"
                    name_map[comp_class_name] = spec.name
                # Fully-qualified form used by IO_RELATION strategy
                full_name = f"{comp.name}.{grouper_name}"
                name_map[full_name] = spec.name
            # ponytail: one bad op (e.g. sys.exit at import) must not abort convert
            except (Exception, SystemExit):
                logger.warning(
                    "Failed to build tool spec for %s.%s, skipping",
                    comp.name,
                    op.name,
                    exc_info=True,
                )
                continue

    _bridge_progress(f"\r   Building tool specs: {spec_idx}/{total_ops} done")

    # ── Phase 4: persist specs (JSON files) ──

    if persist:
        if tool_dir is not None:
            tool_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            target_dir = tool_dir or DEFAULTS.tool_authoring.TOOL_DIR
            spec_path = target_dir / f"{spec.name}.json"
            if not overwrite and spec_path.exists():
                logger.debug("Tool spec already exists, skipping: %s", spec.name)
                continue
            DEFAULTS.tool_authoring.save_spec(spec, tool_dir=tool_dir)
            logger.info("Persisted tool spec: %s -> %s", spec.name, spec_path.parent)

    if skipped_groups:
        skipped_ops = sum(len(ops) for key, ops in groups.items() if key not in script_paths)
        logger.warning(
            "%d wrapper script(s) skipped (%d operations) due to syntax errors",
            len(skipped_groups),
            skipped_ops,
        )

    logger.info(
        "Registered %d tools from %d components (%d wrapper scripts)",
        len(specs),
        len(components),
        len(script_paths),
    )

    # ── Phase 5: L2 lifecycle dependency metadata ──
    if bundle_path is not None:
        try:
            lifecycle_graph = ToolDependencyGraph.detect_from_components(components)
            deps_path = bundle_path / ".clawcodex" / "tool-dependencies.yaml"
            write_tool_dependencies(
                lifecycle_graph,
                deps_path,
                project_name=effective_bundle_id or "",
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Failed to write tool-dependencies.yaml: %s", exc)

    return name_map


# ---------------------------------------------------------------------------
# HTTP Tool Registration
# ---------------------------------------------------------------------------


def _sdk_param_to_json_schema_property(param: SdkParam) -> dict[str, Any]:
    """Convert an SdkParam to a JSON Schema property dict."""
    prop: dict[str, Any] = {
        "type": param.param_type,
    }
    if param.description:
        prop["description"] = param.description
    if param.schema:
        prop.update(param.schema)
        if "type" in param.schema:
            prop["type"] = param.schema["type"]
    return prop


def _build_http_input_schema(method: SdkMethod) -> dict[str, Any]:
    """Build JSON Schema input schema for an HTTP tool."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    has_body_param = any(p.name == "body" and p.location == "body" for p in method.params)

    if has_body_param and method.request_body:
        content = method.request_body.get("content", {})
        for media_type, schema in content.items():
            if isinstance(schema, dict) and "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    properties[prop_name] = prop_schema
                    if schema.get("required") and prop_name in schema.get("required", []):
                        required.append(prop_name)
    else:
        for param in method.params:
            prop = _sdk_param_to_json_schema_property(param)
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        if method.request_body:
            content = method.request_body.get("content", {})
            for media_type, schema in content.items():
                if isinstance(schema, dict):
                    if "properties" in schema:
                        for prop_name, prop_schema in schema["properties"].items():
                            if prop_name not in properties:
                                properties[prop_name] = prop_schema
                                if schema.get("required") and prop_name in schema.get("required", []):
                                    required.append(prop_name)
                    elif "$ref" in schema:
                        pass

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _build_http_call_impl(method: SdkMethod) -> dict[str, str]:
    """Build call_impl dict for HTTP tool."""
    url = method.http_path or ""

    return {
        "method": method.http_method or "GET",
        "url": url,
    }


_HTTP_WRAPPER_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated HTTP wrapper for __TOOL_NAME__ - created by SOP converter."""

import argparse
import json
import sys
from typing import Any


def _build_request(args: argparse.Namespace) -> dict[str, Any]:
    """Build request dict from parsed arguments."""
    request = {}
__REQUEST_BUILDER__
    return request


def _format_url(url: str, path_params: dict[str, Any]) -> str:
    """Replace path parameters in URL template."""
    for key, value in path_params.items():
        url = url.replace("{" + key + "}", str(value))
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description="__DESCRIPTION__")
__ARGPARSE_DEFINITIONS__

    args = parser.parse_args()

    request = _build_request(args)

    # Process URL and body
    path_params = {}
    url_template = "__HTTP_PATH__"
    for k, v in request.items():
        if "{" + k + "}" in url_template:
            path_params[k] = v

    url = _format_url(url_template, path_params)
    body_params = {k: v for k, v in request.items() if k not in path_params}
    data = json.dumps(body_params).encode("utf-8") if body_params else None

    # Print request info
    print("=== Request ===")
    print(f"Method: __HTTP_METHOD__")
    print(f"URL: {url}")
    print('Headers: {"Content-Type": "application/json"}')
    if body_params:
        print(f"Body: {json.dumps(body_params, indent=2, ensure_ascii=False)}")
    print()

    # For testing: make actual HTTP request
    try:
        import urllib.request
        import urllib.error

        headers = {"Content-Type": "application/json"}

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="__HTTP_METHOD__",
        )

        with urllib.request.urlopen(req) as resp:
            response_body = resp.read().decode("utf-8")
            print("=== Response ===")
            print(f"Status: {resp.status}")
            print(f"Body: {response_body}")
            return 0
    except ImportError:
        print("Note: urllib available, skipping actual HTTP request")
        return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.read().decode('utf-8')}")
        return e.code
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''


def _generate_http_wrapper_script(
    method: SdkMethod,
    tool_name: str,
    scripts_dir: Path,
) -> Path:
    """Generate a standalone HTTP wrapper script for testing."""
    script_name = f"{tool_name}.py"
    script_path = scripts_dir / script_name
    scripts_dir.mkdir(parents=True, exist_ok=True)

    argparse_definitions: list[str] = []
    request_builder_lines: list[str] = []

    has_body_param = any(p.name == "body" and p.location == "body" for p in method.params)

    if has_body_param and method.request_body:
        content = method.request_body.get("content", {})
        for media_type, schema in content.items():
            if isinstance(schema, dict) and "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    prop_type = prop_schema.get("type", "string")
                    if prop_type == "integer":
                        arg_type = int
                    elif prop_type == "number":
                        arg_type = float
                    elif prop_type == "boolean":
                        arg_type = bool
                    elif prop_type == "array":
                        required_fields = schema.get("required", [])
                        is_required = prop_name in required_fields
                        if is_required:
                            argparse_definitions.append(
                                f'    parser.add_argument("--{prop_name}", type=str, required=True, help="{prop_schema.get("description", "")}")'
                            )
                        else:
                            argparse_definitions.append(
                                f'    parser.add_argument("--{prop_name}", type=str, default=None, help="{prop_schema.get("description", "")}")'
                            )
                        request_builder_lines.append(
                            f'    if args.{prop_name} is not None:\n        request["{prop_name}"] = [x.strip() for x in args.{prop_name}.split(",")]'
                        )
                        continue
                    else:
                        arg_type = str

                    required_fields = schema.get("required", [])
                    is_required = prop_name in required_fields

                    if is_required:
                        argparse_definitions.append(
                            f'    parser.add_argument("--{prop_name}", type={arg_type.__name__}, required=True, help="{prop_schema.get("description", "")}")'
                        )
                    else:
                        argparse_definitions.append(
                            f'    parser.add_argument("--{prop_name}", type={arg_type.__name__}, default=None, help="{prop_schema.get("description", "")}")'
                        )
                    request_builder_lines.append(
                        f'    if args.{prop_name} is not None:\n        request["{prop_name}"] = args.{prop_name}'
                    )
    else:
        for param in method.params:
            arg_type = param.param_type
            if arg_type == "integer":
                arg_type = int
            elif arg_type == "number":
                arg_type = float
            elif arg_type == "boolean":
                arg_type = bool
            else:
                arg_type = str

            if param.required:
                argparse_definitions.append(
                    f'    parser.add_argument("--{param.name}", type={arg_type.__name__}, required=True, help="{param.description}")'
                )
            else:
                argparse_definitions.append(
                    f'    parser.add_argument("--{param.name}", type={arg_type.__name__}, default=None, help="{param.description}")'
                )
            request_builder_lines.append(
                f'    if args.{param.name} is not None:\n        request["{param.name}"] = args.{param.name}'
            )

        if method.request_body:
            content = method.request_body.get("content", {})
            for media_type, schema in content.items():
                if isinstance(schema, dict) and "properties" in schema:
                    for prop_name, prop_schema in schema["properties"].items():
                        if prop_name not in [p.name for p in method.params]:
                            prop_type = prop_schema.get("type", "string")
                            if prop_type == "integer":
                                arg_type = int
                            elif prop_type == "number":
                                arg_type = float
                            elif prop_type == "boolean":
                                arg_type = bool
                            else:
                                arg_type = str

                            required_fields = schema.get("required", [])
                            is_required = prop_name in required_fields

                            if is_required:
                                argparse_definitions.append(
                                    f'    parser.add_argument("--{prop_name}", type={arg_type.__name__}, required=True, help="{prop_schema.get("description", "")}")'
                                )
                            else:
                                argparse_definitions.append(
                                    f'    parser.add_argument("--{prop_name}", type={arg_type.__name__}, default=None, help="{prop_schema.get("description", "")}")'
                                )
                            request_builder_lines.append(
                                f'    if args.{prop_name} is not None:\n        request["{prop_name}"] = args.{prop_name}'
                            )

    content = _HTTP_WRAPPER_TEMPLATE
    content = content.replace("__TOOL_NAME__", tool_name)
    content = content.replace("__DESCRIPTION__", method.description or method.name)
    content = content.replace("__HTTP_METHOD__", method.http_method or "GET")
    content = content.replace("__HTTP_PATH__", method.http_path or "")
    content = content.replace("__ARGPARSE_DEFINITIONS__", "\n".join(argparse_definitions))
    content = content.replace("__REQUEST_BUILDER__", "\n".join(request_builder_lines))

    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)

    return script_path


def register_http_tools(
    methods: list[SdkMethod],
    *,
    persist: bool = True,
    overwrite: bool = True,
    bundle_dir: str | Path | None = None,
    bundle_id: str | None = None,
    generate_wrappers: bool = True,
) -> dict[str, str]:
    """Register OpenAPI operations as HTTP tools.

    Args:
        methods: List of SdkMethod parsed from OpenAPI spec.
        persist: If True, persist each AgentToolSpec to disk.
        overwrite: If True, overwrite existing specs with the same name.
        bundle_dir: Optional bundle directory for spec persistence.
        bundle_id: Optional bundle identifier.
        generate_wrappers: If True, generate standalone wrapper scripts.

    Returns:
        A mapping from original method names to kebab-case tool names.
    """
    bundle_path = Path(bundle_dir).resolve() if bundle_dir is not None else None
    effective_bundle_id = bundle_id or (bundle_path.name if bundle_path else None)
    tool_dir = DEFAULTS.tool_authoring.bundle_tool_dir(bundle_path) if bundle_path is not None else None
    scripts_dir = DEFAULTS.tool_authoring.scripts_dir_for(tool_dir) if tool_dir is not None else SCRIPTS_DIR

    name_map: dict[str, str] = {}
    specs: list[Any] = []

    for method in methods:
        tool_name = _to_kebab_case(method.name)

        input_schema = _build_http_input_schema(method)
        call_impl = _build_http_call_impl(method)

        tags = tuple(method.tags) if method.tags else ()

        spec = DEFAULTS.tool_authoring.create_spec(
            name=tool_name,
            description=method.description or method.name,
            input_schema=input_schema,
            call_type="http",
            call_impl=call_impl,
            tags=tags,
            source="openapi-converter",
            bundle_id=effective_bundle_id,
        )

        DEFAULTS.tool_authoring.validate_spec(spec)
        specs.append(spec)

        name_map[method.name] = tool_name

        if generate_wrappers:
            _generate_http_wrapper_script(method, tool_name, scripts_dir)

    if persist:
        if tool_dir is not None:
            tool_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            target_dir = tool_dir or DEFAULTS.tool_authoring.TOOL_DIR
            spec_path = target_dir / f"{spec.name}.json"
            if not overwrite and spec_path.exists():
                logger.debug("HTTP tool spec already exists, skipping: %s", spec.name)
                continue
            DEFAULTS.tool_authoring.save_spec(spec, tool_dir=tool_dir)
            logger.info("Persisted HTTP tool spec: %s -> %s", spec.name, spec_path.parent)

    logger.info(
        "Registered %d HTTP tools from OpenAPI spec",
        len(specs),
    )

    return name_map
