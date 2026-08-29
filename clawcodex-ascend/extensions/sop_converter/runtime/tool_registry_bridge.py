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
    WRAPPER_TEAM_DATABASE_COERCION,
    WRAPPER_MESSAGER_COERCION,
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

# --- appended from PR #726 ---

# ruff: noqa
# pylint: skip-file


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


# --- appended from PR #963 ---

# ruff: noqa
# pylint: skip-file

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
