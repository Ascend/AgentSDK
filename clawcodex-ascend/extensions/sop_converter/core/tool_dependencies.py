"""Infer tool prerequisite / postrequisite edges from parameter and return types.

During ``sop convert``, when method *B* takes a parameter whose type is the
return type of method *A*, *A* is recorded as a prerequisite of *B*.

**Surfacing policy (selection vs orchestration)**

- **Orchestration only** — edges are written to each tool's
  ``input_schema.x-sop-dependencies`` (loaded after ToolSearch selects the
  tool) and to ``ORCHESTRATION_ROUTES.md`` via
  :mod:`cross_domain_orchestration`.
- **Not for selection** — dependencies must **not** appear in tool
  ``description``, ToolSearch ``search_hint`` / tags, or Skill task-guide rows;
  those surfaces mislead agents into calling prerequisites before picking the
  right entry tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from extensions.sop_converter.core.import_alias_resolver import ModuleImportIndex, resolve_module_path
from extensions.sop_converter.core.source_parser import ParamSpec, SourceComponent, SourceOperation

_PRIMITIVE_TYPES = frozenset(
    {
        "str",
        "string",
        "int",
        "integer",
        "float",
        "number",
        "bool",
        "boolean",
        "dict",
        "object",
        "list",
        "array",
        "any",
        "none",
        "null",
        "path",
        "bytes",
        "tuple",
        "set",
        "mapping",
        "sequence",
        "iterable",
        "optional",
        "union",
        # Generic callback/async types: returning these must not trigger
        # dependency inference, or the enables list would explode (almost
        # every tool accepts Callable/Any parameters).
        "callable",
        "awaitable",
        "coroutine",
        "asyncgenerator",
        "asynciterator",
        "type",
        "classvar",
        "noreturn",
        # Synchronization primitives are shared across modules, so type
        # matching would produce many spurious dependencies.
        "event",
        "queue",
        "lock",
        "future",
        "task",
        # typing protocols / generic base classes carry no concrete business semantics.
        "protocol",
        "generic",
    }
)


def _split_union(type_hint: str) -> list[str]:
    cleaned = type_hint.strip()
    if not cleaned:
        return []

    if cleaned.startswith("Union[") and cleaned.endswith("]"):
        inner = cleaned[len("Union[") : -1]
        parts = [p.strip() for p in inner.split(",")]
    elif "|" in cleaned:
        parts = [p.strip() for p in cleaned.split("|")]
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


def sanitize_type_name(type_hint: str | None) -> str | None:
    """Normalize a Python type hint to a comparable token (aligned with IO_RELATION)."""
    if not type_hint:
        return None

    s = type_hint.strip()
    if not s:
        return None

    m = re.match(r"^Optional\[(.+)\]$", s)
    if m:
        s = m.group(1).strip()

    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        parts = [re.sub(r"['\"]", "", p) for p in parts if p not in ("None", "NoneType")]
        s = "_".join(parts)

    m = re.match(r"^Union\[(.+)\]$", s)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        parts = [re.sub(r"['\"]", "", p) for p in parts if p not in ("None", "NoneType")]
        s = "_".join(parts)

    generic_set = {"list", "List", "dict", "Dict", "set", "Set", "tuple", "Tuple"}
    for prefix in generic_set:
        m = re.match(rf"^{prefix}\[.+]$", s)
        if m:
            s = prefix.lower()
            break
    else:
        s = re.sub(r"[<\(\[{].*", "", s)

    # Module-qualified generic types (e.g. asyncio.Task / threading.Event)
    # must also be filtered: check the segment after the last "." against
    # the blacklist so the enables list does not explode.
    if "." in s:
        last_seg = s.rsplit(".", 1)[-1].strip()
        if last_seg.lower() in _PRIMITIVE_TYPES:
            return None
    s = s.replace(".", "_")
    s = re.sub(r"[|\\/:*?\"<>']", "", s)
    s = s.lower().strip()
    if not s or s in _PRIMITIVE_TYPES:
        return None
    return s


def extract_type_roots(type_hint: str | None) -> set[str]:
    if not type_hint:
        return set()
    roots: set[str] = set()
    for part in _split_union(type_hint):
        sanitized = sanitize_type_name(part)
        if sanitized:
            roots.add(sanitized)
    return roots


def raw_tool_name(comp_name: str, op: SourceOperation) -> str:
    """Mirror ``operation_to_spec`` naming before kebab conversion."""
    if op.class_name:
        if comp_name:
            return f"{comp_name}.{op.class_name}.{op.name}"
        return f"{op.class_name}.{op.name}"
    if comp_name:
        return f"{comp_name}.{op.name}"
    if op.file_stem:
        return f"{op.file_stem}.{op.name}"
    return op.name


def to_kebab_tool_name(comp_name: str, op: SourceOperation) -> str:
    raw = raw_tool_name(comp_name, op)
    s = raw.replace(".", "-").replace("__", "-").replace("_", "-")
    s = re.sub(r"-+", "-", s)
    return s.strip("-").lower()


@dataclass
class ToolOperationDeps:
    tool_name: str
    requires: list[str] = field(default_factory=list)
    enables: list[str] = field(default_factory=list)
    produces_types: list[str] = field(default_factory=list)
    consumes_types: list[str] = field(default_factory=list)
    is_async_generator: bool = False


def _is_chain_builder_producer(op: SourceOperation) -> bool:
    """Class methods that only mutate and return ``self`` are not cross-tool factories."""
    if not op.class_name:
        return False
    if op.name.startswith("configure_"):
        return True
    if op.return_type:
        returned = op.return_type.strip().strip("'\"")
        if returned == op.class_name:
            return True
    return False


def _operation_module_path(
    comp: SourceComponent,
    source_dir: str,
    op: SourceOperation,
) -> str:
    return resolve_module_path(comp, source_dir, op.file_stem or "unknown")


def _collect_type_tokens(
    comp: SourceComponent,
    op: SourceOperation,
    *,
    resolver: ModuleImportIndex | None,
    source_dir: str | None,
    type_hints: list[str | None],
) -> set[str]:
    types: set[str] = set()
    module_path = _operation_module_path(comp, source_dir, op) if source_dir else ""
    for hint in type_hints:
        if not hint:
            continue
        if resolver and module_path:
            identity = resolver.resolve_type_identity(module_path, hint)
            if identity:
                types.add(identity)
                continue
        types.update(extract_type_roots(hint))
    return types


def _collect_param_types(
    comp: SourceComponent,
    op: SourceOperation,
    *,
    resolver: ModuleImportIndex | None = None,
    source_dir: str | None = None,
) -> set[str]:
    params: list[ParamSpec] = list(op.parameters)
    if op.class_name:
        params = [*comp.class_init_params.get(op.class_name, ()), *params]
    hints = [param.type_hint for param in params if not param.name.startswith("*")]
    return _collect_type_tokens(
        comp,
        op,
        resolver=resolver,
        source_dir=source_dir,
        type_hints=hints,
    )


def build_tool_dependency_index(
    components: list[SourceComponent],
    *,
    source_dir: str | None = None,
) -> dict[str, ToolOperationDeps]:
    """Map kebab tool name → prerequisite / postrequisite metadata."""
    resolver = ModuleImportIndex(source_dir) if source_dir else None
    entries: list[tuple[str, SourceComponent, SourceOperation, set[str], set[str]]] = []
    producers: dict[str, list[str]] = {}
    consumers: dict[str, list[str]] = {}

    for comp in components:
        for op in comp.operations:
            tool_name = to_kebab_tool_name(comp.name, op)
            consumes = _collect_param_types(
                comp,
                op,
                resolver=resolver,
                source_dir=source_dir,
            )
            produces = _collect_type_tokens(
                comp,
                op,
                resolver=resolver,
                source_dir=source_dir,
                type_hints=[op.return_type],
            )
            entries.append((tool_name, comp, op, consumes, produces))

            if not _is_chain_builder_producer(op):
                for type_name in produces:
                    producers.setdefault(type_name, []).append(tool_name)
            for type_name in consumes:
                consumers.setdefault(type_name, []).append(tool_name)

    index: dict[str, ToolOperationDeps] = {}
    for tool_name, _comp, op, consumes, produces in entries:
        requires: list[str] = []
        enables: list[str] = []
        for type_name in sorted(consumes):
            for producer in producers.get(type_name, ()):
                if producer != tool_name and producer not in requires:
                    requires.append(producer)
        for type_name in sorted(produces):
            for consumer in consumers.get(type_name, ()):
                if consumer != tool_name and consumer not in enables:
                    enables.append(consumer)

        index[tool_name] = ToolOperationDeps(
            tool_name=tool_name,
            requires=requires,
            enables=enables,
            produces_types=sorted(produces),
            consumes_types=sorted(consumes),
            is_async_generator=bool(op.is_async_generator),
        )
    return index


def dependency_schema_fragment(deps: ToolOperationDeps | None) -> dict[str, Any] | None:
    if deps is None:
        return None
    if not (deps.requires or deps.enables or deps.produces_types or deps.consumes_types or deps.is_async_generator):
        return None

    payload: dict[str, Any] = {}
    if deps.requires:
        payload["requires"] = [
            {"tool": name, "reason": "parameter type produced by prerequisite tool"} for name in deps.requires
        ]
    if deps.enables:
        payload["enables"] = [
            {"tool": name, "reason": "downstream tool consumes this tool's return type"} for name in deps.enables
        ]
    if deps.produces_types:
        payload["produces_types"] = list(deps.produces_types)
    if deps.consumes_types:
        payload["consumes_types"] = list(deps.consumes_types)
    if deps.is_async_generator:
        payload["yields_stream"] = True
    return payload


def enrich_input_schema_with_dependencies(
    input_schema: dict[str, Any],
    deps: ToolOperationDeps | None,
) -> dict[str, Any]:
    fragment = dependency_schema_fragment(deps)
    if fragment is None:
        return input_schema
    enriched = dict(input_schema)
    enriched["x-sop-dependencies"] = fragment
    return enriched


def dependency_description_suffix(deps: ToolOperationDeps | None) -> str:
    """Deprecated for tool specs — kept for tests/backward compatibility only.

    Dependency hints belong in ``x-sop-dependencies`` and orchestration routes,
    not in ToolSearch-visible descriptions.
    """
    _ = deps
    return ""


def dependency_search_tags(deps: ToolOperationDeps | None) -> tuple[str, ...]:
    """Deprecated for ToolSearch tags — same policy as ``dependency_description_suffix``."""
    _ = deps
    return ()
