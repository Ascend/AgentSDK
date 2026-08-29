#!/usr/bin/env python3
# coding=utf-8

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

"""Session macro overlay: immutable snapshot + COW commit + cleanup/pool."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping, TypeVar

from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import Tool

from ..composite_runtime import CompositeWorkflowSpec
from .errors import MacroConvertError
from .models import MacroDefinition, MacroRoute
from .compiler import compile_macro_definition
from .validation import ValidatedSessionMacro

if TYPE_CHECKING:
    from clawcodex_ext.tool_system.context import ToolContext

_SESSION_MACRO_ATTR = "_session_macro"
_T = TypeVar("_T")

MAX_SESSION_MACROS = 32
MAX_DEFINITION_BYTES = 64 * 1024
MAX_STRING_CHARS = 512
MAX_DESCRIPTION_CHARS = 2 * 1024
MAX_ROUTE_TERMS = 32
RATE_LIMIT_MAX = 5
RATE_LIMIT_WINDOW_SEC = 600.0


def mark_session_macro_tool(tool: Tool) -> Tool:
    """Tag a Tool as overlay-provenance so cleanup can strip by identity."""
    setattr(tool, _SESSION_MACRO_ATTR, True)
    return tool


def is_session_macro_tool(tool: Tool) -> bool:
    return bool(getattr(tool, _SESSION_MACRO_ATTR, False))


def _freeze_lower_map(mapping: Mapping[str, _T]) -> Mapping[str, _T]:
    return MappingProxyType({str(k).lower(): v for k, v in mapping.items()})


def _tool_name_key(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "").lower()


def _read_overlay(context: ToolContext) -> SessionMacroOverlay | None:
    overlay = getattr(context, "session_macro_overlay", None)
    if overlay is None:
        return None
    return overlay if hasattr(overlay, "read") and hasattr(overlay, "commit") else None


def _owner_matching_snapshot(context: ToolContext) -> SessionMacroSnapshot | None:
    overlay = _read_overlay(context)
    if overlay is None:
        return None
    snapshot = overlay.read()
    if snapshot is None:
        return None
    session_id = getattr(context, "session_id", None)
    if not session_id or session_id != snapshot.owner_session_id:
        return None
    return snapshot


def _strip_session_tools_restore_covered(
    tools: list[Any],
    covered_base_tools: Mapping[str, Tool],
) -> list[Any]:
    """Drop session-provenance tools; restore covered base Tools by identity."""
    kept: list[Any] = []
    dropped_names: list[str] = []
    for tool in tools:
        if is_session_macro_tool(tool):
            key = _tool_name_key(tool)
            if key:
                dropped_names.append(key)
            continue
        kept.append(tool)

    present_ids = {id(tool) for tool in kept}
    present_names = {_tool_name_key(tool) for tool in kept if _tool_name_key(tool)}
    for name in dropped_names:
        base = covered_base_tools.get(name)
        if base is None:
            continue
        if id(base) in present_ids:
            continue
        base_key = _tool_name_key(base)
        if base_key and base_key in present_names:
            continue
        kept.append(base)
        present_ids.add(id(base))
        if base_key:
            present_names.add(base_key)
    return kept


@dataclass(frozen=True)
class SessionMacroSnapshot:
    """Immutable per-session overlay state; replace wholesale via COW commit."""

    owner_session_id: str
    generation: int
    definitions: Mapping[str, MacroDefinition] = field(default_factory=dict)
    specs: Mapping[str, CompositeWorkflowSpec] = field(default_factory=dict)
    tools: Mapping[str, Tool] = field(default_factory=dict)
    tool_specs: Mapping[str, AgentToolSpec] = field(default_factory=dict)
    routes: tuple[MacroRoute, ...] = ()
    success_timestamps: tuple[float, ...] = ()
    covered_base_tools: Mapping[str, Tool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", _freeze_lower_map(self.definitions))
        object.__setattr__(self, "specs", _freeze_lower_map(self.specs))
        object.__setattr__(self, "tools", _freeze_lower_map(self.tools))
        object.__setattr__(self, "tool_specs", _freeze_lower_map(self.tool_specs))
        object.__setattr__(self, "covered_base_tools", _freeze_lower_map(self.covered_base_tools))
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "success_timestamps", tuple(self.success_timestamps))


class SessionMacroOverlay:
    """Thread-safe holder for a single immutable snapshot (lock + COW)."""

    def __init__(self, snapshot: SessionMacroSnapshot | None = None) -> None:
        self._lock = threading.RLock()
        self._snapshot = snapshot

    def read(self) -> SessionMacroSnapshot | None:
        with self._lock:
            return self._snapshot

    def commit(self, new_snap: SessionMacroSnapshot | None) -> None:
        """Atomically replace the snapshot under the lock (COW)."""
        with self._lock:
            self._snapshot = new_snap

    def mutate(
        self,
        mutator: Callable[[SessionMacroSnapshot | None], SessionMacroSnapshot],
    ) -> SessionMacroSnapshot:
        """Hold the lock across read → build → commit (TOCTOU-safe)."""
        with self._lock:
            new_snap = mutator(self._snapshot)
            self._snapshot = new_snap
            return new_snap


@dataclass(frozen=True)
class SessionMacroPlanStep:
    step_id: str
    tool: str
    args_template: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args_template", MappingProxyType(dict(self.args_template)))


@dataclass(frozen=True)
class SessionMacroPlan:
    action: Literal["create", "replace", "promote"]
    name: str
    description: str
    catalog_id: str
    owner_session_id: str
    expected_generation: int
    steps: tuple[SessionMacroPlanStep, ...]
    route_summary: Mapping[str, Any]
    originating_user_turn_id: str | None = None
    target_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "route_summary", MappingProxyType(dict(self.route_summary)))


def _raise(code: str, message: str, *, error_field: str = "") -> None:
    raise MacroConvertError(code, message, field=error_field)


def _serialized_size(definition_dict: Mapping[str, Any]) -> int:
    return len(json.dumps(definition_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _check_size_limits(definition_dict: Mapping[str, Any], macro: MacroDefinition) -> None:
    if _serialized_size(definition_dict) > MAX_DEFINITION_BYTES:
        _raise(
            "macro_definition_too_large",
            f"definition exceeds {MAX_DEFINITION_BYTES} bytes",
        )
    if len(macro.name) > MAX_STRING_CHARS:
        _raise("macro_definition_too_large", "name exceeds string limit", error_field="name")
    if len(macro.description or "") > MAX_DESCRIPTION_CHARS:
        _raise(
            "macro_definition_too_large",
            "description exceeds string limit",
            error_field="description",
        )
    route = macro.routing
    for label, values in (
        ("phrases", route.phrases),
        ("keywords", route.keywords),
        ("negative_keywords", route.negative_keywords),
    ):
        for item in values:
            if len(item) > MAX_STRING_CHARS:
                _raise(
                    "macro_definition_too_large",
                    f"routing.{label} entry exceeds string limit",
                    error_field=f"routing.{label}",
                )
    if len(route.phrases) + len(route.keywords) > MAX_ROUTE_TERMS:
        _raise(
            "macro_definition_too_large",
            f"routing phrases+keywords exceed {MAX_ROUTE_TERMS}",
            error_field="routing",
        )


def _check_protected_targets(
    name: str,
    protected_builtin_exclusive_targets: Iterable[str],
) -> None:
    protected = {str(t).lower() for t in protected_builtin_exclusive_targets}
    if name.lower() in protected:
        _raise(
            "macro_route_conflict",
            f"session macro conflicts with protected builtin exclusive target: {name}",
            error_field="name",
        )


def _recent_successes(timestamps: tuple[float, ...], *, now: float) -> tuple[float, ...]:
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    return tuple(ts for ts in timestamps if ts >= cutoff)


def _check_rate_and_count(
    snapshot: SessionMacroSnapshot | None,
    *,
    name_key: str,
    replace: bool,
    now: float,
) -> None:
    timestamps = _recent_successes(
        snapshot.success_timestamps if snapshot is not None else (),
        now=now,
    )
    if len(timestamps) >= RATE_LIMIT_MAX:
        _raise(
            "macro_registration_rate_limited",
            f"at most {RATE_LIMIT_MAX} successful registrations per {int(RATE_LIMIT_WINDOW_SEC)}s",
        )
    definitions = snapshot.definitions if snapshot is not None else {}
    exists = name_key in definitions
    if exists and not replace:
        _raise(
            "macro_already_exists",
            f"session macro already exists: {name_key}",
            error_field="name",
        )
    if not exists and len(definitions) >= MAX_SESSION_MACROS:
        _raise(
            "macro_session_limit_exceeded",
            f"session may have at most {MAX_SESSION_MACROS} macros",
        )


def _build_plan(
    validated: ValidatedSessionMacro,
    *,
    action: Literal["create", "replace", "promote"],
    owner_session_id: str,
    expected_generation: int,
    target_path: str = "",
    catalog_id: str | None = None,
) -> SessionMacroPlan:
    steps = tuple(
        SessionMacroPlanStep(
            step_id=step.id,
            tool=step.callable_ref,
            args_template=dict(step.args),
        )
        for step in validated.workflow.steps
    )
    routing = validated.definition.routing
    return SessionMacroPlan(
        action=action,
        name=validated.definition.name,
        description=validated.definition.description or "",
        catalog_id=catalog_id or f"session:{validated.definition.name}",
        owner_session_id=owner_session_id,
        expected_generation=expected_generation,
        steps=steps,
        route_summary={
            "selection": routing.selection,
            "phrases": list(routing.phrases),
            "keywords": list(routing.keywords),
            "priority": routing.priority,
        },
        target_path=target_path,
    )


def _find_covered_base(context: ToolContext, name_key: str) -> Tool | None:
    options = getattr(context, "options", None)
    for tool in list(getattr(options, "tools", None) or []):
        if is_session_macro_tool(tool):
            continue
        if _tool_name_key(tool) == name_key:
            return tool
    return None


def _success_payload(
    validated: ValidatedSessionMacro,
    *,
    replaced: bool,
) -> dict[str, Any]:
    definition = validated.definition
    routing = definition.routing
    return {
        "registered": True,
        "name": definition.name,
        "catalog_id": f"session:{definition.name}",
        "replaced": replaced,
        "step_count": len(validated.workflow.steps),
        "route": {
            "selection": routing.selection,
            "phrases": list(routing.phrases),
            "keywords": list(routing.keywords),
        },
    }


def register_session_macro(
    context: ToolContext,
    definition_dict: Mapping[str, Any],
    *,
    replace: bool,
    tool_index: Iterable[str] | None,
    workflow_tool_names: Iterable[str] | None,
    protected_builtin_exclusive_targets: Iterable[str] | None,
    create_tool: Callable[[AgentToolSpec], Tool],
) -> dict[str, Any]:
    """Parse → validate → confirm → locked TOCTOU COW commit.

    Does **not** call ``ToolRegistry.register``.
    """
    # 1. Capability gate (universal; before interactive/confirm).
    if not getattr(context, "allow_session_macro_registration", False):
        _raise("macro_capability_denied", "session macro registration is not allowed")

    # 2. Require session_id.
    session_id = getattr(context, "session_id", None)
    if not session_id:
        _raise("macro_session_required", "session_id is required to register a session macro")

    protected = protected_builtin_exclusive_targets or ()
    forbid = workflow_tool_names or ()

    # 3. Strict parse + size limits + validate (MacroCompiler facade).
    if not isinstance(definition_dict, Mapping):
        _raise("macro_schema_invalid", "definition must be a mapping")
    raw = dict(definition_dict)
    if _serialized_size(raw) > MAX_DEFINITION_BYTES:
        _raise(
            "macro_definition_too_large",
            f"definition exceeds {MAX_DEFINITION_BYTES} bytes",
        )
    validated = compile_macro_definition(
        raw,
        tool_index=tool_index,
        forbid_workflow_tools=forbid,
    )
    _check_size_limits(raw, validated.definition)

    # 5. Protected builtin exclusive targets.
    _check_protected_targets(validated.definition.name, protected)

    name_key = validated.definition.name.lower()
    overlay = _read_overlay(context)
    if overlay is None:
        overlay = SessionMacroOverlay()
        context.session_macro_overlay = overlay

    pre = overlay.read()
    if pre is not None and pre.owner_session_id != session_id:
        _raise(
            "macro_stale_session",
            "overlay owner does not match current session_id",
        )

    now = time.monotonic()
    _check_rate_and_count(pre, name_key=name_key, replace=replace, now=now)
    exists = bool(pre is not None and name_key in pre.definitions)
    action: Literal["create", "replace"] = "replace" if (replace and exists) else "create"
    expected_generation = pre.generation if pre is not None else 0

    # 6. Build frozen plan.
    plan = _build_plan(
        validated,
        action=action,
        owner_session_id=str(session_id),
        expected_generation=expected_generation,
    )

    # 7–9. Confirm required (tests inject auto-approve); None/False/exc → deny.
    confirm = getattr(context, "confirm_session_macro_plan", None)
    if confirm is None:
        _raise("macro_registration_denied", "confirm_session_macro_plan is not configured")
    try:
        approved = bool(confirm(plan))
    except Exception as exc:
        raise MacroConvertError(
            "macro_registration_denied",
            f"confirm_session_macro_plan failed: {exc}",
        ) from exc
    if not approved:
        _raise("macro_registration_denied", "session macro registration was denied")

    # Post-confirm session_id check (before lock) for clear stale error.
    if getattr(context, "session_id", None) != plan.owner_session_id:
        _raise("macro_stale_session", "session_id changed during confirm")

    replaced = False

    def _locked_commit(current: SessionMacroSnapshot | None) -> SessionMacroSnapshot:
        nonlocal replaced
        live_session = getattr(context, "session_id", None)
        if live_session != plan.owner_session_id:
            _raise("macro_stale_session", "session_id changed during confirm")
        if current is not None and current.owner_session_id != plan.owner_session_id:
            _raise("macro_stale_session", "overlay owner mismatch after confirm")
        if (current.generation if current is not None else 0) != plan.expected_generation:
            _raise(
                "macro_concurrent_modification",
                "session macro overlay changed during confirm",
            )

        commit_now = time.monotonic()
        _check_rate_and_count(current, name_key=name_key, replace=replace, now=commit_now)
        _check_protected_targets(validated.definition.name, protected)

        live_exists = bool(current is not None and name_key in current.definitions)
        if live_exists and not replace:
            _raise(
                "macro_already_exists",
                f"session macro already exists: {name_key}",
                error_field="name",
            )
        replaced = live_exists and replace

        try:
            tool = create_tool(validated.tool_spec)
        except Exception as exc:
            raise MacroConvertError(
                "macro_create_tool_failed",
                f"create_tool failed: {exc}",
            ) from exc
        mark_session_macro_tool(tool)

        definitions = dict(current.definitions) if current is not None else {}
        specs = dict(current.specs) if current is not None else {}
        tools = dict(current.tools) if current is not None else {}
        tool_specs = dict(current.tool_specs) if current is not None else {}
        covered = dict(current.covered_base_tools) if current is not None else {}
        routes = list(current.routes) if current is not None else []

        if name_key not in covered:
            base = _find_covered_base(context, name_key)
            if base is not None:
                covered[name_key] = base

        definitions[name_key] = validated.definition
        catalog_key = f"session:{validated.definition.name}".lower()
        specs[catalog_key] = validated.workflow
        tools[name_key] = tool
        tool_specs[name_key] = validated.tool_spec

        # Replace any prior route for this target; append new.
        routes = [r for r in routes if str(getattr(r, "target_tool", "") or "").lower() != name_key]
        routes.append(validated.definition.routing)

        timestamps = _recent_successes(
            current.success_timestamps if current is not None else (),
            now=commit_now,
        ) + (commit_now,)

        return SessionMacroSnapshot(
            owner_session_id=plan.owner_session_id,
            generation=(current.generation if current is not None else 0) + 1,
            definitions=definitions,
            specs=specs,
            tools=tools,
            tool_specs=tool_specs,
            routes=tuple(routes),
            success_timestamps=timestamps,
            covered_base_tools=covered,
        )

    # 10–12. Lock TOCTOU + create_tool + COW commit.
    overlay.mutate(_locked_commit)

    # 13. Sync effective tool pool.
    sync_effective_tools(context)

    # 14. Success JSON from validated.definition (not raw input).
    return _success_payload(validated, replaced=replaced)


def clear_session_macros_for_context(context: ToolContext) -> None:
    """Drop overlay snapshot and strip session-provenance tools from the pool.

    Restores covered base Tools by object identity. Clears retrieval plan /
    suppressed on session change (safe default). Rate window lives only on
    the snapshot and is discarded with it.
    """
    overlay = _read_overlay(context)
    snapshot = overlay.read() if overlay is not None else None
    covered: Mapping[str, Tool] = snapshot.covered_base_tools if snapshot is not None else {}

    if overlay is not None:
        overlay.commit(None)

    options = getattr(context, "options", None)
    if options is not None:
        current = list(getattr(options, "tools", None) or [])
        options.tools = _strip_session_tools_restore_covered(current, covered)

    hidden = list(getattr(context, "retrieval_hidden_tools", None) or [])
    context.retrieval_hidden_tools = _strip_session_tools_restore_covered(hidden, covered)

    # Safe default on session change: drop retrieval plan/suppressed entirely.
    context.retrieval_plan = None
    suppressed = getattr(context, "retrieval_suppressed_tools", None)
    if isinstance(suppressed, set):
        suppressed.clear()
    else:
        context.retrieval_suppressed_tools = set()


def iter_effective_tools(context: ToolContext, base_tools: list[Tool] | None) -> list[Tool]:
    """Merge base tools with the owner-matching overlay tool pool.

    Overlay tools win over same-name base entries. Stale session-provenance
    tools in ``base_tools`` are dropped; live overlay tools are appended.
    """
    snapshot = _owner_matching_snapshot(context)
    overlay_tools = list(snapshot.tools.values()) if snapshot is not None else []
    overlay_names = {_tool_name_key(tool) for tool in overlay_tools}

    result: list[Tool] = []
    seen_names: set[str] = set()
    for tool in base_tools or []:
        if is_session_macro_tool(tool):
            continue
        key = _tool_name_key(tool)
        if key and key in overlay_names:
            continue
        result.append(tool)
        if key:
            seen_names.add(key)

    for tool in overlay_tools:
        key = _tool_name_key(tool)
        if key and key in seen_names:
            continue
        result.append(tool)
        if key:
            seen_names.add(key)
    return result


def sync_effective_tools(context: ToolContext) -> None:
    """Refresh ``options.tools`` from the current owner-matching overlay."""
    options = getattr(context, "options", None)
    if options is None:
        return
    options.tools = iter_effective_tools(context, list(options.tools or []))
