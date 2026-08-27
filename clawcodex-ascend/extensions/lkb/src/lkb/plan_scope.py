#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0402
"""Workspace Board / Plan / session scope management for LKB.

A Board is shared by every session in a workspace.  A Plan is a graph inside
that Board and is *not* selected workspace-wide: each session has its own
binding.  Child agents inherit the parent's in-memory binding through
``ToolContext.lkb_plan_id``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .commands import CommandResult
from .ir_hash import canonical_hash
from .json_store import BoardEnvelope
from .refs import NodeRef

PLAN_ACTIVE = "active"
PLAN_SUSPENDED = "suspended"
PLAN_COMPLETED = "completed"
PLAN_ABANDONED = "abandoned"
PLAN_ARCHIVED = "archived"
PLAN_STATES = (
    PLAN_ACTIVE,
    PLAN_SUSPENDED,
    PLAN_COMPLETED,
    PLAN_ABANDONED,
    PLAN_ARCHIVED,
)
PLAN_TERMINAL_STATES = (PLAN_COMPLETED, PLAN_ABANDONED, PLAN_ARCHIVED)
_CONTEXT_FALLBACK_SESSION_ID_ATTR = "_lkb_fallback_session_id"
_LOGGER = logging.getLogger(__name__)


class PlanScopeError(ValueError):
    """Raised when a Plan cannot be resolved, bound, or transitioned."""


@dataclass(frozen=True, slots=True)
class PlanHeader:
    plan_id: str
    title: str
    state: str
    revision: int
    created_at: str = ""
    updated_at: str = ""
    created_by_session_id: str = ""
    session_ids: tuple[str, ...] = ()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_session_id(context: Any | None = None) -> str:
    """Return the trusted current session identity.

    REPL callers normally populate ``ToolContext.session_id``.  TUI and
    compatibility callers may omit it, in which case bootstrap owns the
    canonical process session id.
    """
    value = getattr(context, "session_id", None) if context is not None else None
    if isinstance(value, str) and value:
        return value
    try:
        from src.bootstrap.state import get_session_id
    except ModuleNotFoundError as exc:
        if exc.name not in {"src", "src.bootstrap", "src.bootstrap.state"}:
            raise
        _LOGGER.warning("LKB session-id bootstrap is unavailable; using a context-local fallback")
        return _context_fallback_session_id(context)
    else:
        try:
            value = str(get_session_id())
        except Exception:  # noqa: BLE001 - preserve the bootstrap failure for diagnostics
            _LOGGER.warning("LKB session-id bootstrap failed; using a context-local fallback")
            return _context_fallback_session_id(context)
    if value:
        return value
    # Last-resort process-local identity.  This is intentionally not a
    # workspace-wide constant, so independent processes never share a Plan
    # accidentally when a host forgot to provide a session id.
    return f"process-{os.getpid()}"


def _context_fallback_session_id(context: Any | None) -> str:
    """Return one isolated fallback id for a context with broken bootstrap."""
    if context is None:
        return f"process-{os.getpid()}"
    existing = getattr(context, _CONTEXT_FALLBACK_SESSION_ID_ATTR, None)
    if isinstance(existing, str) and existing:
        return existing
    fallback = f"context-{uuid.uuid4().hex}"
    try:
        setattr(context, _CONTEXT_FALLBACK_SESSION_ID_ATTR, fallback)
    except (AttributeError, TypeError):
        _LOGGER.warning("LKB session-id context cannot retain a fallback identity")
    return fallback


def default_plan_id(session_id: str) -> str:
    """Return the stable private default Plan id for one session."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    return f"plan-{digest}"


def new_plan_id() -> str:
    return f"plan-{uuid.uuid4().hex[:16]}"


def validate_plan_id(plan_id: str) -> str:
    normalized = str(plan_id or "").strip()
    NodeRef(normalized, "_plan", "_id")
    return normalized


def _metadata(graph: dict[str, Any]) -> dict[str, Any]:
    raw = graph.get("plan")
    return dict(raw) if isinstance(raw, dict) else {}


def _header(plan_id: str, graph: dict[str, Any]) -> PlanHeader:
    metadata = _metadata(graph)
    session_ids = metadata.get("session_ids", ())
    if not isinstance(session_ids, (list, tuple)):
        session_ids = ()
    return PlanHeader(
        plan_id=plan_id,
        title=str(metadata.get("title") or plan_id),
        state=str(metadata.get("state") or PLAN_ACTIVE),
        revision=int(graph.get("revision", 0)),
        created_at=str(graph.get("created_at", "")),
        updated_at=str(graph.get("updated_at", "")),
        created_by_session_id=str(metadata.get("created_by_session_id", "")),
        session_ids=tuple(str(item) for item in session_ids if str(item)),
    )


def _load_envelope(repository: Any, board_id: str) -> BoardEnvelope:
    return repository.load_envelope(board_id)


def list_plans(repository: Any, board_id: str) -> tuple[PlanHeader, ...]:
    envelope = _load_envelope(repository, board_id)
    plans = [_header(plan_id, graph) for plan_id, graph in envelope.graphs.items() if graph.get("graph_kind") == "plan"]
    return tuple(sorted(plans, key=lambda item: (item.created_at, item.plan_id)))


def bound_plan_id(repository: Any, board_id: str, session_id: str) -> str | None:
    envelope = _load_envelope(repository, board_id)
    bindings = envelope.board.get("session_plan_bindings", {})
    if not isinstance(bindings, dict):
        return None
    value = bindings.get(session_id)
    return str(value) if isinstance(value, str) and value else None


def _write(
    repository: Any,
    board_id: str,
    *,
    operation: str,
    actor: str,
    payload: dict[str, Any],
    mutate: Any,
) -> None:
    command_id = f"plan-scope-{operation}-{uuid.uuid4().hex[:16]}"
    repository.execute_atomic(
        board_id,
        command_id,
        canonical_hash({"operation": operation, **payload}),
        None,
        mutate,
        actor=actor,
        reason=f"Plan scope: {operation}",
    )


def create_plan(
    repository: Any,
    board_id: str,
    session_id: str,
    *,
    plan_id: str | None = None,
    title: str | None = None,
) -> PlanHeader:
    """Create and bind a fresh active Plan for ``session_id``."""
    selected = validate_plan_id(plan_id or new_plan_id())
    now = _now()
    display_title = str(title or "").strip() or "Untitled plan"

    def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        if selected in envelope.graphs:
            raise PlanScopeError(f"Plan {selected!r} already exists")
        envelope.graphs[selected] = {
            "graph_id": selected,
            "board_id": board_id,
            "graph_kind": "plan",
            "revision": 0,
            "created_at": now,
            "updated_at": now,
            "plan": {
                "plan_id": selected,
                "title": display_title,
                "state": PLAN_ACTIVE,
                "created_by_session_id": session_id,
                "last_bound_session_id": session_id,
                "session_ids": [session_id],
            },
        }
        bindings = envelope.board.setdefault("session_plan_bindings", {})
        if not isinstance(bindings, dict):
            bindings = {}
            envelope.board["session_plan_bindings"] = bindings
        bindings[session_id] = selected
        return envelope, CommandResult(decision="committed", command_id="")

    _write(
        repository,
        board_id,
        operation="create",
        actor=session_id,
        payload={"plan_id": selected, "session_id": session_id, "title": display_title},
        mutate=mutate,
    )
    return get_plan(repository, board_id, selected)


def get_plan(repository: Any, board_id: str, plan_id: str) -> PlanHeader:
    selected = validate_plan_id(plan_id)
    envelope = _load_envelope(repository, board_id)
    graph = envelope.graphs.get(selected)
    if graph is None or graph.get("graph_kind") != "plan":
        raise PlanScopeError(f"Plan {selected!r} does not exist in this Board")
    return _header(selected, graph)


def bind_plan(
    repository: Any,
    board_id: str,
    session_id: str,
    plan_id: str,
    *,
    resume_suspended: bool = False,
) -> PlanHeader:
    """Explicitly bind a session to an existing Plan."""
    selected = validate_plan_id(plan_id)

    def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        graph = envelope.graphs.get(selected)
        if graph is None or graph.get("graph_kind") != "plan":
            raise PlanScopeError(f"Plan {selected!r} does not exist in this Board")
        metadata = graph.setdefault("plan", {})
        if not isinstance(metadata, dict):
            metadata = {}
            graph["plan"] = metadata
        state = str(metadata.get("state") or PLAN_ACTIVE)
        if state in PLAN_TERMINAL_STATES:
            raise PlanScopeError(f"Plan {selected!r} is {state}; reopen it before binding")
        if state == PLAN_SUSPENDED and resume_suspended:
            metadata["state"] = PLAN_ACTIVE
        bindings = envelope.board.setdefault("session_plan_bindings", {})
        if not isinstance(bindings, dict):
            bindings = {}
            envelope.board["session_plan_bindings"] = bindings
        bindings[session_id] = selected
        session_ids = metadata.setdefault("session_ids", [])
        if not isinstance(session_ids, list):
            session_ids = list(session_ids) if isinstance(session_ids, tuple) else []
            metadata["session_ids"] = session_ids
        if session_id not in session_ids:
            session_ids.append(session_id)
        metadata["last_bound_session_id"] = session_id
        graph["updated_at"] = _now()
        return envelope, CommandResult(decision="committed", command_id="")

    current = bound_plan_id(repository, board_id, session_id)
    header = get_plan(repository, board_id, selected)
    if header.state in PLAN_TERMINAL_STATES:
        raise PlanScopeError(f"Plan {selected!r} is {header.state}; reopen it before binding")
    if current == selected and not (resume_suspended and header.state == PLAN_SUSPENDED):
        return header
    _write(
        repository,
        board_id,
        operation="bind",
        actor=session_id,
        payload={"plan_id": selected, "session_id": session_id},
        mutate=mutate,
    )
    return get_plan(repository, board_id, selected)


def resolve_plan(
    repository: Any,
    board_id: str,
    session_id: str,
    *,
    requested_plan_id: str | None = None,
) -> PlanHeader:
    """Resolve this session's Plan without consulting a workspace-global current Plan."""
    selected = requested_plan_id or bound_plan_id(repository, board_id, session_id)
    if selected is None:
        selected = default_plan_id(session_id)
    try:
        header = get_plan(repository, board_id, selected)
    except PlanScopeError:
        if requested_plan_id is not None:
            raise
        try:
            return create_plan(
                repository,
                board_id,
                session_id,
                plan_id=selected,
                title="Session plan",
            )
        except PlanScopeError:
            # Two contexts from the same session can resolve their default
            # Plan concurrently. Both may observe it missing, while only one
            # can create it. Converge the loser onto the now-existing Plan
            # instead of leaking a misleading "already exists" error through
            # /lkb status.
            header = get_plan(repository, board_id, selected)
            if bound_plan_id(repository, board_id, session_id) != selected:
                return bind_plan(repository, board_id, session_id, selected)
            return header
    if bound_plan_id(repository, board_id, session_id) != selected:
        return bind_plan(repository, board_id, session_id, selected)
    return header


def set_plan_state(
    repository: Any,
    board_id: str,
    session_id: str,
    plan_id: str,
    state: str,
) -> PlanHeader:
    """Transition a Plan and release its active claims when it stops."""
    selected = validate_plan_id(plan_id)
    target = str(state).strip().lower()
    if target not in PLAN_STATES:
        raise PlanScopeError(f"Unsupported Plan state {state!r}")

    def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        graph = envelope.graphs.get(selected)
        if graph is None or graph.get("graph_kind") != "plan":
            raise PlanScopeError(f"Plan {selected!r} does not exist in this Board")
        metadata = graph.setdefault("plan", {})
        if not isinstance(metadata, dict):
            metadata = {}
            graph["plan"] = metadata
        metadata["state"] = target
        metadata["last_bound_session_id"] = session_id
        metadata[f"{target}_at"] = _now()
        graph["updated_at"] = _now()
        if target != PLAN_ACTIVE:
            released_at = _now()
            for claim in envelope.claims.values():
                task_ref = str(claim.get("task_ref", ""))
                if claim.get("status") == "active" and task_ref.startswith(f"{selected}:"):
                    claim["status"] = "released"
                    claim["released_at"] = released_at
                    claim["reason"] = f"Plan transitioned to {target}"
            for node in envelope.nodes.values():
                if str(node.get("ref", "")).startswith(f"{selected}:task:"):
                    node["owner"] = None
                    node["updated_at"] = released_at
        bindings = envelope.board.get("session_plan_bindings")
        if not isinstance(bindings, dict):
            bindings = {}
            envelope.board["session_plan_bindings"] = bindings
        bindings[session_id] = selected
        return envelope, CommandResult(decision="committed", command_id="")

    _write(
        repository,
        board_id,
        operation=f"state-{target}",
        actor=session_id,
        payload={"plan_id": selected, "session_id": session_id, "state": target},
        mutate=mutate,
    )
    return get_plan(repository, board_id, selected)
