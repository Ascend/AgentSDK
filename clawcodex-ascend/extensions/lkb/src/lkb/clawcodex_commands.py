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
# pylint: disable=C3001,E1123
# ruff: noqa: E731,E741

"""lkb — Logical Kanban slash command.

Usage::

    /lkb                         Interactive on/off toggle (picker menu)
    /lkb status                  List all tasks with LKB derived status
    /lkb explain <task_id>       Show blockers and invalidation state
    /lkb audit <task_id>         Show audit event log
    /lkb revalidate <task_id>    Revalidate one needs_recheck task

Bare ``/lkb`` opens an interactive picker (the ``/effort`` pattern) that
shows the current state and toggles the merged ``LKB_PLAN_GRAPH`` feature
flag on Enter, persisted to ``~/.clawcodex/features.json``. It is an
``InteractiveCommand`` so the same body drives the REPL numbered menu and
the TUI modal; subcommands are plain text paths with no UI dependency.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from clawcodex_ext.command_system.registry import register_command
from clawcodex_ext.command_system.types import (
    CommandAvailability,
    CommandContext,
    InteractiveCommand,
    InteractiveOutcome,
    UIOption,
)

# ── subcommand handlers ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LkbCommandOutcome:
    """Structured result shared by the interactive command and Agent tool.

    ``value`` is retained as a text-only compatibility accessor for existing
    callers while success classification is carried explicitly rather than
    inferred from user-facing prose.
    """

    text: str
    success: bool
    error_code: str | None = None

    @property
    def value(self) -> str:
        """Compatibility accessor for legacy command-rendering call sites."""
        return self.text


def _ok(text: str) -> LkbCommandOutcome:
    return LkbCommandOutcome(text=text, success=True)


def _failed(text: str, error_code: str) -> LkbCommandOutcome:
    return LkbCommandOutcome(text=text, success=False, error_code=error_code)


def _lkb_call(args: str, ctx: object) -> LkbCommandOutcome:
    """Top-level dispatch for ``/lkb <subcommand> [args]``."""
    parts = args.strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else "status"
    sub_args = parts[1] if len(parts) > 1 else ""

    tool_ctx = getattr(ctx, "tool_context", None)
    if tool_ctx is None:
        return _failed(
            "The /lkb command requires a tool_context (only available in TUI/REPL sessions).",
            "missing_tool_context",
        )

    # Guard: LKB feature flag
    try:
        from lkb.flags import is_plan_graph_enabled

        if not is_plan_graph_enabled():
            return _failed(
                "LKB (Logical Kanban) is not currently enabled. Run bare `/lkb` "
                "to toggle it on, or enable the `LKB_PLAN_GRAPH` feature flag.",
                "feature_disabled",
            )
    except ModuleNotFoundError:
        return _failed("LKB module unavailable.", "module_unavailable")
    except Exception:  # noqa: BLE001 - command must not crash the REPL
        return _failed("LKB feature check failed.", "feature_check_failed")

    switch = {
        "status": _lkb_status,
        "board": _lkb_board,
        "explain": _lkb_explain,
        "audit": _lkb_audit,
        "revalidate": _lkb_revalidate,
        "plan": _lkb_plan,
    }
    handler = switch.get(sub)
    if handler is None:
        return _failed(_usage(), "usage")
    try:
        return handler(tool_ctx, sub_args)
    except Exception:  # noqa: BLE001 - command must not crash the REPL
        return _failed(f"/lkb {sub} failed. See local logs for details.", f"{sub}_failed")


def _usage() -> str:
    return (
        "LKB usage:\n"
        "  /lkb                      Toggle LKB on/off (interactive menu)\n"
        "  /lkb status               List all tasks with LKB derived status\n"
        "  /lkb board [--compact]    Show the unified ASCII LKB board (Plan Graph)\n"
        "  /lkb board compact        Compact old Audit records into history\n"
        "  /lkb explain <task_id>    Show blockers and invalidation state\n"
        "  /lkb audit <task_id>      Show audit event log\n"
        "  /lkb revalidate <task_id> Revalidate one needs_recheck task\n"
        "  /lkb plan current         Show this session's current Plan\n"
        "  /lkb plan list            List Plans in the workspace Board\n"
        "  /lkb plan new [title]     Create and bind a new Plan\n"
        "  /lkb plan use <plan_id>   Explicitly bind an existing Plan\n"
        "  /lkb plan suspend         Suspend the current Plan\n"
        "  /lkb plan complete        Complete the current Plan\n"
        "  /lkb plan abandon         Abandon the current Plan\n"
        "  /lkb plan archive         Archive the current Plan\n"
        "  /lkb plan reopen <id>     Reopen and bind a stopped Plan\n"
    )


def _scope(tool_ctx):
    """Return ``(repository, board_id, session_id, PlanHeader)``."""
    from lkb.plan_scope import current_session_id, resolve_plan
    from lkb.repository import get_repository

    repo = get_repository()
    session_id = current_session_id(tool_ctx)
    board = repo.resolve_board(
        getattr(tool_ctx, "workspace_root", None),
        session_id=session_id,
    )
    requested = getattr(tool_ctx, "lkb_plan_id", None)
    header = resolve_plan(
        repo,
        board.board_id,
        session_id,
        requested_plan_id=requested if isinstance(requested, str) and requested else None,
    )
    try:
        tool_ctx.lkb_plan_id = header.plan_id
    # Some host context implementations intentionally reject optional attributes.
    except Exception:  # nosec B110
        pass
    return repo, board.board_id, session_id, header


def _format_plan(header, *, current: bool = False) -> str:
    marker = "*" if current else "-"
    return f"{marker} {header.plan_id}  [{header.state}]  {header.title}  (rev {header.revision})"


def _lkb_plan(tool_ctx, args: str) -> LkbCommandOutcome:
    """Manage the session-to-Plan binding inside the workspace Board."""
    from lkb.plan_scope import (
        PLAN_ABANDONED,
        PLAN_ARCHIVED,
        PLAN_COMPLETED,
        PLAN_SUSPENDED,
        bind_plan,
        create_plan,
        get_plan,
        list_plans,
        set_plan_state,
    )

    parts = args.strip().split(maxsplit=1)
    action = parts[0].lower() if parts else "current"
    value = parts[1].strip() if len(parts) > 1 else ""
    repo, board_id, session_id, current = _scope(tool_ctx)

    if action == "current":
        return _ok(
            f"Board: {board_id}\n"
            f"Current Plan: {_format_plan(current, current=True)[2:]}\n"
            "Binding scope: this session (child agents inherit it)"
        )
    if action == "list":
        plans = list_plans(repo, board_id)
        lines = [f"Board: {board_id}", "Plans:"]
        lines.extend(_format_plan(plan, current=plan.plan_id == current.plan_id) for plan in plans)
        return _ok("\n".join(lines))
    if action == "new":
        created = create_plan(repo, board_id, session_id, title=value or None)
        tool_ctx.lkb_plan_id = created.plan_id
        return _ok(f"Created and bound Plan: {_format_plan(created, current=True)[2:]}")
    if action == "use":
        if not value:
            return _failed("Usage: /lkb plan use <plan_id>", "usage")
        selected = bind_plan(
            repo,
            board_id,
            session_id,
            value,
            resume_suspended=True,
        )
        tool_ctx.lkb_plan_id = selected.plan_id
        return _ok(f"Bound current session to Plan: {_format_plan(selected, current=True)[2:]}")
    if action in ("suspend", "complete", "abandon", "archive"):
        target = {
            "suspend": PLAN_SUSPENDED,
            "complete": PLAN_COMPLETED,
            "abandon": PLAN_ABANDONED,
            "archive": PLAN_ARCHIVED,
        }[action]
        updated = set_plan_state(
            repo,
            board_id,
            session_id,
            current.plan_id,
            target,
        )
        return _ok(f"Plan updated: {_format_plan(updated, current=True)[2:]}")
    if action == "reopen":
        selected_id = value or current.plan_id
        get_plan(repo, board_id, selected_id)
        updated = set_plan_state(
            repo,
            board_id,
            session_id,
            selected_id,
            "active",
        )
        updated = bind_plan(repo, board_id, session_id, updated.plan_id)
        tool_ctx.lkb_plan_id = updated.plan_id
        return _ok(f"Reopened and bound Plan: {_format_plan(updated, current=True)[2:]}")
    return _failed(_usage(), "usage")


def _lkb_board(tool_ctx, args: str) -> LkbCommandOutcome:
    """``/lkb board`` - render the unified ASCII LKB board (spec §8.4).

    Requires the ``LKB_PLAN_GRAPH`` feature flag so the Graph Store is
    the authority.
    """
    try:
        from lkb.flags import is_plan_graph_enabled
    except Exception:
        is_plan_graph_enabled = lambda: False  # type: ignore[assignment]

    if not is_plan_graph_enabled():
        return _failed(
            "/lkb board requires the LKB_PLAN_GRAPH feature flag (Graph Store authority).",
            "feature_disabled",
        )

    if args.strip().lower() == "compact":
        try:
            repo, board_id, _session_id, _plan = _scope(tool_ctx)
            result = repo.compact_audit(board_id, actor=_command_actor(tool_ctx))
        except Exception:  # noqa: BLE001 - command must not crash the REPL
            return _failed("LKB Audit compaction failed.", "audit_compaction_failed")
        if result.segment_id is None:
            return _ok("LKB Audit is already compact; no history segment was created.")
        return _ok(
            f"Compacted {result.event_count} events, "
            f"{result.processed_command_count} command summaries, and "
            f"{result.validation_run_count} validation runs into {result.file}. "
            f"Active Audit size: {result.bytes_before} -> {result.bytes_after} bytes."
        )

    from lkb.ascii_board import render_board
    from lkb.read_model import build_board_view

    # Resolve the board for the current workspace; if none exists yet,
    # report an empty board rather than crashing.
    try:
        repo, board_id, _session_id, plan = _scope(tool_ctx)
        envelope = repo.load_envelope(board_id)
    except Exception:  # noqa: BLE001 - command must not crash the REPL
        return _failed("No LKB board found for this workspace.", "board_not_found")

    view = build_board_view(envelope, plan.plan_id)
    compact = "--compact" in args
    return _ok(render_board(view, width=100, compact=compact))


def _load_envelope(tool_ctx):
    """Resolve and load the Graph Store envelope for the current workspace.

    Returns ``(envelope, board_id, plan_id)``. Used by
    ``/lkb status|explain|audit`` so every official query entrance reads
    the same Graph Store authority as ``/lkb board`` (issue #6) instead
    of the stale Context Sidecar.
    """
    repo, board_id, _session_id, plan = _scope(tool_ctx)
    envelope = repo.load_envelope(board_id)
    return envelope, board_id, plan.plan_id


def _command_actor(tool_ctx) -> str:
    """Return the host-established actor for a mutating slash command."""
    for attr in ("agent_id", "actor", "session_id"):
        value = getattr(tool_ctx, attr, None)
        if isinstance(value, str) and value:
            return value
    return "operator"


def _execute_plan_task_command(
    tool_ctx,
    kind: str,
    task_id: str,
    *,
    payload: dict | None = None,
):
    """Execute one public ``/lkb`` mutation through the application service."""
    from lkb.application import LkbApplicationService
    from lkb.commands import GraphCommand
    from lkb.plan_graph import plan_command_dispatcher
    from lkb.refs import plan_task_ref

    repo, board_id, _session_id, plan = _scope(tool_ctx)
    routed_payload = {
        **dict(payload or {}),
        "task_id": task_id,
        "plan_id": plan.plan_id,
    }
    command = GraphCommand(
        command_id=f"lkb-{kind}-{uuid.uuid4().hex[:12]}",
        board_id=board_id,
        actor=_command_actor(tool_ctx),
        kind=kind,
        primary_subject_ref=plan_task_ref(task_id, graph_id=plan.plan_id),
        payload=routed_payload,
    )
    dispatcher = plan_command_dispatcher()
    return LkbApplicationService(repository=repo).execute(
        command,
        validate=dispatcher.validate,
        apply=dispatcher.apply,
    )


def _lkb_revalidate(tool_ctx, args: str) -> LkbCommandOutcome:
    """``/lkb revalidate`` — revalidate one task through the public command."""
    parts = args.strip().split()
    if len(parts) != 1:
        return _failed("Usage: /lkb revalidate <task_id>", "usage")
    task_id = parts[0]
    result = _execute_plan_task_command(tool_ctx, "revalidate", task_id)
    if result.decision != "committed":
        return _failed(
            f"Revalidate denied for {task_id}: {result.reason or 'unknown reason'}",
            str(result.error_code or "revalidate_denied"),
        )
    return _ok(f"Revalidated {task_id}")


def _explain_from_store(tool_ctx, task_id: str) -> LkbCommandOutcome:
    """``/lkb explain`` backed by the Graph Store (issue #6)."""
    from lkb.graph_types import NodeRef, PlanSnapshot

    envelope, _board_id, plan_id = _load_envelope(tool_ctx)
    if envelope is None:
        return _failed(f"No LKB board found for task {task_id!r}.", "board_not_found")
    ref = NodeRef(plan_id, "task", task_id)
    snap = envelope.build_graph_snapshot()
    plan = PlanSnapshot.from_graph(snap)
    node_dict = None
    for n in envelope.nodes.values():
        if str(n.get("ref", "")) == ref.to_str():
            node_dict = n
            break
    if node_dict is None:
        return _failed(f"Task {task_id!r} not found in the Board.", "task_not_found")
    payload = node_dict.get("payload") if isinstance(node_dict.get("payload"), dict) else {}
    derived = str(payload.get("derived_status", "") or "")
    state = str(node_dict.get("state", "pending"))
    lines = [f"Task: {task_id}", f"Subject: {node_dict.get('title', '')}"]
    lines.append(f"Base status: {state}")
    lines.append(f"Owner: {node_dict.get('owner') or '-'}")
    if derived:
        lines.append(f"Derived status: {derived}")
    blockers = plan.active_blockers.get(ref, ())
    if blockers:
        lines.append(f"Active blockers: {', '.join(b.id for b in blockers)}")
    cause = payload.get("invalidation_cause")
    reason = payload.get("invalidation_reason")
    if cause or reason:
        lines.append(f"Invalidation cause: {cause} ({reason})")
    last_run = None
    for vr in envelope.validation_runs.values():
        if not isinstance(vr, dict):
            continue
        subj = vr.get("subjectRef") or vr.get("subject_ref")
        if isinstance(subj, dict):
            subj = f"{subj.get('graph')}:{subj.get('kind')}:{subj.get('id')}"
        if subj == ref.to_str():
            last_run = vr
    if last_run:
        lines.append(
            f"Latest validation: {last_run.get('result', '?')} "
            f"({last_run.get('validationRunId', last_run.get('validation_run_id', '?'))})"
        )
    return _ok(_panel("\n".join(lines), f"LKB Explain: {task_id}"))


def _audit_from_store(tool_ctx, task_id: str) -> LkbCommandOutcome:
    """``/lkb audit`` backed by the Graph Store event log (issue #6)."""
    from lkb.graph_types import NodeRef

    envelope, _board_id, plan_id = _load_envelope(tool_ctx)
    if envelope is None:
        return _failed(f"No LKB board found for task {task_id!r}.", "board_not_found")
    ref = NodeRef(plan_id, "task", task_id).to_str()
    events = [
        ev
        for ev in envelope.events
        if isinstance(ev, dict) and (ev.get("subject_ref") == ref or ref in (ev.get("affected_refs") or []))
    ]
    if not events:
        return _ok(f"Task {task_id} has no LKB audit events.")
    lines: list[str] = []
    for ev in events[-20:]:
        ts = (ev.get("timestamp") or "")[11:19]
        etype = ev.get("type", "?")
        actor = ev.get("actor", "?")
        rev = ev.get("store_revision", "?")
        lines.append(f"  [{ts}] {etype}  (actor={actor}, rev={rev})")
        if ev.get("reason"):
            lines.append(f"         reason: {ev['reason']}")
        if ev.get("affected_refs"):
            lines.append(f"         affected: {', '.join(ev['affected_refs'])}")
    return _ok(_panel("\n".join(lines), f"LKB Audit: {task_id}"))


def _panel(body: str, title: str) -> str:
    """Wrap *body* in an ASCII-panel with *title*."""
    lines = body.split("\n")
    width = max(len(l) for l in lines) if lines else 40
    width = min(max(width + 4, 20), 100)
    top = f"╔══ {title} " + "═" * (width - len(title) - 6) + "╗"
    bot = "╚" + "═" * (width - 1) + "╝"
    padded = [f"║  {l:<{width - 4}}  ║" for l in lines]
    return "\n".join([top] + padded + [bot])


def _lkb_status(tool_ctx, _args: str) -> LkbCommandOutcome:
    """``/lkb status`` — list all tasks with LKB derived status."""
    return _lkb_board(tool_ctx, "--compact")


def _lkb_explain(tool_ctx, task_id: str) -> LkbCommandOutcome:
    """``/lkb explain <task_id>`` — show blockers and invalidation state."""
    task_id = task_id.strip()
    if not task_id:
        return _failed("Usage: /lkb explain <task_id>", "usage")

    return _explain_from_store(tool_ctx, task_id)


def _lkb_audit(tool_ctx, task_id: str) -> LkbCommandOutcome:
    """``/lkb audit <task_id>`` — show audit events."""
    task_id = task_id.strip()
    if not task_id:
        return _failed("Usage: /lkb audit <task_id>", "usage")

    return _audit_from_store(tool_ctx, task_id)


# ── interactive toggle (bare /lkb) ────────────────────────────────────────

_LKB_PICKER_TITLE = "Logical Kanban (LKB) — persistent Plan Graph for Task V2:"

_LKB_OPTION_DESCRIPTIONS = {
    "on": "Task V2 tools are backed by the persistent Plan Graph (board.json)",
    "off": "Task V2 tools use the in-memory session store",
}


def _lkb_is_on() -> bool:
    try:
        from lkb.flags import is_plan_graph_enabled

        return is_plan_graph_enabled()
    except Exception:
        return False


def _set_lkb_enabled(enabled: bool) -> None:
    """Persist the merged LKB flag via the feature-gate override channel."""
    from clawcodex_ext.feature_gate import get_registry, register_defaults

    register_defaults()
    reg = get_registry()
    reg.set_override("LKB_PLAN_GRAPH", enabled)
    reg.save_config()


def _toggle_options(current: str) -> list[UIOption]:
    options: list[UIOption] = []
    for value in ("on", "off"):
        desc = _LKB_OPTION_DESCRIPTIONS[value]
        if value == current:
            desc = f"current — {desc}"
        options.append(UIOption(value=value, label=value, description=desc))
    return options


# ── command definition ──────────────────────────────────────────────────


@dataclass(frozen=True)
class LkbCommand(InteractiveCommand):
    """Toggle LKB and inspect its state (the ``/effort`` picker pattern).

    Bare ``/lkb`` opens the on/off picker; subcommands reuse the plain-text
    ``_lkb_call`` dispatcher (no UI dependency).
    """

    async def run(self, args: str, context: CommandContext) -> InteractiveOutcome:
        raw = (args or "").strip()
        if raw:
            result = _lkb_call(raw, context)
            return InteractiveOutcome(message=result.value, display="user")

        current = "on" if _lkb_is_on() else "off"
        picked = await context.ui.select(_LKB_PICKER_TITLE, _toggle_options(current), current=current)
        if picked is None:
            return InteractiveOutcome(message="Cancelled", display="user")
        _set_lkb_enabled(picked == "on")
        if picked == "on":
            message = (
                "LKB enabled (LKB_PLAN_GRAPH persisted): Task V2 tools are now backed by the persistent Plan Graph."
            )
        else:
            message = "LKB disabled (LKB_PLAN_GRAPH persisted): Task V2 tools use the in-memory session store."
        return InteractiveOutcome(message=message, display="user")


LKB_COMMAND: InteractiveCommand = LkbCommand(
    name="lkb",
    description=(
        "Toggle Logical Kanban (LKB) — persistent Plan Graph authority for Task V2; also board/status/audit diagnostics"
    ),
    aliases=["logical-kanban"],
    availability=[CommandAvailability.CONSOLE],
    argument_hint=("[status|board [--compact]|explain <task_id>|audit <task_id>|revalidate <task_id>|plan ...]"),
)

register_command(LKB_COMMAND)

__all__ = ["LKB_COMMAND", "LkbCommand", "LkbCommandOutcome"]
