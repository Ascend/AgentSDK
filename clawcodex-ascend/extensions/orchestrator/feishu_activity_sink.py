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

# ruff: noqa: UP009

"""FeishuActivitySink — translate agent lifecycle into Feishu progress cards.

The orchestrator's :class:`ProgressSink` protocol gives us three hooks
(``on_phase_complete`` / ``on_turn_complete`` / ``on_session_complete``),
and :class:`StatusDashboard` exposes session-start state. Processing
reactions are owned centrally by the IM gateway; this sink only owns the
richer orchestrator progress card:

* On session start — send a placeholder progress card back to the
  same chat. The card's ``message_id`` is cached for subsequent updates.
* On every ``on_phase_complete`` — rebuild the placeholder card with the
  fresh phase progress and ``update_card`` it. Keeps the Feishu-side
  progress bar / header in lock-step with the agent runner.
* On ``on_session_complete`` — rewrite the card title and header template
  to the terminal colour (green / red / grey).

All entry points are synchronous (matches the :class:`ProgressSink`
contract) but each channel API call is ``async``. Calls are scheduled on
the current runner loop and every exception is swallowed. The sink *never*
propagates errors back into the agent runner; :class:`CompositeProgressSink`
already isolates us, but we also try/except inside this class so any direct
caller is safe.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ..api.query import PhaseComplete, SessionComplete, TurnComplete

if TYPE_CHECKING:
    from clawcodex_ext.services.channels.capabilities import CardUpdateCapability

    from .agent_runner import AgentSession
    from .status_dashboard import SessionStatus, StatusDashboard

logger = logging.getLogger(__name__)


# Terminal header colours used by the Feishu card template.
_HEADER_BLUE = "blue"
_HEADER_GREEN = "green"
_HEADER_RED = "red"
_HEADER_GREY = "grey"
_DEFAULT_MAX_PENDING_TASKS = 16
_DEFAULT_OPERATION_TIMEOUT_SECONDS = 10.0


class FeishuActivitySink:
    """Translate agent lifecycle into Feishu reactions + card updates.

    AgentRunner dispatches the :class:`ProgressSink` callbacks through
    :class:`CompositeProgressSink`. Session start is sourced from the
    :class:`StatusDashboard`. The dashboard may already contain the task by
    the time the per-session sink is created, so construction checks current
    state after registering the listener instead of relying only on a future
    callback.
    """

    task_id: str

    def __init__(
        self,
        *,
        task_id: str,
        feishu_adapter: CardUpdateCapability,
        status_dashboard: StatusDashboard | None = None,
        phases_total: int | None = None,
        max_pending_tasks: int = _DEFAULT_MAX_PENDING_TASKS,
        operation_timeout_seconds: float = _DEFAULT_OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        if isinstance(max_pending_tasks, bool) or not isinstance(max_pending_tasks, int):
            # Public configuration rejects invalid types and values uniformly.
            raise ValueError(  # noqa: TRY004
                "max_pending_tasks must be a positive integer"
            )
        if max_pending_tasks <= 0:
            raise ValueError("max_pending_tasks must be a positive integer")
        if (
            isinstance(operation_timeout_seconds, bool)
            or not isinstance(operation_timeout_seconds, (int, float))
            or not math.isfinite(float(operation_timeout_seconds))
            or operation_timeout_seconds <= 0
        ):
            raise ValueError("operation_timeout_seconds must be a positive finite number")
        self.task_id = task_id
        self._adapter = feishu_adapter
        self._max_pending_tasks = max_pending_tasks
        self._operation_timeout_seconds = float(operation_timeout_seconds)
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._remove_session_listener: Callable[[], None] | None = None
        self._session_started = False
        # ``workflow_phases`` is optional — when configured we can show
        # honest "phase 2/4" labels; otherwise we fall back to "phase N".
        self._phases_total = phases_total
        self._placeholder_message_id: str | None = None
        # Freeze the activity destination for this session. Reading the
        # adapter's mutable "last inbound" value on every callback can route
        # one session's card into a newer session's chat.
        try:
            self._inbound_context = feishu_adapter.last_inbound_context()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu activity context lookup failed error_type=%s",
                type(exc).__name__,
            )
            self._inbound_context = None
        # Subscribe to the dashboard's session-start so users see 👀 on
        # their inbound message the moment work begins.
        if status_dashboard is not None:
            self._remove_session_listener = status_dashboard.add_session_start_listener(self._on_session_status)
            # Orchestrator publishes session-start before it creates the
            # per-session progress sink. Replay the already-running task so
            # the placeholder is not lost merely because registration was late.
            current_status = status_dashboard.state().running.get(task_id)
            if current_status is not None:
                self._on_session_status(current_status)

    def _schedule(self, coro: Coroutine[Any, Any, Any], *, operation: str) -> bool:
        """Schedule a bounded operation on the active runner loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            logger.warning("feishu activity operation dropped without a running loop")
            return False
        if len(self._pending_tasks) >= self._max_pending_tasks:
            coro.close()
            logger.warning(
                "feishu activity operation dropped because task queue is full operation=%s",
                operation,
            )
            return False
        task = running_loop.create_task(self._run_with_deadline(coro, operation=operation))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return True

    async def _run_with_deadline(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        operation: str,
    ) -> None:
        try:
            await asyncio.wait_for(coro, timeout=self._operation_timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning("feishu activity operation timed out operation=%s", operation)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu activity operation failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )

    def _detach_listener(self) -> None:
        remove = self._remove_session_listener
        self._remove_session_listener = None
        if remove is not None:
            remove()

    def _on_session_status(self, status: SessionStatus) -> None:
        """Listener wired by ``StatusDashboard.add_session_start_listener``.

        Filters to ``task_id`` so multiple sessions in flight do not
        trigger each other's reactions.
        """
        if status.issue_id != self.task_id or self._session_started:
            return
        self._session_started = True
        # This listener is task-scoped and needed only once. Removing it at
        # the matching start event prevents completed sinks from accumulating
        # on the process-wide dashboard even if the runner exits abnormally.
        self._detach_listener()
        context = self._inbound_context
        if context is None:
            # No inbound context cached yet — usually because the agent
            # was launched out-of-band (e.g. via CLI). Soft skip.
            return
        card = _build_card(
            title="⏳ ClawCodex is processing",
            header_template=_HEADER_BLUE,
            progress=0,
            summary=(
                f"Task {status.issue_identifier or status.issue_id} received, phase 1/{self._phases_total or '?'}"
            ).strip(),
        )
        self._schedule(
            self._emit_session_start(context.chat_id, card),
            operation="session_start",
        )

    async def _emit_session_start(
        self,
        chat_id: str,
        card: dict,
    ) -> None:
        placeholder_id = await self._adapter.send_placeholder_card(chat_id, card)
        if placeholder_id:
            self._placeholder_message_id = placeholder_id

    # ------------------------------------------------------------------
    # ProgressSink (synchronous dispatch from AgentRunner)
    # ------------------------------------------------------------------

    def on_phase_complete(
        self,
        event: PhaseComplete,
        session: AgentSession,
    ) -> None:
        placeholder_id = self._placeholder_message_id
        if not placeholder_id:
            return
        phase_idx = event.phase or 0
        progress = _phase_progress(phase_idx, self._phases_total)
        card = _build_card(
            title=f"⏳ ClawCodex Phase {phase_idx}",
            header_template=_HEADER_BLUE,
            progress=progress,
            summary=_phase_summary(session, phase_idx, self._phases_total),
        )
        self._schedule(
            self._adapter.update_progress_card(placeholder_id, card),
            operation="phase_update",
        )

    def on_turn_complete(
        self,
        event: TurnComplete,
        session: AgentSession,
    ) -> None:
        # Turn events are too noisy for visible UI updates (mirrors
        # :class:`ToolContextProgressSink.on_turn_complete` which also
        # no-ops). Debug log only.
        logger.debug(
            "feishu activity sink turn %d complete for task %s",
            event.turn,
            self.task_id,
        )

    def on_session_complete(
        self,
        event: SessionComplete,
        session: AgentSession,
    ) -> None:
        self._detach_listener()
        placeholder_id = self._placeholder_message_id
        header, title, progress = _terminal_visuals(event.reason)
        # The card might never have been emitted (CLI-launched session);
        # ``update_progress_card`` returns False silently in that case.
        if placeholder_id:
            card = _build_card(
                title=title,
                header_template=header,
                progress=progress,
                summary=f"session: {event.reason}",
            )
            self._schedule(
                self._adapter.update_progress_card(placeholder_id, card),
                operation="session_complete",
            )
        # Drop cached placeholder so a re-run can claim a fresh card.
        self._placeholder_message_id = None
        self._inbound_context = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phase_progress(phase: int, phases_total: int | None) -> int:
    """Mirror :class:`ToolContextProgressSink._phase_progress` semantics."""
    if phase <= 0:
        return 0
    if phases_total and phases_total > 0:
        return max(0, min(100, int(phase / phases_total * 100)))
    # No configured total — fall back to the legacy 25/50/75/100 cadence
    # so the user still sees movement.
    return min(phase * 25, 100)


def _phase_summary(
    session: AgentSession,
    phase: int,
    phases_total: int | None,
) -> str:
    issue_id = getattr(getattr(session, "issue", None), "identifier", "") or ""
    total = phases_total if phases_total and phases_total > 0 else "?"
    issue_part = f"{issue_id} " if issue_id else ""
    return f"{issue_part}phase {phase}/{total}"


def _terminal_visuals(reason: str) -> tuple[str, str, int | None]:
    """Map a session outcome to card header, title and progress."""
    if reason == "success":
        return _HEADER_GREEN, "✅ Completed", 100
    if reason == "paused":
        return _HEADER_GREY, "⏸ Paused", None
    # Everything else (stagnation / loop_detected / max_turns_exceeded /
    # rate_limit_circuit_open / exit_code=N / noop_completed / failed /
    # budget_exhausted …) is a failure for visual purposes.
    return _HEADER_RED, f"❌ Failed: {reason}", None


def _build_card(
    *,
    title: str,
    header_template: str,
    progress: int | None,
    summary: str,
) -> dict:
    """Build a Feishu interactive card payload.

    Mirrors the existing ``feishu_cards.build_permission_card`` shape so
    the resulting JSON matches what the SDK schema expects.
    """
    elements: list[dict] = []
    if summary:
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "plain_text", "content": summary},
            }
        )
    if progress is not None:
        elements.append(
            {
                "tag": "progress",
                "percent": max(0, min(100, progress)),
            }
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": header_template,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": elements,
    }


__all__ = ["FeishuActivitySink"]
