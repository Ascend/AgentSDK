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

"""Status dashboard for orchestrator.

Port of Symphony's StatusDashboard — renders a terminal UI with running
sessions, throughput, TPS sparkline, and retry queue. Phase 4 of the
INTEGRATION.md roadmap.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SPARKLINE_BLOCKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
_RUNNING_EVENT_DEFAULT_WIDTH = 44
_RUNNING_EVENT_MIN_WIDTH = 12
_RUNNING_ROW_CHROME_WIDTH = 10


@dataclass
class SessionStatus:
    """Status of one agent session."""

    issue_id: str
    issue_identifier: str
    status: str = "running"  # running, completed, failed, retrying
    turn_count: int = 0
    max_turns: int = 0
    workspace_path: str = ""
    worker_host: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds_running: int = 0
    last_event: str = ""

    def age_display(self) -> str:
        secs = self.seconds_running
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        return f"{secs // 3600}h {(secs % 3600) // 60}m"

    def tokens_display(self) -> str:
        total = self.total_tokens
        if total < 1000:
            return str(total)
        if total < 1_000_000:
            return f"{total // 1000}k"
        return f"{total // 1_000_000}M"

    def event_truncated(self, width: int | None = None) -> str:
        w = width or _RUNNING_EVENT_DEFAULT_WIDTH
        w = max(w, _RUNNING_EVENT_MIN_WIDTH)
        ev = self.last_event or ""
        if len(ev) <= w:
            return ev
        return ev[: w - 3] + "..."


@dataclass
class DashboardState:
    """Full dashboard state."""

    running: dict[str, SessionStatus] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    retry_queue: list[dict[str, Any]] = field(default_factory=list)
    poll_check_in_progress: bool = False
    next_poll_due_at_ms: int | None = None
    # Clarification status data
    clarifications: list["ClarificationEntry"] = field(default_factory=list)


@dataclass
class ClarificationEntry:
    """A single issue's clarification status for the dashboard panel."""

    issue_id: str
    status: str  # "awaiting_author" | "awaiting_local" | "manual_required" | "resolved"
    open_questions: list[str] = field(default_factory=list)
    round_num: int = 0
    max_rounds: int = 2
    elapsed_seconds: float = 0.0
    author_login: str | None = None


class StatusDashboard:
    """Terminal status dashboard for autonomous orchestrator.

    Renders a live table of running sessions with age, tokens, and last
    event. Shows throughput sparkline, completed count, and retry queue.
    """

    def __init__(
        self,
        refresh_ms: int = 1_000,
        render_interval_ms: int = 16,
        enabled: bool = True,
    ) -> None:
        self._state = DashboardState()
        self._enabled = enabled
        self._refresh_ms = refresh_ms
        self._render_interval_ms = render_interval_ms
        self._token_samples: deque[int] = deque(maxlen=24)
        self._last_tps_value: float = 0.0
        self._last_rendered_at: float = 0.0
        self._pending_lines: list[str] = []
        self._last_snapshot_fingerprint: str = ""
        # Session-start listeners: external sinks (e.g. the Feishu
        # activity sink) can register a callback here without inheriting
        # from or monkey-patching the dashboard. Listeners receive the
        # :class:`SessionStatus` already populated by :meth:`on_session_start`.
        # Failures in any listener are logged + skipped so they never
        # break the dashboard update path.
        self._session_start_listeners: list[Callable[[SessionStatus], None]] = []

    def add_session_start_listener(
        self,
        listener: Callable[[SessionStatus], None],
    ) -> Callable[[], None]:
        """Register ``listener`` to fire after every ``on_session_start``.

        Returns a remove-callable; callers should invoke it when the sink
        is disposed so defunct listeners do not accumulate.
        """
        self._session_start_listeners.append(listener)

        def _remove() -> None:
            try:
                self._session_start_listeners.remove(listener)
            except ValueError:
                pass

        return _remove

    def on_session_start(self, session_status: SessionStatus) -> None:
        self._state.running[session_status.issue_id] = session_status
        if self._enabled:
            logger.info(
                "[dashboard] session start: %s (%s)",
                session_status.issue_identifier,
                session_status.issue_id,
            )
        for listener in list(self._session_start_listeners):
            try:
                listener(session_status)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[dashboard] session_start listener failed: %s", exc)

    def on_session_complete(self, issue_id: str) -> None:
        if issue_id in self._state.running:
            del self._state.running[issue_id]
        self._state.completed.add(issue_id)
        if self._enabled:
            logger.info("[dashboard] session complete: %s", issue_id)

    def on_session_failed(self, issue_id: str, error: str) -> None:
        if issue_id in self._state.running:
            del self._state.running[issue_id]
        self._state.failed.add(issue_id)
        if self._enabled:
            logger.info("[dashboard] session failed: %s error=%s", issue_id, error)

    def on_event(self, event: Any, session: Any) -> None:
        """Handle a streaming event from the agent."""
        if not self._enabled:
            return

        issue_id = getattr(session.issue, "id", None) if session else None
        if issue_id and issue_id in self._state.running:
            status = self._state.running[issue_id]
            # Update last event display
            from ..api.query import (  # pylint: disable=import-error,no-name-in-module
                TextDelta,
                ToolCallEvent,
                ToolResultEvent,
            )

            if isinstance(event, TextDelta):
                status.last_event = f"text: {event.content[-50:]}"
            elif isinstance(event, ToolCallEvent):
                status.last_event = f"tool: {event.tool_name}"
            elif isinstance(event, ToolResultEvent):
                status.last_event = f"result: {event.tool_name}"
                if event.result.get("is_error"):
                    status.last_event += " [ERR]"
            elif hasattr(event, "type"):
                status.last_event = f"event: {event.type}"

    def on_poll_start(self) -> None:
        self._state.poll_check_in_progress = True
        logger.debug("[dashboard] poll started")

    def on_poll_end(self) -> None:
        self._state.poll_check_in_progress = False
        logger.debug("[dashboard] poll ended")

    def on_retry_queue_update(self, retry_items: list[dict[str, Any]]) -> None:
        self._state.retry_queue = retry_items

    def on_clarification_update(self, entries: list["ClarificationEntry"]) -> None:
        """Receive clarification status updates and refresh the dashboard panel."""
        self._state.clarifications = list(entries)

    @property
    def pending_clarifications(self) -> list["ClarificationEntry"]:
        """Clarifications currently requiring operator input (awaiting_local or manual_required)."""
        return [e for e in self._state.clarifications if e.status in ("awaiting_local", "manual_required")]

    def state(self) -> DashboardState:
        return self._state

    def tokens(self) -> dict[str, int]:
        """Return aggregate token counts across all sessions."""
        running = self._state.running
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for sess in running.values():
            totals["input_tokens"] += sess.input_tokens
            totals["output_tokens"] += sess.output_tokens
            totals["total_tokens"] += sess.total_tokens
        return totals

    def tps(self, window_ms: int = 5_000) -> float:
        """Tokens per second averaged over the given window."""
        tokens = sum(s.total_tokens for s in self._state.running.values())
        self._token_samples.append(tokens)
        if len(self._token_samples) < 2:
            return 0.0
        elapsed = window_ms / 1000.0
        current = self._token_samples[-1]
        oldest = self._token_samples[0]
        delta = current - oldest
        return delta / elapsed if elapsed > 0 else 0.0

    def throughput_sparkline(self) -> str:
        """Build throughput sparkline from recent TPS samples."""
        samples = list(self._token_samples)
        if len(samples) < 2:
            return ""

        min_s = min(samples)
        max_s = max(samples)
        range_s = max_s - min_s
        blocks = _SPARKLINE_BLOCKS
        n = len(blocks)

        line = []
        for s in samples[-24:]:
            if range_s == 0:
                idx = n // 2
            else:
                idx = int((s - min_s) / range_s * (n - 1))
            idx = max(0, min(n - 1, idx))
            line.append(blocks[idx])
        return "".join(line)

    def _running_table_row(self, status: SessionStatus) -> str:
        """Render one row of the running sessions table."""
        id_str = status.issue_identifier[-_RUNNING_EVENT_DEFAULT_WIDTH:]
        age = status.age_display()
        tokens = status.tokens_display()
        last = status.event_truncated(40)

        return f"  {id_str:<20} {status.status:<8} {age:<10} {tokens:>8} {last}"

    def render(self) -> str:
        """Render the full dashboard as a multi-line string."""
        lines: list[str] = []
        now = time.time()

        # Header
        lines.append("─" * 80)
        poll_str = "checking..." if self._state.poll_check_in_progress else "idle"
        running_count = len(self._state.running)
        completed_count = len(self._state.completed)
        failed_count = len(self._state.failed)
        retry_count = len(self._state.retry_queue)
        tps_spark = self.throughput_sparkline()

        header = (
            f"  [Symphony] running={running_count} "
            f"completed={completed_count} "
            f"failed={failed_count} "
            f"retry={retry_count} "
            f"poll={poll_str}"
        )
        if tps_spark:
            header += f"  tps={tps_spark}"
        lines.append(header)
        lines.append("─" * 80)

        # Running sessions
        if self._state.running:
            lines.append("  RUNNING SESSIONS")
            lines.append(f"  {'identifier':<20} {'status':<8} {'age':<10} {'tokens':>8}  last_event")
            lines.append("  " + "-" * 72)
            for sess in self._state.running.values():
                lines.append(self._running_table_row(sess))
        else:
            lines.append("  [no running sessions]")

        # Failed sessions
        if self._state.failed:
            lines.append("")
            lines.append("  FAILED SESSIONS")
            lines.append(f"  {len(self._state.failed)} session(s) failed")

        # Retry queue
        if self._state.retry_queue:
            lines.append("")
            lines.append("  RETRY QUEUE")
            for retry in self._state.retry_queue[:10]:
                lines.append(
                    f"  {retry.get('identifier', '?') or retry.get('issue_id', '?')}"
                    f"  attempt={retry.get('attempt', '?')}"
                    f"  delay={retry.get('delay_seconds', 0):.1f}s"
                    f"  error={retry.get('error', '')[:40]}"
                )

        # Clarification panel
        clarification_panel = self._clarification_panel()
        if clarification_panel:
            lines.append("")
            lines.append(clarification_panel)

        self._last_rendered_at = now
        return "\n".join(lines)

    def render_line(self) -> str:
        """Render a single-line summary for logging/metrics."""
        running = len(self._state.running)
        completed = len(self._state.completed)
        retry = len(self._state.retry_queue)
        poll = "active" if self._state.poll_check_in_progress else "idle"
        spark = self.throughput_sparkline()
        parts = [
            f"running={running}",
            f"completed={completed}",
            f"retry_queue={retry}",
            f"poll={poll}",
        ]
        if spark:
            parts.append(f"tps={spark}")
        return f"[dashboard] {' '.join(parts)}"

    # ------------------------------------------------------------------
    # Clarification prompt (Channel 1 — interactive operator input)
    # ------------------------------------------------------------------

    def prompt_clarification(
        self,
        issue_id: str,
        issue_identifier: str,
        question: str,
        options: list[str] | None = None,
        context_summary: str = "",
    ) -> str | None:
        """Display an interactive clarification prompt.

        This is Channel 1 of the three-channel clarification flow.
        Used when the agent encounters ambiguous issue semantics and
        needs immediate operator input.

        Args:
            issue_id: issue identifier
            issue_identifier: human-readable identifier
            question: the clarification question
            options: optional multiple-choice options (1-4)
            context_summary: brief context for the operator

        Returns:
            The operator's answer string, or None if timed out / declined.

        Note: This requires a terminal that supports interactive input.
        In headless mode, this should be called only when the terminal
        supports it, and should return None to trigger Channel 2 fallback.
        """
        import sys

        print(file=sys.stderr)
        print("┌─ 🔵 Clarification Needed ───────────────────────────────┐", file=sys.stderr)
        print(f"│  Issue {issue_id} ({issue_identifier})", file=sys.stderr)
        print("│", file=sys.stderr)
        if context_summary:
            summary_line = context_summary[:50] + ("..." if len(context_summary) > 50 else "")
            print(f"│  Context: {summary_line}", file=sys.stderr)
            print("│", file=sys.stderr)
        print(f"│  Question: {question}", file=sys.stderr)

        if options and len(options) <= 4:
            print("│", file=sys.stderr)
            print("│  Options:", file=sys.stderr)
            for i, opt in enumerate(options, 1):
                print(f"│    [{i}] {opt}", file=sys.stderr)
            print("│", file=sys.stderr)
            print("│  Enter number or text answer, or press Enter to skip", file=sys.stderr)
        else:
            print("│", file=sys.stderr)
            print("│  Enter your answer, or press Enter to skip (forward to author)", file=sys.stderr)

        print("└───────────────────────────────────────────────────────┘", file=sys.stderr)
        print("> ", end="", file=sys.stderr)
        sys.stderr.flush()

        try:
            import select

            # Wait for input with timeout
            if select.select([sys.stdin], [], [], 60)[0]:
                answer = sys.stdin.readline()
                answer = answer.strip()
                if not answer:
                    return None
                # If numeric option, return the text
                if options:
                    try:
                        idx = int(answer)
                        if 1 <= idx <= len(options):
                            return options[idx - 1]
                    except ValueError:
                        pass
                return answer
        except Exception:  # nosec B110
            pass

        return None

    def render_clarification_status(self, issue_id: str, status: str, timeout_seconds: int | None = None) -> str:
        """Render a clarification status indicator for the dashboard."""
        status_icons = {
            "pending": "⏳",
            "awaiting_local": "👤",
            "awaiting_author": "📧",
            "resolved_local": "✅",
            "resolved_author": "✅",
            "timed_out_local": "⏰",
            "timed_out_author": "⏰",
            "exhausted": "❌",
        }
        icon = status_icons.get(status, "?")
        timeout_str = f" (timeout in {timeout_seconds}s)" if timeout_seconds else ""
        return f"  {icon} Issue {issue_id}: clarification {status}{timeout_str}"

    def _clarification_panel(self) -> str:
        """Render the clarification status panel."""
        entries = self._state.clarifications
        if not entries:
            return ""

        awaiting = [e for e in entries if e.status in ("awaiting_author", "awaiting_local")]
        manual = [e for e in entries if e.status == "manual_required"]
        resolved = [e for e in entries if e.status == "resolved"]

        lines: list[str] = ["── Clarification ──────────────────────"]
        if awaiting:
            lines.append(f"  ⏳ Awaiting ({len(awaiting)}):")
            for e in sorted(awaiting, key=lambda x: x.elapsed_seconds, reverse=True):
                icon = "📧" if e.status == "awaiting_author" else "👤"
                q_count = len(e.open_questions)
                lines.append(
                    f"    {icon} #{e.issue_id} Round {e.round_num}/{e.max_rounds} "
                    f"({q_count} Q, {e.elapsed_seconds:.0f}s)"
                )
                if e.open_questions:
                    first_q = e.open_questions[0][:60]
                    lines.append(f"       Q: {first_q}...")
        if manual:
            lines.append(f"  ❌ Manual required ({len(manual)}):")
            for e in sorted(manual, key=lambda x: x.elapsed_seconds, reverse=True)[:5]:
                lines.append(f"    ⚠ #{e.issue_id} (Round {e.round_num}/{e.max_rounds} exhausted)")
        if resolved:
            recent = sorted(resolved, key=lambda x: x.elapsed_seconds, reverse=True)[:3]
            for e in recent:
                lines.append(f"    ✅ #{e.issue_id} resolved")
        return "\n".join(lines)
