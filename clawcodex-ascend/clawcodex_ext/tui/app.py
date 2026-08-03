# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Textual ``App`` subclass that hosts the Claw Codex TUI.

The app owns everything that must outlive a single screen push: the
``Session`` / ``Conversation``, the provider instance, the tool registry,
the tool context, the :class:`AppState`, and the
:class:`AgentBridge` that shuttles events between the agent-loop worker
thread and the UI.

Phase 1 boots the :class:`REPLScreen` on mount and delegates user
submissions to :class:`AgentBridge.submit`. Permission requests from
tools land here as :class:`PermissionRequested` messages, which the
screen then materialises as a modal.
"""

from __future__ import annotations
# pylint: disable=C0302,E0611,R1702,W0201

import asyncio
import time
from pathlib import Path
from typing import Any

from textual.app import App
from src import __version__ as CLAW_VERSION
from src.agent import Session
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.registry import ToolRegistry

from clawcodex_ext.away_summary.controller import AwaySummaryController
from clawcodex_ext.away_summary.config import load_away_summary_config
from clawcodex_ext.away_summary.messages import format_away_summary_for_display
from clawcodex_ext.intent_forecast.config import load_intent_forecast_config
from clawcodex_ext.intent_forecast.controller import IntentForecastController
from clawcodex_ext.intent_forecast.messages import (
    ForecastResult,
    format_forecast_for_display,
)
from clawcodex_ext.intent_forecast.persistence import save_forecast_result
from clawcodex_ext.intent_forecast.service import IntentForecastService
from clawcodex_ext.permissions.runtime import RuntimePermissionController

from .a11y import Announcer, describe_status
from .agent_bridge import AgentBridge
from .commands import (
    CommandDispatchResult,
    CommandSuggestion,
    LOCAL_BUILTINS,
    build_command_suggestions,
    build_command_words,
    dispatch_local_command,
    dispatch_registry_command,
)
from .history_store import HistoryStore  # noqa: F401 (re-exported for tests)
from .messages import (
    CancelRequested,
    PermissionModeChanged,
)
from .screens.cost_threshold import CostThresholdScreen
from .screens.diff_dialog import DiffDialogScreen, FileDiff
from .screens.effort_picker import EffortPickerScreen
from .screens.exit_flow import ExitFlowScreen
from .screens.forecast_picker import ForecastPickerScreen
from .screens.history_search import HistoryEntry, HistorySearchScreen
from .screens.idle_return import IdleReturnScreen
from .screens.mcp_dialogs import McpListScreen, McpServer
from .screens.message_selector import MessageSelectorScreen, TranscriptMessage
from .screens.model_picker import ModelPickerScreen
from .screens.monitor_panel import MonitorPanel
from .screens.permission_mode_picker import PermissionModePickerScreen
from .screens.repl import REPLScreen
from .screens.resume_conversation import ResumeConversation
from .screens.theme_picker import ThemePickerScreen
from .state import AppState
from .terminal_chrome import (
    clear_terminal_title,
    disable_focus_events,
    enable_focus_events,
    ring_bell,
    set_tab_status,
    set_terminal_title,
)
from .theme import (
    get_palette,
    list_theme_names,
    textual_css_overrides,
)
from .widgets.transcript_view import Transcript

_RESUME_BUSY_MESSAGE = "Cannot resume while the agent is running. Press Esc to interrupt first."


def _flatten_message_text(content: Any) -> str:
    """Normalise ``Message.content`` (string or block list) to text.

    Only extracts *text* content — ``tool_use`` blocks are transparent here
    (they are replayed as separate ``ToolEventMessage`` rows in
    ``_replay_history_MARKER``).
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in (None, "text"):
                    parts.append(str(item.get("text") or ""))
                elif kind == "thinking":
                    parts.append(str(item.get("thinking") or ""))
                # tool_use / tool_result are transparent — handled by replay loop
            elif hasattr(item, "type"):
                # Handle dataclass content blocks (TextBlock, ToolUseBlock,
                # ThinkingBlock, etc.) that are neither str nor dict.
                kind = item.type
                if kind in (None, "text"):
                    parts.append(str(getattr(item, "text", "") or ""))
                elif kind == "thinking":
                    parts.append(str(getattr(item, "thinking", "") or ""))
                # tool_use / tool_result are transparent — handled by replay loop
        return "\n".join(p for p in parts if p).strip()
    return str(content)


class ClawCodexTUI(App):
    """Top-level Textual application for Claw Codex."""

    TITLE = "ClawCodex"
    SUB_TITLE = "interactive terminal"

    BINDINGS = [
        ("ctrl+c", "cancel_or_quit", "Cancel / Quit"),
        ("ctrl+d", "request_quit", "Quit"),
        ("ctrl+b", "agent_background", "Background agent"),
        ("ctrl+t", "toggle_thinking", "Toggle thinking"),
        ("shift+tab", "cycle_permission_mode", "Cycle permission mode"),
        ("shift+down", "monitor_panel", "Monitor"),
    ]

    def __init__(
        self,
        *,
        provider,
        provider_name: str,
        workspace_root: Path,
        tool_registry: ToolRegistry | None = None,
        tool_context: ToolContext | None = None,
        session: Session | None = None,
        max_turns: int = 20,
        stream: bool = True,
        theme_name: str | None = None,
        tail_follower: Any | None = None,
        resume_browse: bool = False,
        runtime_context: Any | None = None,
        append_system_prompt: str = "",
        replay_exit_snapshot_from_start: bool = True,
        session_was_resumed: bool = False,
    ) -> None:
        super().__init__()
        self.runtime_context = runtime_context
        self._append_system_prompt = append_system_prompt
        self.provider = provider
        self.provider_name = provider_name
        self.workspace_root = Path(workspace_root)
        self.max_turns = max_turns
        self.stream = stream
        self.model = getattr(provider, "model", "unknown")
        self.session = session or Session.create(provider_name, self.model or "")
        self.tool_registry = tool_registry or build_default_registry(provider=provider)
        self.tool_context = tool_context or self._build_default_tool_context()
        # Theme is resolved once on boot; ``/theme`` can switch it
        # live via :meth:`apply_theme`.
        self.palette = get_palette(theme_name or self._resolve_theme_name())
        self.app_state = AppState(
            model=self.model or "",
            provider=provider_name,
        )
        self._repl_screen: REPLScreen | None = None
        self._command_context: Any | None = None
        self._model_discovery_warning: str | None = None
        self._away_summary_controller: AwaySummaryController | None = None
        self._intent_forecast_controller: IntentForecastController | None = None
        self._pending_system_messages: list[tuple[str, str, str | None]] = []
        # F-9 / ``/goal``: persistent controller instance wired into
        # the agent run lifecycle so continuation/budget-limit injections
        # are drained and enqueued for auto-continuation. Installed by
        # ``_install_goal_controller`` in ``__init__`` and on session resume.
        self._goal_controller: Any | None = None
        # Transcript renderables captured at exit time so entry points
        # can dump them back to the main terminal scrollback after the
        # alt-screen tears down. Mirrors the TS ink behaviour where the
        # conversation the user saw stays on-screen after ``/exit``.
        self.exit_snapshot: list[Any] = []
        self._exit_snapshot_start_index = 0
        self._exit_snapshot_replay_pending = False
        self._replay_exit_snapshot_from_start = replay_exit_snapshot_from_start
        # Persistent prompt history used by the PromptInput (↑/↓) and
        # the /history slash-command dialog. The store is append-only
        # per turn and auto-rotates past ``max_entries``.
        self.history_store = HistoryStore()
        # Seed the in-session history with the most recent entries from
        # the persistent store, so ↑/↓ and ghost-text suggestions carry
        # over across TUI restarts (limited to 20 to avoid context drift).
        try:
            self._initial_history: list[str] = [r.prompt for r in self.history_store.recent(limit=20)]
        except Exception:  # nosec B110
            self._initial_history = []
        self._theme_name = theme_name or self._resolve_theme_name()
        # Index into ``self.stylesheet._sources`` for the theme CSS
        # overrides.  Tracked so ``apply_theme`` can *replace* the
        # source instead of stacking duplicates.
        self._theme_source_idx: int | None = None
        # Screen-reader announcer. The :class:`LiveRegion` widget is
        # bound in :meth:`on_mount` once the REPL screen is composed.
        self.announcer = Announcer(self)
        # Runtime permission controller — single chokepoint for Shift+Tab
        # cycles (``action_cycle_permission_mode``) and picker picks
        # (``_open_permission_mode_picker``). The ``default_handler`` is
        # the agent bridge's own permission handler — captured NOW
        # before any cycle can overwrite it, so a later cycle out of
        # ``bypassPermissions`` restores the exact callable the bridge
        # wired into ``tool_context.permission_handler`` at construction
        # time. The notify hook posts a ``PermissionModeChanged``
        # message so the REPL screen can update the status bar.
        self._runtime_permission_controller = RuntimePermissionController(
            tool_context_factory=lambda: self.tool_context,
            default_handler=None,  # wired below after AgentBridge exists
            app_state_store=None,  # TUI uses direct AppState mutation today
            notify=self._post_permission_mode_changed,
        )
        self._agent_bridge = AgentBridge(
            post_message=self._post_to_screen,
            session=self.session,
            provider=self.provider,
            tool_registry=self.tool_registry,
            tool_context=self.tool_context,
            app_state=self.app_state,
            run_worker=self.run_worker,
            max_turns=self.max_turns,
            stream=self.stream,
            tail_follower=tail_follower,
            append_system_prompt=self._append_system_prompt,
            runtime_permission_controller=self._runtime_permission_controller,
            reset_goal_progress=session_was_resumed,
        )
        # Patch the controller's ``default_handler`` now that the
        # bridge has installed its own ``permission_handler`` on the
        # ``ToolContext``. The cycle-out-of-bypass path reads this to
        # restore the non-bypass handler. Late binding is safe because
        # the controller reads ``default_handler`` lazily inside
        # ``_apply`` (under the lock), not at construction.
        self._runtime_permission_controller._default_handler = self._agent_bridge._permission_handler
        # Also stamp it on the ToolContext so future hand-rolled cycles
        # (none today, but a future hook) can find the same callable.
        if self.tool_context is not None:
            self.tool_context.default_permission_handler = self._agent_bridge._permission_handler
        self._install_away_summary_controller()
        self._install_intent_forecast_controller()
        self._install_goal_controller()
        self._resume_browse = resume_browse
        # Double-press exit guard: Ctrl+C first press clears draft /
        # arms exit; second press within 0.8s actually quits.
        self._last_ctrl_c: float = 0.0
        if self.runtime_context is not None:
            from clawcodex_ext.frontend.tui_extensions import install_tui_extensions

            install_tui_extensions(self, self.runtime_context)

    # The base CSS for the REPL; Phase 1 uses Textual's default theme
    # variables ($primary, $surface, …) — palette overrides sit in
    # ``textual_css_overrides`` and are appended at class build time.
    CSS = ""

    def _resolve_theme_name(self) -> str:
        try:
            from src.config import load_config

            cfg = load_config() or {}
            return cfg.get("theme") or "dark"
        except Exception:  # nosec B110
            return "dark"

    # ---- lifecycle ----
    def on_mount(self) -> None:
        # Apply palette-derived CSS on top of the component defaults so
        # the chrome picks up the correct background / foreground even
        # when Textual's internal theme doesn't cover every slot.
        try:
            self._theme_source_idx = len(self.stylesheet._sources)
            self.stylesheet.add_source(textual_css_overrides(self.palette))
            self.stylesheet.parse()
        except Exception:  # nosec B110
            pass

        self._repl_screen = REPLScreen(
            version=CLAW_VERSION,
            provider=self.provider_name,
            model=self.model,
            workspace_root=self.workspace_root,
            words_provider=self._slash_command_words,
            suggestions_provider=self._slash_command_suggestions,
            message_history_provider=self._message_history_provider,
            agents_provider=self._available_agents,
            # Pass the live BaseProvider so the status line's advisor
            # segment can call ``decide_advisor_mode(provider, ...)``
            # and show the correct mode label (server/client/inactive).
            provider_instance=self.provider,
            initial_history=self._initial_history,
        )
        self.push_screen(self._repl_screen)
        try:
            self.call_after_refresh(self._flush_pending_system_messages)
        except Exception:  # nosec B110
            self._flush_pending_system_messages()
        if self._intent_forecast_controller is not None:
            self._intent_forecast_controller.on_mount()

        # Replay conversation history from a resumed session so the
        # transcript widget shows the prior context immediately. Defer
        # until after the first refresh so REPLScreen has mounted its
        # TranscriptView before rows are appended.
        self._schedule_replay_history_MARKER()
        try:
            from clawcodex_ext.session_intelligence.queue import start_summary_queue_worker

            start_summary_queue_worker()
        except Exception:  # nosec B110
            pass

        # Terminal chrome: set a descriptive title, enable DEC 1004
        # focus reporting, and mark the tab idle. The app-state
        # observer below keeps title + tab status in sync with agent
        # activity.
        self._last_thinking: bool = self.app_state.is_thinking
        self._sync_terminal_title()
        set_tab_status("idle")
        try:
            enable_focus_events()
        except Exception:  # nosec B110
            pass
        self._state_unsub = self.app_state.subscribe(self._on_state_change)

        # If --resume was given without a SESSION_ID, show the session
        # browser so the user can pick a session to resume.
        if self._resume_browse:
            self._show_resume_browser()

    def on_unmount(self) -> None:
        if self._away_summary_controller is not None:
            self._away_summary_controller.close()
        if self._intent_forecast_controller is not None:
            self._intent_forecast_controller.close()
        # Best-effort cleanup so we don't leave stale chrome on the host.
        try:
            self._state_unsub()  # type: ignore[attr-defined]
        except Exception:  # nosec B110
            pass
        try:
            set_tab_status(None)
            clear_terminal_title()
            disable_focus_events()
        except Exception:  # nosec B110
            pass
        # Fallback capture in case ``exit()`` wasn't the path out (e.g.
        # Ctrl+C / SIGTERM). Entry points will print whatever landed
        # here to the host shell after the alt-screen exits.
        if not self.exit_snapshot:
            self._capture_exit_snapshot()
        # Save session as a fallback (if exit() already saved, save is
        # a no-op because the data is already on disk).
        try:
            self.session.save()
        except Exception:  # nosec B110
            pass
        self._enqueue_summary_sidecar_job()

    # ---- exit / snapshot ----------------------------------------------
    def _capture_exit_snapshot(self) -> None:
        """Collect the transcript's renderables into :attr:`exit_snapshot`.

        Called from :meth:`exit` and (as a fallback) :meth:`on_unmount`
        so no matter which shutdown path fires we preserve what the
        user saw. Failures are swallowed — a blank snapshot is fine,
        but raising would mask a normal exit.
        """

        if self.exit_snapshot or self._repl_screen is None:
            return
        try:
            transcript = self._repl_screen.transcript
            if not self._replay_exit_snapshot_from_start and self._exit_snapshot_replay_pending:
                self._exit_snapshot_start_index = transcript.message_count
                self._exit_snapshot_replay_pending = False
            start_index = self._exit_snapshot_start_index
            if start_index > transcript.message_count:
                start_index = 0
            self.exit_snapshot = list(transcript.snapshot(start_index=start_index))
        except Exception:  # nosec B110
            self.exit_snapshot = []

    def exit(self, result=None, return_code=0, message=None):  # type: ignore[override]
        """Capture transcript and save session before handing control back to Textual.

        Overriding ``exit()`` lets the entry-point reprint the
        conversation to the host terminal once the alt-screen unwinds,
        matching the TS ink reference's non-fullscreen UX where
        `/exit` leaves the printed text intact in scrollback.

        Saving the session here ensures it is persisted regardless of
        which exit path fires (``/exit``, Ctrl+D, exit-flow dialog).
        """

        self._capture_exit_snapshot()
        if self._away_summary_controller is not None:
            self._away_summary_controller.close()
        if self._intent_forecast_controller is not None:
            self._intent_forecast_controller.close()
        # Save session state so it can be resumed later.
        try:
            self.session.save()
        except Exception:  # nosec B110
            pass
        self._enqueue_summary_sidecar_job()
        return super().exit(result, return_code=return_code, message=message)

    def _on_state_change(self) -> None:
        """React to :class:`AppState` changes to refresh terminal chrome.

        Title reflects the active verb; tab status flips between
        ``busy`` (agent thinking) and ``idle`` (prompt ready); a
        terminal bell rings on the idle→thinking→idle edge to
        announce turn completion, matching the TS reference's
        idle-notification.
        """

        thinking = self.app_state.is_thinking
        if thinking != self._last_thinking:
            set_tab_status("busy" if thinking else "idle")
            if not thinking:
                # Turn completed — poke the host so the user notices
                # even when they tabbed away.
                try:
                    ring_bell()
                except Exception:  # nosec B110
                    pass
                self.announcer.announce(describe_status("idle"), level="polite", notify=False)
            else:
                self.announcer.announce(
                    describe_status("busy", verb=self.app_state.verb),
                    level="polite",
                    notify=False,
                )
            self._last_thinking = thinking
        self._sync_terminal_title()

    def _sync_terminal_title(self) -> None:
        try:
            state = self.app_state
            verb = state.verb if state.is_thinking else "Ready"
            title = f"ClawCodex — {state.model or self.provider_name}: {verb}"
            set_terminal_title(title)
        except Exception:  # nosec B110
            pass

    # ---- bindings ----
    def action_cancel_or_quit(self) -> None:
        # First press: try to cancel an in-flight agent run.
        if self._agent_bridge.cancel():
            self.announcer.announce("Cancelling…", level="assertive", notify=False)
            self._last_ctrl_c = 0.0  # reset double-press timer on cancel
            return
        # Double-press guard: first Ctrl+C while idle arms exit;
        # second press within 0.8s actually quits.
        now = time.monotonic()
        if now - self._last_ctrl_c < 0.8:
            self.exit()
        else:
            self._last_ctrl_c = now
            self.announcer.announce("Press Ctrl+C again to exit", level="assertive", notify=False)

    def action_request_quit(self) -> None:
        """Ctrl+D: arm the double-press exit flow.

        The stock ``Input`` swallows Ctrl+D (delete-forward), which is
        a no-op on an empty buffer — so the app's ``quit`` binding
        never fires. This action is bound instead, reusing the same
        double-press guard as ``action_cancel_or_quit`` so the user
        experience is consistent across both keys.
        """
        self._last_ctrl_c = time.monotonic()  # reuse same timer
        self.announcer.announce("Press Ctrl+D again to exit", level="assertive", notify=False)

    def action_agent_background(self) -> None:
        """Handle Ctrl+B — signal agent to continue in background, save
        session, and exit to terminal shell.

        Implements the Fork-Continue pattern:
        * If the agent is currently busy, we cancel the foreground run,
          wait briefly for it to settle, then launch the background
          runner so the agent keeps working after the TUI exits.
        * If the agent is idle, we fall back to the simpler
          ``__FULL_EXIT__`` path.

        The exit marker tells the calling entry point whether a
        background agent was actually spawned, so it can print the
        appropriate hint after the alt-screen tears down.
        """
        # Signal agent to continue running in background.
        try:
            from src.agent.background_state import signal_background

            signal_background()
        except Exception:  # nosec B110
            pass
        # Persist session so --resume can find it later.
        try:
            self.session.save()
        except Exception:  # nosec B110
            pass
        sid = getattr(self.session, "session_id", None) or ""

        if self.app_state.is_thinking:
            # Agent is busy — cancel the foreground run and immediately
            # fork into the background runner.
            self._agent_bridge.cancel()
            has_bg_agent = False
            try:
                from src.agent.background_runner import launch_background_runner

                launch_background_runner(
                    session=self.session,
                    provider=self.provider,
                    tool_registry=self.tool_registry,
                    tool_context=self.tool_context,
                    max_turns=self.max_turns,
                )
                has_bg_agent = True
            except Exception:  # nosec B110
                pass
            self.exit(result=("__BACKGROUND_EXIT__", sid, has_bg_agent))
        else:
            # Agent is idle — simple exit with session ID for resume hint.
            self.exit(result=("__FULL_EXIT__", sid))

    def action_toggle_thinking(self) -> None:
        """Ctrl+T: toggle thinking content visibility in all thinking rows."""
        if self._repl_screen is None:
            return
        transcript = self._repl_screen.transcript
        # Toggle all thinking rows
        from src.tui.widgets.messages.assistant_thinking import (
            AssistantThinkingMessage,
        )

        expanded = True
        for row in transcript.query(AssistantThinkingMessage):
            row.toggle()
            expanded = row.expanded

        label = "expanded" if expanded else "collapsed"
        transcript.append_system(f"Thinking content: {label}", style="muted")
        self.announcer.announce(f"Thinking {label}")

    def action_cycle_permission_mode(self) -> None:
        """Shift+Tab: cycle through permission modes.

        Routes through the runtime permission controller so the
        multi-field swap (``permission_context`` +
        ``permission_handler`` + ``allow_docs``) is serialized under a
        single lock; the agent worker thread never sees a torn write.
        The controller's notify hook posts a
        :class:`PermissionModeChanged` message that the screen
        handles to update the status bar and append a transcript line.
        """
        if self._runtime_permission_controller is None:
            return
        self._runtime_permission_controller.cycle()

    def action_monitor_panel(self) -> None:
        """Shift+Down: open the F-88 monitor task panel."""
        if self.tool_context is None:
            return
        self.push_screen(MonitorPanel(self.tool_context))

    def _post_permission_mode_changed(self, mode: str) -> None:
        """Notify hook for the runtime permission controller.

        Posts a :class:`PermissionModeChanged` message so the REPL
        screen can update the status bar and append a transcript line.
        Wrapped in ``try/except`` because the controller calls this
        under the lock; a UI failure must not unwind the swap.
        """
        try:
            self._post_to_screen(PermissionModeChanged(mode=mode))
        except Exception:  # nosec B110
            pass

    # ---- local command dispatcher ----
    def handle_local_slash_command(self, text: str, transcript: Transcript) -> bool:
        """Return ``True`` if the command was handled without hitting the agent.

        The dispatcher tries the local built-ins first (``/exit``,
        ``/help``, …), then falls through to the shared
        :mod:`src.command_system` registry. Commands that produce a
        prompt (``/init``) forward the prompt back to the agent bridge.
        """

        result = dispatch_local_command(
            text,
            session=self.session,
            workspace_root=self.workspace_root,
            tool_registry=self.tool_registry,
        )
        if result.handled:
            self._apply_command_result(result, transcript)
            return True

        # Fall through to the async command registry. We run it via the
        # asyncio loop that Textual already runs on.
        async def _run() -> CommandDispatchResult:
            return await dispatch_registry_command(
                text,
                command_context=self._ensure_command_context(),
            )

        # Schedule the async work on the Textual loop; if it comes back
        # handled we emit the appropriate UI response.
        self.run_worker(self._dispatch_registry_async(text, transcript), exclusive=False, name="slash-cmd")
        return True

    async def _dispatch_registry_async(self, text: str, transcript: Transcript) -> None:
        stripped = text.strip()
        command_name = stripped[1:].split(maxsplit=1)[0].lower() if stripped.startswith("/") else ""
        show_busy = command_name == "recap" and not self.app_state.is_thinking
        if show_busy:
            self.app_state.set_thinking(True, verb="Recapping")
        try:
            result = await dispatch_registry_command(
                text,
                command_context=self._ensure_command_context(),
            )
        finally:
            if show_busy:
                self.app_state.set_thinking(False)
        if not result.handled:
            # Unknown command — try as a skill before falling through
            # to the agent text prompt (matching REPL's
            # ``_try_run_skill_slash``).
            if self._try_run_skill_slash(text, transcript):
                return
            # Fall through to the agent as a plain text prompt.
            if result.error:
                transcript.append_system(result.error, style="error")
                return
            transcript.append_user(text)
            self.submit_to_agent(text)
            return
        self._apply_command_result(result, transcript)

    def _try_run_skill_slash(self, raw: str, transcript: Transcript) -> bool:
        """Try to run an unknown ``/xxx`` command as a skill.

        Mirrors ``clawcodex_ext/repl/core.py:_try_run_skill_slash``.
        Returns ``True`` if the command was consumed as a skill.
        """
        text = raw.strip()
        if not text.startswith("/"):
            return False
        body = text[1:]
        if not body:
            return False
        # Skip if the name matches a known built-in.
        first_word = body.split(maxsplit=1)[0].lower()
        if first_word in {b.lstrip("/").lower() for b in LOCAL_BUILTINS}:
            return False
        parts = body.split(maxsplit=1)
        skill_name = parts[0].strip()
        args = parts[1] if len(parts) > 1 else ""
        if not skill_name:
            return False
        try:
            from clawcodex_ext.tool_system.tools.skill import run_user_invoked_skill

            result = run_user_invoked_skill(skill_name, args, self.tool_context)
        except Exception as e:
            transcript.append_system(f"Skill error: {e}", style="error")
            return True
        payload = result.output if isinstance(result.output, dict) else {}
        if result.is_error or not payload.get("success"):
            err = payload.get("error") if isinstance(payload.get("error"), str) else "Unknown skill error"
            transcript.append_system(err, style="error")
            return True
        if payload.get("status") in {"fork", "forked"}:
            result_text = payload.get("result")
            if isinstance(result_text, str) and result_text.strip():
                transcript.append_assistant(result_text, agent_name=skill_name)
            return True
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            transcript.append_system("Skill produced empty prompt", style="error")
            return True
        meta_parts: list[str] = []
        loaded = payload.get("loadedFrom")
        if isinstance(loaded, str) and loaded:
            meta_parts.append(f"source={loaded}")
        model = payload.get("model")
        if isinstance(model, str) and model:
            meta_parts.append(f"model={model}")
        if meta_parts:
            info = " · ".join(meta_parts)
            transcript.append_system(f"Launching skill: {skill_name}  ({info})", style="info")
        else:
            transcript.append_system(f"Launching skill: {skill_name}", style="info")
        self.submit_to_agent(prompt)
        return True

    def _apply_command_result(
        self,
        result: CommandDispatchResult,
        transcript: Transcript,
    ) -> None:
        if result.error:
            transcript.append_system(result.error, style="error")
            return
        if result.open_dialog:
            self._open_phase2_dialog(result.open_dialog, transcript)
            return
        if result.system_text == "__exit__":
            self._confirm_exit(transcript)
            return
        if result.system_text == "__background__":
            # Ctrl+B: signal agent to continue in background,
            # persist session for --resume, exit with session ID so the
            # calling code prints the resume hint after teardown.
            try:
                from src.agent.background_state import signal_background

                signal_background()
            except Exception:  # nosec B110
                pass
            try:
                self.session.save()
            except Exception:  # nosec B110
                pass
            sid = getattr(self.session, "session_id", None) or ""
            self.exit(result=("__FULL_EXIT__", sid))
            return
        if result.system_text == "__repl__":
            # /repl: cleanly exit TUI page, return to CLI REPL.
            # No dialog, no background signal — just go back.
            self.exit()
            return
        if result.system_text == "__clear__":
            try:
                from clawcodex_ext.goal.service import clear_goal_for_context

                clear_goal_for_context(self.tool_context)
            except Exception as exc:
                transcript.append_system(
                    f"Unable to clear active goal: {exc}",
                    style="error",
                )
                self.announcer.announce("Unable to clear active goal.")
                return
            try:
                self.session.conversation.clear()
            except Exception:  # nosec B110
                pass
            self.app_state.set_goal_status(None)
            transcript.clear_transcript()
            self._agent_bridge.reset_advisor_dedup()
            return
        if result.system_text in (
            "__stream_on__",
            "__stream_off__",
            "__stream_toggle__",
            "__stream_status__",
        ):
            if result.system_text == "__stream_on__":
                self.stream = True
            elif result.system_text == "__stream_off__":
                self.stream = False
            elif result.system_text == "__stream_toggle__":
                self.stream = not self.stream
            self._agent_bridge._stream = self.stream
            status = "enabled" if self.stream else "disabled"
            transcript.append_system(f"Stream mode {status}.")
            return
        if result.transient and result.system_text:
            from clawcodex_ext.tui.screens.goal_status import GoalStatusScreen

            self.push_screen(GoalStatusScreen(result.system_text))
            self.announcer.announce("Opened goal status.", notify=False)
            return
        if result.system_text:
            transcript.append_system(
                result.system_text,
                style="muted",
                render=result.system_render,
            )
        if result.assistant_text:
            transcript.append_assistant(
                result.assistant_text,
                agent_name=result.assistant_name or "",
            )
        if result.prompt_text:
            transcript.append_user(f"(from slash command) {result.prompt_text[:80]}…")
            self.submit_to_agent(result.prompt_text)
        if result.should_query:
            self._agent_bridge.continue_goal_if_idle()

    # ---- Phase 2 dialog dispatcher -------------------------------------
    def _open_phase2_dialog(self, name: str, transcript: Transcript) -> None:
        """Push the modal screen for ``name`` from the slash command.

        ``name`` is one of the values produced by
        :func:`dispatch_local_command`; unknown names degrade to a
        muted system message.
        """

        if name == "model":
            self._open_model_picker(transcript)
        elif name == "effort":
            self._open_effort_picker(transcript)
        elif name == "history":
            self._open_history_search(transcript)
        elif name == "cost":
            self._open_cost_threshold(transcript)
        elif name == "idle":
            self._open_idle_return(transcript)
        elif name == "theme":
            self._open_theme_picker(transcript)
        elif name == "diff":
            self._open_diff_dialog(transcript)
        elif name == "mcp":
            self._open_mcp_list(transcript)
        elif name in ("rewind", "messages"):
            self._open_message_selector(transcript)
        elif name == "tasks":
            self._open_tasks_dialog(transcript)
        elif name == "resume":
            if self._agent_bridge.busy:
                transcript.append_system(_RESUME_BUSY_MESSAGE, style="error")
                self.announcer.announce(_RESUME_BUSY_MESSAGE)
            else:
                self._show_resume_browser()
        elif name == "permission":
            self._open_permission_mode_picker(transcript)
        elif name == "forecast":
            self._open_forecast_picker(transcript)
        else:
            transcript.append_system(f"Dialog '{name}' not available.", style="muted")

    def _open_model_picker(self, transcript: Transcript) -> None:
        provider = self.provider
        provider_name = self.provider_name
        current_model = self.model
        self.run_worker(
            self._discover_and_open_model_picker(
                transcript,
                provider=provider,
                provider_name=provider_name,
                current_model=current_model,
            ),
            exclusive=True,
            group="model-catalog",
            name="model-catalog",
        )

    async def _discover_and_open_model_picker(
        self,
        transcript: Transcript,
        *,
        provider: Any,
        provider_name: str,
        current_model: str,
    ) -> None:
        models, warning = await asyncio.to_thread(
            self._discover_available_models,
            provider,
            provider_name,
            current_model,
        )
        if self.provider is not provider or self.provider_name != provider_name:
            return
        self._model_discovery_warning = warning
        if warning:
            transcript.append_system(warning, style="muted")

        def _on_selected(model_id: str | None) -> None:
            if self.provider is not provider or self.provider_name != provider_name:
                transcript.append_system(
                    "Provider changed while the model picker was open; selection ignored.",
                    style="muted",
                )
                self._restore_prompt_focus()
                return
            if not model_id or model_id == self.model:
                self._restore_prompt_focus()
                return
            self.model = model_id
            try:
                if hasattr(self.provider, "model"):
                    setattr(self.provider, "model", model_id)
            except Exception:  # nosec B110
                pass
            self.app_state.model = model_id
            transcript.append_system(f"Model switched to {model_id}.", style="muted")
            if self._repl_screen is not None:
                self._repl_screen.status_bar.refresh_identity(model=model_id)
            self.announcer.announce(f"Model switched to {model_id}.")
            self._restore_prompt_focus()

        self.announcer.announce("Opened model picker.", notify=False)
        self.push_screen(
            ModelPickerScreen(
                models=models,
                current_model=current_model,
            ),
            callback=_on_selected,
        )

    def _open_effort_picker(self, transcript: Transcript) -> None:
        current = getattr(self.app_state, "effort", None) or None

        def _on_selected(result: tuple[str | None, bool]) -> None:
            effort, persisted = result
            self._restore_prompt_focus()
            if not persisted:
                return
            setattr(self.app_state, "effort", effort)
            transcript.append_system(f"Reasoning effort set to {effort or 'auto'}.", style="muted")
            self.announcer.announce(f"Reasoning effort set to {effort or 'auto'}.")

        self.announcer.announce("Opened effort picker.", notify=False)
        self.push_screen(EffortPickerScreen(current=current), callback=_on_selected)

    def _open_history_search(self, transcript: Transcript) -> None:
        records = self.history_store.recent(limit=500)
        entries = [HistoryEntry(prompt=r.prompt, timestamp=r.timestamp) for r in records]
        if not entries:
            transcript.append_system("History is empty — run some prompts first.", style="muted")
            return

        def _on_selected(result: str | None) -> None:
            self._restore_prompt_focus()
            if not result:
                return
            if self._repl_screen is not None:
                self._repl_screen.prompt_input.set_value(result)
                self._repl_screen.prompt_input.focus_input()
            self.announcer.announce("Prompt restored from history.", notify=False)

        self.announcer.announce("Opened history search.", notify=False)
        self.push_screen(HistorySearchScreen(entries=entries), callback=_on_selected)

    def _open_cost_threshold(self, transcript: Transcript) -> None:
        tokens = self.app_state.usage.get("input_tokens", 0) + self.app_state.usage.get("output_tokens", 0)
        # Rough estimate: $5 per 1M tokens. Phase 2 keeps this simple;
        # real per-model rates land with /cost refactor in Phase 3.
        estimate = (tokens / 1_000_000) * 5.0
        self.announcer.announce(f"Session cost estimate ${estimate:.2f}.", notify=False)
        self.push_screen(
            CostThresholdScreen(provider=self.provider_name, amount_usd=estimate),
            callback=lambda _=None: self._restore_prompt_focus(),
        )

    def _open_idle_return(self, transcript: Transcript) -> None:
        tokens = self.app_state.usage.get("input_tokens", 0)

        def _on_choice(action: str) -> None:
            if action == "clear":
                transcript.clear_transcript()
                self._agent_bridge.reset_advisor_dedup()
                transcript.append_system("Conversation cleared.", style="muted")
                self.announcer.announce("Conversation cleared.")
            elif action == "never":
                transcript.append_system("Idle-return prompts disabled for this session.", style="muted")

        self.announcer.announce("Idle return prompt open.", notify=False)
        self.push_screen(
            IdleReturnScreen(
                idle_minutes=0,
                total_input_tokens=tokens,
                on_choice=_on_choice,
            ),
            callback=lambda _=None: self._restore_prompt_focus(),
        )

    def _open_theme_picker(self, transcript: Transcript) -> None:
        original_theme = self._theme_name

        def _on_preview(name: str | None) -> None:
            # Live-preview the highlighted theme; Esc restores the
            # original one so we don't leak an unintended swap.
            target = name or original_theme
            if target and target != self._theme_name:
                self.apply_theme(target, transcript=None)

        def _on_selected(name: str | None) -> None:
            self._restore_prompt_focus()
            if not name:
                # User cancelled — restore the starting theme.
                if self._theme_name != original_theme:
                    self.apply_theme(original_theme, transcript=None)
                return
            self.apply_theme(name, transcript=transcript)

        self.announcer.announce("Opened theme picker.", notify=False)
        self.push_screen(
            ThemePickerScreen(
                themes=list_theme_names(),
                current=self._theme_name,
                on_preview=_on_preview,
            ),
            callback=_on_selected,
        )

    def _open_permission_mode_picker(self, transcript: Transcript) -> None:
        current_mode = "default"
        try:
            from src.permissions.modes import to_external_permission_mode

            ctx = self.tool_context
            if ctx is not None and ctx.permission_context is not None:
                current_mode = to_external_permission_mode(ctx.permission_context.mode or "default")
        except Exception:  # nosec B110
            pass

        is_bypass_available = False
        try:
            from src.permissions.modes import has_allow_bypass_permissions_mode

            is_bypass_available = has_allow_bypass_permissions_mode()
        except Exception:  # nosec B110
            pass

        def _on_selected(mode: str | None) -> None:
            self._restore_prompt_focus()
            if not mode:
                return
            try:
                if self.tool_context is None or self.tool_context.permission_context is None:
                    return
                # Route through the runtime controller so the same
                # lock / AppState-write / handler-restore logic is
                # shared with Shift+Tab. The controller's notify hook
                # posts ``PermissionModeChanged`` which the screen
                # handles to update the status bar.
                self._runtime_permission_controller.set_mode(mode)
                transcript.append_system(f"Permission mode set to {mode}.", style="muted")
                self.announcer.announce(f"Permission mode: {mode}.")
            except Exception as exc:
                transcript.append_system(f"Failed to set permission mode: {exc}", style="error")

        self.announcer.announce("Opened permission mode picker.", notify=False)
        self.push_screen(
            PermissionModePickerScreen(
                current_mode=current_mode,
                is_bypass_available=is_bypass_available,
                on_select=_on_selected,
            ),
        )

    def _open_forecast_picker(self, transcript: Transcript) -> None:
        controller = self._intent_forecast_controller
        result = controller.last_result if controller is not None else None
        if result is None or not result.generated or not result.suggestions:
            try:
                service = IntentForecastService(
                    conversation=self.session.conversation,
                    provider=self.provider,
                    model=self.model,
                    workspace_root=self.workspace_root,
                    config=load_intent_forecast_config(cwd=self.workspace_root),
                )
                result = service.generate(trigger="manual", force=True)
            except Exception as exc:
                transcript.append_system(f"Forecast failed: {exc}", style="error")
                return
            try:
                save_forecast_result(
                    result,
                    trigger="slash",
                    cwd=self.workspace_root,
                    model=self.model,
                )
            except Exception:  # nosec B110
                pass
        if result is None or not result.generated or not result.suggestions:
            transcript.append_system(
                result.reason if result is not None else "Forecast has no suggestions right now.",
                style="muted",
            )
            return
        if controller is not None:
            controller.remember(result)

        def _on_selected(selection: str | None) -> None:
            self._restore_prompt_focus()
            if selection is None:
                if controller is not None:
                    controller.dismiss()
                transcript.append_system("Forecast dismissed.", style="muted")
                return
            if controller is not None and controller.accept(selection):
                transcript.append_system("Forecast accepted.", style="muted")
                return
            transcript.append_system("Forecast selection is no longer available.", style="muted")

        self.announcer.announce("Forecast suggestions open.", notify=False)
        self.push_screen(ForecastPickerScreen(result), callback=_on_selected)

    # ---- Phase 3 dialogs ----
    def _open_diff_dialog(self, transcript: Transcript) -> None:
        """Show pending file diffs, if the provider can surface them.

        We collect diffs from ``app_state.pending_diffs`` (populated by
        the file-edit tools) or the conversation's most-recent tool
        results. If nothing is available we drop a muted note instead
        of opening an empty dialog.
        """

        files: list[FileDiff] = []
        pending = getattr(self.app_state, "pending_diffs", None) or []
        for entry in pending:
            if isinstance(entry, FileDiff):
                files.append(entry)
            elif isinstance(entry, dict) and "patch" in entry and "path" in entry:
                files.append(FileDiff(path=str(entry["path"]), patch=str(entry["patch"])))

        if not files:
            transcript.append_system("No pending diffs to display.", style="muted")
            return

        self.announcer.announce(f"Diff dialog open. {len(files)} file(s) changed.", notify=False)
        self.push_screen(
            DiffDialogScreen(files=files),
            callback=lambda _=None: self._restore_prompt_focus(),
        )

    def _open_message_selector(self, transcript: Transcript) -> None:
        messages = self._collect_transcript_messages()
        if not messages:
            transcript.append_system("Nothing to rewind — the transcript is empty.", style="muted")
            return

        def _on_choice(result: tuple[int, str]) -> None:
            index, action = result
            self._restore_prompt_focus()
            if action == "cancel" or index < 0:
                return
            selected = next((m for m in messages if m.index == index), None)
            if selected is None:
                return
            if action == "restore" and self._repl_screen is not None:
                self._repl_screen.prompt_input.set_value(selected.text)
                self._repl_screen.prompt_input.focus_input()
                transcript.append_system(f"Restored prompt from message #{index}.", style="muted")
                self.announcer.announce(f"Restored prompt from message {index}.", notify=False)
            elif action == "summarize":
                transcript.append_system(
                    f"Summarise-from-here requested for message #{index}.",
                    style="muted",
                )
                self.announcer.announce(f"Summarise requested for message {index}.")

        self.announcer.announce(f"Message selector open. {len(messages)} message(s).", notify=False)
        self.push_screen(
            MessageSelectorScreen(messages=messages, on_choice=None),
            callback=_on_choice,
        )

    def _collect_transcript_messages(self) -> list[TranscriptMessage]:
        out: list[TranscriptMessage] = []
        try:
            conversation = self.session.conversation
            history = getattr(conversation, "messages", None) or []
        except Exception:  # nosec B110
            return out

        idx = 0
        for msg in history:
            role = getattr(msg, "role", None)
            if role is None and isinstance(msg, dict):
                role = msg.get("role")
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if not role or role not in ("user", "assistant"):
                continue
            text = _flatten_message_text(content)
            if not text.strip():
                continue
            out.append(TranscriptMessage(index=idx, kind=role, text=text))
            idx += 1
        return out

    def _open_mcp_list(self, transcript: Transcript) -> None:
        servers = self._collect_mcp_servers()
        if not servers:
            transcript.append_system("No MCP servers configured.", style="muted")
            return
        self.announcer.announce(f"MCP servers list open. {len(servers)} server(s).", notify=False)
        self.push_screen(
            McpListScreen(servers=servers),
            callback=lambda _=None: self._restore_prompt_focus(),
        )

    def _collect_mcp_servers(self) -> list[McpServer]:
        try:
            from src.config import load_config

            cfg = load_config() or {}
            raw = cfg.get("mcp_servers") or cfg.get("mcpServers") or {}
        except Exception:
            raw = {}
        servers: list[McpServer] = []
        if isinstance(raw, dict):
            for server_id, entry in raw.items():
                name = entry.get("name", server_id) if isinstance(entry, dict) else server_id
                status = entry.get("status", "disconnected") if isinstance(entry, dict) else "disconnected"
                tools = entry.get("tools", []) if isinstance(entry, dict) else []
                servers.append(
                    McpServer(
                        id=str(server_id),
                        name=str(name),
                        status=status,  # type: ignore[arg-type]
                        tools=list(tools) if isinstance(tools, list) else [],
                    )
                )
        return servers

    def _open_tasks_dialog(self, transcript: Transcript) -> None:
        # The background task panel lives on the REPL screen; the slash
        # command just routes focus to it rather than stacking a modal.
        if self._repl_screen is not None and hasattr(self._repl_screen, "focus_task_panel"):
            try:
                self._repl_screen.focus_task_panel()
                return
            except Exception:  # nosec B110
                pass
        transcript.append_system("Task panel focus is not available in this build.", style="muted")

    def _confirm_exit(self, transcript: Transcript) -> None:
        """Push :class:`ExitFlowScreen` instead of quitting immediately."""

        def _on_choice(action: str) -> None:
            if action == "quit":
                self.exit()
            elif action == "quit-clear":
                transcript.clear_transcript()
                self.exit()
            else:  # "cancel"
                self._restore_prompt_focus()

        self.announcer.announce("Exit confirmation open.", level="assertive", notify=False)
        self.push_screen(
            ExitFlowScreen(
                has_inflight_work=self.app_state.is_thinking,
                on_choice=_on_choice,
            )
        )

    # ---- focus helpers ----
    def _restore_prompt_focus(self) -> None:
        """Return keyboard focus to the prompt input after a modal closes.

        Modals that dismiss via the ``callback=`` path don't
        automatically restore focus on the previous screen, so we do
        it explicitly for every dialog close. No-op when the REPL
        screen hasn't mounted yet.
        """

        if self._repl_screen is None:
            return
        try:
            self._repl_screen.prompt_input.focus_input()
        except Exception:  # nosec B110
            pass

    # ---- theme live-switch ----
    def apply_theme(self, name: str, *, transcript: Transcript | None = None) -> None:
        """Hot-swap the palette and refresh the stylesheet overrides."""

        self.palette = get_palette(name)
        self._theme_name = name
        # Persist the user's theme choice to global config so it
        # survives TUI restarts.
        try:
            from src.config import set_theme

            set_theme(name)
        except ImportError:
            pass
        try:
            new_css = textual_css_overrides(self.palette)
            if self._theme_source_idx is not None and self._theme_source_idx < len(self.stylesheet._sources):
                self.stylesheet._sources[self._theme_source_idx] = (new_css, None)
            else:
                self._theme_source_idx = len(self.stylesheet._sources)
                self.stylesheet.add_source(new_css)
            self.stylesheet.parse()
            self.refresh_css()
        except Exception:  # nosec B110
            pass
        if transcript is not None:
            transcript.append_system(f"Theme set to {name}.", style="muted")
            self.announcer.announce(f"Theme set to {name}.", notify=False)

    # ---- model discovery ----
    def _list_available_models(self) -> list[str]:
        """Return a best-effort list of models for the active provider."""

        models, warning = self._discover_available_models(
            self.provider,
            self.provider_name,
            self.model,
        )
        self._model_discovery_warning = warning
        return models

    @staticmethod
    def _discover_available_models(
        provider: Any,
        provider_name: str,
        current_model: str,
    ) -> tuple[list[str], str | None]:
        warning: str | None = None
        try:
            from clawcodex_ext.providers.model_catalog_cache import get_model_catalog

            snapshot = get_model_catalog(provider_name, provider)
            models = list(snapshot.models)
            if snapshot.error:
                shown = "cached" if snapshot.source == "stale-cache" else "fallback"
                warning = (
                    f"Last model catalog refresh failed for {provider_name}: {snapshot.error}; showing {shown} models."
                )
            if models:
                if current_model and current_model not in models:
                    models.insert(0, current_model)
                if warning is None and snapshot.refreshing:
                    shown = "cached" if snapshot.source == "stale-cache" else "fallback"
                    warning = f"Model catalog refresh is running in the background; showing {shown} models."
                return models, warning
            if warning is None:
                warning = f"Model catalog has no models for {provider_name}; showing configured fallback models."
        except Exception as exc:
            warning = f"Model discovery failed for {provider_name}: {exc}; showing configured fallback models."
        try:
            from src.config import get_provider_config

            cfg = get_provider_config(provider_name) or {}
            fallback_models: list[str] = []
            models = cfg.get("models")
            if isinstance(models, list):
                fallback_models.extend(str(model) for model in models if model)
            default = cfg.get("default_model")
            if default and str(default) not in fallback_models:
                fallback_models.append(str(default))
            if current_model and current_model not in fallback_models:
                fallback_models.insert(0, current_model)
            if fallback_models:
                return fallback_models, warning
        except Exception:  # nosec B110
            pass
        # Fallback: just the active model.
        return [current_model or "default"], warning

    def _ensure_command_context(self) -> Any:
        if self._command_context is not None:
            return self._command_context
        try:
            from src.command_system.builtins import register_builtin_commands
            from src.command_system.engine import create_command_context
            from src.cost_tracker import CostTracker
            from src.history import HistoryLog

            register_builtin_commands(None)
            # Register disk skills as PromptCommands in the global
            # command registry so ``dispatch_registry_command`` can
            # resolve ``/<skill-name>`` slash commands. Idempotent
            # via the registry's shadowing guard (builtins win).
            # Failures must never block command dispatch.
            try:
                from src.command_system import load_and_register_skills

                load_and_register_skills(registry=None)
            except Exception:  # nosec B110
                pass
            # F-53: also register dynamic tool commands in the global
            # registry so ``execute_command_sync`` (which looks at the
            # global registry, not the TUI's private one) can route
            # ``/<tool-name>`` slash commands.
            try:
                from clawcodex_ext.cli.tool_cmd import register_tool_commands

                register_tool_commands(None, tool_registry=self.tool_registry)
            except Exception:  # nosec B110
                pass
            # Builtin registration installs the interactive ModelCommand.
            # Reinstall the runtime facade last so TUI `/model <name>` uses
            # the same provider swap and state-sync path as the REPL.
            from clawcodex_ext.cli.runtime_commands import register_runtime_commands

            register_runtime_commands(None)
            self._command_context = create_command_context(
                workspace_root=self.workspace_root,
                conversation=self.session.conversation,
                cost_tracker=CostTracker(),
                history=HistoryLog(),
                provider=self.provider,
                tool_registry=self.tool_registry,
                tool_context=self.tool_context,
                runtime_context=self.runtime_context,
            )
            self._command_context.session = self.session
            self._command_context.intent_forecast_controller = self._intent_forecast_controller
            self._command_context.app_state = self.app_state
        except Exception:  # nosec B110
            self._command_context = None
        return self._command_context

    # ---- agent loop plumbing ----
    def submit_to_agent(self, prompt: str) -> None:
        if self._away_summary_controller is not None:
            self._away_summary_controller.on_user_interaction("submit")
        if self._intent_forecast_controller is not None:
            self._intent_forecast_controller.on_user_interaction("submit")
        # Track last user input in session metadata for the session browser.
        self._update_metadata_last_input(prompt)
        try:
            self.history_store.append(prompt)
        except Exception:  # nosec B110
            pass
        submitted = self._agent_bridge.submit(prompt)
        if not submitted:
            # If the bridge is busy we queue the prompt for the next
            # turn so the user can keep typing. Phase 2 adds a visible
            # queued-prompts pill in the status line.
            self.app_state.queued_prompts.append(prompt)

    def on_cancel_requested(self, _: CancelRequested) -> None:
        """ESC from the prompt — cancel the in-flight agent run, if any."""

        if self._away_summary_controller is not None:
            self._away_summary_controller.on_user_interaction("cancel")
        if self._intent_forecast_controller is not None:
            self._intent_forecast_controller.on_user_interaction("cancel")
        if self._agent_bridge.cancel():
            self.announcer.announce("Cancelling…", level="assertive", notify=False)

    # ---- bash mode ----
    def run_bash_mode(self, command: str, transcript: Any) -> None:
        """Execute a user-typed ``!command`` directly (no agent turn).

        Feeds the bash input + output into the conversation (so the model
        sees what happened on its next turn) and displays the result in
        the transcript. Commands are refused if an agent run is in flight.

        Divergences from TS: sequential only (refused while busy); no
        live progress streaming or ESC cancel yet.
        """
        if self._agent_bridge.busy:
            return  # agent run in flight — refuse silently

        from src.services.bash_mode import run_bash_mode_command

        outcome = run_bash_mode_command(command, self.tool_context)

        # Append conversation texts so the model sees them on next turn.
        for conv_text in outcome.conversation_texts:
            self.session.conversation.add_user_message(conv_text)

        # Display in transcript.
        text = outcome.command
        if outcome.ok:
            if outcome.stdout:
                text += f"\n{outcome.stdout}"
            if outcome.stderr:
                text += f"\n[stderr]\n{outcome.stderr}"
        else:
            text += f"\n[error]\n{outcome.error or outcome.stderr}"
        transcript.append_user(f"! {outcome.command}")
        transcript.append_system(
            text,
            style="success" if outcome.ok else "error",
        )

    # ---- helpers ----
    def _update_metadata_last_input(self, text: str) -> None:
        """Update the ``last_user_input`` field in SessionStorage metadata."""
        if not self.session:
            return
        try:
            from src.services.session_storage import SessionStorage

            storage = SessionStorage(session_id=self.session.session_id)
            storage.update_metadata(last_user_input=text[:200])  # cap at 200 chars
        except Exception:  # nosec B110
            pass

    def _enqueue_summary_sidecar_job(self) -> None:
        try:
            from src.services.session_storage import SESSIONS_DIR
            from clawcodex_ext.session_intelligence.queue import enqueue_summary_job

            sid = str(getattr(self.session, "session_id", "") or "")
            if not sid:
                return
            transcript = SESSIONS_DIR / sid / "transcript.jsonl"
            enqueue_summary_job(
                sid,
                cwd=self.workspace_root,
                transcript_mtime=transcript.stat().st_mtime if transcript.exists() else 0.0,
            )
        except Exception:  # nosec B110
            pass

    def _show_resume_browser(self) -> None:
        """Push the ResumeConversation modal so the user can pick a session."""
        screen = ResumeConversation()
        self.push_screen(screen, self._on_session_selected)

    def _on_session_selected(self, session_id: str | None) -> None:
        """Callback after the user picks a session from the resume browser."""
        if session_id is None:
            # User cancelled — start a fresh session.
            return
        if self._agent_bridge.busy:
            self.announcer.announce(_RESUME_BUSY_MESSAGE)
            return
        try:
            from clawcodex_ext.agent.session_ext import resume_session_with_tail

            resumed, tail = resume_session_with_tail(session_id)
            if resumed is None:
                self.announcer.announce("Unable to resume the selected session.")
                return
            # The bridge owns the definitive busy check. A turn can begin while
            # the resume picker is open, so only publish the new session after
            # the bridge atomically accepts the rebind.
            if not self._agent_bridge.replace_session(resumed):
                self.announcer.announce(_RESUME_BUSY_MESSAGE)
                return
            self.session = resumed
            if self._command_context is not None:
                self._command_context.session = resumed
                self._command_context.conversation = resumed.conversation
                self._command_context.tool_context = self.tool_context
            self._install_away_summary_controller()
            self._install_intent_forecast_controller()
            self._install_goal_controller()
            if tail is not None:
                self._agent_bridge._tail_follower = tail
                self._agent_bridge._start_tail_follower()
            # Replay the restored conversation into the transcript.
            self._schedule_replay_history_MARKER()
        except Exception:  # nosec B110
            pass

    def _build_default_tool_context(self) -> ToolContext:
        # ``ask_user`` and ``permission_handler`` are wired by
        # :class:`AgentBridge.__init__` (see ``tui/agent_bridge.py``)
        # which mounts the bridge-to-UI plumbing that posts modal
        # requests and blocks the worker thread on user input. We
        # intentionally leave the defaults ``None`` here so that any
        # code path that constructs a tool context without the bridge
        # fails loudly instead of silently swallowing user questions
        # the way the old no-op lambda did.
        return ToolContext(workspace_root=self.workspace_root)

    def _append_repl_system_message(
        self,
        text: str,
        *,
        style: str = "light",
        render: str | None = "markdown",
    ) -> None:
        screen = self._repl_screen
        transcript = getattr(screen, "transcript", None) if screen is not None else None
        if transcript is None or not getattr(transcript, "is_mounted", False):
            self._pending_system_messages.append((text, style, render))
            try:
                self.call_after_refresh(self._flush_pending_system_messages)
            except Exception:  # nosec B110
                pass
            return
        transcript.append_system(text, style=style, render=render)
        try:
            self.call_after_refresh(lambda: transcript.scroll_end(animate=False))
        except Exception:  # nosec B110
            pass

    def _flush_pending_system_messages(self) -> None:
        screen = self._repl_screen
        transcript = getattr(screen, "transcript", None) if screen is not None else None
        if transcript is None or not getattr(transcript, "is_mounted", False):
            return
        pending = list(self._pending_system_messages)
        self._pending_system_messages.clear()
        for text, style, render in pending:
            self._append_repl_system_message(text, style=style, render=render)
            if self._pending_system_messages:
                break

    def _install_away_summary_controller(self) -> None:
        if self._away_summary_controller is not None:
            self._away_summary_controller.close()

        def _display(text: str) -> None:
            def _append() -> None:
                display_text = (
                    text.strip()
                    if text.strip().startswith(("Recapitulate", "Recap", "Away Summary"))
                    else format_away_summary_for_display(text)
                )
                self._append_repl_system_message(
                    display_text,
                    style="muted",
                    render="markdown",
                )

            try:
                self.call_from_thread(_append)
            except RuntimeError:
                _append()

        self._away_summary_controller = AwaySummaryController(
            conversation=self.session.conversation,
            provider_getter=lambda: self.provider,
            model_getter=lambda: self.model,
            session_getter=lambda: self.session,
            display=_display,
            config_loader=lambda: load_away_summary_config(cwd=self.workspace_root),
        )

    def _install_intent_forecast_controller(self) -> None:
        if self._intent_forecast_controller is not None:
            self._intent_forecast_controller.close()

        def _display(result: ForecastResult) -> None:
            def _append() -> None:
                self._append_repl_system_message(
                    format_forecast_for_display(result),
                    style="light",
                    render="markdown",
                )

            try:
                self.call_from_thread(_append)
            except RuntimeError:
                _append()

        self._intent_forecast_controller = IntentForecastController(
            provider_getter=lambda: self.provider,
            model_getter=lambda: self.model,
            session_getter=lambda: self.session,
            workspace_root=self.workspace_root,
            display=_display,
            submit=self.submit_to_agent,
            config_loader=lambda: load_intent_forecast_config(cwd=self.workspace_root),
            conversation_getter=lambda: self.session.conversation,
        )
        if self._command_context is not None:
            self._command_context.intent_forecast_controller = self._intent_forecast_controller

    def _install_goal_controller(self) -> None:
        """Legacy GoalController is removed; goal runtime is tool-context based."""
        self._goal_controller = None

    def _schedule_replay_history_MARKER(self) -> None:
        if self._repl_screen is None:
            return
        if not getattr(self.session.conversation, "messages", None):
            return
        try:
            self.call_after_refresh(self._replay_history_MARKER)
        except Exception:  # nosec B110
            self._replay_history_MARKER()

    def _replay_history_MARKER(self) -> None:
        """Replay conversation messages from a resumed session to the transcript.

        Emits ``AssistantMessage`` / ``ToolEventMessage`` for each historical
        message so the transcript widget shows the prior context immediately
        after ``--resume``. Only called from :meth:`on_mount` when the
        session has existing messages.
        """
        agent_type = getattr(self.tool_context, "agent_type", None) or ""
        for msg in self.session.conversation.messages:
            is_meta = bool(msg.get("isMeta", False)) if isinstance(msg, dict) else bool(getattr(msg, "isMeta", False))
            is_virtual = (
                bool(msg.get("isVirtual", False)) if isinstance(msg, dict) else bool(getattr(msg, "isVirtual", False))
            )
            if is_meta or is_virtual:
                continue
            role = getattr(msg, "role", None) or ""
            content = getattr(msg, "content", None) or ""
            if role == "user":
                # Render user messages (S-R2 fix).  User messages are
                # posted through the transcript directly so the resume
                # view shows the full conversation context.
                text = _flatten_message_text(content)
                if text and self._repl_screen is not None:
                    self._repl_screen.transcript.append_user(text)
                continue
            if role == "system":
                subtype = getattr(msg, "subtype", None) or ""
                if subtype == "away_summary" and self._repl_screen is not None:
                    self._repl_screen.transcript.append_system(
                        format_away_summary_for_display(msg),
                        style="muted",
                        render="markdown",
                    )
                elif subtype == "intent_forecast" and self._repl_screen is not None:
                    self._repl_screen.transcript.append_system(
                        getattr(msg, "content", "") or "",
                        style="light",
                        render="markdown",
                    )
                elif (
                    subtype
                    in {
                        "goal_set",
                        "goal_cleared",
                        "goal_evaluation",
                        "goal_achieved",
                        "goal_evaluator_error",
                    }
                    and self._repl_screen is not None
                ):
                    self._repl_screen.transcript.append_system(
                        getattr(msg, "content", "") or "",
                        style=(
                            "error"
                            if subtype == "goal_evaluator_error"
                            else "light"
                            if subtype in {"goal_set", "goal_cleared", "goal_achieved"}
                            else "muted"
                        ),
                        render="markdown",
                    )
                continue
            if role == "assistant":
                text = _flatten_message_text(content)
                # Suppress NO_CONTENT_MESSAGE placeholder injected for empty
                # assistant responses — matches live-chat behaviour.
                from clawcodex_ext.types.messages import NO_CONTENT_MESSAGE

                if text and text != NO_CONTENT_MESSAGE:
                    if self._repl_screen is not None:
                        self._repl_screen.transcript.append_assistant(
                            text,
                            agent_name=agent_type,
                        )
                # Replay tool_use / tool_result / thinking blocks from the content list.
                if isinstance(content, list):
                    for item in content:
                        # Normalise item to a dict so the replay logic below
                        # handles both raw-dict items and dataclass blocks
                        # (TextBlock, ToolUseBlock, etc.) uniformly.
                        if isinstance(item, dict):
                            d = item
                        elif hasattr(item, "type"):
                            # Convert dataclass content block to dict
                            from clawcodex_ext.types.content_blocks import (
                                content_block_to_dict,
                            )

                            d = content_block_to_dict(item)
                        else:
                            continue
                        kind = d.get("type")
                        if kind == "tool_use":
                            if self._repl_screen is not None:
                                self._repl_screen.transcript.append_tool_event(
                                    kind="tool_use",
                                    tool_name=d.get("name", ""),
                                    tool_input=d.get("input"),
                                    tool_output=None,
                                    is_error=False,
                                    error=None,
                                    tool_use_id=d.get("id"),
                                )
                        elif kind == "tool_result":
                            if self._repl_screen is not None:
                                self._repl_screen.transcript.append_tool_event(
                                    kind="tool_result",
                                    tool_name="",
                                    tool_input=None,
                                    tool_output=d.get("content"),
                                    is_error=bool(d.get("is_error")),
                                    error=None,
                                    tool_use_id=d.get("tool_use_id"),
                                )
                        elif kind == "thinking":
                            # Replay thinking content from historical session.
                            thinking_text = d.get("thinking", "") or ""
                            if thinking_text and self._repl_screen is not None:
                                self._repl_screen.transcript.append_thinking_chunk(thinking_text)
                        elif kind == "redacted_thinking":
                            # Replay redacted thinking with redacted=True.
                            data = d.get("data", "") or ""
                            if data and self._repl_screen is not None:
                                self._repl_screen.transcript.append_thinking_chunk(data, redacted=True)
        if not self._replay_exit_snapshot_from_start and self._repl_screen is not None:
            self._exit_snapshot_replay_pending = True
            try:
                self.call_after_refresh(self._mark_exit_snapshot_start)
            except Exception:
                self._mark_exit_snapshot_start()

    def _mark_exit_snapshot_start(self) -> None:
        if self._repl_screen is None:
            return
        self._exit_snapshot_start_index = self._repl_screen.transcript.message_count
        self._exit_snapshot_replay_pending = False

    def _slash_command_words(self) -> list[str]:
        return build_command_words(self.workspace_root, self.tool_context)

    def _slash_command_suggestions(self) -> list[CommandSuggestion]:
        return build_command_suggestions(self.workspace_root, self.tool_context)

    def _message_history_provider(self) -> list[str]:
        """Return previous user messages from the session conversation."""
        try:
            messages = getattr(self.session, "conversation", None)
            if messages is None:
                return []
            msgs = getattr(messages, "messages", [])
            from clawcodex_ext.types.messages import UserMessage

            result: list[str] = []
            for msg in msgs:
                if isinstance(msg, UserMessage):
                    content = msg.content
                    if isinstance(content, str):
                        result.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if hasattr(block, "text"):
                                result.append(block.text)
            return result
        except Exception:
            return []

    def _available_agents(self) -> list[Any]:
        """Return available agent definitions for ``@agent-<type>`` completion."""
        try:
            from clawcodex_ext.agent.agent_definitions import get_built_in_agents
        except Exception:
            return []

        extra = getattr(
            getattr(self.tool_context, "options", None),
            "agent_definitions",
            None,
        )
        if isinstance(extra, dict):
            active = extra.get("active_agents")
            if isinstance(active, list) and active:
                return list(active)

        try:
            from clawcodex_ext.agent.load_agents_dir import get_agents_for_mentions

            return get_agents_for_mentions(
                self.workspace_root,
                tool_context=self.tool_context,
                runtime_context=getattr(self, "runtime_context", None),
            )
        except Exception:
            from clawcodex_ext.agent.agent_definitions import get_built_in_agents

            return list(get_built_in_agents())

    def _post_to_screen(self, message: Any) -> None:
        target = self._repl_screen or self
        try:
            target.post_message(message)
        except Exception:  # nosec B110
            pass


class ClawCodexExtTUI(ClawCodexTUI):
    """Downstream TUI app class.

    Kept as a named subclass so downstream entrypoints can depend on the
    extension surface without changing the upstream-shaped ``ClawCodexTUI``
    facade.
    """

    pass
