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

"""Main REPL screen, rebuilt on top of :class:`FullscreenLayout`.

Composition (Phase 1):

* ``scroll``   — :class:`StartupHeader` + :class:`TranscriptView`.
* ``overlay``  — reserved for the in-region tool permission overlay
  (Phase 2 will mount ``PermissionRequest`` here without taking over
  the whole screen like the Phase 1 modal does).
* ``modal``    — reserved for centered slash-JSX panels (Phase 3).
* ``bottom``   — :class:`StatusLine` + :class:`PromptInput`.

Event wiring:

* :class:`PromptSubmitted` → slash dispatcher → agent bridge.
* :class:`AssistantChunk`  → active streaming row.
* :class:`AssistantMessage` → finalise the streaming row.
* :class:`ToolEventMessage` → transcript tool rows.
* :class:`PermissionRequested` → push :class:`PermissionModal`.
"""

from __future__ import annotations
# pylint: disable=W0201,W0621

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from textual.app import ComposeResult
from textual.screen import Screen

from ..messages import (
    AdvisorEventMessage,
    AgentRunFinished,
    AgentRunStarted,
    AskUserQuestionRequested,
    AskUserQuestionResolved,
    AssistantChunk,
    AssistantMessage,
    MultiModelEvent,
    PermissionModeChanged,
    PermissionModeCycleRequested,
    PermissionRequested,
    PermissionResolved,
    PromptDraftChanged,
    StateChanged,
    SystemNotice,
    ThinkingChunk,
    ToolEventMessage,
)
from ..a11y import LiveRegion, aria_label
from ..widgets.fullscreen_layout import FullscreenLayout
from ..widgets.header import StartupHeader
from ..widgets.prompt_input import PromptInput, PromptSubmitted
from ..widgets.status_line import StatusLine
from ..widgets.transcript_view import Transcript
from ..messages import QueuedPromptReady
from ..widgets.multimodel import MultiModelLivePanel

if TYPE_CHECKING:  # pragma: no cover
    from ..app import ClawCodexTUI
    from ..commands import CommandSuggestion


from textual.binding import Binding


class REPLScreen(Screen):
    """Composes the interactive TUI layout."""

    BINDINGS = [
        ("ctrl+l", "clear_transcript", "Clear transcript"),
        ("ctrl+t", "toggle_thinking", "Toggle thinking"),
        Binding("shift+tab", "cycle_permission_mode", "Cycle permission mode", priority=True),
    ]

    DEFAULT_CSS = """
    REPLScreen {
        layout: vertical;
    }
    """

    def __init__(
        self,
        *,
        version: str,
        provider: str,
        model: str,
        workspace_root: Path,
        words_provider: Callable[[], list[str]],
        suggestions_provider: Callable[[], list["CommandSuggestion"]] | None = None,
        message_history_provider: Callable[[], list[str]] | None = None,
        agents_provider: Callable[[], list[Any]] | None = None,
        provider_instance: object | None = None,
        initial_history: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._version = version
        self._provider = provider
        self._model = model
        self._workspace_root = Path(workspace_root)
        self._words_provider = words_provider
        self._suggestions_provider = suggestions_provider
        self._message_history_provider = message_history_provider
        self._agents_provider = agents_provider
        self._provider_instance = provider_instance
        self._initial_history = initial_history

        self.header_widget = StartupHeader(
            version=version,
            model=model,
            provider=provider,
            workspace_root=self._workspace_root,
        )
        self.transcript = Transcript()
        self.status_bar = StatusLine(
            provider=provider,
            model=model,
            workspace_root=self._workspace_root,
            provider_instance=provider_instance,
        )
        self.prompt_input = PromptInput(
            words_provider=words_provider,
            suggestions_provider=suggestions_provider,
            message_history_provider=message_history_provider,
            agents_provider=agents_provider,
            cwd=self._workspace_root,
            initial_history=initial_history,
        )
        # ARIA live region — stays height: 1 and only announces the
        # most recent status change. Mounted just above the status
        # bar so it's adjacent to the prompt for single-sweep reads.
        self.live_region = LiveRegion(aria_label="Status")
        # Note: intentionally NOT setting tooltip via aria_label on transcript
        # to avoid the hover-tooltip popup that follows the mouse cursor.
        aria_label(self.prompt_input, "Prompt input — type a message, or '/' for commands")
        # ``layout`` is a reserved attribute on Textual screens; use a
        # private attribute for our parity shell.
        self._fullscreen = FullscreenLayout()

    def compose(self) -> ComposeResult:
        yield self._fullscreen

    def on_mount(self) -> None:
        self._fullscreen.scroll_region().mount(self.header_widget)
        self._fullscreen.scroll_region().mount(self.transcript)
        self._fullscreen.bottom_region().mount(self.live_region)
        self._fullscreen.bottom_region().mount(self.status_bar)
        self._fullscreen.bottom_region().mount(self.prompt_input)
        # Bind the status line to app state so the spinner / queue count
        # reflect the authoritative agent state.
        app: "ClawCodexTUI" = self.app  # type: ignore[assignment]
        if hasattr(app, "app_state"):
            self.status_bar.bind_state(app.app_state)
        # Drive the footer's "esc to interrupt" hint off the same busy
        # signal as the spinner (single source of truth — F-38).
        self.status_bar.bind_footer(self.prompt_input._footer)
        # Set the initial permission mode on the status bar.
        try:
            from src.permissions.modes import to_external_permission_mode

            ctx = getattr(app, "tool_context", None)
            if ctx is not None and ctx.permission_context is not None:
                mode = to_external_permission_mode(ctx.permission_context.mode or "default")
                self.status_bar.set_permission_mode(mode)
        except Exception:  # nosec B110
            pass
        # Attach the live region to the app's announcer so every
        # cross-screen announcement mirrors into this REPL.
        if hasattr(app, "announcer"):
            app.announcer.bind_region(self.live_region)
        self.prompt_input.focus_input()
        self.transcript.append_system(
            "Ready. Type a prompt, or '/' for commands. Ctrl+D, /exit, or /repl to leave the Textual TUI.",
        )

    # ---- actions ----
    def action_clear_transcript(self) -> None:
        self.transcript.clear_transcript()

    def action_toggle_thinking(self) -> None:
        """Ctrl+T: toggle thinking content visibility in all thinking rows."""
        from src.tui.widgets.messages.assistant_thinking import (
            AssistantThinkingMessage,
        )

        expanded = True
        for row in self.transcript.query(AssistantThinkingMessage):
            row.toggle()
            expanded = row.expanded

        label = "expanded" if expanded else "collapsed"
        self.transcript.append_system(f"Thinking content: {label}", style="muted")

    def action_cycle_permission_mode(self) -> None:
        """Shift+Tab: cycle permission mode. Delegates to app."""
        app: "ClawCodexTUI" = self.app  # type: ignore[assignment]
        if hasattr(app, "action_cycle_permission_mode"):
            app.action_cycle_permission_mode()

    def on_permission_mode_cycle_requested(self, _: PermissionModeCycleRequested) -> None:
        """Handle Shift+Tab posted from PromptInput."""
        self.action_cycle_permission_mode()

    def on_permission_mode_changed(self, message: PermissionModeChanged) -> None:
        """Handle a mode change posted by the runtime permission
        controller. Updates the status bar and appends a transcript
        line so the user sees the new mode regardless of which entry
        point fired (Shift+Tab, ``/permissions`` picker, or the
        ``permissions_command`` registry path).
        """
        self.status_bar.set_permission_mode(message.mode)
        self.transcript.append_system(f"Permission mode: {message.mode}", style="muted")

    # ---- prompt submission ----
    def on_prompt_submitted(self, message: PromptSubmitted) -> None:
        app: "ClawCodexTUI" = self.app  # type: ignore[assignment]
        text = message.text
        if text.startswith("/"):
            if app.handle_local_slash_command(text, self.transcript):
                return
        if text.startswith("!"):
            # C4 bash-mode: direct execution, no agent turn, no busy
            # spinner (the echo row mounts synchronously; the worker
            # fills in the output when the command finishes).
            app.run_bash_mode(text[1:], self.transcript)
            return
        if text.startswith("#"):
            # C9 memory-note append: persist the note to ~/.claude/CLAUDE.md
            # and show an acknowledgement in the transcript, no agent turn.
            from src.services.memory_append import append_memory_note, pick_saving_message

            note = text[1:]
            ok = append_memory_note(str(Path.home() / ".claude" / "CLAUDE.md"), note)
            if ok:
                msg = pick_saving_message()
                self.transcript.append_system(msg, style="success")
            else:
                self.transcript.append_system("Failed to save memory note", style="error")
            return
        # F-89: expand @agent-name mentions in TUI.
        try:
            from clawcodex_ext.agent.load_agents_dir import get_agents_for_mentions
            from src.command_system.input_processing import (
                expand_agent_mentions,
                find_unknown_agent_mentions,
                format_at_mention_attachments,
                format_unknown_agent_mention_error,
            )

            agents = get_agents_for_mentions(
                getattr(app, "workspace_root", None) or ".",
                tool_context=getattr(app, "tool_context", None),
                runtime_context=getattr(app, "runtime_context", None),
            )
            unknown = find_unknown_agent_mentions(text, agents)
            if unknown:
                self.transcript.append_system(
                    format_unknown_agent_mention_error(unknown, agents),
                    style="error",
                )
                return
            agent_attachments = expand_agent_mentions(text, agents)
            if agent_attachments:
                extra = format_at_mention_attachments(agent_attachments)
                if extra:
                    text = f"{extra}\n{text}"
        except Exception:  # nosec B110
            pass  # best-effort
        self.transcript.append_user(text)
        self.status_bar.set_busy()
        self.status_bar.bump_turn()
        app.submit_to_agent(text)

    def on_prompt_draft_changed(self, message: PromptDraftChanged) -> None:
        app: "ClawCodexTUI" = self.app  # type: ignore[assignment]
        forecast = getattr(app, "_intent_forecast_controller", None)
        if forecast is not None:
            forecast.on_prompt_draft_changed(message.text)

    # ---- queued-prompt drain ----
    def on_queued_prompt_ready(self, _: QueuedPromptReady) -> None:
        """Drain one queued prompt when the bridge becomes idle.

        FIFO, one per turn: after the UI handler verifies the bridge is
        idle and the queue is non-empty, it pops the oldest prompt and
        submits it.  The bridge's ``_finish`` filter is best-effort; the
        UI handler is authoritative, so a spurious post (queue cleared
        by ESC between the worker-side check and this handler) is a
        safe no-op.
        """
        app: "ClawCodexTUI" = self.app  # type: ignore[assignment]
        if app._agent_bridge.busy:
            return
        if not app.app_state.queued_prompts:
            return
        text = app.app_state.queued_prompts.pop(0)
        self.transcript.append_user(text)
        self.status_bar.set_busy()
        self.status_bar.bump_turn()
        app.submit_to_agent(text)

    # ---- agent message handlers ----
    def on_agent_run_started(self, _: AgentRunStarted) -> None:
        self.status_bar.set_busy()
        app = self.app
        controller = getattr(app, "_away_summary_controller", None)
        if controller is not None:
            controller.on_run_start()
        # F-9: start the goal controller so it knows a new assistant
        # turn is about to begin.
        goal_controller = getattr(app, "_goal_controller", None)
        if goal_controller is not None:
            try:
                goal_controller.on_run_start()
            except Exception:  # nosec B110
                pass
        forecast = getattr(app, "_intent_forecast_controller", None)
        if forecast is not None:
            forecast.on_run_start()

    def on_assistant_chunk(self, message: AssistantChunk) -> None:
        self.transcript.append_assistant_chunk(message.text, agent_name=message.agent_name)

    def on_thinking_chunk(self, message: ThinkingChunk) -> None:
        self.transcript.append_thinking_chunk(message.text)

    def on_assistant_message(self, message: AssistantMessage) -> None:
        self.transcript.append_assistant(message.text, agent_name=message.agent_name)

    def on_system_notice(self, message: SystemNotice) -> None:
        self.transcript.append_system(message.text, style=message.style, render="markdown")

    def on_multi_model_event(self, message: MultiModelEvent) -> None:
        if message.kind == "start":
            self._multimodel_panel = MultiModelLivePanel(list(message.results or []))
            self.transcript.mount(self._multimodel_panel)
        panel = getattr(self, "_multimodel_panel", None)
        if panel is None:
            return
        if message.kind == "progress":
            panel.progress(message.slot, message.chunk)
        elif message.kind == "complete" and message.result is not None:
            panel.complete(message.result)

    def on_tool_event_message(self, message: ToolEventMessage) -> None:
        self.transcript.append_tool_event(
            kind=message.kind,
            tool_name=message.tool_name,
            tool_input=message.tool_input,
            tool_output=message.tool_output,
            tool_use_id=message.tool_use_id,
            is_error=message.is_error,
            error=message.error,
        )

    def on_advisor_event_message(self, message: AdvisorEventMessage) -> None:
        self.transcript.append_advisor_event(
            kind=message.kind,
            tool_use_id=message.tool_use_id,
            advisor_model=message.advisor_model,
            text=message.text,
            error_code=message.error_code,
        )

    def on_agent_run_finished(self, message: AgentRunFinished) -> None:
        self.status_bar.set_idle()
        app = self.app
        controller = getattr(app, "_away_summary_controller", None)
        if controller is not None:
            controller.on_run_finish()
        forecast = getattr(app, "_intent_forecast_controller", None)
        if forecast is not None:
            forecast.on_run_finish()
        if message.error:
            self.transcript.append_system(f"error: {message.error}", style="error")
            if hasattr(app, "announcer"):
                app.announcer.announce(  # type: ignore[attr-defined]
                    f"Error: {message.error}", level="assertive"
                )
        elif controller is not None:
            controller.on_assistant_turn_complete()
        # F-9: drain the goal controller's continuation/budget_limit
        # injection and enqueue it so the queued-prompt drain feeds it
        # to the next agent run (auto-continuation).
        goal_controller = getattr(app, "_goal_controller", None)
        if goal_controller is not None:
            try:
                goal_controller.on_run_finish()
                injection = goal_controller.on_assistant_turn_complete()
                if injection is not None:
                    text = injection.get("text", "")
                    if text.strip():
                        app.app_state.queued_prompts.append(text)
                        from ..messages import QueuedPromptReady

                        self.post_message(QueuedPromptReady())
                    goal_controller.drain_pending_injection()
            except Exception:  # nosec B110
                pass
        self.prompt_input.set_enabled(True)
        self.call_after_refresh(self.prompt_input.focus_input)

    # ---- permission modal handlers ----
    def on_permission_requested(self, message: PermissionRequested) -> None:
        app: "ClawCodexTUI" = self.app  # type: ignore[assignment]
        state = getattr(app, "app_state", None)
        if state is None:
            return
        pending = state.pop_next_permission()
        if pending is None or pending.request_id != message.request_id:
            # State drift: search the queue for a matching id.
            for candidate in list(state.pending_permissions):
                if candidate.request_id == message.request_id:
                    pending = candidate
                    break
        if pending is None:
            return
        from .permission_modal import PermissionModal

        self.prompt_input.set_enabled(False)
        if hasattr(app, "announcer"):
            app.announcer.announce(
                f"Permission required: {pending.tool_name}",
                level="assertive",
            )
        app.push_screen(PermissionModal(pending))

    def on_permission_resolved(self, _: PermissionResolved) -> None:
        self.prompt_input.set_enabled(True)
        self.prompt_input.focus_input()

    # ---- AskUserQuestion modal handlers ----
    def on_ask_user_question_requested(self, message: AskUserQuestionRequested) -> None:
        app: "ClawCodexTUI" = self.app  # type: ignore[assignment]
        state = getattr(app, "app_state", None)
        if state is None:
            return
        # Find the pending entry by id; fall back to the head of the
        # queue if the worker thread posted the message before the
        # state enqueue (Textual's message pump order is not strict).
        pending = None
        for candidate in list(state.pending_ask_users):
            if candidate.request_id == message.request_id:
                pending = candidate
                break
        if pending is None and state.pending_ask_users:
            pending = state.pending_ask_users[0]
        if pending is None:
            return
        from .ask_user_question import AskUserQuestionModal

        self.prompt_input.set_enabled(False)
        if hasattr(app, "announcer"):
            app.announcer.announce(
                f"Question: {pending.questions[0].get('question', '') if pending.questions else ''}",
                level="assertive",
            )
        app.push_screen(AskUserQuestionModal(pending))

    def on_ask_user_question_resolved(self, _: AskUserQuestionResolved) -> None:
        self.prompt_input.set_enabled(True)
        self.prompt_input.focus_input()

    def on_state_changed(self, _: StateChanged) -> None:
        # Cheap coalesced refresh. The status line has its own timer so
        # there is nothing to do here yet; reserved for Phase 2 widgets
        # (queued-command pill, idle return dialog, etc.).
        pass
