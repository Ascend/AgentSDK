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
"""Per-run and per-turn preparation for the Orchestrator Agent Runtime."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .agent_session import AgentSession

logger = logging.getLogger(__name__)

NOOP_DETECTION_MAX_TURNS = 5
MAX_READ_ONLY_TURNS = 4
MEGATURN_CHECK_EVERY_S = 60.0
MEGATURN_IDLE_STOP_S = 1800.0

MODIFYING_TOOL_NAMES = frozenset(
    {
        "Write",
        "Edit",
        "FileWrite",
        "FileWriteTool",
        "FileEdit",
        "FileEditTool",
        "WriteTool",
        "EditTool",
    }
)
READ_ONLY_TOOL_NAMES = frozenset({"Read", "Bash", "Grep", "Glob", "WebFetch", "WebSearch", "TodoWrite", "TaskStop"})


@dataclass
class RunState:
    """Counters and guards shared by all turns of one AgentRunner run."""

    sink: Any = None
    diagnostics_callback: Callable[["AgentSession"], None] | None = None
    turn_number: int = 0
    tool_count: int = 0
    consecutive_clean_turns: int = 0
    no_work_streak: int = 0
    read_only_streak: int = 0
    tool_signature_history: list[str] = field(default_factory=list)
    max_no_op_turns: int = 3
    loop_window: int = 5
    loop_threshold: int = 3

    def update_diagnostics(self, session: "AgentSession") -> None:
        session.tool_count = self.tool_count
        if self.diagnostics_callback is None:
            return
        try:
            self.diagnostics_callback(session)
        except Exception:
            logger.exception("run diagnostics callback failed issue_id=%s", session.issue.id)


@dataclass
class TurnState:
    """Mutable state for one QueryRunner.stream invocation."""

    output: str = ""
    has_tool_calls: bool = False
    has_modifying_tool: bool = False
    tool_names: list[str] = field(default_factory=list)
    tool_count: int = 0
    pending_tool_results: int = 0
    cap_reached: bool = False
    megaturn_next_check_at: float = field(default_factory=lambda: time.monotonic() + MEGATURN_CHECK_EVERY_S)
    megaturn_workspace_signature: str | None = None
    megaturn_workspace_changed_at: float = field(default_factory=time.monotonic)
    megaturn_stop: bool = False


class AgentTurnMixin:
    """Prepare run state, prompts, transcript services and QueryRunner."""

    def _create_run_state(
        self,
        progress_reporter: Any = None,
        diagnostics_callback: Callable[["AgentSession"], None] | None = None,
    ) -> RunState:
        """Create guarded counters from the workflow's AgentConfig."""
        return RunState(
            sink=progress_reporter,
            diagnostics_callback=diagnostics_callback,
            max_no_op_turns=max(1, int(getattr(self.agent_config, "max_no_op_turns", 3) or 3)),
            loop_window=max(2, int(getattr(self.agent_config, "loop_detection_window", 5) or 5)),
            loop_threshold=max(
                2,
                int(getattr(self.agent_config, "loop_detection_threshold", 3) or 3),
            ),
        )

    async def _initialize_run(
        self,
        session: "AgentSession",
        workflow: Any,
        *,
        comment_tracker: Any = None,
        progress_reporter: Any = None,
        diagnostics_callback: Callable[["AgentSession"], None] | None = None,
    ) -> tuple[RunState, dict[str, Any]]:
        """Initialise session-scoped infrastructure before the first turn."""
        if session.run_id is None:
            session.run_id = self._build_run_id(session)
        self._initialize_runtime_tasks(session)
        session._snapshot_provider = self.agent_config.provider or ""
        session._snapshot_model = self.agent_config.model or ""
        session._pause_gate = threading.Event()
        session._pause_gate.set()
        if session.state_cache is None:
            from .issue_state_cache import IssueStateCache

            session.state_cache = IssueStateCache(
                stable_skip_turns=max(
                    0,
                    int(
                        getattr(
                            self.agent_config,
                            "perf_should_continue_skip_turns",
                            3,
                        )
                    ),
                )
            )
        if comment_tracker is not None and session.issue.id:
            await self._post_summary_placeholder(session, comment_tracker)

        os.environ["CLAWCODEX_PROVIDER_REQUEST_DELAY_MS"] = str(self.agent_config.delay_between_requests_ms)
        self._resolve_protocols()
        self._set_run_log_context(session)
        context = self._build_session_context(session, workflow)
        session.has_made_progress = False
        state = self._create_run_state(progress_reporter, diagnostics_callback)
        state.update_diagnostics(session)
        return state, context

    @staticmethod
    def _initialize_runtime_tasks(session: "AgentSession") -> None:
        if session._runtime_tasks is not None or not session.run_id:
            return
        try:
            from clawcodex_ext.task_registry import RuntimeTaskRegistry
            from src.tasks.local_agent import LocalAgentTaskState

            registry = RuntimeTaskRegistry()
            registry.upsert(
                LocalAgentTaskState(
                    id=session.run_id,
                    agent_id=session.run_id,
                    status="running",
                    description=(f"orchestrator-{session.issue.identifier or session.issue.id}"),
                    prompt="",
                )
            )
            session._runtime_tasks = registry
        except Exception:
            logger.debug(
                "Failed to init runtime_tasks for run_id=%s",
                session.run_id,
                exc_info=True,
            )

    def _set_run_log_context(self, session: "AgentSession") -> None:
        from .logging_setup import set_log_context

        set_log_context(
            issue_id=str(session.issue.id),
            run_id=str(session.run_id or ""),
            issue_identifier=str(session.issue.identifier),
        )

    def _build_session_context(self, session: "AgentSession", workflow: Any) -> dict[str, Any]:
        from .debug_log import append_debug_event

        workspace = session.workspace
        context = {
            "issue_id": session.issue.id,
            "issue_identifier": session.issue.identifier,
            "workspace_path": str(workspace.path),
            "workflow": workflow,
            "run_id": session.run_id,
            "permission_mode": self.agent_config.permission_mode,
            "audit_log": self.agent_config.audit_log,
        }
        session.tool_events_path = (
            str(workspace.path / ".reports" / f"{session.run_id or 'unknown'}.events.ndjson")
            if self.agent_config.audit_log != "none"
            else None
        )
        session.debug_log_path = str(
            workspace.path / ".orchestrator_control" / "runs" / (session.run_id or "unknown") / "debug.ndjson"
        )
        append_debug_event(
            session.debug_log_path,
            "agent_runner.start",
            issue_id=session.issue.id,
            issue_identifier=session.issue.identifier,
            run_id=session.run_id,
            workspace=str(workspace.path),
            max_turns=self.max_turns,
            provider=self.agent_config.provider,
            permission_mode=self.agent_config.permission_mode,
        )
        return context

    def _build_turn_prompt(
        self,
        session: "AgentSession",
        turn_number: int,
        clarification_resolver: Any = None,
    ) -> str:
        """Render the initial issue prompt or a continuation prompt."""
        from .prompt_builder import PromptBuilder, resolve_python_executable

        issue = session.issue
        workspace = session.workspace
        python_executable = resolve_python_executable(
            workspace_path=getattr(workspace, "path", None),
            agent_cfg=self.agent_config,
            workspace_cfg=self.workspace_cfg,
            issue_executable=getattr(issue, "python_executable", "") or "",
        )
        if turn_number > 0:
            return PromptBuilder.build_continuation_prompt(
                turn_number=turn_number,
                max_turns=self.max_turns,
                issue_context=getattr(session, "_issue_context", None),
                session=session,
                python_executable=python_executable,
            )
        if session.prompt_override:
            session._system_prompt_append = ""
            session._issue_context = session.prompt_override
            return session.prompt_override

        clarification, question, options = self._clarification_context(session, clarification_resolver)
        system_prompt, user_prompt = PromptBuilder.render_parts(
            issue,
            clarification_context=clarification,
            pending_question=question,
            options=options,
            session=session,
            python_executable=python_executable,
            previous_run_ids=getattr(session, "previous_run_ids", None),
            conflict_files=getattr(session, "conflict_files", None),
        )
        session._system_prompt_append = system_prompt
        session._issue_context = user_prompt
        return user_prompt

    @staticmethod
    def _clarification_context(session: "AgentSession", resolver: Any) -> tuple[str, Any, Any]:
        from .prompt_builder import PromptBuilder

        answer = getattr(session, "clarification_answer", None)
        if answer:
            question = getattr(session, "clarification_question", None)
            return (
                PromptBuilder.build_clarification_context(
                    pending_question=question,
                    clarification_answer=answer,
                    answer_source=getattr(session, "clarification_source", None),
                ),
                question,
                None,
            )
        if resolver is None or not session.issue.id:
            return "", None, None
        resolved = resolver.get_answer(session.issue.id)
        item = resolver.get_item(session.issue.id)
        if resolved and resolved.answer:
            question = item.question if item is not None else None
            return (
                PromptBuilder.build_clarification_context(
                    pending_question=question,
                    clarification_answer=resolved.answer,
                    answer_source=resolved.source,
                ),
                question,
                None,
            )
        pending = item or getattr(resolver, "get_pending_feedback", lambda _id: None)(session.issue.id)
        if pending is None:
            return "", None, None
        options = pending.options if getattr(pending, "options", None) else None
        return (
            PromptBuilder.build_clarification_context(
                pending_question=pending.question,
                options=options,
            ),
            pending.question,
            options,
        )

    async def _ensure_turn_services(self, session: "AgentSession", prompt: str) -> None:
        """Initialise transcript/socket once and record the user prompt."""
        if not session.run_id:
            return
        if session._transcript_storage is None:
            try:
                self._resolve_protocols()
                session._transcript_storage = self._session_storage._upstream(session_id=session.run_id)
                session._transcript_storage.init_metadata(
                    model=self.agent_config.model or "",
                    cwd=str(session.workspace.path),
                    title=f"orchestrator-{session.issue.identifier or session.issue.id}",
                )
            except Exception:
                logger.exception("Failed to init transcript storage run_id=%s", session.run_id)
        if session.control_socket is None:
            try:
                from .control_socket import ControlSocket

                socket_path = Path(session.workspace.path) / ".run_control" / f"{session.run_id}.sock"
                control_socket = ControlSocket(socket_path)
                await control_socket.start()
                session.control_socket = control_socket
                session.control_socket_path = str(socket_path)
            except Exception:
                logger.exception("Failed to start control_socket run_id=%s", session.run_id)
                session.control_socket = None
        if session._transcript_storage is not None:
            try:
                from extensions.orchestrator_runtime.utils.messages_impl import (
                    TextBlock,
                    create_user_message,
                )

                session._transcript_storage.write_message(
                    create_user_message(
                        content=[TextBlock(text=prompt)],
                        origin="human",
                    )
                )
            except Exception:
                logger.exception("Failed to write transcript prompt run_id=%s", session.run_id)

    def _build_query_runner(self, session: "AgentSession", prompt: str) -> Any:
        """Build QueryConfig for one turn and return its QueryRunner."""
        from extensions.api.query import QueryConfig, QueryRunner

        config = QueryConfig(
            prompt=prompt,
            workspace=session.workspace.path,
            provider=self.agent_config.provider,
            model=self.agent_config.model,
            max_turns=self.max_turns,
            permission_mode=self.agent_config.permission_mode,
            run_id=session.run_id,
            debug_log_path=session.debug_log_path,
            env={
                **(getattr(self.agent_config, "env", None) or {}),
                "CLAUDE_CODE_COORDINATOR_MODE": ("1" if self._coordinator.is_active() else "0"),
            },
            timeout_s=self.agent_config.run_timeout_ms / 1000.0,
            stall_timeout_s=(getattr(self.agent_config, "stall_timeout_ms", 300_000) / 1000.0),
            stall_warn_s=(getattr(self.agent_config, "stall_warn_ms", 30_000) / 1000.0),
            append_system_prompt=getattr(session, "_system_prompt_append", None),
            agent_id=session.run_id,
            runtime_tasks=session._runtime_tasks,
            control_drain_fn=self._make_control_drain_fn(session),
            pause_wait_fn=self._make_pause_wait_fn(session),
            pause_gate=session._pause_gate,
        )
        return QueryRunner(config)


__all__ = [
    "MAX_READ_ONLY_TURNS",
    "MEGATURN_CHECK_EVERY_S",
    "MEGATURN_IDLE_STOP_S",
    "MODIFYING_TOOL_NAMES",
    "NOOP_DETECTION_MAX_TURNS",
    "READ_ONLY_TOOL_NAMES",
    "AgentTurnMixin",
    "RunState",
    "TurnState",
]
