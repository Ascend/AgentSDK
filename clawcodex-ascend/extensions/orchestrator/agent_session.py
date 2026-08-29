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
"""Session state and retry models for the Orchestrator Agent Runtime."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .provider_routing import RoutingSnapshot

if TYPE_CHECKING:
    from .issue import Issue
    from .issue_state_cache import IssueStateCache
    from .workspace import Workspace

logger = logging.getLogger(__name__)

# ─── pdeath_sig helper ────────────────────────────────────────────────
# When the orchestrator is killed abruptly (SIGKILL, segfault, OOM),
# child processes (hooks, verification) become orphans. PR_SET_PDEATHSIG
# asks the kernel to deliver SIGTERM to children when the parent dies.


def _set_pdeathsig() -> None:
    """Set PR_SET_PDEATHSIG so child receives SIGTERM if parent dies."""
    try:
        import ctypes
        import signal as _signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, _signal.SIGTERM)
    except Exception:  # nosec B110 -- Linux-only best-effort hardening
        pass


# If the agent runs this many consecutive turns without making any
# file changes, the runner assumes it is stuck (e.g. the issue
# deliverables already exist in the base branch / workspace) and
# force-completes the session to avoid wasting API calls and retries.
_NOOP_DETECTION_MAX_TURNS = 5

# F-45: tool-event audit log rotation threshold. When events.ndjson
# exceeds this size on next append, rotate to events.ndjson.1 (single
# generation, overwrite). v2.14 will hook a cron for 7-day cleanup.
_TOOL_EVENT_LOG_ROTATE_BYTES = 50 * 1024 * 1024

# F-40 root-cause fix: after this many consecutive turns where the
# agent makes ONLY read-only tool calls (Bash, Read, Grep, …) without
# a single modifying tool call (Write / Edit / …) AND without changing
# the workspace (no new untracked or modified files), the session is
# considered stuck in an investigation spiral and terminated with
# ``session_end_reason="read_only_loop"``.  The threshold is generous
# because genuine development also involves exploration; the guard is
# meant to catch degenerate cases (F-40's 100+ Python-debug Bash calls
# that spanned multiple outer-loop turns without any code change).
_MAX_READ_ONLY_TURNS = 4

# F-40 root-cause fix: tool names that modify workspace files.
# Only Write / Edit tools count toward ``has_made_progress`` so the
# stagnation guard can distinguish "exploring the codebase" turns
# from actual code-production work.  ``Bash`` is intentionally omitted
# because it can be used for both read (ls / grep / cat) and write
# (git add / rm / mv) and trying to classify it at this level would
# require deep output analysis that is better done elsewhere.
_MODIFYING_TOOL_NAMES = frozenset(
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

# F-40 root-cause fix: tool names that are unambiguously read-only
# (exploration / diagnostics). Bash is deliberately excluded because
# shell commands may modify the workspace and must be checked against
# workspace state before contributing to a read-only streak.
_READ_ONLY_TOOL_NAMES = frozenset(
    {
        "Read",
        "Grep",
        "Glob",
        "WebFetch",
        "WebSearch",
        "TodoWrite",
        "TaskStop",
    }
)

# Mega-turn early-stop tuning (see the run loop). Check cadence is cheap
# (one `git status` per interval); the idle threshold must be long enough
# that a coordinator between worker waves (~1-3 min gaps observed) is not
# cut off, and short enough to save most of the wasted wall-clock before
# the run timeout.
_MEGATURN_CHECK_EVERY_S = 60.0
_MEGATURN_IDLE_STOP_S = 1800.0

_ORCHESTRATOR_INTERNAL_PATH_PREFIXES = (
    ".orchestrator_control/",
    ".run_control/",
    ".reports/",
    ".event_logs/",
)
_ORCHESTRATOR_INTERNAL_PATHS = frozenset(prefix.rstrip("/") for prefix in _ORCHESTRATOR_INTERNAL_PATH_PREFIXES)


def _megaturn_idle_stop_enabled(session: Any) -> bool:
    """Swarm completion is owned by its execution-evidence gate."""
    return str(getattr(session, "run_kind", "")).strip().lower() != "swarm"


def _is_orchestrator_internal_path(path: str | None) -> bool:
    if not path:
        return False
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized in _ORCHESTRATOR_INTERNAL_PATHS or normalized.startswith(
        _ORCHESTRATOR_INTERNAL_PATH_PREFIXES,
    )


def _has_user_visible_status_changes(status_entries: list[Any]) -> bool:
    for entry in status_entries:
        paths = (
            getattr(entry, "path", None),
            getattr(entry, "original_path", None),
        )
        if any(path and not _is_orchestrator_internal_path(path) for path in paths):
            return True
    return False


@dataclass
class AgentSession:
    """One active issue run."""

    issue: Issue
    workspace: Workspace
    turn_count: int = 0
    status: str = "running"  # running, completed, failed
    output_text: str = ""
    # Lifecycle control
    paused: bool = False
    paused_at: float | None = None
    pause_reason: str = ""
    pause_resume_event: "asyncio.Event | None" = None
    # Event stream for CLI tail command
    event_queue: "asyncio.Queue | None" = None
    prompt_override: str | None = None
    # F-124: resolved pre-dispatch clarification context copied from the
    # persistent IssueRecord before the run starts.
    clarification_question: str | None = None
    clarification_answer: str | None = None
    clarification_source: str | None = None
    coordinator_mode: bool | None = None
    # F-49 Phase 1: Unix domain socket for live operator control. None if
    # the socket failed to start (or was disabled by configuration). When
    # set, the runner broadcasts every dispatched event and polls for
    # control commands at turn boundaries. Defensive: all socket ops
    # are wrapped in try/except so a broken socket never kills the
    # agent run.
    control_socket: Any | None = None
    # Public path of the listening socket. Stored on the session so the
    # CLI control commands (pause/resume/stop/inject/takeover) can
    # discover it via the registry without scanning the workspace tree.
    control_socket_path: str | None = None
    run_kind: str = "issue"
    run_id: str | None = None
    # F-129 Phase 4: RuntimeTaskRegistry for this session. When set,
    # the agent_id (== run_id) is wired into the QueryConfig so
    # _drain_pending_user_messages fires at ToolResult boundaries,
    # enabling real-time inject via queue_pending_message.
    _runtime_tasks: Any | None = None
    summary_comment_id: str | None = None
    tool_count: int = 0
    # Set once a source-code edit or another user-visible workspace change
    # has been observed. Declaring it on the session keeps all runtime users
    # on the same contract instead of adding an ad-hoc attribute later.
    has_made_progress: bool = False
    verification_status: str | None = None
    verification_output: str | None = None
    report_path: str | None = None
    # F-105: per-session cache for the tracker poll in ``_should_continue``.
    # Initialised by ``AgentRunner.run()`` from
    # ``agent_config.perf_should_continue_skip_turns``. When ``None`` the
    # runner falls back to the pre-F-105 behaviour of always polling.
    state_cache: "IssueStateCache | None" = None
    # F-120: list of files git left in conflict state. Populated by
    # ``Orchestrator._prepare_rebase_session`` from
    # ``IssueRecord.conflict_files`` when ``run_kind == "agent_rebase"``.
    # The prompt builder injects these into the conflict-resolution
    # prompt so the agent knows exactly which files need ``git add``
    # before ``git rebase --continue``.
    conflict_files: tuple[str, ...] | None = None
    # F-45: canonical path to ~/.clawcodex/tool-events/{run_id}/events.ndjson.
    # Set in AgentRunner.run() at session start; consumed by
    # report_writer.write() to dual-write the NDJSON to the persistent layer.
    tool_events_path: str | None = None
    # F-49 Phase 0.1: session-transcript storage for conversation recording.
    # Lazy-initialized in run() via SessionStorage. The agent_runner buffers
    # per-turn blocks here and emits exactly one AssistantMessage and (if
    # tool calls happened) one UserMessage per LLM turn at end-of-turn.
    _transcript_storage: Any | None = None
    # Accumulated assistant text in the current turn (concatenation of all
    # TextDelta events up to the first tool call OR up to SessionComplete
    # if no tool calls were emitted). Reset by _flush_turn_transcript().
    _transcript_asst_text: str = ""
    # Ordered list of ToolUseBlocks for the current turn, preserved in
    # event arrival order so the final AssistantMessage interleaves text
    # and tool_use blocks exactly as the LLM emitted them. Reset by
    # _flush_turn_transcript().
    _transcript_tool_uses: list[Any] = field(default_factory=list)
    # Pending ToolResultBlocks waiting to be paired with their ToolUseBlock
    # when the LLM turn ends. Keyed by tool_use_id so out-of-order arrivals
    # are handled correctly. Reset by _flush_turn_transcript().
    _transcript_pending_results: dict[str, Any] = field(default_factory=dict)
    # Ordered list of tool_use_ids for which results have been received
    # this turn. Used to emit the final UserMessage's ToolResultBlocks in
    # tool_use order, not arrival order. Reset by _flush_turn_transcript().
    _transcript_result_order: list[str] = field(default_factory=list)
    attempt: int = 1
    issue_attempt: int = 1
    followup_attempt: int = 1
    # 429-aware backoff bookkeeping. ``consecutive_429_count`` is
    # incremented on each rate-limit hit and reset on the next
    # successful turn. ``total_429_backoff_seconds`` is the cumulative
    # sleep time spent in in-turn backoff (visible on the dashboard
    # and useful for cost analysis). ``rate_limit_pending_turn``
    # records the turn number being re-issued after a 429 sleep so
    # the SessionComplete handler skips its turn_number increment.
    consecutive_429_count: int = 0
    total_429_backoff_seconds: float = 0.0
    rate_limit_pending_turn: int | None = None
    debug_log_path: str | None = None
    last_agent_event_at: float | None = None
    last_agent_event: str | None = None
    last_tool_name: str | None = None
    timeout_deadline_at: float | None = None
    # F-09 / F-40 root-cause fix: capture the reason the session ended
    # before the registry writeback. ``session_end_reason`` is one of
    # ``task_complete`` / ``noop_completed`` / ``budget_exhausted`` /
    # ``stagnation`` / ``loop_detected`` / ``failed`` / ``paused`` /
    # ``cancelled``; ``session_end_summary`` is a short human-readable
    # explanation surfaced in dashboard + registry.  The agent_runner
    # sets these on the appropriate exit branch so the orchestrator
    # can pass them to ``IssueRegistry.update_report`` instead of
    # silently inheriting ``status="completed"``.
    session_end_reason: str | None = None
    session_end_summary: str = ""
    # F-?? retry context: list of run_ids from previous failed attempts.
    # Populated by orchestrator._launch_issue from the registry; consumed
    # by PromptBuilder.render() to inject a hint into the agent's prompt
    # so it can Read() past transcripts.
    previous_run_ids: list[str] = field(default_factory=list)
    _routing_snapshot: RoutingSnapshot | None = None
    # F-129: threading.Event used as a "pause gate" — when cleared,
    # the headless session's on_event callback blocks, preventing
    # further LLM API calls while paused. Set = running, clear = paused.
    _pause_gate: Any = None
    # Callback invoked by _drain_control_commands when pause/resume is
    # processed via the socket path, so the orchestrator can sync the
    # registry status. Signature: (issue_id: str, paused: bool, reason: str) -> None.
    # Set by orcheator._launch_issue; None for test sessions.
    _on_pause_state_change: Any | None = None

    def _save_json_snapshot(self) -> None:
        """F-49 Phase 0.4.5: write a ``src.agent.Session``-compatible
        ``.json`` snapshot so ``Session.load()`` can fast-path on
        ``--resume`` instead of replaying the full JSONL transcript.

        The snapshot is built from the JSONL transcript (which is the
        authoritative source) rather than from ``AgentSession`` fields
        that don't carry a full Conversation.  Best-effort: failures
        are logged but never propagated.

        IMPORTANT: this writes the ``{sid}.json`` file directly instead
        of calling ``CoreSession.save()`` to avoid the side-effect in
        ``save_to_session_storage()`` which overwrites the SessionStorage
        metadata (title, cwd, etc.) that ``run()`` already initialised
        via ``session._transcript_storage.init_metadata()``.
        """
        if not self.run_id:
            return
        try:
            import json as _json
            from clawcodex_ext.types.messages import (
                message_from_dict,
                message_to_dict,
            )

            # Phase 3: route through Protocol-injected SessionStorage when
            # available; otherwise fall back to the legacy direct import.
            storage = getattr(self, "_transcript_storage", None)
            messages = []
            if storage is not None:
                try:
                    blocks = storage.load_messages()
                    for blk in blocks:
                        try:
                            msg = message_from_dict(blk)
                            messages.append(message_to_dict(msg))
                        except Exception:
                            logger.debug(
                                "Skipping malformed transcript row run_id=%s",
                                self.run_id,
                                exc_info=True,
                            )
                except Exception:
                    logger.debug(
                        "Failed to load transcript messages run_id=%s",
                        self.run_id,
                        exc_info=True,
                    )

            # Build cost block — mirrors Session._snapshot_cost_block()
            # in src/agent/session.py so restore_cost_state_for_session()
            # can restore bootstrap accumulators on --resume.
            cost_block: dict = {}
            try:
                # Reuse the canonical cost snapshot writer so the resume
                # reader and this compatibility snapshot cannot drift.
                from clawcodex_ext.services.cost_restore import build_cost_block

                cost_block = build_cost_block()
            except Exception:
                # Best-effort: cost block is optional; restore tolerates
                # missing fields with defaults of 0.
                logger.debug(
                    "Failed to build cost snapshot run_id=%s",
                    self.run_id,
                    exc_info=True,
                )

            # Write the .json snapshot directly — do NOT call
            # CoreSession.save() because it triggers
            # save_to_session_storage() which overwrites the
            # SessionStorage metadata (title, cwd) that run()
            # already initialised.
            # Phase 3: Conversation wrapper is no longer needed (we
            # just persist messages+metadata directly).
            routing = self._routing_snapshot or RoutingSnapshot("", "")
            snapshot_data = {
                "session_id": self.run_id,
                "provider": routing.provider_name,
                "model": routing.model,
                "conversation": {"messages": messages, "max_history": 0},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "cost": cost_block,
            }
            session_dir = Path.home() / ".clawcodex" / "sessions" / str(self.run_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = session_dir / "session.json"
            with open(snapshot_path, "w", encoding="utf-8") as f:
                _json.dump(snapshot_data, f, indent=2)
        except Exception:
            logger.exception(
                "F-49 Phase 0.4.5: failed to write .json snapshot run_id=%s",
                self.run_id,
            )


@dataclass
class RetryItem:
    """Item queued for retry."""

    issue_id: str
    attempt: int
    delay_seconds: float
    identifier: str = ""
    error: str = ""
    worker_host: str | None = None
    workspace_path: str = ""
    scheduled_at: float = field(default_factory=time.time)
