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

"""Data models for the issue registry (split from ``issue_registry.py``)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from extensions.orchestrator.tracker import Intent


class IssueStatus(str, Enum):
    """Lifecycle stages of a tracked issue."""

    QUEUED = "queued"  # in candidate queue, awaiting dispatch
    PENDING = "pending"  # claimed, workspace created, not yet synced
    RUNNING = "running"  # agent session actively processing
    SYNCED = "synced"  # git sync completed (commit + push + PR)
    PENDING_REVIEW = "pending_review"  # awaiting human review (LocalTracker only)
    COMPLETED = "completed"  # session finished successfully
    FAILED = "failed"  # session ended with a non-success status
    PAUSED = "paused"  # agent session paused by operator control command
    ABANDONED = "abandoned"  # retry limit reached, gave up
    VERIFICATION_FAILED = "verification_failed"


TERMINAL_STATUSES = frozenset(
    {
        IssueStatus.COMPLETED,
        IssueStatus.FAILED,
        IssueStatus.ABANDONED,
        IssueStatus.VERIFICATION_FAILED,
    }
)


@dataclass
class IssueRecord:
    """One entry in the issue registry."""

    issue_id: str
    issue_identifier: str
    # File-format version guard. All writers ship the same release, so
    # this is a forward guard only: a registry written by a NEWER build
    # refuses to load in this build instead of silently misreading.
    schema_version: int = 1
    branch_name: str | None = None
    commit_sha: str | None = None
    pr_number: str | None = None
    pr_url: str | None = None
    # Wall-clock timestamp of the FIRST PR creation for this issue.
    # Set by mark_synced() only when the record had no pr_number yet, so
    # follow-up / review-feedback runs that reuse the same PR do NOT
    # overwrite the original "first PR created" time. Cleared by
    # reset_for_retry() so a deliberate retry restarts the clock.
    # Absent from registry.json files written before this field — _load()
    # handles back-compat via the known_fields filter (default None).
    pr_created_at: float | None = None
    base_branch: str = "main"
    workspace_strategy: str | None = None
    workspace_path: str | None = None
    base_commit_sha: str | None = None
    start_commit_sha: str | None = None
    previous_issue_id: str | None = None
    sequence_index: int | None = None
    status: IssueStatus = IssueStatus.PENDING
    report_path: str | None = None
    verification_status: str | None = None
    verification_output: str | None = None
    last_hook_error: str | None = None
    summary_comment_id: str | None = None
    # Root-cause fix: explicit end-of-session reason captured by
    # AgentRunner before returning. Possible values:
    #   None | "stagnation" | "loop_detected" | "noop_completed" |
    #   "budget_exhausted" | "user_abort" | "task_complete"
    # ``session_end_summary`` is a short human-readable string the
    # runner appends (e.g. "3 consecutive no-op turns").
    session_end_reason: str | None = None
    session_end_summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    attempt_count: int = 0
    # Clarification-related fields (for three-channel clarification flow)
    clarification_status: str | None = None  # ClarificationStatus value
    question_history: list[str] = field(default_factory=list)
    # Pre-dispatch clarity gate. ``question_history`` remains the
    # append-only audit trail; ``open_questions`` is the current unresolved set.
    open_questions: list[str] = field(default_factory=list)
    clarification_round: int = 0
    clarifier_fingerprint: str | None = None
    clarification_replies: list[str] = field(default_factory=list)
    clarifier_comment_cursor: str | None = None
    author_login: str | None = None
    local_answer: str | None = None
    local_answer_source: str | None = None  # "dashboard" | "clarification_queue"
    first_response_source: str | None = None  # "local" | "author"
    stale_answers: list[str] = field(default_factory=list)
    processed_feedback_ids: list[str] = field(default_factory=list)
    pending_feedback_ids: list[str] = field(default_factory=list)
    # Feedback URL persistence: parallel lookup of the canonical
    # comment/check URL for each pending feedback id, so the IM/CLI
    # ``issue feedback --list`` surface can show a clickable link instead
    # of the internal source-prefixed id. Keyed by the same id string
    # stored in ``pending_feedback_ids``; entries are dropped together
    # with the id in ``mark_feedback_processed`` / ``clear_stale_pending``.
    pending_feedback_urls: dict[str, str] = field(default_factory=dict)
    pending_feedback_since: float | None = None
    feedback_cursor: str | None = None
    followup_attempt_count: int = 0
    last_followup_commit_sha: str | None = None
    last_feedback_checked_at: float | None = None
    # Operator intent + retry bookkeeping.
    intent: Intent = Intent.NONE
    retry_count: int = 0
    last_command: str | None = None
    intent_source: str | None = None  # "label" | "command" | "cli"
    # Comment-command incremental-scan cursor.
    command_cursor: str | None = None
    run_id: str | None = None
    debug_log_path: str | None = None
    run_turn_count: int = 0
    run_tool_count: int = 0
    run_last_event: str | None = None
    run_last_tool: str | None = None
    run_output_len: int = 0
    run_timeout_deadline_at: float | None = None
    run_workspace_dirty: bool | None = None
    # Retry context: list of run_ids from previous attempts for this
    # issue.  The retrying agent can Read() the transcript at
    # ~/.clawcodex/sessions/<run_id>/transcript.jsonl to learn what was
    # attempted before.  Populated by _schedule_retry in orchestrator.py.
    previous_run_ids: list[str] = field(default_factory=list)
    # Collaboration mode chosen for this run. One of the keys in
    # ``orchestrator.modes`` registry — "single" by default so existing
    # records load with byte-identical behavior. Set by
    # ``orchestrator._launch_issue`` after ``ModeSelector.choose``.
    collaboration_mode: str = "single"
    # Why that mode was picked — for operator audit.
    mode_decision_reason: str | None = None
    # PR conflict persistence.
    has_conflict: bool = False
    conflict_files: list[str] = field(default_factory=list)
    rebase_attempt_count: int = 0
    last_rebase_attempt_at: float | None = None
    # Pause reason recorded when the issue is paused by an operator
    # control command. Set by mark_paused(); cleared by mark_resumed().
    pause_reason: str = ""

    def touch(self) -> None:
        """Stamp ``updated_at`` with the current wall-clock time."""
        self.updated_at = time.time()
