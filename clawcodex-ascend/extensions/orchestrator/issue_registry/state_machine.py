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

"""Issue-record lifecycle: queries + state transitions (split from issue_registry.py)."""

from __future__ import annotations

import time

from extensions.orchestrator.issue_registry.models import TERMINAL_STATUSES, IssueRecord, IssueStatus


class StateMachineMixin:
    """Record queries and lifecycle state transitions.

    The host class provides ``_records`` and ``_save`` (from StorageMixin).
    """

    def get(self, issue_id: str) -> IssueRecord | None:
        """Return the record for an issue id, or ``None`` if unknown."""
        return self._records.get(issue_id)

    def get_by_identifier(self, issue_identifier: str) -> IssueRecord | None:
        """Return the record whose ``issue_identifier`` matches, or ``None``."""
        for record in self._records.values():
            if record.issue_identifier == issue_identifier:
                return record
        return None

    def get_by_issue_ref(self, issue_ref: str) -> IssueRecord | None:
        """Resolve a record by issue id OR tracker identifier (either ref form)."""
        return self.get(issue_ref) or self.get_by_identifier(issue_ref)

    def get_by_branch(self, branch_name: str) -> IssueRecord | None:
        """Return the record owning a feature branch, or ``None``."""
        for record in self._records.values():
            if record.branch_name == branch_name:
                return record
        return None

    def has_pr(self, issue_id: str) -> bool:
        """Whether the issue record currently has a registered PR number."""
        record = self._records.get(issue_id)
        return record is not None and record.pr_number is not None

    def is_completed(self, issue_id: str) -> bool:
        """Whether the issue is in the COMPLETED state."""
        record = self._records.get(issue_id)
        return record is not None and record.status == IssueStatus.COMPLETED

    def is_terminal(self, issue_id: str) -> bool:
        """Whether the issue is in any terminal state (won't run again)."""
        record = self._records.get(issue_id)
        return record is not None and record.status in TERMINAL_STATUSES

    def iter_records_with_pr(self) -> list[IssueRecord]:
        """Return records that have both a PR number and a branch name."""
        return [record for record in self._records.values() if record.pr_number and record.branch_name]

    def latest_sequential_record(self) -> IssueRecord | None:
        """Return the highest ``sequence_index`` among sequential records."""
        sequential_records = (
            record
            for record in self._records.values()
            if record.workspace_strategy == "sequential" and record.sequence_index is not None
        )
        return max(
            sequential_records,
            key=lambda record: record.sequence_index or 0,
            default=None,
        )

    def running_records(self) -> list[IssueRecord]:
        """Return all records currently in the RUNNING state."""
        return [record for record in self._records.values() if record.status == IssueStatus.RUNNING]

    def has_processed_feedback(self, issue_id: str, feedback_id: str) -> bool:
        """Whether a feedback id has already been processed for the issue."""
        record = self._records.get(issue_id)
        return record is not None and feedback_id in record.processed_feedback_ids

    def can_follow_up(self, issue_id: str, max_attempts: int) -> bool:
        """Whether the issue's follow-up attempt budget is not exhausted."""
        record = self._records.get(issue_id)
        if record is None:
            return False
        return record.followup_attempt_count < max_attempts

    def register(
        self,
        issue_id: str,
        issue_identifier: str,
        branch_name: str | None = None,
        base_branch: str = "main",
        workspace_strategy: str | None = None,
        workspace_path: str | None = None,
        base_commit_sha: str | None = None,
        start_commit_sha: str | None = None,
        previous_issue_id: str | None = None,
        sequence_index: int | None = None,
        status: IssueStatus | None = None,
        author_login: str | None = None,
    ) -> IssueRecord:
        """Create a pending record for a newly claimed issue.

        Follow-up: ``_launch_issue`` calls ``register`` at the
        start of every run, including re-launches after a previous
        ``mark_synced`` already recorded a ``commit_sha`` /
        ``pr_number`` / ``pr_url``.  Naively overwriting
        ``self._records[issue_id]`` with a fresh ``IssueRecord`` would
        drop those sync-state fields, even though the underlying
        branch state on disk is unchanged.  When the record already
        exists, preserve the sync-state fields so the
        "registerable commit produced by this run" stays visible
        across retry / verification_failed re-dispatches.
        """
        existing = self._records.get(issue_id)
        record = IssueRecord(
            issue_id=issue_id,
            issue_identifier=issue_identifier,
            branch_name=branch_name,
            base_branch=base_branch,
            workspace_strategy=workspace_strategy,
            workspace_path=workspace_path,
            base_commit_sha=base_commit_sha,
            start_commit_sha=start_commit_sha,
            previous_issue_id=previous_issue_id,
            sequence_index=sequence_index,
            status=status or IssueStatus.PENDING,
            author_login=author_login,
        )
        if existing is not None:
            record.commit_sha = existing.commit_sha
            record.pr_number = existing.pr_number
            record.pr_url = existing.pr_url
            record.pr_created_at = existing.pr_created_at
            record.report_path = existing.report_path
            record.verification_status = existing.verification_status
            record.verification_output = existing.verification_output
            record.last_hook_error = existing.last_hook_error
            record.summary_comment_id = existing.summary_comment_id
            # Preserve author_login from existing record if not explicitly provided
            if author_login is None and existing.author_login:
                record.author_login = existing.author_login
            record.last_followup_commit_sha = existing.last_followup_commit_sha
            # Re-register overwrites branch_name with the issue's default
            # (usually "main").  Preserve the actual feature branch name
            # set by a prior `mark_synced` so followup sessions push to
            # the right branch.
            if existing.branch_name:
                record.branch_name = existing.branch_name
            # Preserve operator intent + retry bookkeeping so a
            # label-driven FOLLOWUP/RETRY is not silently wiped.
            record.intent = existing.intent
            record.intent_source = existing.intent_source
            record.retry_count = existing.retry_count
            record.last_command = existing.last_command
            record.command_cursor = existing.command_cursor
            # Preserve review-feedback tracking across re-launches.
            record.processed_feedback_ids = list(existing.processed_feedback_ids)
            record.pending_feedback_ids = list(existing.pending_feedback_ids)
            record.pending_feedback_urls = dict(existing.pending_feedback_urls)
            record.pending_feedback_since = existing.pending_feedback_since
            record.feedback_cursor = existing.feedback_cursor
            record.followup_attempt_count = existing.followup_attempt_count
            record.last_feedback_checked_at = existing.last_feedback_checked_at
            # Preserve rebase-conflict state across re-launches.
            # A retry / followup must not silently wipe has_conflict
            # because the daemon's PR conflict scan needs the flag
            # set until the next rebase succeeds.
            record.has_conflict = existing.has_conflict
            record.conflict_files = list(existing.conflict_files)
            record.rebase_attempt_count = existing.rebase_attempt_count
            record.last_rebase_attempt_at = existing.last_rebase_attempt_at
            # Pre-dispatch clarification is completed before
            # ``_launch_issue`` re-registers the record with workspace data.
            # Preserve the answer and audit state so the normal agent session
            # receives the author's requirements instead of silently losing
            # them at launch time.
            record.clarification_status = existing.clarification_status
            record.question_history = list(existing.question_history)
            record.open_questions = list(existing.open_questions)
            record.clarification_round = existing.clarification_round
            record.clarifier_fingerprint = existing.clarifier_fingerprint
            record.clarification_replies = list(existing.clarification_replies)
            record.clarifier_comment_cursor = existing.clarifier_comment_cursor
            record.local_answer = existing.local_answer
            record.local_answer_source = existing.local_answer_source
            record.first_response_source = existing.first_response_source
            record.stale_answers = list(existing.stale_answers)
        self._records[issue_id] = record
        self._save()
        return record

    def mark_synced(
        self,
        issue_id: str,
        *,
        branch_name: str | None = None,
        commit_sha: str | None = None,
        pr_number: str | None = None,
        pr_url: str | None = None,
    ) -> IssueRecord | None:
        """Update the record after a git sync (commit + push + PR) run.

        Applies the new branch / commit / PR metadata and moves the
        record to SYNCED.  The first PR number seen starts the
        ``pr_created_at`` clock; follow-up runs reusing the same PR do
        not overwrite it.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        if branch_name is not None:
            record.branch_name = branch_name
        if commit_sha is not None:
            record.commit_sha = commit_sha
        if pr_number is not None:
            # First PR creation for this issue: record the wall-clock
            # timestamp. Follow-up / review-feedback runs reuse the same
            # PR and pass the same pr_number — the guard keeps the
            # original "first PR created" time intact.
            if record.pr_number is None and record.pr_created_at is None:
                record.pr_created_at = time.time()
            record.pr_number = pr_number
        if pr_url is not None:
            record.pr_url = pr_url
        record.status = IssueStatus.SYNCED
        record.touch()
        self._save()
        return record

    def mark_running(self, issue_id: str) -> IssueRecord | None:
        """Mark the issue as RUNNING and reset per-run diagnostics.

        Clears ``run_id`` / counters / deadline fields so the next run
        starts from a clean slate.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.RUNNING
        record.run_id = None
        record.debug_log_path = None
        record.run_turn_count = 0
        record.run_tool_count = 0
        record.run_last_event = None
        record.run_last_tool = None
        record.run_output_len = 0
        record.run_timeout_deadline_at = None
        record.run_workspace_dirty = None
        record.touch()
        self._save()
        return record

    def update_run_diagnostics(
        self,
        issue_id: str,
        *,
        run_id: str | None = None,
        debug_log_path: str | None = None,
        turn_count: int | None = None,
        tool_count: int | None = None,
        last_event: str | None = None,
        last_tool: str | None = None,
        output_len: int | None = None,
        timeout_deadline_at: float | None = None,
        workspace_dirty: bool | None = None,
    ) -> IssueRecord | None:
        """Update high-frequency run diagnostics fields (throttled save).

        Only non-``None`` kwargs are applied.  Persists through the
        throttled ``_save_diagnostics`` path because this fires on every
        agent event.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        if run_id is not None:
            record.run_id = run_id
        if debug_log_path is not None:
            record.debug_log_path = debug_log_path
        if turn_count is not None:
            record.run_turn_count = turn_count
        if tool_count is not None:
            record.run_tool_count = tool_count
        if last_event is not None:
            record.run_last_event = last_event
        if last_tool is not None:
            record.run_last_tool = last_tool
        if output_len is not None:
            record.run_output_len = output_len
        if timeout_deadline_at is not None:
            record.run_timeout_deadline_at = timeout_deadline_at
        if workspace_dirty is not None:
            record.run_workspace_dirty = workspace_dirty
        record.touch()
        self._save_diagnostics()
        return record

    def mark_pending_review(self, issue_id: str) -> IssueRecord | None:
        """Mark the issue as PENDING_REVIEW (awaiting human review).

        Used by LocalTracker after the git commit lands, before a human
        merges the branch.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.PENDING_REVIEW
        record.touch()
        self._save()
        return record

    def mark_completed(self, issue_id: str) -> IssueRecord | None:
        """Mark the issue as COMPLETED (session finished successfully)."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.COMPLETED
        record.touch()
        self._save()
        return record

    def mark_failed(self, issue_id: str) -> IssueRecord | None:
        """Mark the issue as FAILED and increment the attempt count."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.FAILED
        record.attempt_count += 1
        record.touch()
        self._save()
        return record

    def mark_failed_with_reason(
        self,
        issue_id: str,
        reason: str,
    ) -> IssueRecord | None:
        """Mark the issue as FAILED with an explicit verification reason.

        Stores the reason in ``verification_status`` / ``verification_output``
        / ``last_hook_error`` and increments the attempt count.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.FAILED
        record.verification_status = "failed"
        record.verification_output = reason
        record.last_hook_error = reason
        record.attempt_count += 1
        record.touch()
        self._save()
        return record

    def mark_verification_failed(
        self,
        issue_id: str,
        *,
        output: str | None = None,
        hook_error: str | None = None,
    ) -> IssueRecord | None:
        """Mark the issue as VERIFICATION_FAILED (pre-commit gate failed).

        Records the verification output / hook error and increments the
        attempt count so retry budgets apply.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.VERIFICATION_FAILED
        record.verification_status = "failed"
        record.verification_output = output
        record.last_hook_error = hook_error
        record.attempt_count += 1
        record.touch()
        self._save()
        return record

    def mark_paused(self, issue_id: str, *, reason: str = "") -> IssueRecord | None:
        """Mark an issue as paused by an operator control command."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.PAUSED
        record.pause_reason = reason
        record.touch()
        self._save()
        return record

    def mark_resumed(self, issue_id: str) -> IssueRecord | None:
        """Restore an issue to running after being paused."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.RUNNING
        record.pause_reason = ""
        record.touch()
        self._save()
        return record

    def mark_abandoned(self, issue_id: str) -> IssueRecord | None:
        """Mark the issue as ABANDONED (retry limit reached, gave up)."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.ABANDONED
        record.touch()
        self._save()
        return record

    def update_branch(self, issue_id: str, branch_name: str) -> IssueRecord | None:
        """Record the feature branch name for an issue."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.branch_name = branch_name
        record.touch()
        self._save()
        return record

    def update_report(
        self,
        issue_id: str,
        *,
        report_path: str | None = None,
        verification_status: str | None = None,
        verification_output: str | None = None,
        summary_comment_id: str | None = None,
        session_end_reason: str | None = None,
        session_end_summary: str | None = None,
    ) -> IssueRecord | None:
        """Update report / verification / session-end fields.

        Only non-``None`` kwargs are applied.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        if report_path is not None:
            record.report_path = report_path
        if verification_status is not None:
            record.verification_status = verification_status
        if verification_output is not None:
            record.verification_output = verification_output
        if summary_comment_id is not None:
            record.summary_comment_id = summary_comment_id
        if session_end_reason is not None:
            record.session_end_reason = session_end_reason
        if session_end_summary is not None:
            record.session_end_summary = session_end_summary
        record.touch()
        self._save()
        return record
