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

"""Operator intent / retry / conflict bookkeeping (split from issue_registry.py)."""

from __future__ import annotations

import time

from ..tracker import Intent  # pylint: disable=relative-beyond-top-level
from .models import IssueRecord, IssueStatus


class IntentMixin:
    """Intent, retry, rebase-conflict and unblock mutations.

    The host class provides ``_records`` and ``_save`` (from StorageMixin).
    """

    def mark_intent(
        self,
        issue_id: str,
        intent: Intent,
        *,
        source: str | None = None,
        command: str | None = None,
    ) -> IssueRecord | None:
        """Record an operator intent on an existing record.

        If the record does not exist yet, this is a no-op — the orchestrator
        creates the record on first claim via `register()`. Callers that need
        to capture intent on a brand-new issue should call `register()` first.

        `source` is informational ("label" | "command" | "cli") and is
        persisted on the record for audit purposes.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.intent = intent
        if source is not None:
            record.intent_source = source
        if command is not None:
            record.last_command = command
        record.touch()
        self._save()
        return record

    def clear_intent(
        self,
        issue_id: str,
        *,
        record_intent_history: bool = False,
    ) -> IssueRecord | None:
        """Reset intent back to NONE.

        Used after the intent has been honored (reset
        succeeded / follow-up commit landed). If the record doesn't
        exist, returns None.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.intent = Intent.NONE
        if not record_intent_history:
            record.intent_source = None
        record.touch()
        self._save()
        return record

    def increment_retry_count(self, issue_id: str) -> IssueRecord | None:
        """Bump retry_count by one (retry rate limiting)."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.retry_count += 1
        record.touch()
        self._save()
        return record

    def mark_conflict(
        self,
        issue_id: str,
        conflict_files: tuple[str, ...] | list[str] = (),
    ) -> IssueRecord | None:
        """Mark an issue record as having a rebase conflict.

        ``conflict_files`` lists the files git reports as unresolved; an
        empty tuple/list means the workspace is dirty but no files are
        attributed (e.g. a non-conflict rebase failure leaving REBASE_HEAD).
        ``has_conflict`` is always set so the daemon picks it up via
        ``_process_pending_rebase_conflicts``.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.has_conflict = True
        record.conflict_files = list(conflict_files) if conflict_files else []
        record.last_rebase_attempt_at = time.time()
        record.touch()
        self._save()
        return record

    def clear_conflict(self, issue_id: str) -> IssueRecord | None:
        """Clear the rebase conflict flag for an issue record.

        Called when ``rebase_for_pr`` succeeds (rebased=True,
        has_conflict=False). Idempotent — safe to call on records
        that have no conflict flag set.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.has_conflict = False
        record.conflict_files = []
        record.touch()
        self._save()
        return record

    def increment_rebase_attempt(self, issue_id: str) -> IssueRecord | None:
        """Bump ``rebase_attempt_count`` by one (rate limiting).

        Used by ``_check_rebase_rate_limit`` before launching a new
        rebase resolution so the count reflects attempts started,
        not attempts succeeded.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.rebase_attempt_count += 1
        record.last_rebase_attempt_at = time.time()
        record.touch()
        self._save()
        return record

    def reset_for_retry(
        self,
        issue_id: str,
        *,
        increment_retry: bool = True,
        reset_retry_count: bool = False,
    ) -> IssueRecord | None:
        """Clear transient PR / commit state for a retry.

        Resets status to PENDING and drops commit_sha / pr_number /
        pr_url / report_path. ``retry_count`` handling (first match wins):

        * ``reset_retry_count=True`` — set to 0 (CLI ``--mode reset``:
          a fresh start that does not count against max_retries).
        * ``increment_retry=False`` — unchanged (legacy test / dry-run knob).
        * Otherwise — ``retry_count += 1`` (daemon-driven retries).

        The intent field is preserved so audit trails can still answer
        "why was this re-run?" after the new run completes.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.status = IssueStatus.PENDING
        record.commit_sha = None
        record.pr_number = None
        record.pr_url = None
        # A deliberate retry restarts the "first PR created" clock.
        record.pr_created_at = None
        record.report_path = None
        record.summary_comment_id = None
        record.verification_status = None
        record.verification_output = None
        record.last_hook_error = None
        record.clarification_status = None
        record.open_questions = []
        record.clarification_round = 0
        record.clarifier_fingerprint = None
        record.clarification_replies = []
        record.clarifier_comment_cursor = None
        if reset_retry_count:
            record.retry_count = 0
        elif increment_retry:
            record.retry_count += 1
        record.touch()
        self._save()
        return record

    def unblock(self, issue_id: str) -> IssueRecord | None:
        """Roll an ABANDONED issue back to PENDING.

        Used by the CLI ``issue retry --mode unblock`` fallback and the
        orchestrator's UNBLOCK comment-command handler.

        Behaviour:
          * Missing record — returns None.
          * ABANDONED record — flip status to PENDING and clear intent.
          * Any other status — no-op (idempotent).

        Note: ``retry_count`` is untouched, so the rate limit still
        applies to the next retry attempt after unblock.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        if record.status is IssueStatus.ABANDONED:
            record.status = IssueStatus.PENDING
            record.intent = Intent.NONE
            record.intent_source = None
            record.last_command = None
            record.touch()
            self._save()
        return record
