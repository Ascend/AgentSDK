#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any


from .agent_runner import AgentRunner
from .events import EventLevel
from .issue_registry import IssueStatus
from .tracker import (
    Intent,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CONTINUATION_RETRY_DELAY_MS = 1_000
_FAILURE_RETRY_BASE_MS = 10_000


def _operator_failure_detail(exc: BaseException) -> str:
    """Return a concise failure detail suitable for IM and registry records."""

    raw = " ".join(str(exc).split())
    body_detail = _extract_error_message_from_body(raw)
    if body_detail:
        status_code = _extract_status_code(raw)
        if raw.startswith("request_failed") and status_code:
            return f"request_failed status={status_code}: {body_detail}"
        return body_detail
    return raw or exc.__class__.__name__


def _extract_status_code(text: str) -> str | None:
    for part in text.split():
        if part.startswith("status="):
            status = part.removeprefix("status=").strip()
            if status:
                return status
    return None


def _extract_error_message_from_body(text: str) -> str | None:
    marker = "body="
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    body = text[marker_index + len(marker) :].strip()
    if not body:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    return _extract_error_message(payload)


def _extract_error_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in (
            "error_message",
            "message",
            "error_description",
            "detail",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return " ".join(error.split())
        nested = _extract_error_message(error)
        if nested:
            return nested
        errors = payload.get("errors")
        if isinstance(errors, list):
            for item in errors:
                nested = _extract_error_message(item)
                if nested:
                    return nested
    return None


class OrchestratorControlMixin:
    def _reset_issue_for_retry(
        self,
        issue_id: str,
        feedback: str,
        *,
        intent: Intent = Intent.RETRY,
        reset_retry_count: bool = False,
        command: str | None = None,
    ) -> bool:
        """Reset review-gated state and queue feedback without requiring a running session."""
        if not issue_id:
            return False

        record = self._registry._records.get(issue_id)
        is_known = bool(
            record
            or issue_id in self._state.running
            or issue_id in self._state.pending_review
            or issue_id in self._state.completed
            or issue_id in self._state.claimed
        )
        if not is_known:
            logger.debug("Retry control for unknown issue %s", issue_id)
            return False

        if feedback:
            question = f"[Human Review Rejected] {feedback}"
            self._clarification_queue.inject_feedback(issue_id, question)

        self._state.pending_review.discard(issue_id)
        self._state.completed.discard(issue_id)
        self._state.claimed.discard(issue_id)
        failed = getattr(self._state, "failed", None)
        if failed is not None:
            failed.discard(issue_id)
        retry_attempts = getattr(self._state, "retry_attempts", None)
        if retry_attempts is not None:
            retry_attempts.pop(issue_id, None)
        retry_queue = getattr(self._state, "retry_queue", None)
        if retry_queue is not None:
            self._state.retry_queue = [retry for retry in retry_queue if retry.issue_id != issue_id]
        if record:
            was_pending_review = record.status is IssueStatus.PENDING_REVIEW
            record.status = IssueStatus.PENDING
            record.intent = intent
            record.intent_source = "cli"
            if reset_retry_count:
                record.retry_count = 0
            if command is not None:
                record.last_command = command
            elif feedback:
                record.last_command = "/issue review --reject"
            if was_pending_review:
                record.attempt_count += 1
            record.touch()
            self._registry._save()

        logger.info(
            "Issue %s queued for retry (attempt %d)",
            issue_id,
            record.attempt_count if record else 1,
        )
        self._emit_im_event(issue_id, "intent.retry", EventLevel.INFO, "retry requested")
        return True

    async def _handle_retry_control(self, issue_id: str, reason: str) -> None:
        """Apply a durable retry request and make the tracker eligible for polling."""
        if not self._reset_issue_for_retry(
            issue_id,
            "",
            reset_retry_count=True,
            command=f"cli:reset:{reason[:64]}",
        ):
            return
        await self._sync_tracker_issue_state(issue_id, "open")

    async def _handle_review_retry_control(self, issue_id: str, feedback: str) -> None:
        """Queue a rejected review as a follow-up that preserves the existing PR."""
        if not self._reset_issue_for_retry(issue_id, feedback, intent=Intent.FOLLOWUP):
            return
        await self._sync_tracker_issue_state(issue_id, "open")

    async def _handle_review_approve_control(self, issue_id: str, comment: str) -> None:
        """Finalize a human approval in registry, daemon state, and remote tracker."""
        record = self._registry.get(issue_id)
        if record is None:
            logger.warning("Review approval ignored for unknown issue %s", issue_id)
            return

        already_completed = record.status is IssueStatus.COMPLETED
        self._registry.mark_completed(issue_id)
        self._state.pending_review.discard(issue_id)
        self._state.claimed.discard(issue_id)
        self._state.completed.add(issue_id)
        tracker_synced = await self._sync_tracker_issue_state(issue_id, "completed")

        if comment and not already_completed:
            try:
                await self.tracker.create_comment(issue_id, f"## Approved\n\n{comment}")
            except Exception as exc:
                logger.warning("Failed to post approval comment issue_id=%s: %s", issue_id, exc)

        if tracker_synced:
            self._emit_im_event(
                issue_id,
                "issue.completed",
                EventLevel.SUCCESS,
                "人工审批通过",
                {
                    "pr": record.pr_url,
                    "branch": record.branch_name,
                    "commit": record.commit_sha,
                },
            )
        else:
            self._emit_im_event(
                issue_id,
                "issue.failed",
                EventLevel.ERROR,
                "审批已记录，但远端 completed 状态同步失败",
                {"pr": record.pr_url},
            )

    def _apply_control_command(self, cmd: str, issue_id: str, extra: str) -> None:
        """Apply a control command, including retries outside running sessions."""
        if cmd == "retry":
            if not self._reset_issue_for_retry(issue_id, extra):
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(self._sync_tracker_issue_state(issue_id, "open"))
            return

        if not issue_id or issue_id not in self._state.running:
            logger.debug("Control %s for unknown issue %s", cmd, issue_id)
            return

        session = self._state.running[issue_id]
        if cmd == "pause":
            AgentRunner._apply_pause_session(session, extra or "operator requested pause")
            logger.info("Paused issue %s: %s", issue_id, session.pause_reason)
            self._emit_im_event(issue_id, "control.pause", EventLevel.INFO, session.pause_reason)
            # Persist paused state to the registry.
            self._registry.mark_paused(issue_id, reason=session.pause_reason)
        elif cmd == "resume":
            AgentRunner._apply_resume_session(session)
            logger.info("Resumed issue %s", issue_id)
            self._emit_im_event(issue_id, "control.resume", EventLevel.INFO, "resumed")
            # Restore running state in the registry.
            self._registry.mark_resumed(issue_id)
        elif cmd == "stop":
            # Request cancellation via task cancel
            logger.info("Stop requested for issue %s", issue_id)
            session.status = "failed"
            session.pause_resume_event.set()  # Unblock if paused
            self._emit_im_event(issue_id, "control.stop", EventLevel.WARN, "stop requested")
            # Root-cause fix: cancel the asyncio task so the
            # CancelledError handler in _run_issue fires immediately
            # instead of leaving the agent running until the next
            # session end check.
            task = self._issue_tasks.get(issue_id)
            if task is not None and not task.done():
                task.cancel()
                logger.info("Cancelled task for issue %s", issue_id)
        elif cmd == "takeover":
            logger.info("Takeover requested for issue %s", issue_id)
            session.status = "failed"
            session.pause_resume_event.set()  # Unblock if paused
            self._emit_im_event(issue_id, "control.takeover", EventLevel.WARN, "takeover requested")
            # Note: REPL takeover requires full session context - handled separately

    def get_event_stream(self, issue_id: str) -> "asyncio.Queue | None":
        """Get the event queue for a running issue session (for CLI tail)."""
        session = self._state.running.get(issue_id)
        if session is None:
            return None
        return session.event_queue

    async def _cancel_all_tasks(self) -> None:
        """Cancel all running agent tasks."""
        if self._tasks:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
