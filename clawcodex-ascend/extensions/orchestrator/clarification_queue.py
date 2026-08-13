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

"""Async clarification queue for operator answers.

File format: JSON, stored at ~/.clawcodex/clarification_queue.json

Architecture:
- ClarificationItem: one pending question awaiting an answer
- ClarificationQueue: file-backed queue with polling support
- Handles conflict detection (DUPLICATE_REJECTED, STALE_REJECTED, CONFLICT_RESOLVED)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Default queue path under user config directory
DEFAULT_QUEUE_PATH = Path.home() / ".clawcodex" / "clarification_queue.json"


class ClarificationStatus(str, Enum):
    """Lifecycle stages of a clarification item."""

    NONE = "none"  # not in clarification flow
    PENDING = "pending"  # awaiting answer (default on enqueue)
    AWAITING_LOCAL = "awaiting_local"  # waiting for local operator answer
    AWAITING_AUTHOR = "awaiting_author"  # waiting for issue author (@mention sent)
    RESOLVED_LOCAL = "resolved_local"  # resolved by local operator
    RESOLVED_AUTHOR = "resolved_author"  # resolved by issue author
    TIMED_OUT_LOCAL = "timed_out_local"  # local timeout, escalated to author
    TIMED_OUT_AUTHOR = "timed_out_author"  # author timeout, escalation triggered
    EXHAUSTED = "exhausted"  # max questions reached, gave up
    # --- conflict handling states ---
    DUPLICATE_REJECTED = "duplicate_rejected"  # duplicate submission, dropped
    STALE_REJECTED = "stale_rejected"  # late answer after escalation
    CONFLICT_RESOLVED = "conflict_resolved"  # simultaneous answers resolved


@dataclass
class ClarificationItem:
    """One entry in the clarification queue."""

    issue_id: str
    issue_identifier: str
    question: str
    options: list[str] = field(default_factory=list)
    context_summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    status: ClarificationStatus = ClarificationStatus.PENDING
    answer: str | None = None
    answer_source: str | None = None  # "dashboard" | "clarification_queue" | "author"
    answered_at: float | None = None
    escalation_notified: bool = False  # operator informed of escalation
    first_response_source: str | None = None  # "dashboard" | "clarification_queue" | "author"
    duplicate_of: str | None = None  # if DUPLICATE_REJECTED, reference original
    stale_answers: list[str] = field(default_factory=list)  # rejected late answers
    # ``review_feedback`` entries are one-shot instructions for a rejected
    # review retry.  Keeping them distinct from real clarification questions
    # prevents a later ordinary retry from replaying stale reviewer feedback.
    kind: str = "clarification"
    last_checked_comment_id: str | None = None
    author_login: str | None = None

    def touch(self) -> None:
        """Refresh ``updated_at`` to the current time."""
        self.updated_at = time.time()

    def is_expired(self, now: float | None = None) -> bool:
        """Return True when ``now`` has reached the item's deadline.

        An item without a deadline (``expires_at`` is None) is never expired.

        Args:
            now: reference timestamp; defaults to the current time.
        """
        if self.expires_at is None:
            return False
        if now is None:
            now = time.time()
        return now >= self.expires_at

    def mark_answered(
        self,
        answer: str,
        source: str,
        answered_at: float | None = None,
    ) -> None:
        """Record the answer text, source and timestamp on the item.

        Args:
            answer: the answer text
            source: where the answer came from ("dashboard",
                "clarification_queue" or "author")
            answered_at: timestamp of the answer; defaults to the current time
        """
        self.answer = answer
        self.answer_source = source
        self.answered_at = answered_at or time.time()
        self.touch()


class ClarificationQueue:
    """File-backed async clarification queue for operator answers.

    Polling mechanism: call poll_pending() each orchestrator poll cycle
    to find items that are awaiting answers.
    """

    def __init__(self, queue_path: Path | None = None) -> None:
        """Initialise the queue from ``queue_path`` (defaults to ``DEFAULT_QUEUE_PATH``)."""
        self._path = queue_path or DEFAULT_QUEUE_PATH
        self._records: dict[str, ClarificationItem] = {}
        self._load()

    def get(self, issue_id: str) -> ClarificationItem | None:
        """Return the item for ``issue_id``, or None when absent."""
        return self._records.get(issue_id)

    def get_pending_feedback(self, issue_id: str) -> ClarificationItem | None:
        """Return an unexpired one-shot review-feedback item, if present."""
        item = self._records.get(issue_id)
        if item is None or item.kind != "review_feedback":
            return None
        if item.status not in (
            ClarificationStatus.PENDING,
            ClarificationStatus.AWAITING_LOCAL,
            ClarificationStatus.AWAITING_AUTHOR,
        ):
            return None
        if item.is_expired():
            return None
        return item

    def list_items(self) -> list[ClarificationItem]:
        """Return all queue items sorted by creation time."""
        return sorted(self._records.values(), key=lambda item: item.created_at)

    def poll_pending(self) -> list[ClarificationItem]:
        """Return all pending items that have not expired."""
        now = time.time()
        return [
            item
            for item in self._records.values()
            if item.kind == "clarification"
            and item.status
            in (
                ClarificationStatus.PENDING,
                ClarificationStatus.AWAITING_LOCAL,
                ClarificationStatus.AWAITING_AUTHOR,
            )
            and not item.is_expired(now)
        ]

    def poll_active(self) -> list[ClarificationItem]:
        """Return unresolved items, including expired ones for escalation."""
        return [
            item
            for item in self._records.values()
            if item.status
            in (
                ClarificationStatus.PENDING,
                ClarificationStatus.AWAITING_LOCAL,
                ClarificationStatus.AWAITING_AUTHOR,
            )
        ]

    def get_resolved(self, issue_id: str) -> ClarificationItem | None:
        """Return the resolved item for an issue, if one exists."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        if item.status in (
            ClarificationStatus.RESOLVED_LOCAL,
            ClarificationStatus.RESOLVED_AUTHOR,
        ):
            return item
        return None

    def get_stale(self, issue_id: str) -> list[str]:
        """Return the stale (rejected) answers recorded for an issue."""
        item = self._records.get(issue_id)
        if item is None:
            return []
        return item.stale_answers

    def enqueue(
        self,
        issue_id: str,
        issue_identifier: str,
        question: str,
        *,
        options: list[str] | None = None,
        context_summary: str = "",
        timeout_seconds: float | None = None,
        since_comment_id: str | None = None,
        author_login: str | None = None,
    ) -> ClarificationItem:
        """Create a new pending clarification item for an issue.

        Args:
            issue_id: internal issue identifier
            issue_identifier: human-readable identifier (e.g. "owner/repo#42")
            question: the clarification question
            options: optional multiple-choice options
            context_summary: issue context shared with the operator/author
            timeout_seconds: how long the item stays answerable; None means
                no deadline
            since_comment_id: lowest comment id already seen for the issue
            author_login: login of the issue author for the @mention channel

        Returns:
            The newly created ClarificationItem.
        """
        now = time.time()
        expires_at = None
        if timeout_seconds is not None:
            expires_at = now + timeout_seconds

        item = ClarificationItem(
            issue_id=issue_id,
            issue_identifier=issue_identifier,
            question=question,
            options=list(options) if options else [],
            context_summary=context_summary,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            status=ClarificationStatus.PENDING,
            last_checked_comment_id=since_comment_id,
            author_login=author_login,
        )
        self._records[issue_id] = item
        self._save()
        return item

    def mark_awaiting_local(self, issue_id: str) -> ClarificationItem | None:
        """Transition to Channel 1/2 (local operator awaiting)."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        item.status = ClarificationStatus.AWAITING_LOCAL
        item.touch()
        self._save()
        return item

    def mark_awaiting_author(
        self,
        issue_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ClarificationItem | None:
        """Transition to Channel 3 (@mention author)."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        item.status = ClarificationStatus.AWAITING_AUTHOR
        if timeout_seconds is not None:
            item.expires_at = time.time() + max(0.0, float(timeout_seconds))
        item.touch()
        self._save()
        return item

    def mark_comment_checked(
        self,
        issue_id: str,
        comment_id: str | None,
    ) -> ClarificationItem | None:
        """Record the highest comment id already inspected for an issue.

        Args:
            issue_id: the issue being tracked
            comment_id: the last comment id seen; None clears the marker

        Returns:
            The updated item, or None when the issue is not in the queue.
        """
        item = self._records.get(issue_id)
        if item is None:
            return None
        item.last_checked_comment_id = comment_id
        item.touch()
        self._save()
        return item

    def resolve(
        self,
        issue_id: str,
        answer: str,
        source: str,
    ) -> ClarificationItem | None:
        """Record an answer from any channel.

        Args:
            issue_id: the issue being clarified
            answer: the answer text
            source: one of "dashboard", "clarification_queue", "author"

        Returns:
            The updated item, or None if not found.
        """
        item = self._records.get(issue_id)
        if item is None:
            return None

        now = time.time()
        item.mark_answered(answer, source, now)

        if item.status in (
            ClarificationStatus.PENDING,
            ClarificationStatus.AWAITING_LOCAL,
            ClarificationStatus.AWAITING_AUTHOR,
        ):
            if source in ("dashboard", "clarification_queue"):
                item.status = ClarificationStatus.RESOLVED_LOCAL
            else:
                item.status = ClarificationStatus.RESOLVED_AUTHOR
        else:
            # Unexpected state — still record answer but keep current status
            logger.warning(
                "resolve() called on issue %s in unexpected status %s",
                issue_id,
                item.status,
            )

        if item.first_response_source is None:
            item.first_response_source = source
        item.touch()
        self._save()
        return item

    def mark_duplicate(
        self,
        issue_id: str,
        duplicate_answer: str,
        original_timestamp: float,
    ) -> ClarificationItem | None:
        """Mark an answer as a duplicate (idempotent deduplication)."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        item.status = ClarificationStatus.DUPLICATE_REJECTED
        item.duplicate_of = str(original_timestamp)
        item.stale_answers.append(duplicate_answer)
        item.touch()
        self._save()
        return item

    def mark_stale(
        self,
        issue_id: str,
        stale_answer: str,
        reason: str = "",
    ) -> ClarificationItem | None:
        """Mark a late answer as stale (after channel escalation)."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        item.status = ClarificationStatus.STALE_REJECTED
        item.stale_answers.append(stale_answer)
        item.touch()
        self._save()
        return item

    def mark_escalation_notified(self, issue_id: str) -> ClarificationItem | None:
        """Mark that the operator has been informed of channel escalation."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        item.escalation_notified = True
        item.touch()
        self._save()
        return item

    def mark_expired(self, issue_id: str) -> ClarificationItem | None:
        """Mark an item as expired (timeout reached, trigger escalation)."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        if item.status == ClarificationStatus.AWAITING_LOCAL:
            item.status = ClarificationStatus.TIMED_OUT_LOCAL
        elif item.status == ClarificationStatus.AWAITING_AUTHOR:
            item.status = ClarificationStatus.TIMED_OUT_AUTHOR
        else:
            item.status = ClarificationStatus.EXHAUSTED
        item.touch()
        self._save()
        return item

    def mark_exhausted(self, issue_id: str) -> ClarificationItem | None:
        """Mark an item as exhausted (max questions reached)."""
        item = self._records.get(issue_id)
        if item is None:
            return None
        item.status = ClarificationStatus.EXHAUSTED
        item.touch()
        self._save()
        return item

    def mark_issue_failed(self, issue_id: str) -> None:
        """Mark an issue as failed due to escalation policy.

        Writes a sentinel file that the orchestrator reads to mark the
        issue as failed on its next poll cycle. The write is atomic
        (temp file + ``os.replace``) so a partial write cannot corrupt
        existing failed-issue markers, and any I/O failure is logged at
        ERROR so the operator sees that the durable "stop" signal was
        never persisted (otherwise the orchestrator would silently
        re-poll the exhausted issue on every cycle).
        """
        sentinel_path = self._path.parent / ".escalated_issues.json"
        tmp_path: Path | None = None
        try:
            sentinel_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if sentinel_path.exists():
                existing = json.loads(sentinel_path.read_text(encoding="utf-8"))
            existing[issue_id] = {"failed_at": time.time()}
            payload = json.dumps(existing, indent=2, ensure_ascii=False)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=sentinel_path.parent,
                prefix=".escalated_issues.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(payload)
            os.replace(tmp_path, sentinel_path)
        except Exception as exc:  # nosec B110
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            logger.error(
                "Failed to persist failure sentinel for issue %s: %s — orchestrator may keep re-polling this issue",
                issue_id,
                exc,
            )

    def remove(self, issue_id: str) -> None:
        """Remove an item from the queue."""
        if issue_id in self._records:
            del self._records[issue_id]
            self._save()

    def consume_feedback(self, issue_id: str) -> ClarificationItem | None:
        """Remove and return review feedback after a successful follow-up.

        Ordinary clarification items are deliberately left untouched.
        """
        item = self._records.get(issue_id)
        if item is None or item.kind != "review_feedback":
            return None
        del self._records[issue_id]
        self._save()
        return item

    def inject_feedback(
        self,
        issue_id: str,
        feedback: str,
    ) -> ClarificationItem | None:
        """Inject feedback from a rejected review to trigger retry.

        This creates a clarification item with the feedback as the question,
        so the agent receives it on the next turn via clarification context.
        """
        item = self._records.get(issue_id)
        now = time.time()
        if item is None:
            # Create a new item for the feedback
            item = ClarificationItem(
                issue_id=issue_id,
                issue_identifier=issue_id,
                question=feedback,
                options=[],
                context_summary="Human review rejection feedback",
                created_at=now,
                updated_at=now,
                status=ClarificationStatus.PENDING,
                kind="review_feedback",
            )
            self._records[issue_id] = item
        else:
            # Update existing item with new question/feedback
            item.question = feedback
            item.options = []
            item.context_summary = "Human review rejection feedback"
            item.status = ClarificationStatus.PENDING
            item.kind = "review_feedback"
            item.expires_at = None
            item.answer = None
            item.answer_source = None
            item.answered_at = None
            item.touch()
        self._save()
        return item

    def _load(self) -> None:
        """Reload the queue from disk; missing or corrupt files start empty."""
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("clarification queue top-level must be an object")
            records: dict[str, ClarificationItem] = {}
            for key, value in data.items():
                if not isinstance(value, dict):
                    raise ValueError(f"clarification queue record {key!r} must be an object")
                item = ClarificationItem(**value)
                if item.options is None:
                    item.options = []
                if item.stale_answers is None:
                    item.stale_answers = []
                records[key] = item
            self._records = records
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to load clarification queue: %s — starting fresh",
                exc,
            )

    def _save(self) -> None:
        """Persist the current queue to disk as JSON, best-effort."""
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {k: asdict(v) for k, v in self._records.items()},
                indent=2,
                ensure_ascii=False,
            )
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
            temporary = None
            if os.name != "nt":
                directory_fd = os.open(self._path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except Exception as exc:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
            logger.warning("Failed to save clarification queue: %s", exc)
