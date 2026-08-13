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

"""ClarificationResolver — orchestrates the three-channel clarification flow.

Channel priority: StatusDashboard prompt → ClarificationQueue file → @mention
issue comments; each timeout escalates to the next channel.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from extensions.orchestrator.clarification_queue import ClarificationQueue, ClarificationStatus
from extensions.orchestrator.tracker import Comment, CommentHistoryCapability, supports

if TYPE_CHECKING:
    from extensions.orchestrator.clarification_queue import ClarificationItem
    from extensions.orchestrator.tracker import TrackerAdapter

logger = logging.getLogger(__name__)

_CLARIFICATION_MARKER_PREFIX = "<!-- clawcodex-clarification:"

# Default tuning values shared by ClarificationConfig and the workflow-agent
# fallbacks in orchestrator.py — keep them in one place to avoid drift.
_DEFAULT_TIMEOUT_LOCAL_SECONDS = 30 * 60  # 30 minutes for local channels
_DEFAULT_TIMEOUT_AUTHOR_SECONDS = 72 * 3600  # 72 hours for author channel
_DEFAULT_MAX_QUESTIONS_PER_ISSUE = 3
_DEFAULT_CONFIDENCE_THRESHOLD = 0.7
_DEFAULT_SIMULTANEOUS_GRACE_MS = 5000  # 5 seconds for "tied" answers

# Unix timestamps in candidate tuples are seconds; grace window is ms.
_MS_PER_SECOND = 1000


@dataclass
class ClarificationConfig:
    """Configuration for clarification flow."""

    enabled: bool = True
    timeout_local_seconds: float = _DEFAULT_TIMEOUT_LOCAL_SECONDS
    timeout_author_seconds: float = _DEFAULT_TIMEOUT_AUTHOR_SECONDS
    max_questions_per_issue: int = _DEFAULT_MAX_QUESTIONS_PER_ISSUE
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD
    operator_priority: bool = True  # operator answers beat author
    simultaneous_grace_ms: float = _DEFAULT_SIMULTANEOUS_GRACE_MS
    stale_notification: str = "all"  # "all" | "operator_only" | "none"

    # Escalation policy when all channels timeout
    escalation: str = "skip"  # "skip" | "mark_failed" | "notify"


@dataclass
class ClarificationResult:
    """Result of a clarification attempt."""

    answer: str | None
    source: str | None  # "dashboard" | "clarification_queue" | "author"
    status: ClarificationStatus


class ClarificationResolver:
    """Orchestrates the three-channel clarification flow.

    Usage:
        resolver = ClarificationResolver(
            clarification_queue=ClarificationQueue(),
            tracker=tracker_adapter,
            config=ClarificationConfig(),
        )

        # In orchestrator poll loop:
        await resolver.poll_clarification_answers()

        # When agent encounters ambiguous semantics:
        result = await resolver.request_clarification(issue, question, context)
    """

    def __init__(
        self,
        clarification_queue: ClarificationQueue,
        tracker: TrackerAdapter,
        config: ClarificationConfig | None = None,
    ) -> None:
        """Store the queue, tracker and configuration for the flow."""
        self._queue = clarification_queue
        self._tracker = tracker
        self._config = config or ClarificationConfig()

    async def poll_clarification_answers(self) -> None:
        """Poll both ClarificationQueue and issue comments for new answers."""
        pending_items = self._queue.poll_active()
        for item in pending_items:
            await self._check_for_answer(item)

    async def request_clarification(
        self,
        issue_id: str,
        issue_identifier: str,
        question: str,
        context: str = "",
        options: list[str] | None = None,
        start_with_author: bool = False,
        since_comment_id: str | None = None,
        author_login: str | None = None,
    ) -> ClarificationResult:
        """Request clarification via the three-channel flow.

        Enqueues a new request when none is in flight, else returns the
        resolved answer / current status for an existing one.
        """
        existing = self._queue.get(issue_id)
        if existing is not None:
            # Already in clarification flow — check if resolved
            resolved = self._queue.get_resolved(issue_id)
            if resolved:
                return ClarificationResult(
                    answer=resolved.answer,
                    source=resolved.answer_source,
                    status=resolved.status,
                )
            # In progress — return current status
            return ClarificationResult(
                answer=None,
                source=None,
                status=existing.status,
            )

        # Enqueue new clarification request
        self._queue.enqueue(
            issue_id=issue_id,
            issue_identifier=issue_identifier,
            question=question,
            options=options,
            context_summary=context,
            timeout_seconds=(
                self._config.timeout_author_seconds if start_with_author else self._config.timeout_local_seconds
            ),
            since_comment_id=since_comment_id,
            author_login=author_login,
        )

        if start_with_author:
            if not author_login:
                # Without the author's identity the @mention channel can
                # never receive an authorized answer; fall back to the
                # local channel instead of publishing a question that is
                # doomed to time out unanswered.
                logger.warning(
                    "start_with_author requested but author_login is missing "
                    "for %s — falling back to the local channel",
                    issue_identifier,
                )
                self._queue.mark_awaiting_local(issue_id)
                return ClarificationResult(
                    answer=None,
                    source=None,
                    status=ClarificationStatus.AWAITING_LOCAL,
                )
            item = self._queue.mark_awaiting_author(
                issue_id,
                timeout_seconds=self._config.timeout_author_seconds,
            )
            if item is not None:
                try:
                    await self._send_author_mention(issue_id, item)
                except Exception:
                    # The request was never delivered. Remove the orphaned
                    # queue item so a later poll can retry from a clean state.
                    self._queue.remove(issue_id)
                    raise
            status = ClarificationStatus.AWAITING_AUTHOR
        else:
            # Start with Channel 1 (Dashboard) — orchestrator will detect pending
            self._queue.mark_awaiting_local(issue_id)
            status = ClarificationStatus.AWAITING_LOCAL

        # Return in-progress status (orchestrator poll will pick up)
        return ClarificationResult(
            answer=None,
            source=None,
            status=status,
        )

    async def _check_for_answer(self, item: "ClarificationItem") -> None:
        """Check both channels for an answer to the given clarification item."""
        issue_id = item.issue_id

        # Collect candidates BEFORE deciding on timeout: a reply posted
        # before the deadline must be honored even when this poll runs
        # after expiry.
        candidates = await self._collect_candidates(issue_id, item)

        if item.is_expired():
            deadline = item.expires_at
            has_pre_deadline_answer = deadline is not None and any(ts <= deadline for _, _, ts in candidates)
            if not has_pre_deadline_answer:
                await self._handle_timeout(issue_id, item)
                return

        if len(candidates) == 0:
            return  # No answer yet

        if len(candidates) == 1:
            winner = candidates[0]
            self._apply_answer(issue_id, winner)
        else:
            # Multiple candidates — resolve conflict across all of them
            winner, losers = self._resolve_conflict(candidates)
            self._apply_answer(issue_id, winner)
            for loser in losers:
                self._notify_rejected(loser, issue_id)

    async def _collect_candidates(
        self,
        issue_id: str,
        item: "ClarificationItem",
    ) -> list[tuple[str, str, float]]:
        """Collect potential answers from all channels as (source, text, ts) tuples."""
        candidates: list[tuple[str, str, float]] = []

        # Channel 2: ClarificationQueue
        if item.status in (
            ClarificationStatus.PENDING,
            ClarificationStatus.AWAITING_LOCAL,
        ):
            queue_answer = self._queue.get_resolved(issue_id)
            if queue_answer and queue_answer.answer:
                candidates.append(
                    (
                        queue_answer.answer_source or "clarification_queue",
                        queue_answer.answer,
                        queue_answer.answered_at or time.time(),
                    )
                )

        # Channel 3: Issue comments
        if item.status in (
            ClarificationStatus.AWAITING_LOCAL,
            ClarificationStatus.AWAITING_AUTHOR,
        ):
            new_comments = await self._fetch_new_author_comments(
                issue_id,
                item.last_checked_comment_id,
            )
            if new_comments:
                last_seen_id = new_comments[-1].id
                if not item.author_login:
                    # Author-first clarification is an authorization boundary.
                    # Adapters that cannot identify the issue author must not
                    # let an arbitrary commenter unblock automated execution.
                    self._queue.mark_comment_checked(issue_id, last_seen_id)
                    logger.warning(
                        "Ignoring issue comments for clarification %s: author identity unavailable",
                        issue_id,
                    )
                    return candidates
                expected_author = item.author_login.strip().casefold()
                authorized = [
                    comment
                    for comment in new_comments
                    if comment.author_login
                    and comment.author_login.strip().casefold() == expected_author
                    and not self._is_own_clarification_comment(comment, item)
                ]
                # Advance across every observed comment, including rejected
                # ones, so unauthorized traffic is not re-fetched forever.
                self._queue.mark_comment_checked(issue_id, last_seen_id)
                if not authorized:
                    return candidates
                latest = authorized[-1]  # Chronologically last authorized reply
                if latest.body and latest.body.strip():
                    ts = self._parse_comment_timestamp(latest.created_at)
                    candidates.append(
                        (
                            "author",
                            latest.body.strip(),
                            ts,
                        )
                    )

        return candidates

    async def _fetch_new_author_comments(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> list[Comment]:
        """Fetch issue comments newer than the last checked comment id."""
        if not supports(self._tracker, CommentHistoryCapability):
            return []
        try:
            comments = await self._tracker.fetch_new_comments_since(
                issue_id,
                since_comment_id,
            )
            return comments
        except Exception as exc:
            logger.warning("Failed to fetch comments for issue %s: %s", issue_id, exc)
            return []

    def _resolve_conflict(
        self,
        candidates: list[tuple[str, str, float]],
    ) -> tuple[tuple[str, str, float], list[tuple[str, str, float]]]:
        """Resolve simultaneous answers from different channels.

        Returns the global winner plus every losing candidate. Any
        candidate outside the simultaneous-grace window of the winner
        is treated as a stale late answer and surfaced separately.

        Rules:
        1. Within simultaneous_grace_ms: operator_priority wins
           (dashboard/clarification_queue beat author)
        2. Otherwise: earliest timestamp wins

        When three or more channels answer in the same poll, the winner
        is chosen by folding the rule across the whole list so no
        candidate is silently dropped.
        """
        if len(candidates) < 2:
            return candidates[0], []

        grace_ms = self._config.simultaneous_grace_ms
        operator_priority = self._config.operator_priority

        def _beats(a: tuple[str, str, float], b: tuple[str, str, float]) -> bool:
            """Return True when ``a`` should win over ``b``."""
            delta_ms = abs(a[2] - b[2]) * _MS_PER_SECOND
            if delta_ms < grace_ms and operator_priority:
                # Within grace window: operator source wins
                a_is_operator = a[0] in ("dashboard", "clarification_queue")
                b_is_operator = b[0] in ("dashboard", "clarification_queue")
                if a_is_operator != b_is_operator:
                    return a_is_operator
            # Normal timestamp comparison — earliest wins
            return a[2] <= b[2]

        # Pick the global winner by folding the pairwise rule.
        winner = candidates[0]
        for candidate in candidates[1:]:
            if _beats(candidate, winner):
                winner = candidate
        losers = [c for c in candidates if c is not winner]
        return winner, losers

    def _apply_answer(
        self,
        issue_id: str,
        winner: tuple[str, str, float],
    ) -> None:
        """Apply the winning answer to the queue."""
        source, answer, _ = winner
        self._queue.resolve(issue_id, answer, source)

    def _notify_rejected(
        self,
        loser: tuple[str, str, float],
        issue_id: str,
    ) -> None:
        """Notify the losing side about the rejection."""
        source, answer, _ = loser
        if source == "author":
            # Author answered but operator already won
            logger.info(
                "Author answer rejected for issue %s (operator priority)",
                issue_id,
            )
            # Could post a comment to inform the author — skip for now
        else:
            # Operator answer came after escalation
            self._queue.mark_stale(issue_id, answer, reason="escalated_to_author")
            logger.info(
                "Operator answer marked stale for issue %s (already escalated)",
                issue_id,
            )

    async def _handle_timeout(self, issue_id: str, item: "ClarificationItem") -> None:
        """Handle clarification timeout — escalate to next channel."""
        current_status = item.status

        if current_status == ClarificationStatus.AWAITING_LOCAL:
            # Channel 1/2 timeout → escalate to Channel 3 (@mention author)
            self._queue.mark_expired(issue_id)
            if not item.author_login:
                # Without the author's identity the @mention channel cannot
                # receive an authorized answer; apply the escalation policy
                # directly instead of posting an unanswerable mention
                # (mirrors the send-failure fallback below).
                logger.warning(
                    "Author identity unavailable for %s — applying escalation policy %s directly",
                    issue_id,
                    self._config.escalation,
                )
                self._handle_escalation(issue_id)
                return
            self._queue.mark_awaiting_author(
                issue_id,
                timeout_seconds=self._config.timeout_author_seconds,
            )

            # Send @mention comment to author. On failure we must NOT mark
            # escalation as notified (the author never received the question),
            # and the @mention channel is effectively dead for this issue —
            # fall straight through to the escalation policy instead of
            # leaving the issue marooned in a terminal-looking state that
            # ``poll_active`` will never return.
            try:
                await self._send_author_mention(issue_id, item)
            except Exception as exc:
                logger.warning(
                    "Failed to send @mention for issue %s: %s — applying escalation policy %s directly",
                    issue_id,
                    exc,
                    self._config.escalation,
                )
                self._queue.mark_expired(issue_id)
                self._handle_escalation(issue_id)
                return

            # @mention was delivered — record it so the operator dashboard
            # truthfully reflects that the author was contacted.
            self._queue.mark_escalation_notified(issue_id)

        elif current_status == ClarificationStatus.AWAITING_AUTHOR:
            # Channel 3 timeout → trigger escalation policy
            self._queue.mark_expired(issue_id)
            self._handle_escalation(issue_id)

    async def _send_author_mention(
        self,
        issue_id: str,
        item: "ClarificationItem",
    ) -> None:
        """Send @mention comment to issue author requesting clarification."""
        body = self._build_mention_body(item)
        mentions = [item.author_login] if item.author_login else []

        try:
            comment = await self._tracker.create_clarification_comment(
                issue_id=issue_id,
                body=body,
                mentions=mentions,
            )
            if comment is not None and getattr(comment, "id", None):
                self._queue.mark_comment_checked(issue_id, comment.id)
            logger.info("Sent @mention clarification for issue %s", issue_id)
        except Exception as exc:
            logger.error("Failed to send @mention for issue %s: %s", issue_id, exc)
            raise

    def _build_mention_body(self, item: "ClarificationItem") -> str:
        """Build the @mention comment body."""
        question = item.question
        marker = f"{_CLARIFICATION_MARKER_PREFIX}{item.issue_id} -->"
        if item.options:
            options_text = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(item.options))
            body = (
                f"## Clarification Needed\n\n"
                f"I need some clarification on this issue:\n\n"
                f"**Question:** {question}\n\n"
                f"**Options:**\n{options_text}\n\n"
                f"Please reply with the number or your answer."
            )
        else:
            body = (
                f"## Clarification Needed\n\n"
                f"I need some clarification on this issue:\n\n"
                f"**Question:** {question}\n\n"
                f"Please reply with your answer."
            )
        return f"{marker}\n\n{body}"

    def _is_own_clarification_comment(
        self,
        comment: Comment,
        item: "ClarificationItem",
    ) -> bool:
        """Reject bot-authored prompts even when bot and author share a login."""
        body = str(comment.body or "")
        marker = f"{_CLARIFICATION_MARKER_PREFIX}{item.issue_id} -->"
        if marker in body:
            return True
        # Compatibility for pending questions posted before markers existed.
        return "## Clarification Needed" in body and "**Question:**" in body and item.question in body

    def _handle_escalation(self, issue_id: str) -> None:
        """Handle escalation policy when all channels have timed out.

        Policies:
          skip      — mark as abandoned (orchestrator will skip on next poll)
          mark_failed — mark as failed immediately
          notify    — mark as failed and send notification
        """
        policy = self._config.escalation
        logger.warning(
            "Clarification exhausted for issue %s — escalation policy: %s",
            issue_id,
            policy,
        )
        self._queue.mark_exhausted(issue_id)

        if policy == "mark_failed":
            # Mark the issue as failed so the orchestrator won't re-launch
            self._queue.mark_issue_failed(issue_id)
        elif policy == "notify":
            # Mark failed + emit a notification event
            self._queue.mark_issue_failed(issue_id)
            self._emit_escalation_notification(issue_id)
        # "skip" policy: just mark as EXHAUSTED, orchestrator will skip via registry check

    def _emit_escalation_notification(self, issue_id: str) -> None:
        """Emit an escalation notification event for external alerting.

        Best-effort: a write failure is logged at WARNING so the operator
        can see the dropped notification, but does not abort the escalation
        flow. The list is written atomically (temp file + ``os.replace``)
        so a partial write cannot corrupt existing notifications.
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        notif_path = Path.home() / ".clawcodex" / ".escalation_notifications.json"
        tmp_path: Path | None = None
        try:
            notif_path.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if notif_path.exists():
                existing = json.loads(notif_path.read_text(encoding="utf-8"))
            existing.append(
                {
                    "issue_id": issue_id,
                    "timestamp": time.time(),
                    "policy": self._config.escalation,
                }
            )
            payload = json.dumps(existing, indent=2, ensure_ascii=False)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=notif_path.parent,
                prefix=".escalation_notifications.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(payload)
            os.replace(tmp_path, notif_path)
        except Exception as exc:  # nosec B110
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            logger.warning(
                "Failed to emit escalation notification for issue %s: %s",
                issue_id,
                exc,
            )

    def _parse_comment_timestamp(self, created_at: str | None) -> float:
        """Parse a tracker ``created_at`` string into a Unix timestamp."""
        if not created_at:
            return time.time()
        try:
            # ISO format with Z suffix
            ts = created_at.replace("Z", "+00:00")
            from datetime import datetime

            dt = datetime.fromisoformat(ts)
            return dt.timestamp()
        except Exception:
            return time.time()

    def get_answer(self, issue_id: str) -> ClarificationResult | None:
        """Get the resolved answer for an issue, if any."""
        resolved = self._queue.get_resolved(issue_id)
        if resolved is None:
            return None
        return ClarificationResult(
            answer=resolved.answer,
            source=resolved.answer_source,
            status=resolved.status,
        )

    def get_pending_feedback(self, issue_id: str) -> "ClarificationItem | None":
        """Return pending rejected-review feedback for an agent retry."""
        return self._queue.get_pending_feedback(issue_id)

    def get_pending_count(self) -> int:
        """Return count of pending clarification items."""
        return len(self._queue.poll_pending())

    def get_item(self, issue_id: str) -> "ClarificationItem | None":
        """Return the queue item for an issue, or None when absent."""
        return self._queue.get(issue_id)

    def clear(self, issue_id: str) -> None:
        """Remove the clarification item for an issue from the queue."""
        self._queue.remove(issue_id)

    def get_stale_answers(self, issue_id: str) -> list[str]:
        """Return stale answers for an issue (for notification)."""
        return self._queue.get_stale(issue_id)
