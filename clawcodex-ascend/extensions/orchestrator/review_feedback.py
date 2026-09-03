#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Pull request review feedback follow-up planning."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .issue import Issue  # pylint: disable=import-error
from .issue_registry import IssueRecord, IssueRegistry  # pylint: disable=import-error
from .tracker import (  # pylint: disable=import-error
    PullRequestFeedback,
    PullRequestRef,
    TrackerAdapter,
    UserIdentityCapability,
    supports,
)

logger = logging.getLogger(__name__)

REPLY_MARKER = "Handled in the latest ClawCodex follow-up commit."
CLAWCODEX_SYSTEM_MARKERS = (
    "## ClawCodex Run Summary",
    "## ClawCodex Automated Change",
    "## ClawCodex Follow-up",
    "## ClawCodex PR Review Follow-up Summary",
    "<!-- metadata: report_path=",
)

_DEFAULT_PENDING_FEEDBACK_TIMEOUT_SECONDS = 600
_DEFAULT_MAX_FOLLOWUP_ATTEMPTS_PER_PR = 5
_DEFAULT_MAX_LOG_CHARS_PER_CHECK = 12_000
_DEFAULT_MAX_FEEDBACK_ITEMS_PER_RUN = 20


@dataclass(frozen=True)
class ReviewFollowup:
    issue: Issue
    record: IssueRecord
    pull_request: PullRequestRef
    feedback: tuple[PullRequestFeedback, ...]
    prompt: str


class ReviewFeedbackService:
    """Find PR feedback that should trigger a follow-up agent run."""

    def __init__(
        self,
        *,
        tracker: TrackerAdapter,
        registry: IssueRegistry,
        config: object,
    ) -> None:
        self.tracker = tracker
        self.registry = registry
        self.config = config
        self._bot_login: str | None = None
        self._bot_login_resolved = False
        self._bot_login_explicit = False
        self._ignored_body_patterns = _compile_patterns(getattr(config, "ignored_body_patterns", []))

    async def collect_followups(self, available_slots: int) -> list[ReviewFollowup]:
        if available_slots <= 0 or not getattr(self.config, "enabled", False):
            return []

        if not self._bot_login_resolved:
            explicit = getattr(self.config, "bot_login", None)
            if explicit:
                self._bot_login = explicit
                self._bot_login_explicit = True
            else:
                if supports(self.tracker, UserIdentityCapability):
                    try:
                        self._bot_login = await self.tracker.get_authenticated_user()
                    except Exception:  # nosec B110
                        pass
            self._bot_login_resolved = True
            if self._bot_login:
                logger.debug("Bot login resolved: %s", self._bot_login)

        followups: list[ReviewFollowup] = []
        for record in self.registry.iter_records_with_pr():
            if len(followups) >= available_slots:
                break
            timeout = getattr(
                self.config,
                "pending_feedback_timeout_seconds",
                _DEFAULT_PENDING_FEEDBACK_TIMEOUT_SECONDS,
            )
            cleared = self.registry.clear_stale_pending(record.issue_id, timeout)
            if cleared:
                logger.info(
                    "Cleared %d stale pending feedback item(s) for issue %s — will re-detect",
                    cleared,
                    record.issue_id,
                )
            if not self.registry.can_follow_up(
                record.issue_id,
                getattr(
                    self.config,
                    "max_followup_attempts_per_pr",
                    _DEFAULT_MAX_FOLLOWUP_ATTEMPTS_PER_PR,
                ),
            ):
                continue

            pull_request = PullRequestRef(
                number=record.pr_number,
                url=record.pr_url,
            )
            fetch_feedback = getattr(self.tracker, "fetch_pull_request_feedback", None)
            if not callable(fetch_feedback):
                continue
            feedback = await fetch_feedback(
                pull_request=pull_request,
                issue_id=record.issue_id,
                include_ci_failures=getattr(self.config, "include_ci_failures", True),
                max_log_chars_per_check=getattr(
                    self.config,
                    "max_log_chars_per_check",
                    _DEFAULT_MAX_LOG_CHARS_PER_CHECK,
                ),
            )
            pending = self._filter_pending(record, feedback, bot_login=self._bot_login)
            if not pending:
                self.registry.mark_feedback_checked(record.issue_id)
                continue

            limit = getattr(
                self.config,
                "max_feedback_items_per_run",
                _DEFAULT_MAX_FEEDBACK_ITEMS_PER_RUN,
            )
            selected = pending[:limit]
            if not selected:
                logger.warning(
                    "max_feedback_items_per_run limit=%s produced empty selection for issue %s",
                    limit,
                    record.issue_id,
                )
                self.registry.mark_feedback_checked(record.issue_id)
                continue
            self.registry.mark_feedback_pending(
                record.issue_id,
                [item.id for item in selected],
                cursor=selected[-1].updated_at or selected[-1].created_at or selected[-1].id,
                feedback_urls={item.id: item.url for item in selected if item.url},
            )
            issue = Issue(
                id=record.issue_id,
                identifier=record.issue_identifier,
                title=None,
                branch_name=record.branch_name,
            )
            followups.append(
                ReviewFollowup(
                    issue=issue,
                    record=record,
                    pull_request=pull_request,
                    feedback=tuple(selected),
                    prompt="",
                )
            )
        return followups

    def _filter_pending(
        self,
        record: IssueRecord,
        feedback: list[PullRequestFeedback],
        *,
        bot_login: str | None = None,
    ) -> list[PullRequestFeedback]:
        ignored_authors = {
            author.strip().lower() for author in getattr(self.config, "ignore_authors", []) if author.strip()
        }
        # Only filter out bot's own comments when bot_login is EXPLICITLY
        # configured (e.g. review_feedback.bot_login: my-bot).
        # Auto-resolved bot_login (get_authenticated_user) is the same as the
        # repo owner — adding it would silently drop all review comments from
        # the repo owner, who is likely the human reviewer.
        if bot_login and self._bot_login_explicit:
            ignored_authors.add(bot_login.strip().lower())
        processed = set(record.processed_feedback_ids)
        already_pending = set(record.pending_feedback_ids)
        ignored_sources = {
            source.strip().lower() for source in getattr(self.config, "ignored_feedback_sources", []) if source.strip()
        }
        ignored_commands = {
            _normalize_command(command)
            for command in getattr(self.config, "ignored_comment_commands", [])
            if _normalize_command(command)
        }
        pending: list[PullRequestFeedback] = []
        for item in feedback:
            if item.id in processed or item.id in already_pending:
                continue
            if item.status in {"resolved", "outdated"}:
                continue
            if item.author_login and item.author_login.strip().lower() in ignored_authors:
                continue
            if _is_clawcodex_system_comment(item.body):
                continue
            if item.source.strip().lower() in ignored_sources:
                logger.debug("Skipping feedback %s: ignored source %s", item.id, item.source)
                continue
            if _is_standalone_command(item.body, ignored_commands):
                logger.debug("Skipping feedback %s: ignored comment command", item.id)
                continue
            if _matches_ignored_body_pattern(item.body, self._ignored_body_patterns):
                logger.debug("Skipping feedback %s: ignored body pattern", item.id)
                continue
            pending.append(item)
        return pending


def _is_clawcodex_system_comment(body: str | None) -> bool:
    if not body:
        return False
    if REPLY_MARKER in body:
        return True
    lines = body.splitlines()
    for marker in CLAWCODEX_SYSTEM_MARKERS:
        if marker.startswith("<!--"):
            if marker in body:
                return True
        else:
            if any(line.strip().startswith(marker) for line in lines):
                return True
    return False


def _normalize_command(command: object) -> str:
    """Normalize a configured slash command for exact comment matching."""
    if not isinstance(command, str):
        return ""
    command = command.strip().lower()
    if not command:
        return ""
    return command if command.startswith("/") else f"/{command}"


def _is_standalone_command(body: str | None, commands: set[str]) -> bool:
    """Return whether a comment consists only of an ignored slash command.

    An optional leading mention is accepted so ``@clawcodex /lgtm`` is
    treated identically to ``/lgtm``. Commands embedded in prose, or with
    additional instructions, deliberately remain actionable.
    """
    if not body or not commands:
        return False
    normalized = re.sub(r"^@[A-Za-z0-9][A-Za-z0-9_.-]*\s+", "", body.strip())
    return normalized.lower() in commands


def _compile_patterns(patterns: object) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns if isinstance(patterns, list) else []:
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            logger.warning("Ignoring invalid review-feedback body pattern: %r", pattern)
    return tuple(compiled)


def _matches_ignored_body_pattern(body: str | None, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Match configured patterns against the entire normalized body only."""
    return bool(body and any(pattern.fullmatch(body.strip()) for pattern in patterns))
