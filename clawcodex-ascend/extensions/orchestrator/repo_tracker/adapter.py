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

"""Tracker adapter for repository-backed issue trackers."""

# This split feature branch is linted without the aggregate package initializer.
# pylint: disable=relative-beyond-top-level

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from ..issue import Issue

from ..tracker import (
    Comment,
    DEFAULT_INTENT_LABELS,
    Intent,
    MergeableStatus,
    PullRequestFeedback,
    PullRequestRef,
    TrackerAdapter,
    default_active_states_for_kind,
    default_terminal_states_for_kind,
    intent_from_label_set,
)
from .client import RepositoryIssueClient
from .normalizers import _extract_comment_author

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from ..tracker import CommandIntent

_LIFECYCLE_LABELS = frozenset(
    {
        "pending",
        "queued",
        "running",
        "synced",
        "pending_review",
        "completed",
        "failed",
        "abandoned",
        "verification_failed",
        "cancelled",
        "canceled",
    }
)


class RepositoryTrackerAdapter(TrackerAdapter):
    """Repository-backed issue tracker adapter."""

    def __init__(
        self,
        *,
        platform: str,
        owner: str,
        repo: str,
        api_key: str | None = None,
        endpoint: str | None = None,
        active_states: list[str] | None = None,
        terminal_states: list[str] | None = None,
        assignee: str | None = None,
        intent_labels: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        skip_labels: list[str] | None = None,
        require_any_labels: list[str] | None = None,
        title_prefixes: list[str] | None = None,
        title_prefix_match: str = "any",
    ) -> None:
        """Initialize the repository-backed tracker adapter.

        Args:
            platform: Repository platform name ("github", "gitee", "gitcode").
            owner: Repository owner (user or organization).
            repo: Repository name.
            api_key: Optional API key for authenticated requests.
            endpoint: Optional API endpoint override.
            active_states: Issue states treated as active candidates.
            terminal_states: Issue states treated as terminal.
            assignee: Optional assignee filter for candidate issues.
            intent_labels: Mapping of label names to intents.
            http_client: Optional shared httpx async client.
            skip_labels: Labels that exclude an issue from candidates.
            require_any_labels: Labels; at least one must be present.
            title_prefixes: Optional title prefixes for candidate filtering.
            title_prefix_match: "any" or "all" match policy for prefixes.
        """
        self.platform = platform
        self.owner = owner
        self.repo = repo
        self.assignee = assignee
        self.active_states = active_states or default_active_states_for_kind(platform)
        self.terminal_states = terminal_states or default_terminal_states_for_kind(platform)
        self.skip_labels: list[str] = list(skip_labels or [])
        self.require_any_labels: list[str] = list(require_any_labels or [])
        self.title_prefixes: list[str] = list(title_prefixes or [])
        self.title_prefix_match = title_prefix_match
        # Intent label conventions (operator-driven retry/followup/blocked).
        # If caller passes None, fall back to the standard "agent:*" set.
        self.intent_labels: dict[str, str] = dict(intent_labels) if intent_labels else dict(DEFAULT_INTENT_LABELS)
        self.client = RepositoryIssueClient(
            platform=platform,
            owner=owner,
            repo=repo,
            api_key=api_key,
            endpoint=endpoint,
            http_client=http_client,
            skip_labels=skip_labels,
            require_any_labels=require_any_labels,
            title_prefixes=title_prefixes,
            title_prefix_match=title_prefix_match,
        )

    def configure_title_prefix_filter(self, prefixes: list[str] | None, match: str = "any") -> None:
        """Configure the title-prefix filter used when polling issues."""
        self.title_prefixes = list(prefixes or [])
        self.title_prefix_match = match
        self.client.configure_title_prefix_filter(prefixes, match)

    async def extract_intent_from_labels(
        self,
        labels: list[str] | None,
    ) -> Intent:
        """Resolve the operator intent from a set of issue labels."""
        return intent_from_label_set(labels, self.intent_labels)

    async def get_authenticated_user(self) -> str | None:
        """Return the login of the authenticated token owner, or None."""
        return await self.client.get_authenticated_user()

    async def close_pull_request(
        self,
        pull_request: PullRequestRef,
    ) -> bool:
        """Close a remote pull request, best-effort.

        Returns:
            True when the close succeeded (or was already merged),
            False on failure.
        """
        return await self.client.close_pull_request(pull_request)

    async def fetch_issue_command_intent(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> "CommandIntent | None":
        """Scan new issue comments for a parseable agent command.

        Returns:
            The first ``CommandIntent`` found, or None when none is present
            or the comment fetch fails.
        """
        from ..tracker import CommandIntent, parse_agent_command

        try:
            comments = await self.fetch_new_comments_since(issue_id, since_comment_id)
        except Exception as exc:
            logger.warning(
                "fetch_issue_command_intent(%s) failed: %s",
                issue_id,
                exc,
            )
            return None
        for comment in comments:
            body = comment.body or ""
            command = parse_agent_command(body)
            if command is not None:
                return CommandIntent(
                    command=command,
                    author_login=comment.author_login,
                    comment_id=comment.id,
                    comment_body=body,
                )
        return None

    async def fetch_candidate_issues(self) -> list[Issue]:
        """Fetch issues in active states matching assignee and filters."""
        return await self.client.fetch_candidate_issues(
            active_states=self.active_states,
            assignee=self.assignee,
        )

    async def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> dict[str, Issue]:
        """Fetch current snapshots for the requested issue IDs."""
        issues = await self.client.fetch_issue_states_by_ids(
            issue_ids,
            active_states=self.active_states,
            assignee=self.assignee,
        )
        return {issue.id: issue for issue in issues if issue.id}

    async def create_comment(self, issue_id: str, body: str) -> Comment | None:
        """Create a comment on an issue and return the created comment."""
        created = await self.client.create_comment(issue_id, body)
        if created is None:
            return None
        return Comment(
            id=str(created.get("id", "")),
            body=created.get("body"),
            author_login=_extract_comment_author(created),
            created_at=created.get("created_at"),
            updated_at=created.get("updated_at"),
            in_reply_to_id=created.get("in_reply_to_id"),
        )

    async def update_comment(
        self,
        issue_id: str,
        comment_id: str,
        body: str,
    ) -> Comment | None:
        """Update an existing comment body and return the updated comment."""
        updated = await self.client.update_comment(comment_id, body)
        if updated is None:
            return None
        return Comment(
            id=str(updated.get("id", "")),
            body=updated.get("body"),
            author_login=_extract_comment_author(updated),
            created_at=updated.get("created_at"),
            updated_at=updated.get("updated_at"),
            in_reply_to_id=updated.get("in_reply_to_id"),
        )

    async def update_issue_state(self, issue_id: str, state: str) -> None:
        """Update a remote issue's state, mirroring it via lifecycle labels."""
        issue = await self.client.fetch_issue_states_by_ids(
            [issue_id],
            active_states=self.active_states,
            assignee=None,
        )
        current = issue[0] if issue else None

        labels = list(current.labels) if current is not None else []
        normalized_state = state.strip().lower()
        known_state_labels = {
            item.strip().lower() for item in [*self.active_states, *self.terminal_states] if item.strip()
        }
        known_state_labels.update(_LIFECYCLE_LABELS)
        labels = [label for label in labels if label.strip().lower() not in known_state_labels]
        if normalized_state and normalized_state not in {
            "open",
            "opened",
            "closed",
            "close",
        }:
            labels.append(state)

        await self.client.update_issue(
            issue_id,
            state=state,
            # ``[]`` means "remove every label" while ``None`` means
            # "leave labels unchanged".  Reopening an issue whose only
            # label is a terminal lifecycle marker must preserve that
            # distinction or the remote issue stays labelled failed/
            # pending_review after the daemon has reset it locally.
            # If the pre-read could not find the issue, keep the historical
            # safe behaviour for an ``open`` transition and do not clear
            # unknown labels.  An empty list is authoritative only when the
            # current issue was actually fetched.
            labels=labels if current is not None or labels else None,
        )

    async def find_pull_request(
        self,
        *,
        head_branch: str,
        base_branch: str,
    ) -> PullRequestRef | None:
        """Find an open pull request matching head and base branches."""
        return await self.client.find_pull_request(
            head_branch=head_branch,
            base_branch=base_branch,
        )

    async def ensure_pull_request(
        self,
        *,
        issue: Issue,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> PullRequestRef | None:
        """Ensure a pull request exists for the given branches.

        Returns the existing PR when one matches, otherwise creates a new one.

        Args:
            issue: The issue the pull request belongs to.
            head_branch: Branch carrying the changes.
            base_branch: Branch the changes target.
            title: Pull request title.
            body: Pull request description.

        Returns:
            The matched or created ``PullRequestRef``, or None on failure.
        """
        existing = await self.client.find_pull_request(
            head_branch=head_branch,
            base_branch=base_branch,
        )
        if existing is not None:
            return existing
        return await self.client.create_pull_request(
            title=title,
            head_branch=head_branch,
            base_branch=base_branch,
            body=body,
        )

    async def update_pull_request(
        self,
        *,
        pull_request: PullRequestRef,
        title: str | None = None,
        body: str | None = None,
    ) -> PullRequestRef | None:
        """Update a pull request's title and/or body.

        Returns:
            The updated ``PullRequestRef``, or None when the PR has no
            remote number.
        """
        return await self.client.update_pull_request(
            pull_request=pull_request,
            title=title,
            body=body,
        )

    async def fetch_pull_request_mergeable(
        self,
        pull_request: PullRequestRef,
    ) -> "MergeableStatus | None":
        """Delegate to ``RepositoryIssueClient`` and translate
        to the platform-normalized ``MergeableStatus``.

        The PullRequestRef's ``number`` is the PR number on the
        remote side; the local short-id form is not used here.
        """
        if pull_request.number is None:
            return None
        return await self.client.fetch_pull_request_mergeable(
            pull_request=pull_request,
        )

    async def fetch_pull_request_feedback(
        self,
        *,
        pull_request: PullRequestRef,
        issue_id: str | None = None,
        include_ci_failures: bool = True,
        max_log_chars_per_check: int = 12_000,
    ) -> list[PullRequestFeedback]:
        """Fetch all available feedback for a pull request."""
        return await self.client.fetch_pull_request_feedback(
            pull_request=pull_request,
            issue_id=issue_id,
            include_ci_failures=include_ci_failures,
            max_log_chars_per_check=max_log_chars_per_check,
        )

    async def reply_to_pull_request_feedback(
        self,
        *,
        pull_request: PullRequestRef,
        feedback: PullRequestFeedback,
        body: str,
        issue_id: str | None = None,
    ) -> Comment | None:
        """Reply to a pull request feedback item with the given body."""
        created = await self.client.reply_to_pull_request_feedback(
            pull_request=pull_request,
            feedback=feedback,
            body=body,
            issue_id=issue_id,
        )
        if created is None:
            return None
        return Comment(
            id=str(created.get("id", "")),
            body=created.get("body"),
            author_login=_extract_comment_author(created),
            created_at=created.get("created_at"),
            updated_at=created.get("updated_at"),
            in_reply_to_id=feedback.id,
        )

    async def fetch_issue_comments(self, issue_id: str) -> list[Comment]:
        """Return all non-empty comments on an issue, oldest first."""
        raw_comments = await self.client.fetch_comments(issue_id)
        return [
            Comment(
                id=str(c.get("id", "")),
                body=c.get("body"),
                author_login=_extract_comment_author(c),
                created_at=c.get("created_at"),
                updated_at=c.get("updated_at"),
                in_reply_to_id=c.get("in_reply_to_id"),
            )
            for c in raw_comments
            if c.get("body")
        ]

    async def fetch_new_comments_since(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> list[Comment]:
        """Return non-empty comments newer than ``since_comment_id``.

        Returns:
            All comments when ``since_comment_id`` is None, otherwise the
            comments that follow the cursor.
        """
        raw_comments = await self.client.fetch_comments_since(
            issue_id,
            since_comment_id,
        )
        return [
            Comment(
                id=str(c.get("id", "")),
                body=c.get("body"),
                author_login=_extract_comment_author(c),
                created_at=c.get("created_at"),
                updated_at=c.get("updated_at"),
                in_reply_to_id=c.get("in_reply_to_id"),
            )
            for c in raw_comments
            if c.get("body")
        ]

    async def create_clarification_comment(
        self,
        issue_id: str,
        body: str,
        mentions: list[str] | None = None,
    ) -> Comment | None:
        """Create a comment that mentions users and asks for clarification.

        Uses the POST response directly to avoid races with fast replies.
        """
        mention_prefix = " ".join(f"@{login.strip()}" for login in (mentions or []) if login.strip())
        comment_body = f"{mention_prefix}\n\n{body}" if mention_prefix else body
        # Use the POST response directly. Re-fetching and selecting the last
        # comment races with a fast author reply: the reply can become the
        # cursor and then be skipped forever by incremental polling.
        created = await self.client.create_comment(issue_id, comment_body)
        if created:
            return Comment(
                id=str(created.get("id", "")),
                body=created.get("body"),
                author_login=_extract_comment_author(created),
                created_at=created.get("created_at"),
                updated_at=created.get("updated_at"),
                in_reply_to_id=created.get("in_reply_to_id"),
            )
        return None
