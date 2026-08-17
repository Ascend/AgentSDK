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

"""Filesystem-backed tracker adapter for local issue documents."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

# pylint: disable=relative-beyond-top-level
from ..issue import Issue
from ..title_prefix_filter import (
    matches_title_prefixes,
    normalize_title_prefix_match,
    normalize_title_prefixes,
)
# pylint: enable=relative-beyond-top-level

if TYPE_CHECKING:
    from ..tracker import CommandIntent

# pylint: disable=relative-beyond-top-level
from ..tracker import (
    Comment,
    DEFAULT_INTENT_LABELS,
    Intent,
    PullRequestRef,
    TrackerAdapter,
    intent_from_label_set,
)

# pylint: enable=relative-beyond-top-level
# pylint: disable=relative-beyond-top-level,deprecated-module
from .parser import (
    LocalIssueDocument,
    parse_markdown_issue,
    utc_now_iso,
    write_markdown_frontmatter,
)
# pylint: enable=relative-beyond-top-level,deprecated-module

logger = logging.getLogger(__name__)

# Length of the sha256 digest suffix appended to filesystem-safe stems.
_STEM_DIGEST_LENGTH = 12


class LocalTrackerAdapter(TrackerAdapter):
    """Tracker adapter that stores issues and comments in local files."""

    def __init__(
        self,
        issues_path: str | Path,
        active_states: list[str] | None = None,
        terminal_states: list[str] | None = None,
        intent_labels: dict[str, str] | None = None,
        title_prefixes: list[str] | None = None,
        title_prefix_match: str = "any",
    ) -> None:
        """Initialize the local filesystem tracker adapter.

        Args:
            issues_path: Directory holding the local issue markdown files.
            active_states: Issue states treated as active candidates.
            terminal_states: Issue states treated as terminal.
            intent_labels: Mapping of label names to intents.
            title_prefixes: Optional title prefixes for candidate filtering.
            title_prefix_match: "any" or "all" match policy for prefixes.
        """
        self.issues_path = Path(issues_path).expanduser()
        self._active_states = tuple(active_states if active_states is not None else ["open", "ready"])
        self._terminal_states = tuple(
            terminal_states
            if terminal_states is not None
            else ["completed", "closed", "cancelled", "failed", "abandoned"]
        )
        self._active_state_set = _normalize_states(self._active_states)
        self.configure_title_prefix_filter(title_prefixes, title_prefix_match)
        # same label conventions as the repository-backed adapters.
        self.intent_labels: dict[str, str] = dict(intent_labels) if intent_labels else dict(DEFAULT_INTENT_LABELS)

    @property
    def active_states(self) -> list[str]:
        """Return the list of configured active issue states."""
        return list(self._active_states)

    @property
    def terminal_states(self) -> list[str]:
        """Return the list of configured terminal issue states."""
        return list(self._terminal_states)

    def configure_title_prefix_filter(self, prefixes: list[str] | None, match: str = "any") -> None:
        """Configure the title-prefix filter used when polling issues."""
        self._title_prefixes = normalize_title_prefixes(prefixes)
        self._title_prefix_match = normalize_title_prefix_match(match)

    async def fetch_candidate_issues(self) -> list[Issue]:
        """Fetch issues in an active state, matching the title-prefix filter.

        Returns:
            The candidate issues sorted by priority then identifier.
        """
        documents = self._load_documents()
        issues = []
        for document in documents:
            if not document.issue.state:
                logger.warning(
                    "Issue %s has no state field, defaulting to active",
                    document.issue.id or document.path.stem,
                )
            if _normalize_state(document.issue.state or "open") in self._active_state_set:
                if matches_title_prefixes(document.issue.title, self._title_prefixes, self._title_prefix_match):
                    issues.append(document.issue)
        return sorted(
            issues,
            key=lambda issue: (
                issue.priority is None,
                issue.priority if issue.priority is not None else 0,
                issue.identifier or issue.id or "",
            ),
        )

    async def fetch_issue_states_by_ids(
        self,
        issue_ids: list[str],
    ) -> dict[str, Issue]:
        """Fetch current snapshots for the requested local issues."""
        requested = set(issue_ids)
        issues: dict[str, Issue] = {}
        for document in self._load_documents():
            issue = document.issue
            if issue.id in requested:
                issues[issue.id or ""] = issue
        return issues

    async def create_comment(self, issue_id: str, body: str) -> Comment | None:
        """Append a comment to the issue's local comment log."""
        return self._append_comment(issue_id, body)

    async def update_comment(
        self,
        issue_id: str,
        comment_id: str,
        body: str,
    ) -> Comment | None:
        """Update a stored comment body in place.

        Returns:
            The updated comment, or None when the issue has no comment log
            or the comment ID was not found.
        """
        comment_path = self._comments_path(issue_id)
        if not comment_path.exists():
            return None
        now = utc_now_iso()
        updated_comment: Comment | None = None
        lines: list[str] = []
        for line in comment_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                lines.append(line)
                continue
            if str(payload.get("id")) == str(comment_id):
                payload["body"] = body
                payload["updated_at"] = now
                updated_comment = Comment(
                    id=_string_or_none(payload.get("id")),
                    body=_string_or_none(payload.get("body")),
                    author_login=_string_or_none(payload.get("author_login")),
                    created_at=_string_or_none(payload.get("created_at")),
                    updated_at=_string_or_none(payload.get("updated_at")),
                    in_reply_to_id=_string_or_none(payload.get("in_reply_to_id")),
                )
            lines.append(json.dumps(payload, ensure_ascii=False))
        if updated_comment is None:
            return None
        tmp_path = comment_path.with_suffix(comment_path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp_path, comment_path)
        return updated_comment

    async def update_issue_state(self, issue_id: str, state: str) -> None:
        """Persist a new state and updated timestamp in the issue frontmatter."""
        document = self._document_for_issue(issue_id)
        write_markdown_frontmatter(
            document.path,
            {
                "state": state,
                "updated_at": utc_now_iso(),
            },
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
        """Record PR metadata in the issue frontmatter and locate it back.

        Args:
            issue: The issue the pull request belongs to.
            head_branch: Branch carrying the changes.
            base_branch: Branch the changes target.
            title: Pull request title.
            body: Pull request description.

        Returns:
            The matching ``PullRequestRef`` after the metadata is written.
        """
        issue_id = issue.id or issue.identifier
        if issue_id:
            document = self._document_for_issue(issue_id)
            write_markdown_frontmatter(
                document.path,
                {
                    "branch_name": head_branch,
                    "base_branch": base_branch,
                    "pr_title": title,
                },
            )
        return await self.find_pull_request(
            head_branch=head_branch,
            base_branch=base_branch,
        )

    async def find_pull_request(
        self,
        *,
        head_branch: str,
        base_branch: str,
    ) -> PullRequestRef | None:
        """Find a local issue whose frontmatter records a matching PR.

        Returns:
            The matching ``PullRequestRef``, or None when no issue pairs
            the head branch, base branch and a recorded PR URL.
        """
        for document in self._load_documents():
            issue = document.issue
            if issue.branch_name != head_branch:
                continue
            if document.base_branch != base_branch:
                continue
            if not document.pr_url:
                continue
            return PullRequestRef(
                number=document.pr_number,
                url=document.pr_url,
                title=_string_or_none(document.metadata.get("pr_title")),
            )
        return None

    async def list_pull_requests(self) -> list[PullRequestRef]:
        """Local tracker has no remote PRs — return empty list."""
        return []

    async def fetch_issue_comments(self, issue_id: str) -> list[Comment]:
        """Return all comments stored for an issue, oldest first."""
        comment_path = self._comments_path(issue_id)
        if not comment_path.exists():
            return []

        comments: list[Comment] = []
        for line in comment_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            comments.append(
                Comment(
                    id=_string_or_none(payload.get("id")),
                    body=_string_or_none(payload.get("body")),
                    author_login=_string_or_none(payload.get("author_login")),
                    created_at=_string_or_none(payload.get("created_at")),
                    updated_at=_string_or_none(payload.get("updated_at")),
                    in_reply_to_id=_string_or_none(payload.get("in_reply_to_id")),
                )
            )
        return comments

    async def fetch_new_comments_since(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> list[Comment]:
        """Return comments newer than ``since_comment_id`` for polling.

        Returns:
            All comments when ``since_comment_id`` is None, otherwise the
            comments that follow the cursor (inclusive of nothing beyond).
        """
        comments = await self.fetch_issue_comments(issue_id)
        if since_comment_id is None:
            return comments
        for index, comment in enumerate(comments):
            if comment.id == since_comment_id:
                return comments[index + 1 :]
        return comments

    async def create_clarification_comment(
        self,
        issue_id: str,
        body: str,
        mentions: list[str] | None = None,
    ) -> Comment | None:
        """Create a comment that mentions users and asks for clarification."""
        prefix = " ".join(f"@{mention}" for mention in mentions or [])
        comment_body = f"{prefix}\n\n{body}".strip() if prefix else body
        return self._append_comment(issue_id, comment_body)

    async def extract_intent_from_labels(
        self,
        labels: list[str] | None,
    ) -> Intent:
        """Resolve the operator intent from a set of issue labels."""
        return intent_from_label_set(labels, self.intent_labels)

    async def add_label(self, issue_id: str, label: str) -> bool:
        """Append ``label`` to the issue's frontmatter list.

        Idempotent: adding a label that is already present returns
        True without rewriting the file. Missing issues return False
        (we never auto-create issues from a label mutation).
        """
        try:
            document = self._document_for_issue(issue_id)
        except FileNotFoundError:
            logger.warning("LocalTrackerAdapter.add_label: issue %s not found", issue_id)
            return False
        existing = list(document.issue.labels or [])
        if label in existing:
            return True
        existing.append(label)
        write_markdown_frontmatter(
            document.path,
            {"labels": existing, "updated_at": utc_now_iso()},
        )
        logger.info(
            "LocalTrackerAdapter.add_label: added %r to issue %s",
            label,
            issue_id,
        )
        return True

    async def remove_label(self, issue_id: str, label: str) -> bool:
        """Drop ``label`` from the issue's frontmatter list.

        Idempotent: removing a label that is already absent returns
        True without rewriting the file. Missing issues return False.
        """
        try:
            document = self._document_for_issue(issue_id)
        except FileNotFoundError:
            logger.warning("LocalTrackerAdapter.remove_label: issue %s not found", issue_id)
            return False
        existing = list(document.issue.labels or [])
        if label not in existing:
            return True
        existing.remove(label)
        write_markdown_frontmatter(
            document.path,
            {"labels": existing, "updated_at": utc_now_iso()},
        )
        logger.info(
            "LocalTrackerAdapter.remove_label: removed %r from issue %s",
            label,
            issue_id,
        )
        return True

    async def close_pull_request(
        self,
        pull_request: PullRequestRef,
    ) -> bool:
        """Close a local pull request.

        The local tracker has no remote PR; closing is a no-op because
        the orchestrator resets the frontmatter fields directly.

        Returns:
            True, since the operation always succeeds.
        """
        # LocalTracker has no remote PR — issue_registry tracks the
        # "PR" via frontmatter (pr_number / pr_url). Closing a local
        # PR is a no-op; the orchestrator's reset_for_retry will clear
        # the frontmatter fields directly.
        logger.info(
            "LocalTrackerAdapter.close_pull_request: no-op (pr_number=%s)",
            pull_request.number,
        )
        return True

    async def fetch_issue_command_intent(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> "CommandIntent | None":
        """Scan new comments for a parseable agent command.

        Returns:
            The first ``CommandIntent`` found among comments newer than
            ``since_comment_id``, or None when none is present.
        """
        from ..tracker import CommandIntent, parse_agent_command  # pylint: disable=relative-beyond-top-level

        comments = await self.fetch_new_comments_since(issue_id, since_comment_id)
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

    def _load_documents(self) -> list[LocalIssueDocument]:
        """Load and parse all non-ignored issue documents, sorted by path."""
        if not self.issues_path.exists():
            return []
        documents: list[LocalIssueDocument] = []
        for path in sorted(self.issues_path.glob("*.md")):
            if _is_ignored_issue_path(path):
                continue
            documents.append(parse_markdown_issue(path))
        return documents

    def _document_for_issue(self, issue_id: str) -> LocalIssueDocument:
        """Return the document whose id or identifier matches ``issue_id``.

        Raises:
            FileNotFoundError: When no local issue matches ``issue_id``.
        """
        for document in self._load_documents():
            issue = document.issue
            if issue_id in (issue.id, issue.identifier):
                return document
        raise FileNotFoundError(f"Local issue not found: {issue_id}")

    def _append_comment(self, issue_id: str, body: str) -> Comment:
        """Append a comment record to the issue's NDJSON comment log."""
        self.issues_path.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        comment = Comment(
            id=str(uuid.uuid4()),
            body=body,
            author_login="clawcodex",
            created_at=now,
            updated_at=now,
        )
        payload = {
            "id": comment.id,
            "body": comment.body,
            "author_login": comment.author_login,
            "created_at": comment.created_at,
            "updated_at": comment.updated_at,
        }
        with self._comments_path(issue_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return comment

    def _comments_path(self, issue_id: str) -> Path:
        """Return the NDJSON comment-log path for an issue ID."""
        return self.issues_path / f"{_safe_file_stem(issue_id)}.comments.ndjson"


def _normalize_states(states: tuple[str, ...]) -> set[str]:
    """Normalize a collection of state names into a lowercased set."""
    return {_normalize_state(state) for state in states if _normalize_state(state)}


def _normalize_state(state: str | None) -> str:
    """Normalize a state name to a stripped, lowercased string."""
    return (state or "").strip().lower()


def _is_ignored_issue_path(path: Path) -> bool:
    """Return whether a file name is a non-issue artifact (temp/comment)."""
    name = path.name
    return name.startswith(".") or name.endswith(".tmp") or name.endswith(".comments.md") or ".comments." in name


def _safe_file_stem(value: str) -> str:
    """Build a filesystem-safe stem from an arbitrary issue identifier."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    stem = safe.strip("-._") or "issue"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_STEM_DIGEST_LENGTH]
    return f"{stem}-{digest}"


def _string_or_none(value: Any) -> str | None:
    """Return a stripped string, or None for empty or missing values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
