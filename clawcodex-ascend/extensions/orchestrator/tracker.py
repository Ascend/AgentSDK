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

"""Tracker adapter protocol for issue tracker backends.

The tracker layer is split into a **core contract** (``TrackerAdapter``,
methods every backend must implement) and **capability protocols**
(``PullRequestCapability``, ``LabelCapability``, ...) for optional feature
groups.  Callers must check ``supports(tracker, SomeCapability)`` before
invoking a capability method; unsupported trackers simply do not have the
method instead of returning a silent no-op.

This replaces the previous design where every optional method defaulted to
``None`` / ``False`` — callers could not distinguish "not supported" from
"failed".  The refactor assumes all adapters ship together in the same
release, so cross-version back-compat defaults are no longer needed.

This module is the single-file home of the adapter contract, the
normalized data models, and the capability protocols.  Two sibling
modules hold the rest of the former ``tracker.py`` contents and are
re-exported here for back-compat:

  - :mod:`extensions.orchestrator.intent` — ``Intent`` / ``Command``
    semantics, label/comment parsing, priority merging.
  - :mod:`extensions.orchestrator.tracker_kinds` — tracker kind registry,
    config validation, and the ``create_tracker_adapter`` factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .issue import Issue
from .intent import (
    DEFAULT_INTENT_LABELS,
    Command,
    Intent,
    command_to_intent,
    intent_from_label_set,
    merge_intents,
    merge_intents_with_cli,
    parse_agent_command,
)
from .tracker_kinds import (
    SUPPORTED_TRACKERS,
    TrackerConfigError,
    TrackerKindInfo,
    create_tracker_adapter,
    default_active_states_for_kind,
    default_terminal_states_for_kind,
    normalize_tracker_kind,
    repository_clone_url_for_tracker,
    tracker_kind_info,
    validate_tracker_config,
)

__all__ = [
    # intent.py re-exports
    "DEFAULT_INTENT_LABELS",
    "Command",
    "Intent",
    "command_to_intent",
    "intent_from_label_set",
    "merge_intents",
    "merge_intents_with_cli",
    "parse_agent_command",
    # tracker_kinds.py re-exports
    "SUPPORTED_TRACKERS",
    "TrackerConfigError",
    "TrackerKindInfo",
    "create_tracker_adapter",
    "default_active_states_for_kind",
    "default_terminal_states_for_kind",
    "normalize_tracker_kind",
    "repository_clone_url_for_tracker",
    "tracker_kind_info",
    "validate_tracker_config",
    # data models
    "Comment",
    "CommandIntent",
    "MergeableStatus",
    "PullRequestFeedback",
    "PullRequestRef",
    # core contract + capability protocols
    "CommentHistoryCapability",
    "CommandIntentCapability",
    "LabelCapability",
    "PullRequestCapability",
    "PullRequestFeedbackCapability",
    "PullRequestMaintenanceCapability",
    "TrackerAdapter",
    "UserIdentityCapability",
    "supports",
]

# ---------------------------------------------------------------------------
# Normalized data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comment:
    """Normalized issue comment."""

    id: str | None = None
    body: str | None = None
    author_login: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    in_reply_to_id: str | None = None  # for threading


@dataclass(frozen=True)
class CommandIntent:
    """A parsed ``/agent ...`` command plus provenance.

    The orchestrator needs the author login to perform the role check
    ("only the issue author or a maintainer may trigger
    ``/agent retry``"). Older callers that only need the command value
    should use ``intent.command``.
    """

    command: Command
    author_login: str | None = None
    comment_id: str | None = None
    comment_body: str | None = None


@dataclass(frozen=True)
class PullRequestFeedback:
    """Normalized pull request review feedback."""

    id: str
    source: Literal["conversation", "inline_review", "review_summary", "ci"]
    body: str
    author_login: str | None = None
    file_path: str | None = None
    line: int | None = None
    diff_hunk: str | None = None
    severity: Literal["info", "warning", "error"] | None = None
    status: Literal["open", "resolved", "outdated"] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    commit_sha: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class MergeableStatus:
    """Normalized PR mergeability report.

    The three platforms we target expose this differently:

    - **GitHub** — ``pull.mergeable`` (bool | None) and
      ``pull.mergeable_state`` (clean/dirty/blocked/unstable/dirty).
    - **Gitee** — same shape as GitHub.
    - **GitCode** — ``mergeable`` is often ``None`` because the
      page is JS-rendered. ``mergeable_state`` may be a nested object
      with a ``conflict_passed`` boolean.

    ``has_conflicts`` is a derived convenience flag set explicitly by
    the adapter (it is NOT auto-computed from ``mergeable`` /
    ``mergeable_state`` here because the meaning of "conflict"
    differs slightly per platform). On GitCode, when the relevant
    fields are missing, the adapter returns
    ``MergeableStatus(mergeable=None, has_conflicts=False)`` to
    signal "unknown, treat as no-op".
    """

    mergeable: bool | None = None
    mergeable_state: str | None = None
    behind_by: int | None = None
    ahead_by: int | None = None
    has_conflicts: bool = False
    # ``raw`` carries platform-specific debug payload that the adapter
    # forwards verbatim.  It must NOT participate in ``__eq__`` or
    # ``__hash__``: the dict is mutable (and thus unhashable), and two
    # instances with the same normalized mergeability but different raw
    # payloads are still the same mergeable status.
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


class TrackerAdapter(ABC):
    """Core adapter contract: methods every tracker backend must implement.

    Optional feature groups live in the capability protocols below
    (``PullRequestCapability`` etc.).  Use :func:`supports` to test
    whether a concrete adapter implements a capability before calling it —
    unsupported trackers simply lack the method.
    """

    @abstractmethod
    async def fetch_candidate_issues(self) -> list[Issue]:
        """Poll the tracker for issues currently in active states.

        Called on every poll cycle; the orchestrator filters and ranks
        the returned candidates before dispatching any of them.
        """

    @abstractmethod
    async def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> dict[str, Issue]:
        """Refresh the current state of running issues by ID.

        Args:
            issue_ids: the issue ids to refresh.

        Returns:
            Mapping from each issue id to its freshest ``Issue`` snapshot.
        """

    @abstractmethod
    async def create_comment(self, issue_id: str, body: str) -> "Comment | None":
        """Post a comment on an issue (used by the agent to report progress).

        Returns:
            The created comment, or ``None`` if the backend cannot
            create comments.
        """

    @abstractmethod
    async def update_issue_state(self, issue_id: str, state: str) -> None:
        """Transition an issue to a new tracker state (e.g. open → in_progress)."""

    # -- comment & clarification (implemented by every backend) --

    @abstractmethod
    async def update_comment(
        self,
        issue_id: str,
        comment_id: str,
        body: str,
    ) -> "Comment | None":
        """Update the body of an existing issue comment.

        Returns:
            The updated comment, or ``None`` if the backend cannot
            update comments.
        """

    @abstractmethod
    async def create_clarification_comment(
        self,
        issue_id: str,
        body: str,
        mentions: list[str] | None = None,
    ) -> "Comment | None":
        """Post a clarification request comment with @mention notifications.

        Args:
            issue_id: the issue to comment on
            body: comment body (should include @mention for authors)
            mentions: list of usernames to @mention

        Returns:
            The created comment, or None if not supported.
        """

    @abstractmethod
    async def extract_intent_from_labels(
        self,
        labels: list[str] | None,
    ) -> Intent:
        """Resolve an operator Intent from the issue's label set.

        Adapters apply platform-specific label conventions; the shared
        priority rules live in ``intent_from_label_set``.
        """

    @abstractmethod
    async def close_pull_request(
        self,
        pull_request: "PullRequestRef",
    ) -> bool:
        """Close a remote pull request (used by the reset/retry path).

        Returns:
            ``True`` if the PR was closed (or was already closed),
            ``False`` if the platform does not support PR closure.
        """


@dataclass(frozen=True)
class PullRequestRef:
    """Normalized pull request reference."""

    number: str | None = None
    url: str | None = None
    title: str | None = None


# ---------------------------------------------------------------------------
# Capability protocols — optional feature groups
# ---------------------------------------------------------------------------
#
# A concrete adapter structurally satisfies a capability when it implements
# every method of the protocol (duck-typed via ``runtime_checkable``).
# Callers gate capability use behind :func:`supports` — the structural
# check itself is documented in :func:`supports` (and the module docstring).


@runtime_checkable
class PullRequestCapability(Protocol):
    """PR creation & lookup.  Implemented by local + repo trackers."""

    async def ensure_pull_request(
        self,
        *,
        issue: Issue,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> "PullRequestRef | None":
        """Ensure a pull request exists for the branch, creating one if needed.

        Returns:
            The PR reference, or ``None`` if the adapter cannot
            create pull requests.
        """

    async def find_pull_request(
        self,
        *,
        head_branch: str,
        base_branch: str,
    ) -> "PullRequestRef | None":
        """Look up an existing pull request for the given branch pair.

        Returns:
            The PR reference if found, otherwise ``None``.
        """


@runtime_checkable
class PullRequestMaintenanceCapability(Protocol):
    """PR metadata update / mergeability probing.

    Implemented by repo trackers (GitHub / Gitee / GitCode).
    """

    async def update_pull_request(
        self,
        *,
        pull_request: "PullRequestRef",
        title: str | None = None,
        body: str | None = None,
    ) -> "PullRequestRef | None":
        """Update pull request metadata (title and/or body).

        Returns:
            The updated PR reference, or ``None`` if the update
            is not supported.
        """

    async def fetch_pull_request_mergeable(
        self,
        pull_request: "PullRequestRef",
    ) -> "MergeableStatus | None":
        """Fetch a normalized PR mergeability report.

        Returns:
            The mergeability report, or ``None`` when the platform
            cannot report mergeability for the PR.
        """


@runtime_checkable
class PullRequestFeedbackCapability(Protocol):
    """PR review feedback fetch & reply.

    Implemented by repo trackers (GitHub / Gitee / GitCode).
    """

    async def fetch_pull_request_feedback(
        self,
        *,
        pull_request: "PullRequestRef",
        issue_id: str | None = None,
        include_ci_failures: bool = True,
        max_log_chars_per_check: int = 12_000,
    ) -> list["PullRequestFeedback"]:
        """Fetch review feedback and CI failures for a pull request.

        Args:
            pull_request: the PR to fetch feedback for.
            issue_id: optional originating issue id for context.
            include_ci_failures: whether to include CI check failures.
            max_log_chars_per_check: cap on CI log characters per check.

        Returns:
            Normalized feedback items across all supported sources.
        """

    async def reply_to_pull_request_feedback(
        self,
        *,
        pull_request: "PullRequestRef",
        feedback: "PullRequestFeedback",
        body: str,
        issue_id: str | None = None,
    ) -> "Comment | None":
        """Reply to a pull request feedback item after a follow-up run.

        Returns:
            The created comment, or ``None`` if the backend cannot
            post replies.
        """


@runtime_checkable
class UserIdentityCapability(Protocol):
    """Platform identity of the token owner (repo trackers)."""

    async def get_authenticated_user(self) -> str | None:
        """Return the platform login of the token owner.

        Returns:
            The login string, or ``None`` if it cannot be detected.
        """


@runtime_checkable
class LabelCapability(Protocol):
    """Issue label management.  Implemented by local + repo trackers."""

    async def add_label(self, issue_id: str, label: str) -> bool:
        """Add a single label to an issue.

        Returns True if the label is now present, False if the adapter
        cannot modify labels.
        """

    async def remove_label(self, issue_id: str, label: str) -> bool:
        """Remove a single label from an issue.

        Returns True if the label is now absent, False if the adapter
        cannot modify labels.
        """


@runtime_checkable
class CommandIntentCapability(Protocol):
    """Issue comment command scanning."""

    async def fetch_issue_command_intent(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> "CommandIntent | None":
        """Scan recent issue comments for a ``/agent ...`` command.

        Returns the first ``CommandIntent`` found, or ``None`` if no
        command is present in the unscanned portion of the comment
        stream.  The returned ``CommandIntent`` MUST include the
        comment's ``author_login`` (for the role check) and
        ``comment_id`` (for the ``command_cursor``).
        """


@runtime_checkable
class CommentHistoryCapability(Protocol):
    """Issue comment history for clarification polling."""

    async def fetch_issue_comments(self, issue_id: str) -> list["Comment"]:
        """Fetch all comments on an issue (used by clarification polling)."""

    async def fetch_new_comments_since(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> list["Comment"]:
        """Fetch comments newer than a given comment ID (incremental polling).

        Returns comments sorted oldest-first so the caller can process
        them in order.
        """


def supports(adapter: TrackerAdapter, capability: type) -> bool:
    """Return True if ``adapter`` implements the given capability protocol.

    Structural check: an adapter satisfies a capability when it defines
    every method of the protocol.  Unsupported trackers simply lack the
    method — callers must gate capability use behind this check.

    Uses ``getattr`` (not ``runtime_checkable``'s static ``isinstance``)
    so dynamically-mocked adapters (``MagicMock`` in tests) are detected
    correctly: ``__getattr__``-generated methods count as present, and
    real adapters are matched by the same duck-typed rule.
    """
    required = getattr(capability, "__protocol_attrs__", None)
    if required is None:
        return isinstance(adapter, capability)
    return all(callable(getattr(adapter, name, None)) for name in required)
