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

"""Linear-backed tracker adapter."""

# pylint: disable=relative-beyond-top-level

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..tracker import Comment, TrackerAdapter
from ..issue import Issue
from .client import LinearGraphQLClient

if TYPE_CHECKING:
    from ..tracker import Intent, PullRequestRef

logger = logging.getLogger(__name__)

LINEAR_ENDPOINT = "https://api.linear.app/graphql"
LINEAR_ACTIVE_STATES_LIST = ["Todo", "In Progress"]

_CREATE_COMMENT_MUTATION = """
mutation SymphonyCreateComment($issueId: String!, $body: String!) {
  commentCreate(input: {issueId: $issueId, body: $body}) {
    success
    comment {
      id
      body
      createdAt
      updatedAt
      user { name displayName email }
    }
  }
}
"""

_UPDATE_COMMENT_MUTATION = """
mutation SymphonyUpdateComment($commentId: String!, $body: String!) {
  commentUpdate(id: $commentId, input: {body: $body}) {
    success
    comment {
      id
      body
      createdAt
      updatedAt
      user { name displayName email }
    }
  }
}
"""

_UPDATE_STATE_MUTATION = """
mutation SymphonyUpdateIssueState($issueId: String!, $stateId: String!) {
  issueUpdate(id: $issueId, input: {stateId: $stateId}) {
    success
  }
}
"""

_STATE_LOOKUP_QUERY = """
query SymphonyResolveStateId($issueId: String!, $stateName: String!) {
  issue(id: $issueId) {
    team {
      states(filter: {name: {eq: $stateName}}, first: 1) {
        nodes { id }
      }
    }
  }
}
"""


class LinearAdapter(TrackerAdapter):
    """Linear-backed issue tracker via GraphQL API."""

    def __init__(
        self,
        api_key: str,
        project_slug: str | None = None,
        endpoint: str = LINEAR_ENDPOINT,
        active_states: list[str] | None = None,
        assignee: str | None = None,
        intent_labels: dict[str, str] | None = None,
        title_prefixes: list[str] | None = None,
        title_prefix_match: str = "any",
    ) -> None:
        """Initialize the Linear-backed tracker adapter.

        Args:
            api_key: Linear API key used for GraphQL authentication.
            project_slug: Project slug to poll candidate issues from.
            endpoint: Linear GraphQL endpoint URL.
            active_states: Issue states treated as active candidates.
            assignee: Optional assignee filter ("me" or a display name).
            intent_labels: Mapping of label names to intents.
            title_prefixes: Optional title prefixes for candidate filtering.
            title_prefix_match: "any" or "all" match policy for prefixes.
        """
        self.client = LinearGraphQLClient(
            api_key=api_key,
            endpoint=endpoint,
            title_prefixes=title_prefixes,
            title_prefix_match=title_prefix_match,
        )
        self.project_slug = project_slug
        self.active_states = active_states or LINEAR_ACTIVE_STATES_LIST
        self.assignee = assignee
        # same label conventions as the other adapters.
        from ..tracker import DEFAULT_INTENT_LABELS, intent_from_label_set

        self.intent_labels: dict[str, str] = dict(intent_labels) if intent_labels else dict(DEFAULT_INTENT_LABELS)
        self._resolve_intent = intent_from_label_set
        self._assignee_filter: dict[str, Any] | None = None

    def configure_title_prefix_filter(self, prefixes: list[str] | None, match: str = "any") -> None:
        """Configure the title-prefix filter used when polling issues."""
        self.client.configure_title_prefix_filter(prefixes, match)

    async def extract_intent_from_labels(
        self,
        labels: list[str] | None,
    ) -> "Intent":
        """Resolve the operator intent from a set of issue labels."""
        from ..tracker import Intent

        return self._resolve_intent(labels, self.intent_labels) or Intent.NONE

    async def close_pull_request(
        self,
        pull_request: "PullRequestRef",
    ) -> bool:
        """Close a remote pull request.

        Linear has no explicit pull-request model, so this reports the
        operation as unsupported and lets the orchestrator record the
        intent as failed while still resetting the local registry.

        Returns:
            False, because Linear does not support remote PR closing.
        """
        # pylint: disable=fixme
        # TODO LinearAdapter does not have a
        # remote GitHub-style PR to close. Linear's PR model is
        # implicit; the issue's state transitions to "Cancelled" or
        # similar. For now we report the operation as unsupported and
        # rely on the orchestrator to record the intent as FAILED
        # (or move on without remote close, since the registry will
        # still reset the local state).
        logger.warning("LinearAdapter.close_pull_request is not implemented; local registry will still be reset")
        return False

    async def fetch_candidate_issues(self) -> list[Issue]:
        """Fetch issues matching the project, active states and assignee."""
        assignee_filter = await self._resolve_assignee_filter()
        return await self.client.fetch_candidate_issues(
            project_slug=self.project_slug or "",
            active_states=self.active_states,
            assignee_filter=assignee_filter,
        )

    async def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> dict[str, Issue]:
        """Fetch current state snapshots for the given issue IDs."""
        issues = await self.client.fetch_issue_states_by_ids(issue_ids)
        return {issue.id: issue for issue in issues if issue.id}

    async def create_comment(self, issue_id: str, body: str) -> Comment | None:
        """Create a comment on an issue and return the created comment."""
        body_resp = await self.client.graphql(
            _CREATE_COMMENT_MUTATION,
            {"issueId": issue_id, "body": body},
        )
        result = (body_resp.get("data") or {}).get("commentCreate", {})
        if result.get("success") is not True:
            raise LinearAdapterError("comment_create_failed")
        return _comment_from_node(result.get("comment"))

    async def update_comment(
        self,
        issue_id: str,
        comment_id: str,
        body: str,
    ) -> Comment | None:
        """Update an existing comment body and return the updated comment."""
        body_resp = await self.client.graphql(
            _UPDATE_COMMENT_MUTATION,
            {"commentId": comment_id, "body": body},
        )
        result = (body_resp.get("data") or {}).get("commentUpdate", {})
        if result.get("success") is not True:
            raise LinearAdapterError("comment_update_failed")
        return _comment_from_node(result.get("comment"))

    async def create_clarification_comment(
        self,
        issue_id: str,
        body: str,
        mentions: list[str] | None = None,
    ) -> Comment | None:
        """Create a comment that mentions users and asks for clarification."""
        mention_prefix = " ".join(f"@{login.strip()}" for login in (mentions or []) if login.strip())
        comment_body = f"{mention_prefix}\n\n{body}" if mention_prefix else body
        return await self.create_comment(issue_id, comment_body)

    async def update_issue_state(self, issue_id: str, state: str) -> None:
        """Update an issue to the given workflow state."""
        # Map orchestrator-internal terminal states to Linear-compatible
        # workflow state names. Linear teams commonly name their final
        # state "Cancelled" (or a variant); falling back to the raw
        # state name preserves backward compatibility for users who
        # define matching custom states.
        _LINEAR_STATE_FALLBACKS: dict[str, str] = {
            "failed": "Cancelled",
            "abandoned": "Cancelled",
            "verification_failed": "Cancelled",
            "cancelled": "Cancelled",
            "canceled": "Cancelled",
        }
        resolved = _LINEAR_STATE_FALLBACKS.get(state.strip().lower(), state)
        state_id = await self._resolve_state_id(issue_id, resolved)
        body = await self.client.graphql(
            _UPDATE_STATE_MUTATION,
            {"issueId": issue_id, "stateId": state_id},
        )
        success = (body.get("data") or {}).get("issueUpdate", {}).get("success") is True
        if not success:
            raise LinearAdapterError("issue_update_failed")

    async def _resolve_state_id(self, issue_id: str, state_name: str) -> str:
        """Resolve a Linear state ID for the issue's team by state name."""
        body = await self.client.graphql(
            _STATE_LOOKUP_QUERY,
            {"issueId": issue_id, "stateName": state_name},
        )
        data = body.get("data") or {}
        issue = data.get("issue") or {}
        team = issue.get("team") or {}
        states = team.get("states") or {}
        nodes = states.get("nodes") or []
        if nodes and isinstance(nodes, list):
            state_id = nodes[0].get("id")
            if state_id:
                return state_id
        raise LinearAdapterError(f"state_not_found: {state_name}")

    async def _resolve_assignee_filter(self) -> dict[str, Any] | None:
        """Resolve and cache the assignee filter for candidate queries."""
        if self._assignee_filter is not None:
            return self._assignee_filter

        if not self.assignee:
            self._assignee_filter = None
            return None

        normalized = self.assignee.strip()
        if normalized.lower() == "me":
            viewer_id = await self.client.resolve_viewer_id()
            if viewer_id:
                self._assignee_filter = {
                    "configured_assignee": "me",
                    "match_values": {viewer_id},
                }
            else:
                logger.warning("Could not resolve Linear viewer for assignee='me'")
                self._assignee_filter = None
        else:
            self._assignee_filter = {
                "configured_assignee": normalized,
                "match_values": {normalized},
            }
        return self._assignee_filter


def _comment_from_node(node: Any) -> Comment | None:
    """Build a normalized ``Comment`` from a Linear GraphQL comment node."""
    if not isinstance(node, dict):
        return None
    user = node.get("user") if isinstance(node.get("user"), dict) else {}
    return Comment(
        id=_string_or_none(node.get("id")),
        body=_string_or_none(node.get("body")),
        author_login=(
            _string_or_none(user.get("displayName"))
            or _string_or_none(user.get("name"))
            or _string_or_none(user.get("email"))
        ),
        created_at=_string_or_none(node.get("createdAt")),
        updated_at=_string_or_none(node.get("updatedAt")),
    )


def _string_or_none(value: Any) -> str | None:
    """Return a stripped non-empty string, or None for falsy/empty input."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class LinearAdapterError(Exception):
    """Raised when a Linear adapter operation fails."""
