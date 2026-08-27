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

"""Thin Linear GraphQL client for polling candidate issues."""

# pylint: disable=relative-beyond-top-level

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from ..issue import Issue
from ..title_prefix_filter import matches_title_prefixes, normalize_title_prefix_match, normalize_title_prefixes

logger = logging.getLogger(__name__)

LINEAR_ENDPOINT = "https://api.linear.app/graphql"
_ISSUE_PAGE_SIZE = 50
_MAX_ERROR_BODY_LOG_BYTES = 1_000
_HTTP_TIMEOUT_SECONDS = 30

_CANDIDATE_QUERY = """
query SymphonyLinearPoll($projectSlug: String!, $stateNames: [String!]!, $first: Int!, $relationFirst: Int!, $after: String) {
  issues(filter: {project: {slugId: {eq: $projectSlug}}, state: {name: {in: $stateNames}}}, first: $first, after: $after) {
    nodes {
      id
      identifier
      title
      description
      priority
      state { name }
      branchName
      url
      assignee { id }
      labels { nodes { name } }
      inverseRelations(first: $relationFirst) {
        nodes {
          type
          issue { id identifier state { name } }
        }
      }
      createdAt
      updatedAt
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

_QUERY_BY_IDS = """
query SymphonyLinearIssuesById($ids: [ID!]!, $first: Int!, $relationFirst: Int!) {
  issues(filter: {id: {in: $ids}}, first: $first) {
    nodes {
      id
      identifier
      title
      description
      priority
      state { name }
      branchName
      url
      assignee { id }
      labels { nodes { name } }
      inverseRelations(first: $relationFirst) {
        nodes {
          type
          issue { id identifier state { name } }
        }
      }
      createdAt
      updatedAt
    }
  }
}
"""

_VIEWER_QUERY = """
query SymphonyLinearViewer {
  viewer { id }
}
"""


class LinearGraphQLClient:
    """Async Linear GraphQL client."""

    def __init__(
        self,
        api_key: str,
        endpoint: str = LINEAR_ENDPOINT,
        title_prefixes: list[str] | None = None,
        title_prefix_match: str = "any",
    ) -> None:
        """Initialize the Linear GraphQL client.

        Args:
            api_key: Linear API key used as the authorization header.
            endpoint: Linear GraphQL endpoint URL.
            title_prefixes: Optional title prefixes for candidate filtering.
            title_prefix_match: "any" or "all" match policy for prefixes.
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.configure_title_prefix_filter(title_prefixes, title_prefix_match)

    def configure_title_prefix_filter(self, prefixes: list[str] | None, match: str = "any") -> None:
        """Configure the title-prefix filter used when polling issues."""
        self._title_prefixes = normalize_title_prefixes(prefixes)
        self._title_prefix_match = normalize_title_prefix_match(match)

    async def graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL request against the Linear API.

        Args:
            query: The GraphQL query or mutation string.
            variables: Optional variable bindings for the request.
            operation_name: Optional named operation selector.

        Returns:
            The parsed JSON response body.

        Raises:
            LinearAPIError: On HTTP or transport failures.
        """
        payload: dict[str, Any] = {"query": query, "variables": variables or {}}
        if operation_name:
            payload["operationName"] = operation_name

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SECONDS),
                ) as resp:
                    body = await resp.json()
                    if resp.status != 200:
                        logger.error(
                            "Linear GraphQL request failed status=%s %s",
                            resp.status,
                            _summarize_error_body(body),
                        )
                        raise LinearAPIError(f"Linear API returned status {resp.status}")
                    # GraphQL business errors (invalid slug, permissions,
                    # rate limits, ...) come back with HTTP 200 and an
                    # ``errors`` array. Detect them here so callers never
                    # dereference ``data: null`` and crash with
                    # AttributeError instead of a LinearAPIError.
                    if isinstance(body, dict) and body.get("errors"):
                        logger.error(
                            "Linear GraphQL errors: %s",
                            _summarize_error_body(body["errors"]),
                        )
                        raise LinearAPIError(
                            f"Linear API returned GraphQL errors: {_summarize_error_body(body['errors'])}"
                        )
                    return body
            except aiohttp.ClientError as exc:
                logger.error("Linear GraphQL request failed: %s", exc)
                raise LinearAPIError(f"Linear API request failed: {exc}") from exc

    async def fetch_candidate_issues(
        self,
        project_slug: str,
        active_states: list[str],
        assignee_filter: dict[str, Any] | None = None,
    ) -> list[Issue]:
        """Fetch candidate issues for a project, paginating until exhausted.

        Args:
            project_slug: Linear project slug to query.
            active_states: Issue states considered active candidates.
            assignee_filter: Optional assignee restriction from the adapter.

        Returns:
            The list of normalized candidate issues.
        """
        if not self.api_key:
            raise LinearAPIError("Missing Linear API token")
        if not project_slug:
            raise LinearAPIError("Missing Linear project slug")

        all_issues: list[Issue] = []
        after_cursor: str | None = None
        while True:
            body = await self.graphql(
                _CANDIDATE_QUERY,
                {
                    "projectSlug": project_slug,
                    "stateNames": active_states,
                    "first": _ISSUE_PAGE_SIZE,
                    "relationFirst": _ISSUE_PAGE_SIZE,
                    "after": after_cursor,
                },
            )
            issues, page_info = _decode_page(body, assignee_filter)
            issues = [
                issue
                for issue in issues
                if matches_title_prefixes(issue.title, self._title_prefixes, self._title_prefix_match)
            ]
            all_issues.extend(issues)
            if page_info.get("has_next_page") and page_info.get("end_cursor"):
                after_cursor = page_info["end_cursor"]
            else:
                break
        return all_issues

    async def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> list[Issue]:
        """Fetch issue snapshots by IDs, preserving the requested order.

        This is a targeted state refresh: unlike candidate discovery it
        does not apply project / assignee / active-state / title-prefix
        filters, so an issue that was reassigned or renamed in Linear is
        still returned with its current state.

        Args:
            issue_ids: IDs of the issues to fetch.

        Returns:
            The normalized issues, ordered like ``issue_ids`` (deduplicated).
        """
        if not issue_ids:
            return []

        all_issues: list[Issue] = []
        ids = list(dict.fromkeys(issue_ids))  # preserve order, dedupe
        for i in range(0, len(ids), _ISSUE_PAGE_SIZE):
            batch = ids[i : i + _ISSUE_PAGE_SIZE]
            body = await self.graphql(
                _QUERY_BY_IDS,
                {
                    "ids": batch,
                    "first": len(batch),
                    "relationFirst": _ISSUE_PAGE_SIZE,
                },
            )
            issues, _ = _decode_page(body, None)
            all_issues.extend(issues)

        # Sort by original requested order
        order = {issue_id: idx for idx, issue_id in enumerate(ids)}
        all_issues.sort(key=lambda i: order.get(i.id or "", len(ids)))
        return all_issues

    async def resolve_viewer_id(self) -> str | None:
        """Return the ID of the authenticated Linear viewer, or None."""
        body = await self.graphql(_VIEWER_QUERY, {})
        viewer = (body.get("data") or {}).get("viewer")
        if viewer and isinstance(viewer, dict):
            return viewer.get("id")
        return None


class LinearAPIError(Exception):
    """Raised when a Linear API request fails."""


def _decode_page(
    body: dict[str, Any],
    assignee_filter: dict[str, Any] | None,
) -> tuple[list[Issue], dict[str, Any]]:
    """Decode one page of a Linear issues response.

    Args:
        body: The raw GraphQL response body.
        assignee_filter: Optional assignee restriction to apply.

    Returns:
        A tuple of the normalized issues and page info
        (``has_next_page`` / ``end_cursor``).
    """
    data = (body.get("data") or {}).get("issues", {})
    nodes = data.get("nodes", [])
    page_info = data.get("pageInfo", {})

    issues = []
    for node in nodes:
        issue = _normalize_issue(node, assignee_filter)
        if issue is not None:
            issues.append(issue)

    return issues, {
        "has_next_page": page_info.get("hasNextPage", False),
        "end_cursor": page_info.get("endCursor"),
    }


def _normalize_issue(
    issue: dict[str, Any],
    assignee_filter: dict[str, Any] | None,
) -> Issue | None:
    """Normalize a raw Linear issue node into an ``Issue``."""
    assignee = issue.get("assignee")
    return Issue(
        id=issue.get("id"),
        identifier=issue.get("identifier"),
        title=issue.get("title"),
        description=issue.get("description"),
        priority=_parse_priority(issue.get("priority")),
        state=issue.get("state", {}).get("name") if issue.get("state") else None,
        branch_name=issue.get("branchName"),
        url=issue.get("url"),
        assignee_id=assignee.get("id") if isinstance(assignee, dict) else None,
        blocked_by=_extract_blockers(issue),
        labels=_extract_labels(issue),
        assigned_to_worker=_assigned_to_worker(assignee, assignee_filter),
        created_at=_parse_datetime(issue.get("createdAt")),
        updated_at=_parse_datetime(issue.get("updatedAt")),
    )


def _extract_labels(issue: dict[str, Any]) -> list[str]:
    """Extract lowercased label names from a Linear issue node."""
    labels = issue.get("labels", {}).get("nodes", [])
    return [str(label["name"]).lower() for label in labels if isinstance(label, dict) and label.get("name")]


def _extract_blockers(issue: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract blocking-relation issues from a Linear issue node."""
    relations = issue.get("inverseRelations", {}).get("nodes", [])
    blockers = []
    for rel in relations:
        if isinstance(rel, dict) and str(rel.get("type", "")).strip().lower() == "blocks":
            blocker_issue = rel.get("issue")
            if isinstance(blocker_issue, dict):
                blockers.append(
                    {
                        "id": blocker_issue.get("id"),
                        "identifier": blocker_issue.get("identifier"),
                        "state": blocker_issue.get("state", {}).get("name") if blocker_issue.get("state") else None,
                    }
                )
    return blockers


def _assigned_to_worker(
    assignee: dict[str, Any] | None,
    assignee_filter: dict[str, Any] | None,
) -> bool:
    """Return whether the assignee satisfies the optional assignee filter."""
    if assignee_filter is None:
        return True
    match_values = assignee_filter.get("match_values")
    if not match_values:
        return False
    aid = assignee.get("id") if isinstance(assignee, dict) else None
    return aid in match_values


def _parse_priority(value: Any) -> int | None:
    """Return the issue priority when it is an integer, else None."""
    if isinstance(value, int):
        return value
    return None


def _parse_datetime(value: str | None) -> Any:
    """Parse an ISO-8601 timestamp into a timezone-aware datetime, or None."""
    from datetime import datetime

    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _summarize_error_body(body: Any) -> str:
    """Collapse an error response body into a compact, truncated string."""
    text = json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else str(body)
    text = " ".join(text.split())
    if len(text) > _MAX_ERROR_BODY_LOG_BYTES:
        text = text[:_MAX_ERROR_BODY_LOG_BYTES] + "...<truncated>"
    return text
