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

"""Generic repository issue client for GitHub/Gitee/GitCode.

This module hosts the HTTP transport layer and issue-level operations.
Pull-request lifecycle methods live in
:class:`~extensions.orchestrator.repo_tracker.pull_requests.RepositoryPullRequestMixin`
(consumed here via inheritance), and payload normalization lives in
:mod:`extensions.orchestrator.repo_tracker.normalizers`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

# pylint: disable=relative-beyond-top-level
from ..issue import Issue
from ..title_prefix_filter import (
    matches_title_prefixes,
    normalize_title_prefix_match,
    normalize_title_prefixes,
)
from .normalizers import (
    _PAGE_SIZE,
    _build_issue_update_payload,
    _comment_sort_key,
    _extract_labels,
    _matches_assignee,
    _normalize_issue,
    _repository_label_filter,
    RepositoryTrackerError,
)
from .pull_requests import RepositoryPullRequestMixin

logger = logging.getLogger(__name__)

# Max characters of an HTTP error-response body kept in a raised
# RepositoryTrackerError message; longer bodies are truncated.
_SUMMARIZE_BODY_MAX_LEN = 500
_HTTP_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class RepositoryPlatform:
    """Static per-platform behavior for repository-backed trackers."""

    name: str
    default_endpoint: str
    auth_mode: str
    open_state: str
    closed_state: str
    # API parameter name for filtering issues by state (all platforms use "state")
    state_param: str = "state"
    accept_header: str | None = None
    supports_ci_statuses: bool = True
    # Web host for building human-facing comment/issue URLs when the API
    # response omits ``html_url`` (GitCode issue-comments do). Derived from
    # the platform, not the API endpoint.
    web_host: str = ""


_PLATFORMS: dict[str, RepositoryPlatform] = {
    "github": RepositoryPlatform(
        name="github",
        default_endpoint="https://api.github.com",
        auth_mode="bearer",
        open_state="open",
        closed_state="closed",
        accept_header="application/vnd.github+json",
        web_host="https://github.com",
    ),
    "gitee": RepositoryPlatform(
        name="gitee",
        default_endpoint="https://gitee.com/api/v5",
        auth_mode="access_token",
        open_state="open",
        closed_state="closed",
        accept_header="application/json",
        web_host="https://gitee.com",
    ),
    "gitcode": RepositoryPlatform(
        name="gitcode",
        default_endpoint="https://api.gitcode.com/api/v5",
        auth_mode="access_token",
        open_state="open",
        closed_state="closed",
        accept_header="application/json",
        supports_ci_statuses=False,
        web_host="https://gitcode.com",
    ),
}


class RepositoryIssueClient(RepositoryPullRequestMixin):
    """Issue API wrapper for repository-backed trackers."""

    def __init__(
        self,
        *,
        platform: str,
        owner: str,
        repo: str,
        api_key: str | None,
        endpoint: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        skip_labels: list[str] | None = None,
        require_any_labels: list[str] | None = None,
        title_prefixes: list[str] | None = None,
        title_prefix_match: str = "any",
    ) -> None:
        """Initialize the repository issue client.

        Args:
            platform: Repository platform name ("github", "gitee", "gitcode").
            owner: Repository owner (user or organization).
            repo: Repository name.
            api_key: Optional API key for authenticated requests.
            endpoint: Optional API endpoint override.
            http_client: Optional shared httpx async client.
            skip_labels: Labels that exclude an issue from candidates.
            require_any_labels: Labels; at least one must be present.
            title_prefixes: Optional title prefixes for candidate filtering.
            title_prefix_match: "any" or "all" match policy for prefixes.
        """
        try:
            self.platform = _PLATFORMS[platform]
        except KeyError as exc:
            raise RepositoryTrackerError(f"unsupported platform: {platform}") from exc
        self.owner = owner
        self.repo = repo
        self.api_key = api_key or ""
        self.endpoint = (endpoint or self.platform.default_endpoint).rstrip("/")
        self._http_client = http_client
        # Labels for denylist / allowlist filtering (case-insensitive).
        self._skip_labels: frozenset[str] = frozenset(
            label.strip().lower() for label in (skip_labels or []) if label and label.strip()
        )
        self._require_any_labels: frozenset[str] = frozenset(
            label.strip().lower() for label in (require_any_labels or []) if label and label.strip()
        )
        self.configure_title_prefix_filter(title_prefixes, title_prefix_match)

    def configure_title_prefix_filter(self, prefixes: list[str] | None, match: str = "any") -> None:
        """Configure the title-prefix filter used when polling issues."""
        self._title_prefixes = normalize_title_prefixes(prefixes)
        self._title_prefix_match = normalize_title_prefix_match(match)

    async def fetch_candidate_issues(
        self,
        *,
        active_states: list[str],
        assignee: str | None = None,
    ) -> list[Issue]:
        """Fetch open issues as candidates, paginating until exhausted.

        Args:
            active_states: Issue states treated as active candidates.
            assignee: Optional assignee filter.

        Returns:
            The normalized issues passing the title-prefix and label filters.
        """
        page = 1
        issues: list[Issue] = []
        labels = _repository_label_filter(active_states)

        while True:
            params = {
                "state": self.platform.open_state,
                "per_page": _PAGE_SIZE,
                "page": page,
            }
            if labels:
                params["labels"] = ",".join(labels)

            payload = await self._request_json(
                "GET",
                f"/repos/{self.owner}/{self.repo}/issues",
                params=params,
            )
            if not isinstance(payload, list):
                raise RepositoryTrackerError("invalid_issue_list_response")

            for issue in (_normalize_issue(item, active_states=active_states) for item in payload):
                if issue is None or not _matches_assignee(issue, assignee):
                    continue
                if not self._matches_title_prefixes(issue):
                    logger.info("skip_issue_title_prefix issue_id=%s", issue.id)
                    continue
                if not self._matches_any_required_label(issue):
                    logger.info(
                        "skip_issue_require_any issue_id=%s have=%s",
                        issue.id,
                        sorted({label.lower() for label in issue.labels} & self._require_any_labels) or "<none>",
                    )
                    continue
                skipped = self._matched_skip_label(issue)
                if skipped is not None:
                    logger.info(
                        "skip_issue_label issue_id=%s label=%s",
                        issue.id,
                        skipped,
                    )
                    continue
                issues.append(issue)
            if len(payload) < _PAGE_SIZE:
                break
            page += 1

        return issues

    async def fetch_issue_states_by_ids(
        self,
        issue_ids: list[str],
        *,
        active_states: list[str],
        assignee: str | None = None,
    ) -> list[Issue]:
        """Fetch current snapshots for the requested issue IDs.

        Args:
            issue_ids: IDs of the issues to fetch.
            active_states: Issue states treated as active candidates.
            assignee: Optional assignee filter.

        Returns:
            The normalized issues passing the title-prefix and label filters.
        """
        issues: list[Issue] = []
        for issue_id in dict.fromkeys(issue_ids):
            payload = await self._request_json(
                "GET",
                f"/repos/{self.owner}/{self.repo}/issues/{issue_id}",
            )
            issue = _normalize_issue(payload, active_states=active_states)
            if issue is None or not _matches_assignee(issue, assignee):
                continue
            if not self._matches_title_prefixes(issue):
                logger.info("skip_issue_title_prefix issue_id=%s", issue.id)
                continue
            if not self._matches_any_required_label(issue):
                logger.info(
                    "skip_issue_require_any issue_id=%s have=%s",
                    issue.id,
                    sorted({label.lower() for label in issue.labels} & self._require_any_labels) or "<none>",
                )
                continue
            skipped = self._matched_skip_label(issue)
            if skipped is not None:
                logger.info(
                    "skip_issue_label issue_id=%s label=%s",
                    issue.id,
                    skipped,
                )
                continue
            issues.append(issue)
        return issues

    async def create_comment(self, issue_id: str, body: str) -> dict[str, Any] | None:
        """Create a comment on an issue.

        Returns:
            The created comment payload, or None on a non-dict response.
        """
        data: dict[str, Any] = {"body": body}
        result = await self._request_json(
            "POST",
            f"/repos/{self.owner}/{self.repo}/issues/{issue_id}/comments",
            json=data if self.platform.auth_mode == "bearer" else None,
            data=data if self.platform.auth_mode != "bearer" else None,
        )
        return result if isinstance(result, dict) else None

    async def update_comment(self, comment_id: str, body: str) -> dict[str, Any] | None:
        """Update an existing comment's body.

        Returns:
            The updated comment payload, or None on a non-dict response.
        """
        data: dict[str, Any] = {"body": body}
        result = await self._request_json(
            "PATCH",
            f"/repos/{self.owner}/{self.repo}/issues/comments/{comment_id}",
            json=data if self.platform.auth_mode == "bearer" else None,
            data=data if self.platform.auth_mode != "bearer" else None,
        )
        return result if isinstance(result, dict) else None

    async def fetch_comments(self, issue_id: str) -> list[dict[str, Any]]:
        """Fetch all comments on an issue."""
        page = 1
        comments: list[dict[str, Any]] = []
        while True:
            params = {"per_page": _PAGE_SIZE, "page": page}
            payload = await self._request_json(
                "GET",
                f"/repos/{self.owner}/{self.repo}/issues/{issue_id}/comments",
                params=params,
            )
            if not isinstance(payload, list):
                break
            comments.extend(payload)
            if len(payload) < _PAGE_SIZE:
                break
            page += 1
        if self.platform.name == "gitcode":
            # GitCode returns issue comments newest-first, unlike GitHub.
            # Incremental cursor scanning requires a stable oldest-first
            # sequence or every reply newer than the cursor is skipped.
            comments.sort(key=_comment_sort_key)
        return comments

    async def fetch_comments_since(
        self,
        issue_id: str,
        since_comment_id: str | None,
    ) -> list[dict[str, Any]]:
        """Fetch comments newer than since_comment_id for incremental polling."""
        if since_comment_id is None:
            return await self.fetch_comments(issue_id)

        all_comments = await self.fetch_comments(issue_id)

        # GitHub returns comments in chronological order (oldest first).
        # Find the comment with since_comment_id and return newer ones.
        for index, comment in enumerate(all_comments):
            if str(comment.get("id")) == str(since_comment_id):
                return all_comments[index + 1 :]

        # Cursor comment not found (deleted, trimmed, or a stale cursor).
        # Returning an empty list would silently skip every later comment —
        # including operator /agent commands — on each subsequent poll.
        # Mirror the local tracker fallback: return all comments so the
        # caller can re-scan from the beginning.
        logger.warning(
            "fetch_comments_since: cursor comment %s not found for issue %s; falling back to returning all comments",
            since_comment_id,
            issue_id,
        )
        return all_comments

    async def create_issue(
        self,
        *,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Create a new issue on the remote repository.

        Returns:
            The created issue payload, or None on a non-dict response.
        """
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        result = await self._request_json(
            "POST",
            f"/repos/{self.owner}/{self.repo}/issues",
            json=payload if self.platform.auth_mode == "bearer" else None,
            data=payload if self.platform.auth_mode != "bearer" else None,
        )
        return result if isinstance(result, dict) else None

    async def update_issue(
        self,
        issue_id: str,
        *,
        state: str | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Update a remote issue's state and/or labels.

        Best-effort: when the platform is GitCode (``access_token`` auth)
        and the only change is ``state_event=close``, a known platform
        limitation prevents the issue from actually closing (HTTP 200 is
        returned but state stays ``open``). The error is logged at
        WARNING level and swallowed — the caller is not interrupted.
        """
        payload = _build_issue_update_payload(
            state=state,
            labels=labels,
            platform=self.platform,
        )
        if not payload:
            return
        try:
            await self._request_json(
                "PATCH",
                f"/repos/{self.owner}/{self.repo}/issues/{issue_id}",
                json=payload if self.platform.auth_mode == "bearer" else None,
                data=payload if self.platform.auth_mode != "bearer" else None,
            )
        except RepositoryTrackerError as exc:
            # GitCode limitation: state_event=close alone requires at
            # least one extra content field, and even then the close
            # doesn't take effect. Degrade gracefully.
            if (
                self.platform.auth_mode != "bearer"
                and list(payload.keys()) == ["state_event"]
                and "state_event" in str(exc)
                and "400" in str(exc)
            ):
                logger.warning(
                    "update_issue: GitCode API does not support close via "
                    "state_event=close (known platform limitation). "
                    "issue_id=%s state=%s — ignoring.",
                    issue_id,
                    state,
                )
                return
            raise

    async def update_issue_body(
        self,
        issue_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Update an issue's title, body and/or labels.

        Returns:
            The updated issue payload, or None when there is nothing to
            update or the response is not a dict.
        """
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if labels is not None:
            payload["labels"] = labels
        if not payload:
            return None
        result = await self._request_json(
            "PATCH",
            f"/repos/{self.owner}/{self.repo}/issues/{issue_id}",
            json=payload if self.platform.auth_mode == "bearer" else None,
            data=payload if self.platform.auth_mode != "bearer" else None,
        )
        return result if isinstance(result, dict) else None

    async def add_label(self, issue_id: str, label: str) -> bool:
        """Mirror the CLI retry intent onto the remote issue.

        Read-modify-write: fetch the current labels, append ``label``
        if not present, then PATCH the new list. Idempotent: adding
        a label that is already present is a no-op (still returns
        True). On any HTTP error, returns False so the caller can
        log and continue — the registry.intent is the authoritative
        local source of truth, this is just a best-effort mirror.
        """
        current = await self._fetch_issue_labels(issue_id)
        if label in current:
            return True
        new_labels = list(current) + [label]
        try:
            await self.update_issue(issue_id, labels=new_labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "add_label: PATCH failed for issue %s label=%r: %s",
                issue_id,
                label,
                exc,
            )
            return False
        return True

    async def remove_label(self, issue_id: str, label: str) -> bool:
        """Drop a label from the remote issue.

        Read-modify-write symmetric to :meth:`add_label`. Idempotent.
        On HTTP error, returns False — the local registry.intent is
        the source of truth, the remote label is best-effort.
        """
        current = await self._fetch_issue_labels(issue_id)
        if label not in current:
            return True
        new_labels = [item for item in current if item != label]
        try:
            await self.update_issue(issue_id, labels=new_labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "remove_label: PATCH failed for issue %s label=%r: %s",
                issue_id,
                label,
                exc,
            )
            return False
        return True

    async def find_issue_by_title(
        self,
        title: str,
        *,
        state: str = "open",
    ) -> dict[str, Any] | None:
        """Find an open issue by exact title.

        Returns:
            The first matching issue payload, or None when not found.
        """
        page = 1
        while True:
            payload = await self._request_json(
                "GET",
                f"/repos/{self.owner}/{self.repo}/issues",
                params={
                    "state": state or self.platform.open_state,
                    "per_page": _PAGE_SIZE,
                    "page": page,
                },
            )
            if not isinstance(payload, list):
                return None
            for item in payload:
                if not isinstance(item, dict) or item.get("pull_request"):
                    continue
                if item.get("title") == title:
                    return item
            if len(payload) < _PAGE_SIZE:
                return None
            page += 1

    async def get_authenticated_user(self) -> str | None:
        """Return the login of the authenticated token owner, or None."""
        try:
            payload = await self._request_json("GET", "/user")
        except RepositoryTrackerError:
            return None
        if isinstance(payload, dict):
            return payload.get("login") or payload.get("username") or payload.get("name")
        return None

    def _matches_title_prefixes(self, issue: Issue) -> bool:
        """Return whether the issue title matches the configured prefixes."""
        return matches_title_prefixes(issue.title, self._title_prefixes, self._title_prefix_match)

    def _matched_skip_label(self, issue: Issue) -> str | None:
        """Return the first skip_label hit on this issue, or None."""
        if not self._skip_labels:
            return None
        issue_labels = {label.lower() for label in issue.labels}
        matched = issue_labels & self._skip_labels
        return next(iter(matched), None) if matched else None

    def _matches_any_required_label(self, issue: Issue) -> bool:
        """True if at least one required label is on the issue (OR)."""
        if not self._require_any_labels:
            return True
        issue_labels = {label.lower() for label in issue.labels}
        return bool(issue_labels & self._require_any_labels)

    async def _fetch_issue_labels(self, issue_id: str) -> list[str]:
        """Read the current label list for one issue.

        Returns the list of label names as strings. On HTTP error
        or unparseable payload, returns an empty list (the caller
        treats this as "no labels" rather than as a hard failure).
        """
        try:
            payload = await self._request_json(
                "GET",
                f"/repos/{self.owner}/{self.repo}/issues/{issue_id}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_fetch_issue_labels: GET failed for issue %s: %s",
                issue_id,
                exc,
            )
            return []
        if not isinstance(payload, dict):
            return []
        return _extract_labels(payload)

    async def _fetch_paginated(self, path: str) -> list[dict[str, Any]]:
        """Fetch every page of a paginated list endpoint.

        Returns:
            The collected dict payloads across all pages.
        """
        page = 1
        items: list[dict[str, Any]] = []
        while True:
            payload = await self._request_json(
                "GET",
                path,
                params={"per_page": _PAGE_SIZE, "page": page},
            )
            if not isinstance(payload, list):
                break
            items.extend(item for item in payload if isinstance(item, dict))
            if len(payload) < _PAGE_SIZE:
                break
            page += 1
        return items

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        """Perform an authenticated request and parse the JSON response.

        Args:
            method: HTTP method to use.
            path: API path relative to the endpoint.
            params: Optional query parameters.
            json: Optional JSON body (bearer-auth platforms).
            data: Optional form body (access-token platforms).

        Returns:
            The parsed JSON response, or None for empty success bodies.

        Raises:
            RepositoryTrackerError: On invalid JSON response bodies.
        """
        headers = {"User-Agent": "clawcodex-orchestrator"}
        if self.platform.accept_header:
            headers["Accept"] = self.platform.accept_header

        merged_params = dict(params or {})
        if self.platform.name == "gitcode":
            # GitCode accepts bearer auth while still requiring form bodies
            # for mutation endpoints. Keeping credentials out of query
            # parameters prevents httpx/access logs from leaking tokens.
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.platform.auth_mode == "bearer":
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.api_key:
            merged_params["access_token"] = self.api_key

        response = await self._request(
            method,
            f"{self.endpoint}{path}",
            headers=headers,
            params=merged_params,
            json=json,
            data=data,
        )
        try:
            return response.json()
        except ValueError:
            # Some endpoints (e.g. GitCode comment PATCH) return 200 with
            # an empty body. Return None gracefully instead of raising.
            if response.status_code < 400 and not response.content:
                return None
            raise RepositoryTrackerError(f"invalid_json_response status={response.status_code}") from None

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Execute an HTTP request, raising on transport or HTTP errors.

        Raises:
            RepositoryTrackerError: On transport failures or status >= 400.
        """
        client = self._http_client
        should_close = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise RepositoryTrackerError(f"request_failed: {exc}") from exc
        finally:
            if should_close:
                await client.aclose()

        if response.status_code >= 400:
            raise RepositoryTrackerError(
                f"request_failed status={response.status_code} body={_summarize_body(response)}"
            )
        return response


def _summarize_body(response: httpx.Response) -> str:
    """Collapse a response body into a compact, truncated string."""
    text = " ".join(response.text.split())
    if len(text) > _SUMMARIZE_BODY_MAX_LEN:
        return text[:_SUMMARIZE_BODY_MAX_LEN] + "...<truncated>"
    return text
