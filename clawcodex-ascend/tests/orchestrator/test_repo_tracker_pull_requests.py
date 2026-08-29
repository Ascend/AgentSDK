#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

from __future__ import annotations

import os
import sys
import types
import unittest
from collections import deque
from dataclasses import dataclass
from typing import Any

_tracker_mod = types.ModuleType("extensions.orchestrator.tracker")


@dataclass(frozen=True)
class MergeableStatus:
    mergeable: bool | None
    mergeable_state: str | None
    ahead_by: int | None
    behind_by: int | None
    has_conflicts: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class PullRequestFeedback:
    id: str
    source: str
    body: str
    author_login: str | None = None
    file_path: str | None = None
    line: int | None = None
    diff_hunk: str | None = None
    severity: str = "info"
    status: str = "open"
    created_at: str | None = None
    updated_at: str | None = None
    commit_sha: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class PullRequestRef:
    number: str | None = None
    url: str | None = None
    title: str | None = None


_tracker_mod.MergeableStatus = MergeableStatus
_tracker_mod.PullRequestFeedback = PullRequestFeedback
_tracker_mod.PullRequestRef = PullRequestRef

# --- Stub: extensions.orchestrator.repo_tracker.normalizers ---
_normalizers_mod = types.ModuleType("extensions.orchestrator.repo_tracker.normalizers")


class RepositoryTrackerError(Exception):
    """Raised when repository issue tracker operations fail."""


_normalizers_mod.RepositoryTrackerError = RepositoryTrackerError
_normalizers_mod._PAGE_SIZE = 100


@dataclass
class RepositoryPlatform:
    name: str = "github"
    web_host: str = "https://github.com"
    auth_mode: str = "bearer"
    open_state: str = "open"
    closed_state: str = "closed"
    supports_ci_statuses: bool = True


_normalizers_mod.RepositoryPlatform = RepositoryPlatform


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _normalize_pull_request(payload: Any) -> PullRequestRef | None:
    if not isinstance(payload, dict):
        return None
    number = payload.get("number") or payload.get("iid") or payload.get("id")
    url = payload.get("html_url") or payload.get("url")
    title = payload.get("title")
    return PullRequestRef(
        number=str(number) if number is not None else None,
        url=url if isinstance(url, str) else None,
        title=title if isinstance(title, str) else None,
    )


def _extract_ref_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        for key in ("ref", "name", "branch", "label"):
            ref = value.get(key)
            if not isinstance(ref, str) or not ref:
                continue
            return ref.rsplit(":", 1)[-1] if key == "label" else ref
    return None


def _pull_request_matches(payload: dict[str, Any], *, head_branch: str, base_branch: str) -> bool:
    head = _extract_ref_name(payload.get("head") or payload.get("source_branch"))
    base = _extract_ref_name(payload.get("base") or payload.get("target_branch"))
    if head is None and base is None:
        return False
    if head is not None and head != head_branch:
        return False
    if base is not None and base != base_branch:
        return False
    return True


def _payload_has_branch_fields(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("head", "base", "source_branch", "target_branch"))


def _find_pull_request_in_payload(
    payload: Any,
    *,
    head_branch: str,
    base_branch: str,
    allow_unique_unmatched: bool = False,
) -> PullRequestRef | None:
    if not isinstance(payload, list):
        return None
    unmatched: list[PullRequestRef] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        pr = _normalize_pull_request(item)
        if pr is None:
            continue
        if _pull_request_matches(item, head_branch=head_branch, base_branch=base_branch):
            return pr
        if not _payload_has_branch_fields(item):
            unmatched.append(pr)
    if allow_unique_unmatched and len(unmatched) == 1:
        return unmatched[0]
    return None


def _build_issue_comment_url(
    platform: RepositoryPlatform,
    owner: str,
    repo: str,
    number: str,
    comment_id: str,
) -> str | None:
    if not platform.web_host or not owner or not repo or not number or not comment_id:
        return None
    if platform.name == "github":
        return f"{platform.web_host}/{owner}/{repo}/issues/{number}#issuecomment-{comment_id}"
    return f"{platform.web_host}/{owner}/{repo}/issues/{number}#tid-{comment_id}"


def _extract_comment_author(comment: dict[str, Any]) -> str | None:
    user = comment.get("user") or comment.get("author")
    if isinstance(user, dict):
        return user.get("login") or user.get("username") or user.get("name")
    if isinstance(user, str) and user.strip():
        return user
    return None


def _normalize_feedback_status(payload: dict[str, Any]) -> str:
    if payload.get("resolved") is True:
        return "resolved"
    if payload.get("outdated") is True:
        return "outdated"
    return "open"


def _normalize_conversation_feedback(payload: dict[str, Any]) -> PullRequestFeedback | None:
    body = payload.get("body")
    feedback_id = payload.get("id")
    if not isinstance(body, str) or not body.strip() or feedback_id is None:
        return None
    return PullRequestFeedback(
        id=f"conversation:{feedback_id}",
        source="conversation",
        body=body,
        author_login=_extract_comment_author(payload),
        status="open",
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        url=_string_value(payload.get("html_url") or payload.get("url")),
    )


def _normalize_inline_feedback(payload: dict[str, Any]) -> PullRequestFeedback | None:
    body = payload.get("body")
    feedback_id = payload.get("id")
    if not isinstance(body, str) or not body.strip() or feedback_id is None:
        return None
    return PullRequestFeedback(
        id=f"inline_review:{feedback_id}",
        source="inline_review",
        body=body,
        author_login=_extract_comment_author(payload),
        file_path=_string_value(payload.get("path") or payload.get("file_path")),
        line=_int_value(payload.get("line") or payload.get("new_line") or payload.get("position")),
        diff_hunk=_string_value(payload.get("diff_hunk")),
        severity="warning",
        status=_normalize_feedback_status(payload),
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        commit_sha=_string_value(payload.get("commit_id") or payload.get("commit_sha")),
        url=_string_value(payload.get("html_url") or payload.get("url")),
    )


def _normalize_review_feedback(payload: dict[str, Any]) -> PullRequestFeedback | None:
    body = payload.get("body")
    feedback_id = payload.get("id")
    if not isinstance(body, str) or not body.strip() or feedback_id is None:
        return None
    state = str(payload.get("state") or "").strip().lower()
    severity = "error" if state in {"changes_requested", "request_changes"} else "info"
    return PullRequestFeedback(
        id=f"review_summary:{feedback_id}",
        source="review_summary",
        body=body,
        author_login=_extract_comment_author(payload),
        severity=severity,
        status="open",
        created_at=payload.get("submitted_at") or payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        commit_sha=_string_value(payload.get("commit_id") or payload.get("commit_sha")),
        url=_string_value(payload.get("html_url") or payload.get("url")),
    )


def _normalize_ci_feedback(
    payload: dict[str, Any],
    *,
    commit_sha: str,
    max_log_chars_per_check: int,
) -> PullRequestFeedback | None:
    state = str(payload.get("conclusion") or payload.get("state") or "").strip().lower()
    if state not in {"failure", "failed", "error", "cancelled", "timed_out"}:
        return None
    name = _string_value(payload.get("name") or payload.get("context")) or "CI check"
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    summary = _string_value(output.get("summary")) if isinstance(output, dict) else None
    text = _string_value(output.get("text")) if isinstance(output, dict) else None
    description = _string_value(payload.get("description"))
    details_url = _string_value(payload.get("details_url") or payload.get("html_url") or payload.get("target_url"))
    parts = [f"{name} reported {state}."]
    if description:
        parts.append(description)
    if summary:
        parts.append(summary)
    if text:
        parts.append(f"Output:\n{text}")
    body = "\n\n".join(parts)
    if len(body) > max_log_chars_per_check:
        body = body[:max_log_chars_per_check] + "\n...<truncated>"
    feedback_id = payload.get("id") or payload.get("context") or name
    return PullRequestFeedback(
        id=f"ci:{commit_sha}:{feedback_id}",
        source="ci",
        body=body,
        severity="error",
        status="open",
        created_at=payload.get("started_at") or payload.get("created_at"),
        updated_at=payload.get("completed_at") or payload.get("updated_at"),
        commit_sha=commit_sha,
        url=details_url,
    )


def _normalize_mergeable_status(
    payload: Any,
    *,
    platform: str,
    raw: dict[str, Any] | None = None,
) -> MergeableStatus:
    if not isinstance(payload, dict):
        return MergeableStatus(
            mergeable=None,
            mergeable_state=None,
            ahead_by=None,
            behind_by=None,
            has_conflicts=False,
            raw={"platform": platform, "payload": {}},
        )
    mergeable_raw = payload.get("mergeable")
    if isinstance(mergeable_raw, bool):
        mergeable = mergeable_raw
    elif isinstance(mergeable_raw, str):
        lowered = mergeable_raw.strip().lower()
        mergeable = True if lowered in {"true", "1"} else False if lowered in {"false", "0"} else None
    else:
        mergeable = None
    state_raw = payload.get("mergeable_state")
    if isinstance(state_raw, dict):
        msg = state_raw.get("message")
        mergeable_state = str(msg).strip() if isinstance(msg, str) and msg.strip() else None
    elif isinstance(state_raw, str):
        mergeable_state = state_raw.strip() or None
    else:
        mergeable_state = None
    has_conflicts = mergeable is False or mergeable_state == "dirty"
    return MergeableStatus(
        mergeable=mergeable,
        mergeable_state=mergeable_state,
        ahead_by=None,
        behind_by=None,
        has_conflicts=has_conflicts,
        raw={"platform": platform, "payload": payload},
    )


def _is_not_found_error(exc: RepositoryTrackerError) -> bool:
    return "status=404" in str(exc)


_normalizers_mod._build_issue_comment_url = _build_issue_comment_url
_normalizers_mod._find_pull_request_in_payload = _find_pull_request_in_payload
_normalizers_mod._pull_request_matches = _pull_request_matches
_normalizers_mod._payload_has_branch_fields = _payload_has_branch_fields
_normalizers_mod._extract_ref_name = _extract_ref_name
_normalizers_mod._normalize_pull_request = _normalize_pull_request
_normalizers_mod._normalize_conversation_feedback = _normalize_conversation_feedback
_normalizers_mod._normalize_inline_feedback = _normalize_inline_feedback
_normalizers_mod._normalize_review_feedback = _normalize_review_feedback
_normalizers_mod._normalize_ci_feedback = _normalize_ci_feedback
_normalizers_mod._normalize_feedback_status = _normalize_feedback_status
_normalizers_mod._normalize_mergeable_status = _normalize_mergeable_status
_normalizers_mod._string_value = _string_value
_normalizers_mod._int_value = _int_value
_normalizers_mod._extract_comment_author = _extract_comment_author
_normalizers_mod._is_not_found_error = _is_not_found_error

_REPO_TRACKER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "extensions", "orchestrator", "repo_tracker")
)
_EXTENSIONS_PATH = os.path.abspath(os.path.join(_REPO_TRACKER_PATH, "..", "..", ".."))
_ORCHESTRATOR_PATH = os.path.abspath(os.path.join(_REPO_TRACKER_PATH, ".."))

_pkg_ext = types.ModuleType("extensions")
_pkg_ext.__path__ = [_EXTENSIONS_PATH]
_pkg_orch = types.ModuleType("extensions.orchestrator")
_pkg_orch.__path__ = [_ORCHESTRATOR_PATH]
_pkg_repo = types.ModuleType("extensions.orchestrator.repo_tracker")
_pkg_repo.__path__ = [_REPO_TRACKER_PATH]

sys.modules["extensions"] = _pkg_ext
sys.modules["extensions.orchestrator"] = _pkg_orch
sys.modules["extensions.orchestrator.tracker"] = _tracker_mod
sys.modules["extensions.orchestrator.repo_tracker"] = _pkg_repo
sys.modules["extensions.orchestrator.repo_tracker.normalizers"] = _normalizers_mod
_pkg_ext.orchestrator = _pkg_orch
_pkg_orch.tracker = _tracker_mod
_pkg_orch.repo_tracker = _pkg_repo
_pkg_repo.normalizers = _normalizers_mod

# NOW import the system under test.
from extensions.orchestrator.repo_tracker import pull_requests as pr  # noqa: E402

# =============================================================================
# Phase 2: Stub host class providing the mixin's host-class contract
# =============================================================================


class _StubHost:
    """Minimal host class: provides attributes and async helpers the mixin reads."""

    owner: str = "acme"
    repo: str = "widget"

    def __init__(self, platform: RepositoryPlatform | None = None) -> None:
        self.platform = platform or RepositoryPlatform()
        self.requests: list[dict[str, Any]] = []
        # Per-endpoint FIFO: first queued response is consumed first.
        # Allows sequencing multiple calls to the same endpoint (e.g. POST
        # then GET on /repos/.../pulls during ``create_pull_request``).
        self._responses: dict[str, deque[Any]] = {}
        self._paginated: dict[str, list[dict[str, Any]]] = {}
        self._comments: dict[str, list[dict[str, Any]]] = {}

    def queue_response(self, endpoint: str, payload: Any) -> None:
        """Append a payload to the FIFO queue for ``endpoint``."""
        self._responses.setdefault(endpoint, deque()).append(payload)

    def queue_paginated(self, endpoint: str, items: list[dict[str, Any]]) -> None:
        self._paginated[endpoint] = items

    def queue_comments(self, pr_number: str, items: list[dict[str, Any]]) -> None:
        self._comments[pr_number] = items

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        self.requests.append({"method": method, "endpoint": endpoint, "params": params, "json": json, "data": data})
        queue = self._responses.get(endpoint)
        if queue:
            value = queue.popleft()
            if isinstance(value, Exception):
                raise value
            return value
        return {}

    async def _fetch_paginated(self, endpoint: str) -> list[dict[str, Any]]:
        self.requests.append({"method": "PAGINATED", "endpoint": endpoint})
        return list(self._paginated.get(endpoint, []))

    async def fetch_comments(self, pr_number: str) -> list[dict[str, Any]]:
        self.requests.append({"method": "FETCH_COMMENTS", "pr_number": pr_number})
        return list(self._comments.get(pr_number, []))


class _PRHost(_StubHost, pr.RepositoryPullRequestMixin):
    """Test host: stub class mixed with the SUT."""


class TestModuleImport(unittest.TestCase):
    def test_module_loads(self) -> None:
        """Sanity: pull_requests.py imports cleanly with our stubs."""
        self.assertTrue(hasattr(pr, "RepositoryPullRequestMixin"))
        self.assertTrue(callable(pr.RepositoryPullRequestMixin))


class TestFetchPullRequestFeedback(unittest.IsolatedAsyncioTestCase):
    async def test_aggregates_conversation_inline_review(self) -> None:
        host = _PRHost()
        host.queue_comments(
            "9",
            [{"id": 1, "body": "comment", "user": {"login": "u"}}],
        )
        host.queue_paginated(
            "/repos/acme/widget/pulls/9/comments",
            [{"id": 2, "body": "inline", "path": "x.py", "line": 1, "user": {"login": "u"}}],
        )
        host.queue_paginated(
            "/repos/acme/widget/pulls/9/reviews",
            [{"id": 3, "state": "approved", "body": "ok", "user": {"login": "u"}}],
        )
        host.queue_response(
            "/repos/acme/widget/pulls/9",
            {"head": {"sha": "abc123"}},
        )
        host.queue_paginated("/repos/acme/widget/commits/abc123/check-runs", [])

        feedbacks = await host.fetch_pull_request_feedback(
            pull_request=PullRequestRef(number="9"),
        )
        sources = sorted({f.source for f in feedbacks})
        self.assertEqual(sources, ["conversation", "inline_review", "review_summary"])

    async def test_returns_empty_when_pr_number_missing(self) -> None:
        host = _PRHost()
        result = await host.fetch_pull_request_feedback(
            pull_request=PullRequestRef(number=None),
        )
        self.assertEqual(result, [])


class TestFetchPullRequestMergeable(unittest.IsolatedAsyncioTestCase):
    async def test_returns_normalized_status(self) -> None:
        host = _PRHost()
        host.queue_response(
            "/repos/acme/widget/pulls/9",
            {"mergeable": False, "mergeable_state": "dirty"},
        )
        result = await host.fetch_pull_request_mergeable(
            pull_request=PullRequestRef(number="9"),
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.mergeable)
        self.assertTrue(result.has_conflicts)

    async def test_returns_none_on_transport_error(self) -> None:
        host = _PRHost()
        host.queue_response(
            "/repos/acme/widget/pulls/9",
            RepositoryTrackerError("HTTP error: status=500"),
        )
        result = await host.fetch_pull_request_mergeable(
            pull_request=PullRequestRef(number="9"),
        )
        self.assertIsNone(result)


class TestBackfillFeedbackUrl(unittest.TestCase):
    def test_keeps_existing_url(self) -> None:
        host = _PRHost()
        item = PullRequestFeedback(
            id="conversation:42",
            source="conversation",
            body="b",
            url="https://existing/42",
        )
        self.assertEqual(host._backfill_feedback_url(item, "9"), item)

    def test_skips_review_summary_source(self) -> None:
        """review_summary items never get a backfill — they have their own html_url."""
        host = _PRHost()
        item = PullRequestFeedback(
            id="review_summary:42",
            source="review_summary",
            body="b",
        )
        result = host._backfill_feedback_url(item, "9")
        self.assertIsNone(result.url)

    def test_skips_ci_source(self) -> None:
        host = _PRHost()
        item = PullRequestFeedback(
            id="ci:abc:42",
            source="ci",
            body="b",
        )
        result = host._backfill_feedback_url(item, "9")
        self.assertIsNone(result.url)


if __name__ == "__main__":
    unittest.main()
