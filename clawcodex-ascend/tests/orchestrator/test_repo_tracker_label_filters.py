#!/usr/bin/env python3

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

"""Tests for skip_labels / require_any_labels filtering on RepositoryIssueClient.

Covers both the candidate-fetch path and the by-id refresh path, with
case-insensitive matching. `require_any_labels` uses OR semantics — the
issue must carry at least one of the required labels.
"""

from __future__ import annotations

import unittest
from typing import Any

import httpx
from extensions.orchestrator.repo_tracker.adapter import RepositoryTrackerAdapter


def _build_handler(payloads_by_path: dict[str, Any]) -> Any:
    """Return a mock handler that maps request path to canned JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/issues") and request.method == "GET":
            return httpx.Response(200, json=payloads_by_path.get("list", []))
        for key, payload in payloads_by_path.items():
            if key == "list":
                continue
            if request.url.path.endswith(f"/issues/{key}"):
                return httpx.Response(200, json=payload)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    return handler


def _issue(
    number: int,
    labels: list[str],
    state: str = "open",
    title: str = "Test issue",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "state": state,
        "labels": [{"name": name} for name in labels],
        "html_url": f"https://example.test/issues/{number}",
    }


class TestSkipLabels(unittest.IsolatedAsyncioTestCase):
    async def test_skip_label_excludes_issue(self) -> None:
        """An issue with a label in skip_labels must not appear in results."""
        payloads = [
            _issue(1, [], title="clean"),
            _issue(2, ["completed"], title="completed"),
            _issue(3, ["bug"], title="bug"),
        ]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                skip_labels=["completed"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["1", "3"])

    async def test_skip_label_is_case_insensitive(self) -> None:
        payloads = [
            _issue(1, ["Completed"]),
            _issue(2, ["WONTFIX"]),
        ]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                skip_labels=["completed", "wontfix"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual(issues, [])

    async def test_empty_skip_labels_is_noop(self) -> None:
        payloads = [_issue(1, ["anything"]), _issue(2, ["completed"])]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["1", "2"])

    async def test_multiple_skip_labels_use_or_semantics(self) -> None:
        """Any matching label triggers exclusion (denylist = OR)."""
        payloads = [
            _issue(1, ["completed"]),
            _issue(2, ["wontfix"]),
            _issue(3, ["duplicate"]),
            _issue(4, ["unrelated"]),
        ]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                skip_labels=["completed", "wontfix", "duplicate"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["4"])

    async def test_fetch_issue_states_by_ids_respects_skip_labels(self) -> None:
        payloads = {
            "1": _issue(1, ["completed"]),
            "2": _issue(2, []),
        }
        handler = _build_handler(payloads)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                skip_labels=["completed"],
            )
            issues = await adapter.fetch_issue_states_by_ids(["1", "2"])
        self.assertEqual(list(issues.keys()), ["2"])


class TestRequireAnyLabels(unittest.IsolatedAsyncioTestCase):
    async def test_require_any_labels_or_semantics(self) -> None:
        """Issue must carry at least one of the required labels (OR)."""
        payloads = [
            _issue(1, ["priority/high"]),
            _issue(2, ["priority/urgent"]),
            _issue(3, ["team/ai"]),
            _issue(4, []),
        ]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                require_any_labels=["priority/high", "priority/urgent"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual(sorted(i.id for i in issues), ["1", "2"])

    async def test_require_any_labels_case_insensitive(self) -> None:
        payloads = [_issue(1, ["PRIORITY/HIGH"])]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                require_any_labels=["priority/high"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["1"])

    async def test_empty_require_any_labels_is_noop(self) -> None:
        payloads = [_issue(1, []), _issue(2, ["anything"])]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["1", "2"])

    async def test_require_any_labels_with_unrelated_labels(self) -> None:
        """Issue with labels that don't intersect the require set is dropped."""
        payloads = [
            _issue(1, ["unrelated"]),
            _issue(2, ["priority/high"]),
        ]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                require_any_labels=["priority/high"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["2"])

    async def test_fetch_issue_states_by_ids_respects_require_any_labels(self) -> None:
        payloads = {
            "1": _issue(1, []),
            "2": _issue(2, ["priority/high"]),
        }
        handler = _build_handler(payloads)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                require_any_labels=["priority/high"],
            )
            issues = await adapter.fetch_issue_states_by_ids(["1", "2"])
        self.assertEqual(list(issues.keys()), ["2"])


class TestSkipAndRequireCombined(unittest.IsolatedAsyncioTestCase):
    async def test_require_any_evaluated_before_skip(self) -> None:
        """An issue failing require_any_labels is dropped, regardless of skip_labels."""
        payloads = [
            _issue(1, ["completed"]),
            _issue(2, ["priority/high", "completed"]),
            _issue(3, ["priority/high"]),
        ]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                require_any_labels=["priority/high"],
                skip_labels=["completed"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["3"])

    async def test_intent_label_not_confused_with_skip(self) -> None:
        payloads = [
            _issue(1, ["agent:retry"]),
            _issue(2, ["agent:retry", "completed"]),
        ]
        handler = _build_handler({"list": payloads})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-token",
                http_client=client,
                skip_labels=["completed"],
            )
            issues = await adapter.fetch_candidate_issues()
        self.assertEqual([i.id for i in issues], ["1"])
