#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import json
import unittest
from typing import Any

import httpx

from extensions.orchestrator.issue import Issue
from extensions.orchestrator.repo_tracker.adapter import RepositoryTrackerAdapter


class TestRepositoryTrackerAdapterState(unittest.IsolatedAsyncioTestCase):
    async def test_update_issue_state_replaces_conflicting_lifecycle_labels(
        self,
    ) -> None:
        updates: list[tuple[str, str, list[str] | None]] = []

        class _Client:
            async def fetch_issue_states_by_ids(self, *args, **kwargs):
                return [
                    Issue(
                        id="5",
                        identifier="#5",
                        state="opened",
                        labels=["bug", "failed", "completed", "pending_review"],
                    )
                ]

            async def update_issue(
                self,
                issue_id: str,
                *,
                state: str,
                labels: list[str] | None,
            ) -> None:
                updates.append((issue_id, state, labels))

        adapter = RepositoryTrackerAdapter(
            platform="gitcode",
            owner="acme",
            repo="widget",
            api_key="token",
        )
        adapter.client = _Client()

        await adapter.update_issue_state("5", "failed")
        await adapter.update_issue_state("5", "completed")

        self.assertEqual(
            updates,
            [
                ("5", "failed", ["bug", "failed"]),
                ("5", "completed", ["bug", "completed"]),
            ],
        )

    async def test_update_issue_state_can_remove_the_last_lifecycle_label(self) -> None:
        updates: list[tuple[str, str, list[str] | None]] = []

        class _Client:
            async def fetch_issue_states_by_ids(self, *args, **kwargs):
                return [
                    Issue(
                        id="5",
                        identifier="#5",
                        state="opened",
                        labels=["pending_review"],
                    )
                ]

            async def update_issue(
                self,
                issue_id: str,
                *,
                state: str,
                labels: list[str] | None,
            ) -> None:
                updates.append((issue_id, state, labels))

        adapter = RepositoryTrackerAdapter(
            platform="gitcode",
            owner="acme",
            repo="widget",
            api_key="token",
        )
        adapter.client = _Client()

        await adapter.update_issue_state("5", "open")

        self.assertEqual(updates, [("5", "open", [])])

    async def test_open_without_a_prefetched_issue_does_not_clear_unknown_labels(
        self,
    ) -> None:
        updates: list[tuple[str, str, list[str] | None]] = []

        class _Client:
            async def fetch_issue_states_by_ids(self, *args, **kwargs):
                return []

            async def update_issue(
                self,
                issue_id: str,
                *,
                state: str,
                labels: list[str] | None,
            ) -> None:
                updates.append((issue_id, state, labels))

        adapter = RepositoryTrackerAdapter(
            platform="gitcode",
            owner="acme",
            repo="widget",
            api_key="token",
        )
        adapter.client = _Client()

        await adapter.update_issue_state("5", "open")

        self.assertEqual(updates, [("5", "open", None)])

    async def test_github_candidate_fetch_normalizes_and_filters_issues(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/repos/acme/widget/issues":
                payload = [
                    {
                        "number": 12,
                        "title": "Fix failing build",
                        "body": "details",
                        "state": "open",
                        "labels": [{"name": "In Progress"}],
                        "assignee": {"login": "codex-bot"},
                        "html_url": "https://github.com/acme/widget/issues/12",
                    },
                    {
                        "number": 13,
                        "title": "PR masquerading as issue",
                        "state": "open",
                        "pull_request": {"url": "https://api.github.com/repos/acme/widget/pulls/13"},
                    },
                    {
                        "number": 14,
                        "title": "Assigned elsewhere",
                        "state": "open",
                        "labels": [{"name": "In Progress"}],
                        "assignee": {"login": "someone-else"},
                    },
                ]
                return httpx.Response(200, json=payload)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                active_states=["In Progress"],
                assignee="codex-bot",
                http_client=client,
            )

            issues = await adapter.fetch_candidate_issues()

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].id, "12")
        self.assertEqual(issues[0].identifier, "#12")
        self.assertEqual(issues[0].state, "in progress")
        self.assertEqual(issues[0].labels, ["in progress"])
        self.assertEqual(issues[0].assignee_id, "codex-bot")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer gh-test-token")

    async def test_github_issue_branch_is_extracted_from_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 15,
                        "title": "Fix branch workflow",
                        "body": "Branch: feature/issue-15\n\nDo the work.",
                        "state": "open",
                    }
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=client,
            )
            issues = await adapter.fetch_candidate_issues()

        self.assertEqual(issues[0].branch_name, "feature/issue-15")

    async def test_gitee_comment_uses_access_token_query_param(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["query"] = dict(request.url.params)
            body = request.content.decode("utf-8")
            seen["body"] = body
            return httpx.Response(201, json={"id": 1})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="gitee",
                owner="acme",
                repo="widget",
                api_key="gitee-token",
                http_client=client,
            )
            await adapter.create_comment("99", "job finished")

        self.assertEqual(
            seen["path"],
            "/api/v5/repos/acme/widget/issues/99/comments",
        )
        self.assertEqual(seen["query"]["access_token"], "gitee-token")
        self.assertIn("body=job+finished", seen["body"])

    async def test_github_update_comment_uses_issue_comment_patch(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "id": 123,
                    "body": "updated summary",
                    "user": {"login": "clawcodex"},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=client,
            )
            comment = await adapter.update_comment("99", "123", "updated summary")

        self.assertEqual(seen["method"], "PATCH")
        self.assertEqual(seen["path"], "/repos/acme/widget/issues/comments/123")
        self.assertEqual(seen["payload"], {"body": "updated summary"})
        assert comment is not None
        self.assertEqual(comment.id, "123")
        self.assertEqual(comment.body, "updated summary")

    async def test_gitee_update_comment_uses_access_token_form_patch(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["query"] = dict(request.url.params)
            seen["body"] = request.content.decode("utf-8")
            return httpx.Response(200, json={"id": 321, "body": "updated"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="gitee",
                owner="acme",
                repo="widget",
                api_key="gitee-token",
                http_client=client,
            )
            await adapter.update_comment("99", "321", "updated")

        self.assertEqual(seen["method"], "PATCH")
        self.assertEqual(seen["path"], "/api/v5/repos/acme/widget/issues/comments/321")
        self.assertEqual(seen["query"]["access_token"], "gitee-token")
        self.assertIn("body=updated", seen["body"])

    async def test_github_refresh_by_ids_returns_mapping(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            issue_no = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                json={
                    "number": int(issue_no),
                    "title": f"Issue {issue_no}",
                    "state": "open",
                    "labels": [{"name": "Todo"}],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                active_states=["Todo"],
                http_client=client,
            )
            issues = await adapter.fetch_issue_states_by_ids(["7", "8"])

        self.assertEqual(sorted(issues), ["7", "8"])
        self.assertEqual(issues["7"].state, "todo")
