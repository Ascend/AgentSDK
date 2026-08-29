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
from extensions.orchestrator.repo_tracker.adapter import RepositoryTrackerAdapter
from extensions.orchestrator.tracker import PullRequestRef


class TestRepositoryTrackerAdapterPr(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_pull_request_uses_existing_open_pr(self) -> None:
        seen_requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append((request.method, request.url.path))
            if request.method == "GET" and request.url.path == "/repos/acme/widget/pulls":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "number": 21,
                            "title": "Existing PR",
                            "html_url": "https://github.com/acme/widget/pull/21",
                        }
                    ],
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=client,
            )
            pr = await adapter.ensure_pull_request(
                issue=None,  # type: ignore[arg-type]
                head_branch="feature/issue-1",
                base_branch="main",
                title="PR title",
                body="PR body",
            )

        self.assertEqual(
            pr,
            PullRequestRef(
                number="21",
                title="Existing PR",
                url="https://github.com/acme/widget/pull/21",
            ),
        )
        self.assertEqual(seen_requests, [("GET", "/repos/acme/widget/pulls")])

    async def test_ensure_pull_request_creates_when_missing(self) -> None:
        seen_payloads: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/repos/acme/widget/pulls":
                return httpx.Response(200, json=[])
            if request.method == "POST" and request.url.path == "/repos/acme/widget/pulls":
                seen_payloads.append(json.loads(request.content.decode("utf-8")))
                return httpx.Response(
                    201,
                    json={
                        "number": 22,
                        "title": "Created PR",
                        "html_url": "https://github.com/acme/widget/pull/22",
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=client,
            )
            pr = await adapter.ensure_pull_request(
                issue=None,  # type: ignore[arg-type]
                head_branch="feature/issue-2",
                base_branch="main",
                title="PR title",
                body="PR body",
            )

        self.assertEqual(pr.number, "22")
        self.assertEqual(
            seen_payloads[0],
            {
                "title": "PR title",
                "head": "feature/issue-2",
                "base": "main",
                "body": "PR body",
            },
        )

    async def test_gitcode_find_pull_request_matches_broad_list_by_branch(self) -> None:
        seen_queries: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/pulls":
                seen_queries.append(dict(request.url.params))
                if "head" in request.url.params:
                    return httpx.Response(200, json=[])
                return httpx.Response(
                    200,
                    json=[
                        {
                            "number": 31,
                            "title": "Other PR",
                            "html_url": "https://gitcode.test/acme/widget/pulls/31",
                            "head": {"ref": "feature/other"},
                            "base": {"ref": "main"},
                        },
                        {
                            "number": 33,
                            "title": "Matched PR",
                            "html_url": "https://gitcode.test/acme/widget/pulls/33",
                            "head": {"ref": "feature/issue-3"},
                            "base": {"ref": "main"},
                        },
                    ],
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="gitcode",
                owner="acme",
                repo="widget",
                api_key="gitcode-token",
                http_client=client,
            )
            pr = await adapter.find_pull_request(
                head_branch="feature/issue-3",
                base_branch="main",
            )

        self.assertEqual(pr.number if pr else None, "33")
        self.assertEqual(len(seen_queries), 2)

    async def test_gitcode_create_pull_request_recovers_missing_number_from_list(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.method)
            if request.method == "POST" and request.url.path == "/api/v5/repos/acme/widget/pulls":
                return httpx.Response(201, json={"title": "Created PR"})
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/pulls":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "number": 34,
                            "title": "Created PR",
                            "html_url": "https://gitcode.test/acme/widget/pulls/34",
                            "head": {"ref": "feature/issue-4"},
                            "base": {"ref": "main"},
                        }
                    ],
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="gitcode",
                owner="acme",
                repo="widget",
                api_key="gitcode-token",
                http_client=client,
            )
            pr = await adapter.client.create_pull_request(
                title="Created PR",
                head_branch="feature/issue-4",
                base_branch="main",
                body="PR body",
            )

        self.assertEqual(
            pr,
            PullRequestRef(
                number="34",
                title="Created PR",
                url="https://gitcode.test/acme/widget/pulls/34",
            ),
        )
        self.assertEqual(requests, ["POST", "GET"])

    async def test_update_pull_request_uses_pull_patch(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "number": 22,
                    "title": "Updated PR",
                    "html_url": "https://github.com/acme/widget/pull/22",
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
            pr = await adapter.update_pull_request(
                pull_request=PullRequestRef(number="22", title="Old PR"),
                title="Updated PR",
                body="updated body",
            )

        self.assertEqual(seen["method"], "PATCH")
        self.assertEqual(seen["path"], "/repos/acme/widget/pulls/22")
        self.assertEqual(seen["payload"], {"title": "Updated PR", "body": "updated body"})
        self.assertEqual(
            pr,
            PullRequestRef(
                number="22",
                title="Updated PR",
                url="https://github.com/acme/widget/pull/22",
            ),
        )
