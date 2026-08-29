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
from extensions.orchestrator.tracker import PullRequestFeedback, PullRequestRef


class TestRepositoryTrackerAdapterPrFeedback(unittest.IsolatedAsyncioTestCase):
    async def test_gitcode_fetch_pull_request_feedback_normalizes_comments_and_ci(self) -> None:
        requests: list[httpx.Request] = []
        long_summary = "x" * 40

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertIsNone(request.url.params.get("access_token"))
            self.assertEqual(request.headers.get("Authorization"), "Bearer gitcode-token")
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/issues/42/comments":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": 101,
                            "body": "Please update docs",
                            "user": {"login": "reviewer"},
                            "html_url": "https://gitcode.test/comment/101",
                            "created_at": "2026-01-01T00:00:00Z",
                        }
                    ],
                )
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/pulls/9/comments":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": 202,
                            "body": "This branch is wrong",
                            "user": {"login": "reviewer"},
                            "path": "src/app.py",
                            "line": 12,
                            "diff_hunk": "@@ -1 +1 @@",
                            "commit_id": "headsha",
                            "outdated": False,
                        }
                    ],
                )
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/pulls/9/reviews":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": 303,
                            "body": "Changes requested",
                            "state": "changes_requested",
                            "user": {"login": "lead"},
                            "commit_id": "headsha",
                        }
                    ],
                )
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/pulls/9":
                return httpx.Response(200, json={"head": {"sha": "headsha"}})
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/commits/headsha/statuses":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "ci-1",
                            "context": "pytest",
                            "state": "failed",
                            "description": "Unit tests failed",
                            "target_url": "https://ci.test/1",
                        },
                        {
                            "id": "ci-2",
                            "context": "lint",
                            "state": "success",
                        },
                        {
                            "id": "ci-3",
                            "context": "integration",
                            "state": "error",
                            "description": long_summary,
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
            feedback = await adapter.fetch_pull_request_feedback(
                pull_request=PullRequestRef(number="9"),
                issue_id="42",
                max_log_chars_per_check=30,
            )

        self.assertEqual(
            [item.id for item in feedback],
            [
                "conversation:101",
                "inline_review:202",
                "review_summary:303",
            ],
        )
        self.assertEqual(feedback[1].file_path, "src/app.py")
        self.assertEqual(feedback[1].line, 12)
        self.assertEqual(feedback[1].status, "open")
        self.assertEqual(feedback[2].severity, "error")
        self.assertEqual(len(requests), 4)

    async def test_fetch_pull_request_feedback_skips_missing_optional_review_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/issues/9/comments":
                return httpx.Response(
                    200,
                    json=[{"id": 101, "body": "Please update docs"}],
                )
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/pulls/9/comments":
                return httpx.Response(200, json=[])
            if request.method == "GET" and request.url.path == "/api/v5/repos/acme/widget/pulls/9/reviews":
                return httpx.Response(404, json={"message": "Not Found"})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="gitcode",
                owner="acme",
                repo="widget",
                api_key="gitcode-token",
                http_client=client,
            )
            feedback = await adapter.fetch_pull_request_feedback(
                pull_request=PullRequestRef(number="9"),
                include_ci_failures=False,
            )

        self.assertEqual([item.id for item in feedback], ["conversation:101"])

    async def test_gitcode_reply_to_inline_feedback_strips_normalized_prefix(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["query"] = dict(request.url.params)
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = request.content.decode("utf-8")
            return httpx.Response(
                201,
                json={
                    "id": 404,
                    "body": "Handled",
                    "user": {"login": "clawcodex"},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="gitcode",
                owner="acme",
                repo="widget",
                api_key="gitcode-token",
                http_client=client,
            )
            comment = await adapter.reply_to_pull_request_feedback(
                pull_request=PullRequestRef(number="9"),
                feedback=PullRequestFeedback(
                    id="inline_review:202",
                    source="inline_review",
                    body="Fix this",
                ),
                body="Handled",
            )

        self.assertEqual(
            seen["path"],
            "/api/v5/repos/acme/widget/pulls/9/comments/202/replies",
        )
        self.assertNotIn("access_token", seen["query"])
        self.assertEqual(seen["auth"], "Bearer gitcode-token")
        self.assertIn("body=Handled", seen["body"])
        assert comment is not None
        self.assertEqual(comment.in_reply_to_id, "inline_review:202")

    async def test_conversation_feedback_reply_posts_pr_issue_comment(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(201, json={"id": 505, "body": "Handled"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = RepositoryTrackerAdapter(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=client,
            )
            await adapter.reply_to_pull_request_feedback(
                pull_request=PullRequestRef(number="9"),
                feedback=PullRequestFeedback(
                    id="conversation:101",
                    source="conversation",
                    body="Please update docs",
                ),
                body="Handled",
                issue_id="42",
            )

        self.assertEqual(seen["path"], "/repos/acme/widget/issues/42/comments")
        self.assertEqual(seen["payload"], {"body": "Handled"})
