from __future__ import annotations

import json
import unittest
from typing import Any

import httpx

from extensions.orchestrator.repo_tracker.client import RepositoryIssueClient


class TestRepositoryTrackerAdapterClient(unittest.IsolatedAsyncioTestCase):
    async def test_repository_client_create_issue_uses_github_json_bearer(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("Authorization")
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(201, json={"number": 42, "title": "Telemetry"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = RepositoryIssueClient(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=http_client,
            )
            issue = await client.create_issue(title="Telemetry", body="summary")

        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["path"], "/repos/acme/widget/issues")
        self.assertEqual(seen["auth"], "Bearer gh-test-token")
        self.assertEqual(seen["payload"], {"title": "Telemetry", "body": "summary"})
        assert issue is not None
        self.assertEqual(issue["number"], 42)

    async def test_repository_client_update_issue_body_uses_github_json_bearer(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("Authorization")
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"number": 42, "body": "updated"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = RepositoryIssueClient(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=http_client,
            )
            issue = await client.update_issue_body("42", title="Telemetry", body="updated")

        self.assertEqual(seen["method"], "PATCH")
        self.assertEqual(seen["path"], "/repos/acme/widget/issues/42")
        self.assertEqual(seen["auth"], "Bearer gh-test-token")
        self.assertEqual(seen["payload"], {"title": "Telemetry", "body": "updated"})
        assert issue is not None
        self.assertEqual(issue["body"], "updated")

    async def test_repository_client_gitee_issue_writes_use_token_query_and_form_body(self) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "query": dict(request.url.params),
                    "body": request.content.decode("utf-8"),
                }
            )
            return httpx.Response(200, json={"number": 7, "title": "Telemetry"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = RepositoryIssueClient(
                platform="gitee",
                owner="acme",
                repo="widget",
                api_key="gitee-token",
                http_client=http_client,
            )
            await client.create_issue(title="Telemetry", body="daily summary")
            await client.update_issue_body("7", body="updated summary")

        self.assertEqual(seen[0]["method"], "POST")
        self.assertEqual(seen[0]["path"], "/api/v5/repos/acme/widget/issues")
        self.assertEqual(seen[0]["query"]["access_token"], "gitee-token")
        self.assertIn("title=Telemetry", seen[0]["body"])
        self.assertIn("body=daily+summary", seen[0]["body"])
        self.assertEqual(seen[1]["method"], "PATCH")
        self.assertEqual(seen[1]["path"], "/api/v5/repos/acme/widget/issues/7")
        self.assertEqual(seen[1]["query"]["access_token"], "gitee-token")
        self.assertIn("body=updated+summary", seen[1]["body"])

    async def test_repository_client_find_issue_by_title_ignores_pull_requests(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 1,
                        "title": "Telemetry Inbox",
                        "pull_request": {"url": "https://api.github.com/repos/acme/widget/pulls/1"},
                    },
                    {"number": 2, "title": "Other"},
                    {"number": 3, "title": "Telemetry Inbox"},
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = RepositoryIssueClient(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="gh-test-token",
                http_client=http_client,
            )
            issue = await client.find_issue_by_title("Telemetry Inbox")

        assert issue is not None
        self.assertEqual(issue["number"], 3)
        self.assertEqual(requests[0].url.params["state"], "open")
        self.assertEqual(requests[0].url.params["per_page"], "100")

    async def test_gitcode_issue_create_uses_bearer_header_and_form_body(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["query"] = dict(request.url.params)
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = request.content.decode("utf-8")
            return httpx.Response(201, json={"id": 9, "title": "Telemetry"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = RepositoryIssueClient(
                platform="gitcode",
                owner="acme",
                repo="widget",
                api_key="gitcode-token",
                http_client=http_client,
            )
            issue = await client.create_issue(title="Telemetry", body="daily summary")

        self.assertEqual(seen["path"], "/api/v5/repos/acme/widget/issues")
        self.assertNotIn("access_token", seen["query"])
        self.assertEqual(seen["auth"], "Bearer gitcode-token")
        self.assertIn("title=Telemetry", seen["body"])
        self.assertIn("body=daily+summary", seen["body"])
        assert issue is not None
        self.assertEqual(issue["id"], 9)

    async def test_gitcode_comment_cursor_normalizes_newest_first_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers.get("Authorization"), "Bearer gitcode-token")
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 3,
                        "body": "new author reply",
                        "created_at": "2026-07-13T15:38:40+08:00",
                    },
                    {
                        "id": 2,
                        "body": "clarification cursor",
                        "created_at": "2026-07-13T15:37:21+08:00",
                    },
                    {
                        "id": 1,
                        "body": "older comment",
                        "created_at": "2026-07-13T15:30:00+08:00",
                    },
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = RepositoryIssueClient(
                platform="gitcode",
                owner="acme",
                repo="widget",
                api_key="gitcode-token",
                http_client=http_client,
            )
            comments = await client.fetch_comments_since("30", "2")

        self.assertEqual([comment["id"] for comment in comments], [3])
