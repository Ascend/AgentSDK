from __future__ import annotations

import unittest
import httpx

from extensions.orchestrator.repo_tracker.client import RepositoryIssueClient


class TestClientDirect(unittest.IsolatedAsyncioTestCase):
    """Lower-level tests against RepositoryIssueClient (no adapter layer)."""

    async def test_client_construction_stores_frozensets(self) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as client:
            c = RepositoryIssueClient(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="x",
                http_client=client,
                skip_labels=["Completed", "wontfix"],
                require_any_labels=["P0"],
            )
        self.assertEqual(c._skip_labels, frozenset({"completed", "wontfix"}))
        self.assertEqual(c._require_any_labels, frozenset({"p0"}))

    async def test_client_default_has_empty_label_sets(self) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as client:
            c = RepositoryIssueClient(
                platform="github",
                owner="acme",
                repo="widget",
                api_key="x",
                http_client=client,
            )
        self.assertEqual(c._skip_labels, frozenset())
        self.assertEqual(c._require_any_labels, frozenset())
