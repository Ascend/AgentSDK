from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from extensions.orchestrator.repo_tracker.adapter import RepositoryTrackerAdapter
from extensions.orchestrator.tracker import MergeableStatus, PullRequestRef


class TestRepositoryTrackerAdapterDelegate(unittest.TestCase):
    """Verifies that the adapter delegates to the client and returns
    the same MergeableStatus (does not transform).
    """

    def _make(self, client: Any) -> Any:
        # Use real adapter without spinning up the real client.

        adapter = RepositoryTrackerAdapter.__new__(RepositoryTrackerAdapter)
        adapter.client = client
        return adapter

    def test_adapter_returns_client_status(self) -> None:
        expected = MergeableStatus(mergeable=False, mergeable_state="dirty")
        client = MagicMock()
        client.fetch_pull_request_mergeable = AsyncMock(return_value=expected)
        adapter = self._make(client)
        pr = PullRequestRef(number=42, url="https://example/pr/42")
        # Run the async method synchronously.
        import asyncio

        result = asyncio.run(adapter.fetch_pull_request_mergeable(pull_request=pr))
        self.assertIs(result, expected)
        client.fetch_pull_request_mergeable.assert_awaited_once_with(
            pull_request=pr,
        )


if __name__ == "__main__":
    unittest.main()
