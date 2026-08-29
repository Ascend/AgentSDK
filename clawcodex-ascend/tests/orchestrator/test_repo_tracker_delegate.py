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
