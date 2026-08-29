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
