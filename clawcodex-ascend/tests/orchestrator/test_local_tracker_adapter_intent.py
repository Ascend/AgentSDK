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
import tempfile

from extensions.orchestrator.local_tracker.adapter import LocalTrackerAdapter
from extensions.orchestrator.tracker import Intent


class TestLocalTrackerAdapterIntent(unittest.IsolatedAsyncioTestCase):
    async def test_default_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LocalTrackerAdapter(issues_path=tmp)
            self.assertIs(
                await adapter.extract_intent_from_labels(["agent:retry"]),
                Intent.RETRY,
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["agent:blocked", "agent:follow-up"]),
                Intent.BLOCKED,
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["agent:retry", "agent:follow-up"]),
                Intent.FOLLOWUP,
            )

    async def test_custom_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LocalTrackerAdapter(
                issues_path=tmp,
                intent_labels={
                    "retry": "rerun",
                    "followup": "followup",
                    "blocked": "blocked",
                },
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["rerun"]),
                Intent.RETRY,
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["blocked"]),
                Intent.BLOCKED,
            )


# ---------------------------------------------------------------------------
# IssueRecord new fields
# ---------------------------------------------------------------------------
