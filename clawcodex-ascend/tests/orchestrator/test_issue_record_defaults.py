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
from extensions.orchestrator.issue_registry.models import IssueRecord
from extensions.orchestrator.tracker import Intent


class TestIssueRecordDefaults(unittest.TestCase):
    def test_default_intent_is_none(self) -> None:
        record = IssueRecord(issue_id="1", issue_identifier="ISSUE-1")
        self.assertEqual(record.intent, Intent.NONE)

    def test_default_retry_count_is_zero(self) -> None:
        record = IssueRecord(issue_id="1", issue_identifier="ISSUE-1")
        self.assertEqual(record.retry_count, 0)

    def test_default_last_command_is_none(self) -> None:
        record = IssueRecord(issue_id="1", issue_identifier="ISSUE-1")
        self.assertIsNone(record.last_command)

    def test_default_intent_source_is_none(self) -> None:
        record = IssueRecord(issue_id="1", issue_identifier="ISSUE-1")
        self.assertIsNone(record.intent_source)

    def test_can_set_explicit_intent(self) -> None:
        record = IssueRecord(
            issue_id="1",
            issue_identifier="ISSUE-1",
            intent=Intent.RETRY,
            retry_count=2,
            last_command="/agent retry",
            intent_source="label",
        )
        self.assertEqual(record.intent, Intent.RETRY)
        self.assertEqual(record.retry_count, 2)
        self.assertEqual(record.last_command, "/agent retry")
        self.assertEqual(record.intent_source, "label")


# ---------------------------------------------------------------------------
# IssueRegistry round-trip
# ---------------------------------------------------------------------------
