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

import time
import unittest
from typing import Any

from extensions.orchestrator.issue_registry.models import (
    IssueRecord,
    IssueStatus,
    TERMINAL_STATUSES,
)
from extensions.orchestrator.tracker import Intent


def _make_record(
    issue_id: str = "i1",
    issue_identifier: str = "owner/repo#1",
    **kwargs: Any,
) -> IssueRecord:
    return IssueRecord(issue_id=issue_id, issue_identifier=issue_identifier, **kwargs)


class TestIssueRecord(unittest.TestCase):
    def test_defaults(self) -> None:
        record = _make_record()
        self.assertEqual(record.issue_id, "i1")
        self.assertEqual(record.issue_identifier, "owner/repo#1")
        self.assertEqual(record.status, IssueStatus.PENDING)
        self.assertEqual(record.attempt_count, 0)
        self.assertEqual(record.intent, Intent.NONE)
        self.assertEqual(record.retry_count, 0)
        self.assertEqual(record.followup_attempt_count, 0)
        self.assertEqual(record.processed_feedback_ids, [])
        self.assertEqual(record.pending_feedback_ids, [])
        self.assertEqual(base_branch := record.base_branch, "main")
        del base_branch  # silence unused

    def test_touch_updates_updated_at(self) -> None:
        record = _make_record()
        before = record.updated_at
        time.sleep(0.005)
        record.touch()
        self.assertGreater(record.updated_at, before)


class TestIssueStatusEnum(unittest.TestCase):
    def test_terminal_statuses(self) -> None:
        # Sanity-check the documented terminal set.
        for status in (
            IssueStatus.COMPLETED,
            IssueStatus.FAILED,
            IssueStatus.ABANDONED,
            IssueStatus.VERIFICATION_FAILED,
        ):
            self.assertIn(status, TERMINAL_STATUSES)
        for status in (
            IssueStatus.PENDING,
            IssueStatus.RUNNING,
            IssueStatus.SYNCED,
            IssueStatus.PENDING_REVIEW,
            IssueStatus.QUEUED,
        ):
            self.assertNotIn(status, TERMINAL_STATUSES)


# ---------------------------------------------------------------------------
# IssueRegistry — persistence
# ---------------------------------------------------------------------------
