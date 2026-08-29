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
import json
import tempfile
from pathlib import Path

from extensions.orchestrator.issue_registry import (
    IssueRegistry,
    IssueStatus,
)
from extensions.orchestrator.tracker import Intent


class TestIssueRegistryIntentFields(unittest.TestCase):
    def test_json_round_trip_preserves_intent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            reg = IssueRegistry(path)
            reg.register(issue_id="1", issue_identifier="ISSUE-1")
            reg.mark_intent("1", Intent.RETRY, source="label", command="/agent retry")
            reg.increment_retry_count("1")
            reg.increment_retry_count("1")
            reg.mark_completed("1")

            # Reload from disk
            reloaded = IssueRegistry(path)
            record = reloaded.get("1")
            assert record is not None
            self.assertEqual(record.intent, Intent.RETRY)
            self.assertEqual(record.retry_count, 2)
            self.assertEqual(record.last_command, "/agent retry")
            self.assertEqual(record.intent_source, "label")
            self.assertEqual(record.status, IssueStatus.COMPLETED)

    def test_backward_compat_old_json(self) -> None:
        """A registry.json written before this feature existed (no `intent` field) must
        load cleanly with Intent.NONE / retry_count=0 defaults.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            old_payload = {
                "1": {
                    "issue_id": "1",
                    "issue_identifier": "ISSUE-1",
                    "branch_name": "main",
                    "status": "completed",
                    "pr_number": "7",
                    "pr_url": "https://example.test/pr/7",
                    "base_branch": "main",
                }
            }
            path.write_text(json.dumps(old_payload), encoding="utf-8")

            reg = IssueRegistry(path)
            record = reg.get("1")
            assert record is not None
            self.assertEqual(record.intent, Intent.NONE)
            self.assertEqual(record.retry_count, 0)
            self.assertIsNone(record.last_command)
            self.assertIsNone(record.intent_source)
            self.assertEqual(record.status, IssueStatus.COMPLETED)
            self.assertEqual(record.pr_number, "7")
            # The record still has_pr() — that's the default from before this feature existed;
            # Sub-A must not break this 4-layer defense.
            self.assertTrue(reg.has_pr("1"))
            self.assertTrue(reg.is_completed("1"))

    def test_mark_intent_on_missing_record_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "registry.json")
            self.assertIsNone(reg.mark_intent("missing", Intent.RETRY))

    def test_clear_intent_resets_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "registry.json")
            reg.register(issue_id="1", issue_identifier="ISSUE-1")
            reg.mark_intent("1", Intent.RETRY, source="label")
            reg.clear_intent("1")
            record = reg.get("1")
            assert record is not None
            self.assertEqual(record.intent, Intent.NONE)
            self.assertIsNone(record.intent_source)

    def test_clear_intent_preserves_history_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "registry.json")
            reg.register(issue_id="1", issue_identifier="ISSUE-1")
            reg.mark_intent("1", Intent.RETRY, source="label")
            reg.clear_intent("1", record_intent_history=True)
            record = reg.get("1")
            assert record is not None
            self.assertEqual(record.intent, Intent.NONE)
            self.assertEqual(record.intent_source, "label")  # preserved

    def test_increment_retry_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "registry.json")
            reg.register(issue_id="1", issue_identifier="ISSUE-1")
            reg.increment_retry_count("1")
            reg.increment_retry_count("1")
            reg.increment_retry_count("1")
            record = reg.get("1")
            assert record is not None
            self.assertEqual(record.retry_count, 3)

    def test_increment_retry_count_on_missing_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "registry.json")
            self.assertIsNone(reg.increment_retry_count("missing"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
