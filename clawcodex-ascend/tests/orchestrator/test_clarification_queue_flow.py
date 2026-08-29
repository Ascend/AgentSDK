#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.orchestrator.clarification_queue import (
    ClarificationQueue,
    ClarificationStatus,
)


class TestClarificationQueueConflict(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "queue.json"
        self.queue = ClarificationQueue(queue_path=self.path)

    def test_mark_duplicate(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        result = self.queue.mark_duplicate("i1", "dup-answer", 12345.0)
        self.assertEqual(result.status, ClarificationStatus.DUPLICATE_REJECTED)
        self.assertEqual(result.duplicate_of, "12345.0")
        self.assertIn("dup-answer", result.stale_answers)
        self.assertEqual(self.queue.get_stale("i1"), ["dup-answer"])

    def test_mark_duplicate_missing(self) -> None:
        self.assertIsNone(self.queue.mark_duplicate("missing", "x", 0.0))

    def test_mark_stale(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        result = self.queue.mark_stale("i1", "stale", reason="escalated_to_author")
        self.assertEqual(result.status, ClarificationStatus.STALE_REJECTED)
        self.assertIn("stale", result.stale_answers)

    def test_mark_stale_missing(self) -> None:
        self.assertIsNone(self.queue.mark_stale("missing", "x"))

    def test_mark_escalation_notified(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        result = self.queue.mark_escalation_notified("i1")
        self.assertTrue(result.escalation_notified)

    def test_mark_escalation_notified_missing(self) -> None:
        self.assertIsNone(self.queue.mark_escalation_notified("missing"))

    def test_mark_expired_transitions_by_status(self) -> None:
        # AWAITING_LOCAL → TIMED_OUT_LOCAL
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.queue.mark_awaiting_local("i1")
        result = self.queue.mark_expired("i1")
        self.assertEqual(result.status, ClarificationStatus.TIMED_OUT_LOCAL)

        # AWAITING_AUTHOR → TIMED_OUT_AUTHOR
        self.queue.enqueue(issue_id="i2", issue_identifier="x", question="?")
        self.queue.mark_awaiting_author("i2")
        result = self.queue.mark_expired("i2")
        self.assertEqual(result.status, ClarificationStatus.TIMED_OUT_AUTHOR)

        # Other status → EXHAUSTED
        self.queue.enqueue(issue_id="i3", issue_identifier="x", question="?")
        result = self.queue.mark_expired("i3")
        self.assertEqual(result.status, ClarificationStatus.EXHAUSTED)

    def test_mark_expired_missing(self) -> None:
        self.assertIsNone(self.queue.mark_expired("missing"))

    def test_mark_exhausted(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        result = self.queue.mark_exhausted("i1")
        self.assertEqual(result.status, ClarificationStatus.EXHAUSTED)

    def test_mark_issue_failed_writes_sentinel(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.queue.mark_issue_failed("i1")
        sentinel = self.path.parent / ".escalated_issues.json"
        self.assertTrue(sentinel.exists())
        data = json.loads(sentinel.read_text())
        self.assertIn("i1", data)
        self.assertIn("failed_at", data["i1"])

    def test_mark_issue_failed_appends_existing_sentinel(self) -> None:
        self.queue.mark_issue_failed("i1")
        self.queue.mark_issue_failed("i2")
        sentinel = self.path.parent / ".escalated_issues.json"
        data = json.loads(sentinel.read_text())
        self.assertIn("i1", data)
        self.assertIn("i2", data)

    def test_remove(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.assertIsNotNone(self.queue.get("i1"))
        self.queue.remove("i1")
        self.assertIsNone(self.queue.get("i1"))

    def test_remove_missing_is_silent(self) -> None:
        self.queue.remove("missing")  # should not raise


class TestInjectFeedback(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "queue.json"
        self.queue = ClarificationQueue(queue_path=self.path)

    def test_inject_creates_new_item_when_missing(self) -> None:
        result = self.queue.inject_feedback("i1", "please fix the lint")
        self.assertEqual(result.issue_id, "i1")
        self.assertEqual(result.question, "please fix the lint")
        self.assertEqual(result.status, ClarificationStatus.PENDING)
        self.assertEqual(result.context_summary, "Human review rejection feedback")
        self.assertEqual(result.kind, "review_feedback")
        self.assertIs(self.queue.get_pending_feedback("i1"), result)
        self.assertEqual(self.queue.poll_pending(), [])

    def test_inject_resets_existing_item(self) -> None:
        # First, create an item with a question.
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="old?")
        self.queue.mark_awaiting_local("i1")
        self.queue.resolve("i1", "old-answer", "dashboard")
        # Now inject feedback — should reset state.
        result = self.queue.inject_feedback("i1", "new feedback")
        self.assertEqual(result.question, "new feedback")
        self.assertEqual(result.options, [])
        self.assertEqual(result.status, ClarificationStatus.PENDING)
        self.assertEqual(result.kind, "review_feedback")
        self.assertIsNone(result.expires_at)
        self.assertIsNone(result.answer)
        self.assertIsNone(result.answer_source)

    def test_consume_feedback_does_not_remove_real_clarification(self) -> None:
        clarification = self.queue.enqueue(
            issue_id="i1",
            issue_identifier="x",
            question="which mode?",
        )

        self.assertIsNone(self.queue.consume_feedback("i1"))
        self.assertIs(self.queue.get("i1"), clarification)

    def test_consume_feedback_removes_one_shot_item(self) -> None:
        feedback = self.queue.inject_feedback("i1", "please fix the lint")

        self.assertIs(self.queue.consume_feedback("i1"), feedback)
        self.assertIsNone(self.queue.get("i1"))


class TestSaveFailure(unittest.TestCase):
    def test_save_error_does_not_propagate(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "queue.json"
        queue = ClarificationQueue(queue_path=path)
        queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        # Force _save to fail by patching the atomic-write fdopen to raise.
        with patch("os.fdopen", side_effect=OSError("disk full")):
            with self.assertLogs("extensions.orchestrator.clarification_queue", level="WARNING"):
                # Should not raise — write errors are logged.
                queue.enqueue(issue_id="i2", issue_identifier="x", question="?")
        # In-memory state still updated.
        self.assertIsNotNone(queue.get("i2"))


if __name__ == "__main__":
    unittest.main()
