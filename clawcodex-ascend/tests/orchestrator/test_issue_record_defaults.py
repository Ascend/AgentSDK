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
