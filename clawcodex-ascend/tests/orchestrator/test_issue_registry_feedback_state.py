from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from extensions.orchestrator.issue_registry import IssueRegistry


class TestIssueRegistryFeedbackState(unittest.TestCase):
    def test_feedback_state_is_persisted_and_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry = IssueRegistry(registry_path)
            registry.register(
                "42",
                "#42",
                branch_name="clawcodex/issue-42",
            )
            registry.mark_synced(
                "42",
                branch_name="clawcodex/issue-42",
                pr_number="7",
                pr_url="https://example.test/pr/7",
            )

            registry.mark_feedback_pending(
                "42",
                ["conversation:1", "inline_review:2", "conversation:1"],
                cursor="cursor-1",
            )
            registry.mark_feedback_processed(
                "42",
                ["conversation:1"],
                commit_sha="abc123",
            )
            registry.increment_followup_attempt("42")

            reloaded = IssueRegistry(registry_path)
            record = reloaded.get("42")

        assert record is not None
        self.assertEqual(record.pending_feedback_ids, ["inline_review:2"])
        self.assertEqual(record.processed_feedback_ids, ["conversation:1"])
        self.assertEqual(record.feedback_cursor, "cursor-1")
        self.assertEqual(record.last_followup_commit_sha, "abc123")
        self.assertEqual(record.followup_attempt_count, 1)
        self.assertTrue(reloaded.can_follow_up("42", 2))
        self.assertFalse(reloaded.can_follow_up("42", 1))
        self.assertEqual(len(reloaded.iter_records_with_pr()), 1)

    def test_registry_load_rejects_unknown_fields(self) -> None:
        # 严格加载：registry.json 中出现 dataclass 未知字段 → TypeError
        # （旧实现是静默过滤；重构后数据损坏/外来文件应显式暴露）。
        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "42": {
                            "issue_id": "42",
                            "issue_identifier": "#42",
                            "branch_name": "clawcodex/issue-42",
                            "unknown_future_field": "ignored",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(TypeError):
                IssueRegistry(registry_path)
