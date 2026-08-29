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
import time
import unittest
from pathlib import Path

from extensions.orchestrator.issue_registry import IssueRegistry


class TestIssueRecordF120Defaults(unittest.TestCase):
    def test_f120_fields_have_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="1", issue_identifier="ISSUE-1")
            record = reg.get("1")
            assert record is not None
            self.assertFalse(record.has_conflict)
            self.assertEqual(tuple(record.conflict_files), ())
            self.assertEqual(record.rebase_attempt_count, 0)
            self.assertIsNone(record.last_rebase_attempt_at)


class TestMarkClearConflict(unittest.TestCase):
    def test_mark_conflict_sets_files_and_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            record = reg.mark_conflict("7", ("src/a.py", "src/b.py"))
            assert record is not None
            self.assertTrue(record.has_conflict)
            self.assertEqual(tuple(record.conflict_files), ("src/a.py", "src/b.py"))
            # Persisted to disk.
            reloaded = IssueRegistry(Path(tmp) / "r.json").get("7")
            assert reloaded is not None
            self.assertTrue(reloaded.has_conflict)
            self.assertEqual(tuple(reloaded.conflict_files), ("src/a.py", "src/b.py"))

    def test_mark_conflict_empty_clears_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.mark_conflict("7", ("src/a.py",))
            reg.mark_conflict("7", ())
            record = reg.get("7")
            assert record is not None
            self.assertTrue(record.has_conflict)
            self.assertEqual(tuple(record.conflict_files), ())

    def test_clear_conflict_resets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.mark_conflict("7", ("src/a.py",))
            reg.clear_conflict("7")
            record = reg.get("7")
            assert record is not None
            self.assertFalse(record.has_conflict)
            self.assertEqual(tuple(record.conflict_files), ())

    def test_clear_conflict_unknown_id_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            self.assertIsNone(reg.clear_conflict("missing"))

    def test_mark_conflict_unknown_id_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            self.assertIsNone(reg.mark_conflict("missing", ("x.py",)))


class TestIncrementRebaseAttempt(unittest.TestCase):
    def test_increment_from_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            record = reg.increment_rebase_attempt("7")
            assert record is not None
            self.assertEqual(record.rebase_attempt_count, 1)
            self.assertIsNotNone(record.last_rebase_attempt_at)

    def test_increment_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            for expected in (1, 2, 3, 4):
                record = reg.increment_rebase_attempt("7")
                assert record is not None
                self.assertEqual(record.rebase_attempt_count, expected)

    def test_increment_unknown_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            self.assertIsNone(reg.increment_rebase_attempt("missing"))


class TestReregisterPreservesF120Fields(unittest.TestCase):
    def test_reregister_keeps_conflict_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.mark_conflict("7", ("src/a.py",))
            # Re-register with new metadata — rebase fields preserved.
            reg.register(issue_id="7", issue_identifier="ISSUE-7-renamed")
            record = reg.get("7")
            assert record is not None
            self.assertTrue(record.has_conflict)
            self.assertEqual(tuple(record.conflict_files), ("src/a.py",))

    def test_reregister_keeps_rebase_attempt_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.increment_rebase_attempt("7")
            reg.increment_rebase_attempt("7")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            record = reg.get("7")
            assert record is not None
            self.assertEqual(record.rebase_attempt_count, 2)


class TestBackwardCompatPreF120Registry(unittest.TestCase):
    def test_legacy_registry_loads_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg_path = Path(tmp) / "r.json"
            legacy_record = {
                "issue_id": "1",
                "issue_identifier": "ISSUE-1",
                "branch_name": "feature/old",
                "status": "completed",
                "attempt_count": 1,
                "retry_count": 0,
            }
            reg_path.write_text(
                json.dumps({"1": legacy_record}),
                encoding="utf-8",
            )
            reg = IssueRegistry(reg_path)
            record = reg.get("1")
            assert record is not None
            self.assertFalse(record.has_conflict)
            self.assertEqual(tuple(record.conflict_files), ())
            self.assertEqual(record.rebase_attempt_count, 0)
            self.assertIsNone(record.last_rebase_attempt_at)


class TestClearStalePending(unittest.TestCase):
    def test_drops_only_expired_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.mark_feedback_pending("7", ["old"], feedback_urls={"old": "https://example/u1"})
            reg.mark_feedback_pending("7", ["new"], feedback_urls={"new": "https://example/u2"})
            record = reg.get("7")
            assert record is not None
            now = time.time()
            record.pending_feedback_since_map["old"] = now - 601
            record.pending_feedback_since_map["new"] = now - 1
            self.assertEqual(reg.clear_stale_pending("7", timeout_seconds=600), 1)
            record = reg.get("7")
            assert record is not None
            self.assertEqual(record.pending_feedback_ids, ["new"])
            self.assertNotIn("old", record.pending_feedback_urls)
            self.assertIn("new", record.pending_feedback_urls)
            self.assertNotIn("old", record.pending_feedback_since_map)
            self.assertIn("new", record.pending_feedback_since_map)

    def test_falls_back_to_legacy_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.mark_feedback_pending("7", ["a", "b"])
            record = reg.get("7")
            assert record is not None
            record.pending_feedback_since_map.clear()
            record.pending_feedback_since = time.time() - 601
            self.assertEqual(reg.clear_stale_pending("7", timeout_seconds=600), 2)
            record = reg.get("7")
            assert record is not None
            self.assertEqual(record.pending_feedback_ids, [])
            self.assertIsNone(record.pending_feedback_since)

    def test_nothing_stale_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.mark_feedback_pending("7", ["a"])
            record = reg.get("7")
            assert record is not None
            record.pending_feedback_since_map["a"] = time.time() - 10
            self.assertEqual(reg.clear_stale_pending("7", timeout_seconds=600), 0)
            self.assertIsNone(reg.clear_stale_pending("missing"))

    def test_clears_maps_when_all_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = IssueRegistry(Path(tmp) / "r.json")
            reg.register(issue_id="7", issue_identifier="ISSUE-7")
            reg.mark_feedback_pending("7", ["a"], feedback_urls={"a": "https://example/u"})
            record = reg.get("7")
            assert record is not None
            record.pending_feedback_since_map["a"] = time.time() - 601
            self.assertEqual(reg.clear_stale_pending("7", timeout_seconds=600), 1)
            record = reg.get("7")
            assert record is not None
            self.assertEqual(record.pending_feedback_ids, [])
            self.assertEqual(record.pending_feedback_urls, {})
            self.assertEqual(record.pending_feedback_since_map, {})
            self.assertIsNone(record.pending_feedback_since)


if __name__ == "__main__":
    unittest.main()
