#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
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

"""Unit tests for :mod:`extensions.orchestrator.issue_registry`.

Covers the issue→commit→PR mapping registry, including:

* :class:`IssueRecord` dataclass behaviour (``touch``).
* :class:`IssueRegistry` persistence (load / save round-trip, malformed
  JSON, atomic write via temp+replace, back-compat with missing fields).
* The :meth:`IssueRegistry.register` re-entry semantics — re-registering
  an existing record must preserve prior sync-state, intent, retry
  bookkeeping, and feedback tracking.
* All status mutations: ``mark_synced``, ``mark_running``,
  ``mark_completed``, ``mark_failed``, ``mark_failed_with_reason``,
  ``mark_verification_failed``, ``mark_abandoned``, ``mark_pending_review``.
* Run diagnostics updates: ``update_run_diagnostics``.
* Feedback mutations: ``mark_feedback_pending``, ``mark_feedback_processed``,
  ``clear_stale_pending``, ``mark_feedback_checked``,
  ``increment_followup_attempt``.
* intent / retry bookkeeping: ``mark_intent``, ``clear_intent``,
  ``increment_retry_count``, ``reset_for_retry``, ``unblock``.
* Query helpers: ``get``, ``get_by_identifier``, ``get_by_issue_ref``,
  ``get_by_branch``, ``is_completed``, ``is_terminal``,
  ``iter_records_with_pr``, ``latest_sequential_record``,
  ``running_records``, ``has_processed_feedback``, ``can_follow_up``.
* Clarification-related mutations: ``update_clarification``,
  ``add_stale_answer``.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from extensions.orchestrator.issue_registry import (
    IssueRecord,
    IssueRegistry,
    IssueStatus,
)
from extensions.orchestrator.tracker import Intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(tmp_dir: str | Path) -> IssueRegistry:
    return IssueRegistry(storage_path=Path(tmp_dir) / "registry.json")


def _make_record(
    issue_id: str = "i1",
    issue_identifier: str = "owner/repo#1",
    **kwargs: Any,
) -> IssueRecord:
    return IssueRecord(issue_id=issue_id, issue_identifier=issue_identifier, **kwargs)


# ---------------------------------------------------------------------------
# IssueRecord dataclass
# ---------------------------------------------------------------------------


class TestIssueRegistryPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)

    def test_empty_registry(self) -> None:
        self.assertIsNone(self.registry.get("any"))
        self.assertEqual(list(self.registry._records.values()), [])

    def test_round_trip_persists(self) -> None:
        self.registry.register("i1", "owner/repo#1", branch_name="feat/x")
        reloaded = IssueRegistry(storage_path=Path(self.tmp.name) / "registry.json")
        record = reloaded.get("i1")
        self.assertIsNotNone(record)
        self.assertEqual(record.issue_identifier, "owner/repo#1")
        self.assertEqual(record.branch_name, "feat/x")

    def test_malformed_json_raises(self) -> None:
        # 严格加载：非法 JSON 直接抛出，不再吞异常后空启动。
        path = Path(self.tmp.name) / "registry.json"
        path.write_text("not-valid-json", encoding="utf-8")
        with self.assertRaises(Exception):
            IssueRegistry(storage_path=path)

    def test_invalid_status_string_raises(self) -> None:
        path = Path(self.tmp.name) / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "i1": {
                        "issue_id": "i1",
                        "issue_identifier": "x",
                        "status": "definitely-not-a-real-status",
                    }
                }
            ),
            encoding="utf-8",
        )
        # 严格加载：无效 status 值抛 ValueError，不再静默回退 PENDING。
        with self.assertRaises(ValueError):
            IssueRegistry(storage_path=path)

    def test_invalid_intent_string_raises(self) -> None:
        path = Path(self.tmp.name) / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "i1": {
                        "issue_id": "i1",
                        "issue_identifier": "x",
                        "intent": "definitely-not-a-real-intent",
                    }
                }
            ),
            encoding="utf-8",
        )
        # 严格加载：无效 intent 值抛 ValueError，不再静默回退 NONE。
        with self.assertRaises(ValueError):
            IssueRegistry(storage_path=path)

    def test_unknown_fields_raise(self) -> None:
        # 严格加载：未知字段抛 TypeError（旧实现是静默丢弃 forward-compat）。
        path = Path(self.tmp.name) / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "i1": {
                        "issue_id": "i1",
                        "issue_identifier": "x",
                        "future_field_unknown_to_loader": "whatever",
                    }
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(TypeError):
            IssueRegistry(storage_path=path)


# ---------------------------------------------------------------------------
# IssueRegistry — register() and re-register semantics
# ---------------------------------------------------------------------------


class TestRegister(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)

    def test_register_creates_pending_record(self) -> None:
        record = self.registry.register("i1", "owner/repo#1", branch_name="feat/x", base_branch="main")
        self.assertEqual(record.status, IssueStatus.PENDING)
        self.assertEqual(record.branch_name, "feat/x")
        self.assertEqual(record.base_branch, "main")

    def test_re_register_preserves_sync_state(self) -> None:
        # First registration then mark_synced.
        self.registry.register("i1", "owner/repo#1", branch_name="feat/x")
        self.registry.mark_synced(
            "i1",
            branch_name="feat/x",
            commit_sha="abc123",
            pr_number=42,
            pr_url="https://x/y/pull/42",
        )
        # Re-register with the SAME issue (simulating a re-dispatch).
        re_registered = self.registry.register("i1", "owner/repo#1", branch_name="main")
        # Sync-state must be preserved.
        self.assertEqual(re_registered.commit_sha, "abc123")
        self.assertEqual(re_registered.pr_number, 42)
        self.assertEqual(re_registered.pr_url, "https://x/y/pull/42")
        # Branch name from the prior sync is kept (existing.branch_name wins).
        self.assertEqual(re_registered.branch_name, "feat/x")

    def test_re_register_preserves_intent_and_retry(self) -> None:
        self.registry.register("i1", "owner/repo#1")
        self.registry.mark_intent("i1", Intent.FOLLOWUP, source="label")
        self.registry.increment_retry_count("i1")
        # Re-register and verify intent / retry_count survive.
        re_registered = self.registry.register("i1", "owner/repo#1")
        self.assertEqual(re_registered.intent, Intent.FOLLOWUP)
        self.assertEqual(re_registered.intent_source, "label")
        self.assertEqual(re_registered.retry_count, 1)

    def test_re_register_preserves_feedback_tracking(self) -> None:
        self.registry.register("i1", "owner/repo#1")
        self.registry.mark_feedback_processed("i1", ["fb-1"], commit_sha="c1")
        self.registry.mark_feedback_pending("i1", ["fb-2"], cursor="cur", feedback_urls={"fb-2": "u2"})
        re_registered = self.registry.register("i1", "owner/repo#1")
        self.assertEqual(re_registered.processed_feedback_ids, ["fb-1"])
        self.assertEqual(re_registered.pending_feedback_ids, ["fb-2"])
        self.assertEqual(re_registered.pending_feedback_urls, {"fb-2": "u2"})
        self.assertEqual(re_registered.feedback_cursor, "cur")
        self.assertEqual(re_registered.last_followup_commit_sha, "c1")


# ---------------------------------------------------------------------------
# IssueRegistry — status mutations
# ---------------------------------------------------------------------------


class TestStatusMutations(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)
        self.registry.register("i1", "owner/repo#1")

    def test_mark_synced_updates_and_sets_status(self) -> None:
        result = self.registry.mark_synced(
            "i1",
            branch_name="feat/x",
            commit_sha="abc",
            pr_number=99,
            pr_url="https://x/y/pull/99",
        )
        self.assertEqual(result.status, IssueStatus.SYNCED)
        self.assertEqual(result.branch_name, "feat/x")
        self.assertEqual(result.commit_sha, "abc")
        self.assertEqual(result.pr_number, 99)
        self.assertEqual(result.pr_url, "https://x/y/pull/99")

    def test_mark_synced_only_updates_provided_fields(self) -> None:
        # First sync sets everything.
        self.registry.mark_synced("i1", branch_name="feat/x", commit_sha="a")
        # Second sync only changes commit_sha — other fields must survive.
        result = self.registry.mark_synced("i1", commit_sha="b")
        self.assertEqual(result.commit_sha, "b")
        self.assertEqual(result.branch_name, "feat/x")

    def test_mark_synced_records_pr_created_at_once(self) -> None:
        # First PR creation stamps pr_created_at.
        result = self.registry.mark_synced(
            "i1", branch_name="feat/x", commit_sha="a", pr_number=1, pr_url="https://x/pull/1"
        )
        self.assertIsNotNone(result.pr_created_at)
        first_created = result.pr_created_at
        # Re-sync with the SAME PR must NOT overwrite the first timestamp.
        result2 = self.registry.mark_synced("i1", commit_sha="b", pr_number=1)
        self.assertEqual(result2.pr_created_at, first_created)
        # A different PR number (retry reusing a new PR) resets the clock.
        result3 = self.registry.mark_synced("i1", commit_sha="c", pr_number=2)
        self.assertIsNotNone(result3.pr_created_at)
        self.assertGreaterEqual(result3.pr_created_at, first_created)

    def test_pr_created_at_survives_re_register_and_retry_reset(self) -> None:
        # Stamp a PR first.
        self.registry.mark_synced("i1", branch_name="feat/x", commit_sha="a", pr_number=1, pr_url="https://x/pull/1")
        # Re-register (e.g. daemon relaunch) must preserve pr_created_at.
        self.registry.register("i1", "ISSUE-1", branch_name="feat/x", base_branch="main")
        rec = self.registry.get("i1")
        self.assertIsNotNone(rec.pr_created_at)
        # reset_for_retry must clear pr_created_at so the clock restarts.
        self.registry.reset_for_retry("i1")
        rec2 = self.registry.get("i1")
        self.assertIsNone(rec2.pr_created_at)
        self.assertIsNone(rec2.pr_number)

    def test_mark_synced_missing_returns_none(self) -> None:
        self.assertIsNone(self.registry.mark_synced("missing"))

    def test_mark_running_resets_run_diagnostics(self) -> None:
        # Inject stale run diagnostics.
        self.registry.update_run_diagnostics("i1", run_id="r1", turn_count=10, tool_count=5, last_event="x")
        result = self.registry.mark_running("i1")
        self.assertEqual(result.status, IssueStatus.RUNNING)
        self.assertIsNone(result.run_id)
        self.assertEqual(result.run_turn_count, 0)
        self.assertEqual(result.run_tool_count, 0)
        self.assertIsNone(result.run_last_event)

    def test_mark_running_missing(self) -> None:
        self.assertIsNone(self.registry.mark_running("missing"))

    def test_mark_completed(self) -> None:
        result = self.registry.mark_completed("i1")
        self.assertEqual(result.status, IssueStatus.COMPLETED)
        self.assertTrue(self.registry.is_completed("i1"))
        self.assertTrue(self.registry.is_terminal("i1"))

    def test_mark_failed_increments_attempt(self) -> None:
        self.registry.register("i2", "x/y#2")
        before = self.registry.get("i2").attempt_count
        result = self.registry.mark_failed("i2")
        self.assertEqual(result.status, IssueStatus.FAILED)
        self.assertEqual(result.attempt_count, before + 1)

    def test_mark_failed_with_reason(self) -> None:
        result = self.registry.mark_failed_with_reason("i1", "boom")
        self.assertEqual(result.status, IssueStatus.FAILED)
        self.assertEqual(result.verification_status, "failed")
        self.assertEqual(result.verification_output, "boom")
        self.assertEqual(result.last_hook_error, "boom")

    def test_mark_verification_failed(self) -> None:
        result = self.registry.mark_verification_failed("i1", output="pytest failed", hook_error="exit 1")
        self.assertEqual(result.status, IssueStatus.VERIFICATION_FAILED)
        self.assertEqual(result.verification_status, "failed")
        self.assertEqual(result.verification_output, "pytest failed")
        self.assertEqual(result.last_hook_error, "exit 1")

    def test_mark_abandoned(self) -> None:
        result = self.registry.mark_abandoned("i1")
        self.assertEqual(result.status, IssueStatus.ABANDONED)
        self.assertTrue(self.registry.is_terminal("i1"))

    def test_mark_pending_review(self) -> None:
        result = self.registry.mark_pending_review("i1")
        self.assertEqual(result.status, IssueStatus.PENDING_REVIEW)

    def test_missing_record_returns_none(self) -> None:
        for method, args in [
            (self.registry.mark_completed, ("missing",)),
            (self.registry.mark_failed, ("missing",)),
            (self.registry.mark_abandoned, ("missing",)),
            (self.registry.mark_pending_review, ("missing",)),
            (self.registry.update_branch, ("missing", "x")),
        ]:
            with self.subTest(method=method.__name__):
                self.assertIsNone(method(*args))


# ---------------------------------------------------------------------------
# IssueRegistry — update_branch / update_report
# ---------------------------------------------------------------------------


class TestUpdateMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)
        self.registry.register("i1", "owner/repo#1")

    def test_update_branch(self) -> None:
        result = self.registry.update_branch("i1", "new-branch")
        self.assertEqual(result.branch_name, "new-branch")

    def test_update_report(self) -> None:
        result = self.registry.update_report(
            "i1",
            report_path="/tmp/report.md",
            verification_status="passed",
            verification_output="ok",
            summary_comment_id="c-1",
            session_end_reason="task_complete",
            session_end_summary="done",
        )
        self.assertEqual(result.report_path, "/tmp/report.md")
        self.assertEqual(result.verification_status, "passed")
        self.assertEqual(result.verification_output, "ok")
        self.assertEqual(result.summary_comment_id, "c-1")
        self.assertEqual(result.session_end_reason, "task_complete")
        self.assertEqual(result.session_end_summary, "done")

    def test_update_report_only_changes_provided(self) -> None:
        self.registry.update_report("i1", report_path="/tmp/r.md")
        result = self.registry.update_report("i1", verification_status="passed")
        # report_path kept, verification_status updated.
        self.assertEqual(result.report_path, "/tmp/r.md")
        self.assertEqual(result.verification_status, "passed")


# ---------------------------------------------------------------------------
# IssueRegistry — update_run_diagnostics
# ---------------------------------------------------------------------------


class TestUpdateRunDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)
        self.registry.register("i1", "owner/repo#1")

    def test_update_run_diagnostics(self) -> None:
        result = self.registry.update_run_diagnostics(
            "i1",
            run_id="r1",
            debug_log_path="/tmp/dbg.ndjson",
            turn_count=3,
            tool_count=7,
            last_event="phase",
            last_tool="bash",
            output_len=2000,
            timeout_deadline_at=12345.0,
            workspace_dirty=True,
        )
        self.assertEqual(result.run_id, "r1")
        self.assertEqual(result.debug_log_path, "/tmp/dbg.ndjson")
        self.assertEqual(result.run_turn_count, 3)
        self.assertEqual(result.run_tool_count, 7)
        self.assertEqual(result.run_last_event, "phase")
        self.assertEqual(result.run_last_tool, "bash")
        self.assertEqual(result.run_output_len, 2000)
        self.assertEqual(result.run_timeout_deadline_at, 12345.0)
        self.assertTrue(result.run_workspace_dirty)

    def test_update_run_diagnostics_missing(self) -> None:
        self.assertIsNone(self.registry.update_run_diagnostics("missing", run_id="x"))


# ---------------------------------------------------------------------------
# IssueRegistry — throttled diagnostics save (P2.3-a)
# ---------------------------------------------------------------------------


class TestThrottledDiagnosticsSave(unittest.TestCase):
    """``update_run_diagnostics`` coalesces disk writes; ``flush`` drains.

    Status / PR mutations must still persist immediately and clear the
    pending throttle flag. Setting ``diagnostics_min_save_interval_s=0``
    disables throttling entirely (every call writes).
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "registry.json"

    def _registry(self, interval: float) -> IssueRegistry:
        reg = IssueRegistry(
            storage_path=self.path,
            diagnostics_min_save_interval_s=interval,
        )
        reg.register("i1", "owner/repo#1")
        # register() just performed a durable save and primed the throttle
        # window to "now". Rewind it past the interval so the *first*
        # diagnostics call is deterministically allowed regardless of how
        # long register() took.
        reg._last_diagnostics_save_monotonic = time.monotonic() - (interval + 1.0)
        return reg

    def _on_disk(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_second_call_within_interval_is_throttled(self) -> None:
        # A large interval guarantees the second call falls inside the
        # window. The register() above already performed a durable save
        # and primed the throttle timestamp.
        reg = self._registry(interval=3600.0)
        reg.update_run_diagnostics("i1", turn_count=1)
        # In-memory record is current after the (first, allowed) write.
        self.assertEqual(reg.get("i1").run_turn_count, 1)
        reg.update_run_diagnostics("i1", turn_count=2)
        # Second update is throttled: memory current, disk stale.
        self.assertEqual(reg.get("i1").run_turn_count, 2)
        self.assertTrue(reg._pending_diagnostics_save)
        self.assertEqual(self._on_disk()["i1"].get("run_turn_count"), 1)

    def test_flush_persists_pending_diagnostics(self) -> None:
        reg = self._registry(interval=3600.0)
        reg.update_run_diagnostics("i1", turn_count=1)
        reg.update_run_diagnostics("i1", turn_count=5)
        self.assertTrue(reg._pending_diagnostics_save)
        reg.flush()
        self.assertFalse(reg._pending_diagnostics_save)
        self.assertEqual(self._on_disk()["i1"].get("run_turn_count"), 5)

    def test_flush_without_pending_is_noop(self) -> None:
        reg = self._registry(interval=3600.0)
        reg.update_run_diagnostics("i1", turn_count=1)  # allowed first write
        self.assertFalse(reg._pending_diagnostics_save)
        # No pending write — flush must not error and disk stays as-is.
        reg.flush()
        self.assertEqual(self._on_disk()["i1"].get("run_turn_count"), 1)

    def test_status_mutation_persists_immediately_and_clears_pending(self) -> None:
        reg = self._registry(interval=3600.0)
        reg.update_run_diagnostics("i1", turn_count=1)
        reg.update_run_diagnostics("i1", turn_count=9)  # throttled
        self.assertTrue(reg._pending_diagnostics_save)
        # A durable status mutation goes straight to disk and, because it
        # serialises the whole in-memory record, carries the pending
        # diagnostics with it and resets the throttle flag.
        reg.mark_completed("i1")
        self.assertFalse(reg._pending_diagnostics_save)
        on_disk = self._on_disk()["i1"]
        self.assertEqual(on_disk.get("status"), IssueStatus.COMPLETED.value)
        self.assertEqual(on_disk.get("run_turn_count"), 9)

    def test_zero_interval_disables_throttling(self) -> None:
        reg = self._registry(interval=0.0)
        reg.update_run_diagnostics("i1", turn_count=1)
        reg.update_run_diagnostics("i1", turn_count=2)
        # Every call writes immediately; nothing is ever left pending.
        self.assertFalse(reg._pending_diagnostics_save)
        self.assertEqual(self._on_disk()["i1"].get("run_turn_count"), 2)

    def test_default_interval_used_when_unset(self) -> None:
        reg = IssueRegistry(storage_path=self.path)
        self.assertEqual(
            reg._diagnostics_min_save_interval_s,
            IssueRegistry._DIAGNOSTICS_MIN_SAVE_INTERVAL_S,
        )


# ---------------------------------------------------------------------------
# IssueRegistry — feedback mutations
# ---------------------------------------------------------------------------


class TestFeedbackMutations(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)
        self.registry.register("i1", "owner/repo#1")

    def test_mark_feedback_pending_records_since(self) -> None:
        before = time.time()
        result = self.registry.mark_feedback_pending("i1", ["fb-1", "fb-2"])
        self.assertGreaterEqual(result.pending_feedback_since, before)
        self.assertEqual(result.pending_feedback_ids, ["fb-1", "fb-2"])
        self.assertIsNotNone(result.last_feedback_checked_at)

    def test_mark_feedback_pending_skips_duplicates(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1"])
        # fb-1 is already pending → should not re-add.
        result = self.registry.mark_feedback_pending("i1", ["fb-1", "fb-2"])
        self.assertEqual(result.pending_feedback_ids, ["fb-1", "fb-2"])

    def test_mark_feedback_pending_skips_already_processed(self) -> None:
        self.registry.mark_feedback_processed("i1", ["fb-1"])
        # fb-1 already processed → skip.
        result = self.registry.mark_feedback_pending("i1", ["fb-1", "fb-2"])
        self.assertEqual(result.pending_feedback_ids, ["fb-2"])

    def test_mark_feedback_pending_updates_cursor(self) -> None:
        result = self.registry.mark_feedback_pending("i1", ["fb-1"], cursor="fb-cursor")
        self.assertEqual(result.feedback_cursor, "fb-cursor")

    def test_mark_feedback_processed_moves_ids(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1", "fb-2"])
        result = self.registry.mark_feedback_processed("i1", ["fb-1"], commit_sha="c1")
        self.assertIn("fb-1", result.processed_feedback_ids)
        self.assertNotIn("fb-1", result.pending_feedback_ids)
        self.assertEqual(result.last_followup_commit_sha, "c1")

    def test_mark_feedback_processed_clears_since_when_empty(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1"])
        result = self.registry.mark_feedback_processed("i1", ["fb-1"])
        self.assertIsNone(result.pending_feedback_since)

    def test_mark_feedback_checked(self) -> None:
        result = self.registry.mark_feedback_checked("i1")
        self.assertIsNotNone(result.last_feedback_checked_at)

    def test_clear_stale_pending_returns_count(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1"])
        # Force the pending_since into the past by patching.
        record = self.registry.get("i1")
        record.pending_feedback_since = time.time() - 1000
        self.registry._save()
        count = self.registry.clear_stale_pending("i1", timeout_seconds=10)
        self.assertEqual(count, 1)
        self.assertEqual(self.registry.get("i1").pending_feedback_ids, [])

    def test_clear_stale_pending_no_action_within_window(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1"])
        count = self.registry.clear_stale_pending("i1", timeout_seconds=10_000)
        self.assertEqual(count, 0)
        self.assertEqual(self.registry.get("i1").pending_feedback_ids, ["fb-1"])

    def test_clear_stale_pending_no_action_when_none(self) -> None:
        self.assertEqual(self.registry.clear_stale_pending("i1"), 0)

    def test_mark_feedback_pending_persists_urls(self) -> None:
        result = self.registry.mark_feedback_pending(
            "i1",
            ["fb-1", "fb-2"],
            feedback_urls={"fb-1": "https://example.com/issues/1#tid-1"},
        )
        # Only ids with a truthy url are stored; fb-2 had no url entry.
        self.assertEqual(result.pending_feedback_urls, {"fb-1": "https://example.com/issues/1#tid-1"})

    def test_mark_feedback_pending_skips_empty_url(self) -> None:
        result = self.registry.mark_feedback_pending("i1", ["fb-1"], feedback_urls={"fb-1": ""})
        self.assertEqual(result.pending_feedback_urls, {})

    def test_mark_feedback_pending_backfills_url_for_existing_id(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1"])
        result = self.registry.mark_feedback_pending(
            "i1",
            ["fb-1"],
            feedback_urls={"fb-1": "https://example.com/issues/1#tid-1"},
        )
        self.assertEqual(result.pending_feedback_ids, ["fb-1"])
        self.assertEqual(
            result.pending_feedback_urls,
            {"fb-1": "https://example.com/issues/1#tid-1"},
        )

    def test_mark_feedback_processed_drops_url(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1", "fb-2"], feedback_urls={"fb-1": "u1", "fb-2": "u2"})
        result = self.registry.mark_feedback_processed("i1", ["fb-1"])
        self.assertNotIn("fb-1", result.pending_feedback_ids)
        self.assertNotIn("fb-1", result.pending_feedback_urls)
        self.assertEqual(result.pending_feedback_urls, {"fb-2": "u2"})

    def test_clear_stale_pending_clears_urls(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1"], feedback_urls={"fb-1": "u1"})
        record = self.registry.get("i1")
        record.pending_feedback_since = time.time() - 1000
        self.registry._save()
        self.registry.clear_stale_pending("i1", timeout_seconds=10)
        cleared = self.registry.get("i1")
        self.assertEqual(cleared.pending_feedback_ids, [])
        self.assertEqual(cleared.pending_feedback_urls, {})

    def test_increment_followup_attempt(self) -> None:
        before = self.registry.get("i1").followup_attempt_count
        result = self.registry.increment_followup_attempt("i1")
        self.assertEqual(result.followup_attempt_count, before + 1)


# ---------------------------------------------------------------------------
# IssueRegistry — intent / retry bookkeeping
# ---------------------------------------------------------------------------


class TestIntentAndRetry(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)
        self.registry.register("i1", "owner/repo#1")

    def test_mark_intent(self) -> None:
        result = self.registry.mark_intent("i1", Intent.RETRY, source="command", command="/agent retry")
        self.assertEqual(result.intent, Intent.RETRY)
        self.assertEqual(result.intent_source, "command")
        self.assertEqual(result.last_command, "/agent retry")

    def test_mark_intent_missing_returns_none(self) -> None:
        self.assertIsNone(self.registry.mark_intent("missing", Intent.RETRY))

    def test_clear_intent(self) -> None:
        self.registry.mark_intent("i1", Intent.RETRY, source="command")
        result = self.registry.clear_intent("i1")
        self.assertEqual(result.intent, Intent.NONE)
        self.assertIsNone(result.intent_source)

    def test_clear_intent_preserves_history(self) -> None:
        self.registry.mark_intent("i1", Intent.RETRY, source="command")
        result = self.registry.clear_intent("i1", record_intent_history=True)
        # When preserving history, intent_source is kept.
        self.assertEqual(result.intent_source, "command")

    def test_increment_retry_count(self) -> None:
        self.registry.increment_retry_count("i1")
        self.registry.increment_retry_count("i1")
        self.assertEqual(self.registry.get("i1").retry_count, 2)

    def test_reset_for_retry_clears_pr_state(self) -> None:
        self.registry.mark_synced(
            "i1",
            branch_name="feat/x",
            commit_sha="abc",
            pr_number=10,
            pr_url="https://x/y/pull/10",
        )
        self.registry.update_report("i1", report_path="/tmp/r.md")
        result = self.registry.reset_for_retry("i1")
        self.assertEqual(result.status, IssueStatus.PENDING)
        self.assertIsNone(result.commit_sha)
        self.assertIsNone(result.pr_number)
        self.assertIsNone(result.pr_url)
        self.assertIsNone(result.report_path)
        self.assertIsNone(result.summary_comment_id)
        self.assertIsNone(result.verification_status)
        self.assertIsNone(result.verification_output)
        self.assertIsNone(result.last_hook_error)
        # retry_count is incremented by default.
        self.assertEqual(result.retry_count, 1)

    def test_reset_for_retry_preserves_intent(self) -> None:
        self.registry.mark_intent("i1", Intent.FOLLOWUP, source="label")
        result = self.registry.reset_for_retry("i1", increment_retry=False)
        self.assertEqual(result.intent, Intent.FOLLOWUP)
        self.assertEqual(result.intent_source, "label")
        # retry_count not incremented.
        self.assertEqual(result.retry_count, 0)

    def test_unblock_abandoned_returns_to_pending(self) -> None:
        self.registry.register("i1", "owner/repo#1")
        self.registry.mark_abandoned("i1")
        self.registry.mark_intent("i1", Intent.BLOCKED)
        result = self.registry.unblock("i1")
        self.assertEqual(result.status, IssueStatus.PENDING)
        self.assertEqual(result.intent, Intent.NONE)

    def test_unblock_other_status_is_idempotent(self) -> None:
        # Already PENDING → unblock is a full no-op: status and intent
        # are both left untouched (per the unblock docstring, only
        # ABANDONED records are rolled back and have intent cleared).
        self.registry.mark_intent("i1", Intent.FOLLOWUP)
        result = self.registry.unblock("i1")
        self.assertEqual(result.status, IssueStatus.PENDING)
        self.assertEqual(result.intent, Intent.FOLLOWUP)

    def test_unblock_missing_returns_none(self) -> None:
        self.assertIsNone(self.registry.unblock("missing"))


# ---------------------------------------------------------------------------
# IssueRegistry — queries
# ---------------------------------------------------------------------------


class TestQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)
        # Build a small fixture.
        self.registry.register("i1", "owner/repo#1", branch_name="feat/1")
        self.registry.register("i2", "owner/repo#2", branch_name="feat/2")
        self.registry.register("i3", "owner/repo#3", branch_name="feat/3")
        self.registry.mark_synced("i1", branch_name="feat/1", pr_number=1, pr_url="x")
        self.registry.mark_synced("i2", branch_name="feat/2", pr_number=2, pr_url="x")
        # i3 has no PR.
        self.registry.mark_completed("i2")
        self.registry.register(
            "i-seq-1",
            "owner/repo#seq1",
            branch_name="feat/seq1",
            workspace_strategy="sequential",
            sequence_index=1,
        )
        self.registry.register(
            "i-seq-2",
            "owner/repo#seq2",
            branch_name="feat/seq2",
            workspace_strategy="sequential",
            sequence_index=2,
        )
        self.registry.register("i-run", "owner/repo#run", branch_name="feat/run")
        self.registry.mark_running("i-run")

    def test_get(self) -> None:
        self.assertEqual(self.registry.get("i1").issue_id, "i1")
        self.assertIsNone(self.registry.get("missing"))

    def test_get_by_identifier(self) -> None:
        self.assertEqual(self.registry.get_by_identifier("owner/repo#2").issue_id, "i2")
        self.assertIsNone(self.registry.get_by_identifier("nope"))

    def test_get_by_issue_ref_accepts_both(self) -> None:
        self.assertEqual(self.registry.get_by_issue_ref("i1").issue_id, "i1")
        self.assertEqual(self.registry.get_by_issue_ref("owner/repo#1").issue_id, "i1")
        self.assertIsNone(self.registry.get_by_issue_ref("nope"))

    def test_get_by_branch(self) -> None:
        self.assertEqual(self.registry.get_by_branch("feat/2").issue_id, "i2")
        self.assertIsNone(self.registry.get_by_branch("nope"))

    def test_has_pr(self) -> None:
        self.assertTrue(self.registry.has_pr("i1"))
        self.assertFalse(self.registry.has_pr("i3"))
        self.assertFalse(self.registry.has_pr("missing"))

    def test_is_completed(self) -> None:
        self.assertTrue(self.registry.is_completed("i2"))
        self.assertFalse(self.registry.is_completed("i1"))
        self.assertFalse(self.registry.is_completed("missing"))

    def test_is_terminal(self) -> None:
        # i2 is COMPLETED.
        self.assertTrue(self.registry.is_terminal("i2"))
        # i1 is SYNCED → not terminal.
        self.assertFalse(self.registry.is_terminal("i1"))
        # i-run is RUNNING → not terminal.
        self.assertFalse(self.registry.is_terminal("i-run"))

    def test_iter_records_with_pr(self) -> None:
        records = self.registry.iter_records_with_pr()
        ids = {r.issue_id for r in records}
        self.assertEqual(ids, {"i1", "i2"})

    def test_latest_sequential_record(self) -> None:
        result = self.registry.latest_sequential_record()
        self.assertIsNotNone(result)
        self.assertEqual(result.issue_id, "i-seq-2")

    def test_latest_sequential_record_no_records(self) -> None:
        # Use a fresh storage path so the fixture records from setUp
        # don't bleed in.
        empty_registry = IssueRegistry(storage_path=Path(self.tmp.name) / "empty-registry.json")
        self.assertIsNone(empty_registry.latest_sequential_record())

    def test_running_records(self) -> None:
        records = self.registry.running_records()
        self.assertEqual([r.issue_id for r in records], ["i-run"])

    def test_has_processed_feedback(self) -> None:
        self.registry.mark_feedback_processed("i1", ["fb-1"])
        self.assertTrue(self.registry.has_processed_feedback("i1", "fb-1"))
        self.assertFalse(self.registry.has_processed_feedback("i1", "fb-2"))
        self.assertFalse(self.registry.has_processed_feedback("missing", "fb-1"))

    def test_can_follow_up(self) -> None:
        # Default followup_attempt_count is 0, max=5 → can.
        self.assertTrue(self.registry.can_follow_up("i1", 5))
        # Bump attempts to 5 → cannot.
        for _ in range(5):
            self.registry.increment_followup_attempt("i1")
        self.assertFalse(self.registry.can_follow_up("i1", 5))
        # Missing record → cannot.
        self.assertFalse(self.registry.can_follow_up("missing", 5))


# ---------------------------------------------------------------------------
# IssueRegistry — clarification mutations
# ---------------------------------------------------------------------------
