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

"""Unit tests for :class:`extensions.orchestrator.clarification_queue`.

Covers the file-backed clarification queue state machine:

* :class:`ClarificationItem` lifecycle helpers (``touch``, ``is_expired``,
  ``mark_answered``).
* :class:`ClarificationQueue` enqueue / mark-awaiting-local / awaiting-author
  / resolve / mark-duplicate / mark-stale / mark-expired / mark-exhausted /
  mark-issue-failed transitions.
* Persistence round-trip via :func:`_load` / :func:`_save`.
* Failure modes: missing file, malformed JSON, write errors.
* :meth:`ClarificationQueue.inject_feedback` for review-rejection feedback.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.orchestrator.clarification_queue import (
    ClarificationItem,
    ClarificationQueue,
    ClarificationStatus,
    DEFAULT_QUEUE_PATH,
)


# ---------------------------------------------------------------------------
# ClarificationItem
# ---------------------------------------------------------------------------


class TestClarificationItem(unittest.TestCase):
    def test_defaults(self) -> None:
        item = ClarificationItem(issue_id="i1", issue_identifier="owner/repo#1", question="why?")
        self.assertEqual(item.issue_id, "i1")
        self.assertEqual(item.options, [])
        self.assertEqual(item.context_summary, "")
        self.assertEqual(item.status, ClarificationStatus.PENDING)
        self.assertIsNone(item.answer)
        self.assertIsNone(item.answer_source)
        self.assertIsNone(item.answered_at)
        self.assertFalse(item.escalation_notified)
        self.assertEqual(item.stale_answers, [])

    def test_touch_updates_updated_at(self) -> None:
        item = ClarificationItem(issue_id="i1", issue_identifier="x", question="?")
        original = item.updated_at
        time.sleep(0.005)
        item.touch()
        self.assertGreater(item.updated_at, original)

    def test_is_expired_when_no_deadline(self) -> None:
        item = ClarificationItem(issue_id="i1", issue_identifier="x", question="?")
        # No expires_at set → never expired.
        self.assertFalse(item.is_expired())

    def test_is_expired_before_deadline(self) -> None:
        item = ClarificationItem(
            issue_id="i1",
            issue_identifier="x",
            question="?",
            expires_at=time.time() + 100,
        )
        self.assertFalse(item.is_expired())

    def test_is_expired_after_deadline(self) -> None:
        item = ClarificationItem(
            issue_id="i1",
            issue_identifier="x",
            question="?",
            expires_at=time.time() - 5,
        )
        self.assertTrue(item.is_expired())

    def test_is_expired_with_explicit_now(self) -> None:
        item = ClarificationItem(
            issue_id="i1",
            issue_identifier="x",
            question="?",
            expires_at=100.0,
        )
        self.assertTrue(item.is_expired(now=200.0))
        self.assertFalse(item.is_expired(now=50.0))

    def test_mark_answered_records_source_and_time(self) -> None:
        item = ClarificationItem(issue_id="i1", issue_identifier="x", question="?")
        item.mark_answered("yes", "dashboard", answered_at=123.0)
        self.assertEqual(item.answer, "yes")
        self.assertEqual(item.answer_source, "dashboard")
        self.assertEqual(item.answered_at, 123.0)

    def test_mark_answered_defaults_to_current_time(self) -> None:
        item = ClarificationItem(issue_id="i1", issue_identifier="x", question="?")
        before = time.time()
        item.mark_answered("yes", "author")
        after = time.time()
        self.assertIsNotNone(item.answered_at)
        self.assertGreaterEqual(item.answered_at, before)
        self.assertLessEqual(item.answered_at, after)


# ---------------------------------------------------------------------------
# ClarificationQueue — construction / persistence
# ---------------------------------------------------------------------------


class TestClarificationQueuePersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "queue.json"

    def test_empty_file_loads_empty(self) -> None:
        # File doesn't exist → empty records.
        queue = ClarificationQueue(queue_path=self.path)
        self.assertEqual(queue.poll_pending(), [])
        self.assertIsNone(queue.get("any"))

    def test_round_trip_persists_records(self) -> None:
        queue = ClarificationQueue(queue_path=self.path)
        queue.enqueue(
            issue_id="i1",
            issue_identifier="owner/repo#1",
            question="which?",
            options=["a", "b"],
            context_summary="ctx",
            timeout_seconds=120,
        )
        # File should be written.
        self.assertTrue(self.path.exists())
        # Reload from disk.
        reloaded = ClarificationQueue(queue_path=self.path)
        item = reloaded.get("i1")
        self.assertIsNotNone(item)
        self.assertEqual(item.question, "which?")
        self.assertEqual(item.options, ["a", "b"])
        self.assertEqual(item.context_summary, "ctx")
        self.assertIsNotNone(item.expires_at)

    def test_malformed_json_loads_empty(self) -> None:
        self.path.write_text("not-valid-json", encoding="utf-8")
        with self.assertLogs("extensions.orchestrator.clarification_queue", level="WARNING"):
            queue = ClarificationQueue(queue_path=self.path)
        self.assertEqual(queue.poll_pending(), [])

    def test_load_top_level_list_starts_empty(self) -> None:
        # A JSON array at the top level is not a record map; treat the
        # file as corrupt and start fresh instead of crashing.
        self.path.write_text('["not", "an", "object"]', encoding="utf-8")
        with self.assertLogs("extensions.orchestrator.clarification_queue", level="WARNING"):
            queue = ClarificationQueue(queue_path=self.path)
        self.assertEqual(queue.poll_pending(), [])

    def test_load_non_object_record_starts_empty(self) -> None:
        self.path.write_text('{"i1": ["not", "an", "object"]}', encoding="utf-8")
        with self.assertLogs("extensions.orchestrator.clarification_queue", level="WARNING"):
            queue = ClarificationQueue(queue_path=self.path)
        self.assertEqual(queue.poll_pending(), [])

    def test_load_null_list_fields_are_sanitized(self) -> None:
        # A hand-edited or schema-drifted file may carry null list fields;
        # they must load as empty lists so mark_stale / mark_duplicate
        # never hit AttributeError on .append().
        self.path.write_text(
            json.dumps(
                {
                    "i1": {
                        "issue_id": "i1",
                        "issue_identifier": "x",
                        "question": "?",
                        "options": None,
                        "stale_answers": None,
                    }
                }
            ),
            encoding="utf-8",
        )
        queue = ClarificationQueue(queue_path=self.path)
        item = queue.get("i1")
        self.assertIsNotNone(item)
        self.assertEqual(item.options, [])
        self.assertEqual(item.stale_answers, [])
        queue.mark_stale("i1", "late answer")
        self.assertEqual(queue.get("i1").stale_answers, ["late answer"])

    def test_load_preserves_first_response_source(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "i1": {
                        "issue_id": "i1",
                        "issue_identifier": "x",
                        "question": "?",
                        "first_response_source": "dashboard",
                    }
                }
            ),
            encoding="utf-8",
        )

        item = ClarificationQueue(queue_path=self.path).get("i1")

        self.assertIsNotNone(item)
        self.assertEqual(item.first_response_source, "dashboard")

    @unittest.skipIf(os.name == "nt", "directory fsync is unavailable on Windows")
    def test_save_fsyncs_parent_directory_after_replace(self) -> None:
        queue = ClarificationQueue(queue_path=self.path)
        queue._records = {"i1": ClarificationItem("i1", "x", "?")}

        with (
            patch("extensions.orchestrator.clarification_queue.os.open", side_effect=os.open) as open_mock,
            patch("extensions.orchestrator.clarification_queue.os.fsync", wraps=os.fsync) as fsync_mock,
        ):
            queue._save()

        self.assertEqual(open_mock.call_args_list[-1].args, (self.path.parent, os.O_RDONLY))
        self.assertGreaterEqual(fsync_mock.call_count, 2)

    def test_save_preserves_existing_file_on_replace_failure(self) -> None:
        original = '{"existing": {"issue_id": "existing", "issue_identifier": "x", "question": "?"}}'
        self.path.write_text(original, encoding="utf-8")
        queue = ClarificationQueue(queue_path=self.path)
        queue._records = {"i1": ClarificationItem("i1", "x", "?")}

        with patch("extensions.orchestrator.clarification_queue.os.replace", side_effect=OSError):
            with self.assertLogs("extensions.orchestrator.clarification_queue", level="WARNING"):
                queue._save()

        self.assertEqual(self.path.read_text(encoding="utf-8"), original)
        self.assertEqual(list(self.path.parent.glob(f".{self.path.name}.*.tmp")), [])

    def test_default_path_uses_home(self) -> None:
        # DEFAULT_QUEUE_PATH is a module-level constant under ~/.clawcodex.
        # It must always resolve under the user's home.
        self.assertTrue(str(DEFAULT_QUEUE_PATH).startswith(str(Path.home())))


# ---------------------------------------------------------------------------
# ClarificationQueue — enqueue / mark / resolve lifecycle
# ---------------------------------------------------------------------------


class TestClarificationQueueLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "queue.json"
        self.queue = ClarificationQueue(queue_path=self.path)

    def test_enqueue_creates_pending_item(self) -> None:
        item = self.queue.enqueue(
            issue_id="i1",
            issue_identifier="x",
            question="?",
            options=["a", "b"],
            timeout_seconds=60,
        )
        self.assertEqual(item.status, ClarificationStatus.PENDING)
        self.assertIsNotNone(item.expires_at)
        # Without a mark_awaiting_*, the item is still pending.
        self.assertEqual(len(self.queue.poll_pending()), 1)

    def test_enqueue_without_options(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        item = self.queue.get("i1")
        self.assertEqual(item.options, [])

    def test_enqueue_without_timeout_has_no_expiry(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        item = self.queue.get("i1")
        self.assertIsNone(item.expires_at)

    def test_mark_awaiting_local(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        result = self.queue.mark_awaiting_local("i1")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, ClarificationStatus.AWAITING_LOCAL)

    def test_mark_awaiting_local_missing_returns_none(self) -> None:
        self.assertIsNone(self.queue.mark_awaiting_local("missing"))

    def test_mark_awaiting_author(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        result = self.queue.mark_awaiting_author("i1")
        self.assertEqual(result.status, ClarificationStatus.AWAITING_AUTHOR)

    def test_resolve_local_channel_from_local(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.queue.mark_awaiting_local("i1")
        self.queue.resolve("i1", "yes", "clarification_queue")
        item = self.queue.get("i1")
        self.assertEqual(item.status, ClarificationStatus.RESOLVED_LOCAL)
        self.assertEqual(item.answer, "yes")
        self.assertEqual(item.first_response_source, "clarification_queue")

    def test_resolve_author_from_local(self) -> None:
        # Status AWAITING_LOCAL + source "author" → RESOLVED_AUTHOR.
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.queue.mark_awaiting_local("i1")
        self.queue.resolve("i1", "answer from author", "author")
        self.assertEqual(self.queue.get("i1").status, ClarificationStatus.RESOLVED_AUTHOR)

    def test_resolve_from_author_channel(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.queue.mark_awaiting_author("i1")
        self.queue.resolve("i1", "answer", "author")
        self.assertEqual(self.queue.get("i1").status, ClarificationStatus.RESOLVED_AUTHOR)

    def test_resolve_from_local_channel_while_awaiting_author(self) -> None:
        for source in ("dashboard", "clarification_queue"):
            with self.subTest(source=source):
                self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
                self.queue.mark_awaiting_author("i1")
                self.queue.resolve("i1", "answer", source)
                self.assertEqual(self.queue.get("i1").status, ClarificationStatus.RESOLVED_LOCAL)

    def test_resolve_preserves_first_response_source(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.queue.resolve("i1", "first", "dashboard")
        self.queue.resolve("i1", "second", "author")

        item = self.queue.get("i1")
        self.assertEqual(item.first_response_source, "dashboard")
        self.assertEqual(item.answer_source, "author")

    def test_resolve_missing_returns_none(self) -> None:
        self.assertIsNone(self.queue.resolve("missing", "x", "dashboard"))

    def test_resolve_from_unexpected_status_warns(self) -> None:
        # Resolve from EXHAUSTED → still records answer but keeps status.
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.queue.mark_exhausted("i1")
        with self.assertLogs("extensions.orchestrator.clarification_queue", level="WARNING"):
            self.queue.resolve("i1", "late", "dashboard")
        item = self.queue.get("i1")
        self.assertEqual(item.answer, "late")
        self.assertEqual(item.status, ClarificationStatus.EXHAUSTED)

    def test_get_resolved_only_for_resolved_statuses(self) -> None:
        self.queue.enqueue(issue_id="i1", issue_identifier="x", question="?")
        self.assertIsNone(self.queue.get_resolved("i1"))  # still PENDING
        self.queue.mark_awaiting_local("i1")
        self.queue.resolve("i1", "x", "dashboard")
        self.assertIsNotNone(self.queue.get_resolved("i1"))

    def test_get_resolved_missing_returns_none(self) -> None:
        self.assertIsNone(self.queue.get_resolved("nope"))

    def test_poll_pending_excludes_expired(self) -> None:
        self.queue.enqueue(
            issue_id="i1",
            issue_identifier="x",
            question="?",
            timeout_seconds=0,
        )
        # Force expiration: the expires_at was set to time.time() + 0 → effectively now.
        time.sleep(0.01)
        pending = self.queue.poll_pending()
        self.assertEqual(pending, [])
