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

"""Unit tests for :mod:`extensions.orchestrator.review_feedback`.

Covers the PR review feedback follow-up planning:

* :func:`_is_clawcodex_system_comment` — marker matching for the
  ``REPLY_MARKER`` and ``CLAWCODEX_SYSTEM_MARKERS`` substrings.
* :meth:`ReviewFeedbackService._filter_pending` — exclusion rules
  (ignore_authors, bot login, processed/pending IDs, resolved/outdated
  status, system comments).
* :meth:`ReviewFeedbackService.collect_followups` — early returns
  (no slots, disabled), bot-login resolution (explicit and via
  tracker), stale-pending clearing, followup-attempt cap, and
  feedback selection.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from extensions.orchestrator.issue_registry import IssueRegistry  # pylint: disable=import-error
from extensions.orchestrator.review_feedback import (
    CLAWCODEX_SYSTEM_MARKERS,
    ReviewFeedbackService,
    _is_clawcodex_system_comment,
)
from extensions.orchestrator.tracker import (  # pylint: disable=import-error
    PullRequestFeedback,
    PullRequestRef,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feedback(
    id: str = "fb-1",
    *,
    body: str = "please fix this",
    author_login: str | None = "alice",
    status: str | None = "open",
    source: str = "conversation",
    created_at: str = "2026-01-01T00:00:00Z",
    updated_at: str = "2026-01-01T00:00:00Z",
) -> PullRequestFeedback:
    return PullRequestFeedback(
        id=id,
        source=source,
        body=body,
        author_login=author_login,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
    )


def _build_registry(
    tmp_dir: str | Path,
    *,
    issues: list[dict[str, Any]] | None = None,
) -> IssueRegistry:
    """Build a registry with PR-bearing records for testing."""
    registry = IssueRegistry(storage_path=Path(tmp_dir) / "registry.json")
    for entry in issues or []:
        registry.register(
            issue_id=entry["issue_id"],
            issue_identifier=entry["issue_identifier"],
            branch_name=entry["branch_name"],
        )
        registry.mark_synced(
            entry["issue_id"],
            branch_name=entry["branch_name"],
            pr_number=entry.get("pr_number", 1),
            pr_url=entry.get("pr_url", "https://x/y/pull/1"),
        )
    return registry


class _FakeTracker:
    """Minimal async tracker stand-in for ReviewFeedbackService tests."""

    def __init__(self, feedback_by_issue: dict[str, list[PullRequestFeedback]] | None = None) -> None:
        self._feedback = feedback_by_issue or {}
        self.user_calls = 0
        self.feedback_calls: list[tuple[str, int | None]] = []

    async def fetch_pull_request_feedback(
        self,
        pull_request: PullRequestRef,
        issue_id: str,
        include_ci_failures: bool = True,
        max_log_chars_per_check: int = 12_000,
    ) -> list[PullRequestFeedback]:
        self.feedback_calls.append((issue_id, pull_request.number if pull_request.number else None))
        return list(self._feedback.get(issue_id, []))

    async def get_authenticated_user(self) -> str:
        self.user_calls += 1
        return "clawcodex-bot"


def _config(**overrides: Any) -> SimpleNamespace:
    base = dict(
        enabled=True,
        bot_login="",
        pending_feedback_timeout_seconds=600,
        max_followup_attempts_per_pr=5,
        max_feedback_items_per_run=20,
        include_ci_failures=True,
        max_log_chars_per_check=12_000,
        ignore_authors=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# _is_clawcodex_system_comment
# ---------------------------------------------------------------------------


class TestIsClawcodexSystemComment(unittest.TestCase):
    def test_empty_body_returns_false(self) -> None:
        self.assertFalse(_is_clawcodex_system_comment(""))

    def test_none_body_returns_false(self) -> None:
        self.assertFalse(_is_clawcodex_system_comment(None))

    def test_normal_comment_returns_false(self) -> None:
        self.assertFalse(_is_clawcodex_system_comment("Please fix the lint"))

    def test_reply_marker_detected(self) -> None:
        body = "Some prose.\n\nHandled in the latest ClawCodex follow-up commit.\n"
        self.assertTrue(_is_clawcodex_system_comment(body))

    def test_each_system_marker_detected(self) -> None:
        for marker in CLAWCODEX_SYSTEM_MARKERS:
            with self.subTest(marker=marker):
                self.assertTrue(_is_clawcodex_system_comment(marker))

    def test_marker_in_middle_of_body(self) -> None:
        body = "I checked\n\n## ClawCodex Run Summary\n\nSome details."
        self.assertTrue(_is_clawcodex_system_comment(body))

    def test_case_sensitive(self) -> None:
        # Marker is case-sensitive.
        self.assertFalse(_is_clawcodex_system_comment("clawcodex run summary"))


# ---------------------------------------------------------------------------
# ReviewFeedbackService._filter_pending
# ---------------------------------------------------------------------------


class TestFilterPending(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _build_registry(self.tmp.name)
        record = self.registry.register("i1", "owner/repo#1", branch_name="feat/x")
        self.record = record
        self.service = ReviewFeedbackService(
            tracker=_FakeTracker(),
            registry=self.registry,
            config=_config(),
        )

    def test_empty_feedback_returns_empty(self) -> None:
        self.assertEqual(self.service._filter_pending(self.record, []), [])

    def test_open_feedback_kept(self) -> None:
        result = self.service._filter_pending(self.record, [_feedback(id="fb-1")])
        self.assertEqual([f.id for f in result], ["fb-1"])

    def test_resolved_feedback_dropped(self) -> None:
        result = self.service._filter_pending(self.record, [_feedback(status="resolved")])
        self.assertEqual(result, [])

    def test_outdated_feedback_dropped(self) -> None:
        result = self.service._filter_pending(self.record, [_feedback(status="outdated")])
        self.assertEqual(result, [])

    def test_already_processed_dropped(self) -> None:
        self.registry.mark_feedback_processed("i1", ["fb-1"])
        result = self.service._filter_pending(self.record, [_feedback(id="fb-1")])
        self.assertEqual(result, [])

    def test_already_pending_dropped(self) -> None:
        self.registry.mark_feedback_pending("i1", ["fb-1"])
        result = self.service._filter_pending(self.record, [_feedback(id="fb-1")])
        self.assertEqual(result, [])

    def test_bot_login_excluded(self) -> None:
        # Use a service with explicitly configured bot_login so the
        # _bot_login_explicit flag is set and the filter applies.
        service_with_explicit_bot = ReviewFeedbackService(
            tracker=_FakeTracker(),
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        service_with_explicit_bot._bot_login_explicit = True
        result = service_with_explicit_bot._filter_pending(
            self.record,
            [_feedback(author_login="clawcodex-bot")],
            bot_login="clawcodex-bot",
        )
        self.assertEqual(result, [])

    def test_ignore_authors_excluded(self) -> None:
        service = ReviewFeedbackService(
            tracker=_FakeTracker(),
            registry=self.registry,
            config=_config(ignore_authors=["ignored-user"]),
        )
        result = service._filter_pending(self.record, [_feedback(author_login="ignored-user")])
        self.assertEqual(result, [])

    def test_ignore_authors_case_insensitive(self) -> None:
        service = ReviewFeedbackService(
            tracker=_FakeTracker(),
            registry=self.registry,
            config=_config(ignore_authors=["IGNORED-USER"]),
        )
        result = service._filter_pending(self.record, [_feedback(author_login="ignored-user")])
        self.assertEqual(result, [])

    def test_system_comment_dropped(self) -> None:
        body = "## ClawCodex Run Summary\nAll green"
        result = self.service._filter_pending(self.record, [_feedback(body=body)])
        self.assertEqual(result, [])

    def test_reply_marker_dropped(self) -> None:
        body = "Handled in the latest ClawCodex follow-up commit."
        result = self.service._filter_pending(self.record, [_feedback(body=body)])
        self.assertEqual(result, [])

    def test_multiple_feedback_filtered(self) -> None:
        service_with_explicit_bot = ReviewFeedbackService(
            tracker=_FakeTracker(),
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        service_with_explicit_bot._bot_login_explicit = True
        result = service_with_explicit_bot._filter_pending(
            self.record,
            [
                _feedback(id="fb-keep-1", author_login="alice"),
                _feedback(id="fb-keep-2", author_login="bob"),
                _feedback(id="fb-resolved", status="resolved"),
                _feedback(id="fb-bot", author_login="clawcodex-bot"),
            ],
            bot_login="clawcodex-bot",
        )
        self.assertEqual([f.id for f in result], ["fb-keep-1", "fb-keep-2"])

    def test_no_author_login_kept(self) -> None:
        # Missing author_login → no author-based exclusion.
        result = self.service._filter_pending(
            self.record,
            [_feedback(author_login=None)],
            bot_login="clawcodex-bot",
        )
        self.assertEqual([f.id for f in result], ["fb-1"])


# ---------------------------------------------------------------------------
# ReviewFeedbackService.collect_followups
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an async coroutine in a fresh event loop.

    Uses :func:`asyncio.run` so each call gets a brand-new loop —
    critical because preceding tests (e.g. agent_runner under
    pytest-asyncio) may have closed the previous event loop, which
    would break the older ``asyncio.get_event_loop()`` pattern.
    """
    return asyncio.run(coro)


class TestCollectFollowups(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _build_registry(
            self.tmp.name,
            issues=[
                {
                    "issue_id": "i1",
                    "issue_identifier": "owner/repo#1",
                    "branch_name": "feat/1",
                    "pr_number": 1,
                }
            ],
        )

    def test_disabled_returns_empty(self) -> None:
        tracker = _FakeTracker()
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(enabled=False),
        )
        result = _run(service.collect_followups(available_slots=5))
        self.assertEqual(result, [])
        # No tracker calls should be made when disabled.
        self.assertEqual(tracker.feedback_calls, [])

    def test_zero_slots_returns_empty(self) -> None:
        tracker = _FakeTracker()
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(),
        )
        result = _run(service.collect_followups(available_slots=0))
        self.assertEqual(result, [])

    def test_negative_slots_returns_empty(self) -> None:
        tracker = _FakeTracker()
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(),
        )
        result = _run(service.collect_followups(available_slots=-1))
        self.assertEqual(result, [])

    def test_explicit_bot_login_not_overridden(self) -> None:
        # With explicit bot_login, get_authenticated_user is not called.
        tracker = _FakeTracker()
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="explicit-bot"),
        )
        _run(service.collect_followups(available_slots=1))
        self.assertEqual(tracker.user_calls, 0)

    def test_no_pending_feedback_marks_checked(self) -> None:
        # No feedback → record marked as checked, no followup.
        tracker = _FakeTracker(feedback_by_issue={"i1": []})
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        result = _run(service.collect_followups(available_slots=1))
        self.assertEqual(result, [])
        record = self.registry.get("i1")
        self.assertIsNotNone(record.last_feedback_checked_at)

    def test_pending_feedback_returns_followup(self) -> None:
        tracker = _FakeTracker(feedback_by_issue={"i1": [_feedback(id="fb-1", author_login="alice")]})
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        result = _run(service.collect_followups(available_slots=1))
        self.assertEqual(len(result), 1)
        followup = result[0]
        self.assertEqual(followup.record.issue_id, "i1")
        self.assertEqual(followup.feedback[0].id, "fb-1")
        # Pending recorded.
        record = self.registry.get("i1")
        self.assertIn("fb-1", record.pending_feedback_ids)

    def test_followup_attempt_cap_enforced(self) -> None:
        # Pre-load with max attempts.
        for _ in range(5):
            self.registry.increment_followup_attempt("i1")
        tracker = _FakeTracker(feedback_by_issue={"i1": [_feedback(id="fb-1", author_login="alice")]})
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot", max_followup_attempts_per_pr=5),
        )
        result = _run(service.collect_followups(available_slots=1))
        self.assertEqual(result, [])

    def test_max_feedback_items_per_run_truncates(self) -> None:
        # 5 feedback, max 2 per run.
        feedback = [_feedback(id=f"fb-{i}", author_login="alice") for i in range(5)]
        tracker = _FakeTracker(feedback_by_issue={"i1": feedback})
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot", max_feedback_items_per_run=2),
        )
        result = _run(service.collect_followups(available_slots=1))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].feedback), 2)

    def test_stale_pending_cleared(self) -> None:
        # Pre-mark feedback as pending long ago.
        self.registry.mark_feedback_pending("i1", ["stale-fb"])
        record = self.registry.get("i1")
        # Force stale.
        record.pending_feedback_since = 0.0  # ancient
        record.pending_feedback_since_map["stale-fb"] = 0.0
        self.registry._save()

        tracker = _FakeTracker(feedback_by_issue={"i1": [_feedback(id="new-fb", author_login="alice")]})
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot", pending_feedback_timeout_seconds=600),
        )
        result = _run(service.collect_followups(available_slots=1))
        self.assertEqual(len(result), 1)
        # The stale pending was cleared.
        record = self.registry.get("i1")
        self.assertNotIn("stale-fb", record.pending_feedback_ids)
        # The new feedback is pending.
        self.assertIn("new-fb", record.pending_feedback_ids)

    def test_bot_login_resolved_via_tracker_once(self) -> None:
        tracker = _FakeTracker(feedback_by_issue={"i1": [_feedback(id="fb-1", author_login="alice")]})
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(),  # bot_login empty → resolved via tracker
        )
        # First call: bot login resolved, no auth call yet.
        _run(service.collect_followups(available_slots=1))
        # Subsequent calls: cached, no second auth call.
        _run(service.collect_followups(available_slots=1))
        self.assertEqual(tracker.user_calls, 1)

    def test_cursor_set_to_updated_at(self) -> None:
        tracker = _FakeTracker(
            feedback_by_issue={"i1": [_feedback(id="fb-1", author_login="alice", updated_at="2026-05-01T00:00:00Z")]}
        )
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        _run(service.collect_followups(available_slots=1))
        record = self.registry.get("i1")
        self.assertEqual(record.feedback_cursor, "2026-05-01T00:00:00Z")

    def test_cursor_falls_back_to_created_at(self) -> None:
        tracker = _FakeTracker(
            feedback_by_issue={
                "i1": [
                    _feedback(
                        id="fb-1",
                        author_login="alice",
                        created_at="2026-04-01T00:00:00Z",
                        updated_at=None,
                    )
                ]
            }
        )
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        _run(service.collect_followups(available_slots=1))
        record = self.registry.get("i1")
        self.assertEqual(record.feedback_cursor, "2026-04-01T00:00:00Z")

    def test_cursor_falls_back_to_id(self) -> None:
        tracker = _FakeTracker(
            feedback_by_issue={"i1": [_feedback(id="fb-99", author_login="alice", created_at=None, updated_at=None)]}
        )
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        _run(service.collect_followups(available_slots=1))
        record = self.registry.get("i1")
        self.assertEqual(record.feedback_cursor, "fb-99")

    def test_ignores_issues_without_pr(self) -> None:
        # Add an extra issue with no PR — should be skipped.
        self.registry.register("i-no-pr", "owner/repo#np", branch_name="feat/np")
        self.registry.mark_running("i-no-pr")
        # The mark_running reset run_id, but the record still has no PR.
        tracker = _FakeTracker(feedback_by_issue={"i1": [_feedback()]})
        service = ReviewFeedbackService(
            tracker=tracker,
            registry=self.registry,
            config=_config(bot_login="clawcodex-bot"),
        )
        result = _run(service.collect_followups(available_slots=5))
        # Only i1 should be processed.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].record.issue_id, "i1")


if __name__ == "__main__":
    unittest.main()
