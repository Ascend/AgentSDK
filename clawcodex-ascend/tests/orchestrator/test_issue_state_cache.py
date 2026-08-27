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

"""Unit tests for ``extensions.orchestrator.issue_state_cache``.

Exercises the skip policy, the forced-poll conditions, and the
invalidation paths in isolation from the orchestrator. Mirrors the
style of ``tests/test_orchestrator_*.py`` (unittest, hand-written
stubs, no ``unittest.mock.AsyncMock``).
"""

from __future__ import annotations

import unittest

from extensions.orchestrator.issue_state_cache import IssueStateCache


class TestIssueStateCacheSkipPolicy(unittest.TestCase):
    """The core skip logic with default ``stable_skip_turns=3``."""

    def setUp(self) -> None:
        self.cache = IssueStateCache(stable_skip_turns=3)

    def test_first_poll_never_skips(self) -> None:
        # No snapshots recorded yet.
        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=0))

    def test_two_consecutive_active_states_do_not_skip(self) -> None:
        # Two snapshots is not enough — default is N=3.
        for turn in range(2):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=2))

    def test_three_consecutive_active_states_skip(self) -> None:
        for turn in range(3):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        # Called at turn=3, last snapshot at turn=2 → consecutive window
        # 0,1,2 all active with state="open" → skip.
        self.assertTrue(self.cache.should_skip_poll("ISSUE-1", turn=3))

    def test_stable_skip_continues_across_more_polls(self) -> None:
        for turn in range(5):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        # Skip should remain True at turn=5 because the last 3
        # snapshots are still active and identical.
        self.assertTrue(self.cache.should_skip_poll("ISSUE-1", turn=5))

    def test_state_change_resets_skip_window(self) -> None:
        # First two snapshots say "open", then a transition to "review".
        for turn in (0, 1):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        for turn in (2, 3):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="review",
                observed_at_turn=turn,
            )
        # Even though we have 4 active snapshots, the last 3 share
        # state="review" — they ARE eligible to skip. But the turn at
        # which we ask matters: the last snapshot is from turn=3, so
        # the skip window covers turns 1,2,3 with states
        # ("open","review","review") — states are NOT all identical,
        # so no skip.
        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=4))

    def test_gap_in_polling_turns_blocks_skip(self) -> None:
        # Snapshot history covers turns 0, 1, 5 (gap at 2,3,4). The
        # defensive turn-consecutive check must force a re-poll.
        for turn in (0, 1, 5):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        # At turn=6, last snapshot is at turn=5 — consecutive window
        # turns 4,5 only spans 2 turns, not 3. Skip must be False.
        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=6))

    def test_inactive_state_never_skips(self) -> None:
        for turn in range(3):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=False,
                state="closed",
                observed_at_turn=turn,
            )
        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=3))

    def test_inactive_snapshot_outside_window_allows_skip(self) -> None:
        # Snapshot 0: inactive; snapshots 1-3: active.
        # Only the last N=3 snapshots matter for the skip decision.
        # Window [1,2,3] is all active+same state → skip is True.
        self.cache.record(
            issue_id="ISSUE-1",
            is_active=False,
            state="closed",
            observed_at_turn=0,
        )
        for turn in (1, 2, 3):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        self.assertTrue(self.cache.should_skip_poll("ISSUE-1", turn=4))

    def test_inactive_snapshot_within_window_blocks_skip(self) -> None:
        # Snapshot 0: active; snapshot 1: active; snapshot 2: inactive.
        # Window [0,1,2] contains an inactive → skip is False.
        for turn in (0, 1):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        self.cache.record(
            issue_id="ISSUE-1",
            is_active=False,
            state="closed",
            observed_at_turn=2,
        )
        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=3))

    def test_last_snapshot_not_from_previous_turn_blocks_skip(self) -> None:
        # Snapshots from turns 0, 2, 3 — calling at turn=4 the last
        # snapshot is turn=3 (gap), so the window 1,2,3 is not
        # consecutive starting from turn-1.
        for turn in (0, 2, 3):
            self.cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=4))


class TestIssueStateCacheHasRecentInactive(unittest.TestCase):
    """``has_recent_inactive`` powers the forced-poll branch in
    ``AgentRunner._should_continue``.
    """

    def setUp(self) -> None:
        self.cache = IssueStateCache(stable_skip_turns=3)

    def test_empty_history_returns_false(self) -> None:
        self.assertFalse(self.cache.has_recent_inactive("ISSUE-1", turn=0))

    def test_active_snapshot_returns_false(self) -> None:
        self.cache.record(
            issue_id="ISSUE-1",
            is_active=True,
            state="open",
            observed_at_turn=2,
        )
        self.assertFalse(self.cache.has_recent_inactive("ISSUE-1", turn=2))

    def test_inactive_at_asked_turn_returns_true(self) -> None:
        self.cache.record(
            issue_id="ISSUE-1",
            is_active=False,
            state="closed",
            observed_at_turn=2,
        )
        self.assertTrue(self.cache.has_recent_inactive("ISSUE-1", turn=2))

    def test_inactive_at_different_turn_returns_false(self) -> None:
        self.cache.record(
            issue_id="ISSUE-1",
            is_active=False,
            state="closed",
            observed_at_turn=1,
        )
        self.assertFalse(self.cache.has_recent_inactive("ISSUE-1", turn=2))


class TestIssueStateCacheInvalidate(unittest.TestCase):
    """Cache invalidation paths."""

    def setUp(self) -> None:
        self.cache = IssueStateCache(stable_skip_turns=2)

    def test_invalidate_specific_issue_clears_only_that_history(self) -> None:
        for issue_id in ("ISSUE-1", "ISSUE-2"):
            for turn in range(2):
                self.cache.record(
                    issue_id=issue_id,
                    is_active=True,
                    state="open",
                    observed_at_turn=turn,
                )
        # Both issues should be eligible to skip
        self.assertTrue(self.cache.should_skip_poll("ISSUE-1", turn=2))
        self.assertTrue(self.cache.should_skip_poll("ISSUE-2", turn=2))

        self.cache.invalidate("ISSUE-1")

        self.assertFalse(self.cache.should_skip_poll("ISSUE-1", turn=2))
        self.assertTrue(self.cache.should_skip_poll("ISSUE-2", turn=2))

    def test_invalidate_all_clears_every_history(self) -> None:
        for issue_id in ("ISSUE-1", "ISSUE-2"):
            for turn in range(2):
                self.cache.record(
                    issue_id=issue_id,
                    is_active=True,
                    state="open",
                    observed_at_turn=turn,
                )
        self.cache.invalidate()
        self.assertEqual(self.cache.stats()["tracked_issues"], 0)
        self.assertEqual(self.cache.stats()["total_snapshots"], 0)

    def test_invalidate_unknown_issue_is_noop(self) -> None:
        self.cache.invalidate("NONEXISTENT")
        self.assertEqual(self.cache.stats()["tracked_issues"], 0)


class TestIssueStateCacheDisableSwitch(unittest.TestCase):
    """``stable_skip_turns=0`` disables the cache."""

    def test_zero_stable_skip_turns_never_skips(self) -> None:
        cache = IssueStateCache(stable_skip_turns=0)
        for turn in range(10):
            cache.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        # Even with 10 identical active snapshots, the cache never
        # advises skipping when disabled.
        for turn in range(1, 10):
            self.assertFalse(cache.should_skip_poll("ISSUE-1", turn=turn))

    def test_negative_stable_skip_turns_normalised_to_zero(self) -> None:
        cache = IssueStateCache(stable_skip_turns=-5)
        self.assertEqual(cache.stable_skip_turns, 0)


class TestIssueStateCacheConcurrentSessions(unittest.TestCase):
    """Per-instance isolation: two ``AgentSession`` instances must never
    share state.
    """

    def test_two_caches_have_independent_state(self) -> None:
        cache_a = IssueStateCache(stable_skip_turns=2)
        cache_b = IssueStateCache(stable_skip_turns=2)
        # Populate both caches with the same issue
        for turn in range(2):
            cache_a.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
            cache_b.record(
                issue_id="ISSUE-1",
                is_active=True,
                state="open",
                observed_at_turn=turn,
            )
        # Both caches are full — both must skip
        self.assertTrue(cache_a.should_skip_poll("ISSUE-1", turn=2))
        self.assertTrue(cache_b.should_skip_poll("ISSUE-1", turn=2))

        # invalidate cache_a — cache_b must remain populated
        cache_a.invalidate()
        self.assertFalse(cache_a.should_skip_poll("ISSUE-1", turn=2))
        self.assertTrue(cache_b.should_skip_poll("ISSUE-1", turn=2))


if __name__ == "__main__":
    unittest.main()
