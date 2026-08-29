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

"""Tests for the per-turn REPL buffer cleanup added in OOM mitigation.

Covers ``clear_pending_turn_buffers`` and the ``deque(maxlen=...)`` bounds
on ``_queued_prompts`` / ``_thinking_chunks`` introduced to keep the REPL
from accumulating state across turns on memory-constrained hosts (the
WSL2 3.8 GB OOM repro).

These tests intentionally construct a bare REPL via ``__new__`` to
bypass the heavy ``__init__`` (provider wiring, session, tool registry)
— the cleanup helper only touches three buffer attributes, none of
which require real provider / session state.
"""

from __future__ import annotations

import unittest
from collections import deque
from threading import Lock

from src.repl import ClawcodexREPL


class TestTurnBufferCleanup(unittest.TestCase):
    """Verify turn-boundary buffer reset and maxlen bounds."""

    def _make_repl(self) -> ClawcodexREPL:
        """Bypass __init__ and install only the buffer attributes the
        cleanup helper reads / writes. ``ClawCodexExtREPL.__init__`` is
        heavy (provider, session, tool context); the test only cares
        about the three buffer attributes.
        """
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._queued_prompts = deque(maxlen=100)
        repl._cron_queued_prompts = deque(maxlen=100)
        repl._queued_prompts_lock = Lock()
        repl._thinking_chunks = deque(maxlen=1000)
        repl._expandable_blocks = deque(maxlen=20)
        return repl

    # -- queued prompt bound --

    def test_queued_prompts_bounded_by_maxlen(self) -> None:
        repl = self._make_repl()
        for i in range(150):
            repl._enqueue_prompt(f"prompt-{i}")
        self.assertEqual(len(repl._queued_prompts), 100)
        # Oldest 50 prompts should have been silently dropped.
        self.assertNotIn("prompt-0", repl._queued_prompts)
        self.assertNotIn("prompt-49", repl._queued_prompts)
        self.assertIn("prompt-50", repl._queued_prompts)
        self.assertIn("prompt-149", repl._queued_prompts)

    def test_queued_prompts_fifo_order_preserved_with_deque(self) -> None:
        """``popleft`` replaces the old ``pop(0)`` — FIFO order must hold."""
        repl = self._make_repl()
        repl._enqueue_prompt("first")
        repl._enqueue_prompt("second")
        repl._enqueue_prompt("third")
        self.assertEqual(repl._pop_queued_prompt(), ("first", "user"))
        self.assertEqual(repl._pop_queued_prompt(), ("second", "user"))
        self.assertEqual(repl._pop_queued_prompt(), ("third", "user"))
        self.assertIsNone(repl._pop_queued_prompt())

    def test_queued_prompts_count_zero_when_empty(self) -> None:
        repl = self._make_repl()
        self.assertEqual(repl._queued_count(), 0)
        self.assertIsNone(repl._pop_queued_prompt())

    def test_queued_prompts_enqueue_strips_and_skips_empty(self) -> None:
        repl = self._make_repl()
        repl._enqueue_prompt("   ")
        repl._enqueue_prompt("")
        repl._enqueue_prompt(None)  # type: ignore[arg-type]
        self.assertEqual(len(repl._queued_prompts), 0)

    # -- thinking chunks bound & cleanup --

    def test_thinking_chunks_bounded_by_maxlen(self) -> None:
        repl = self._make_repl()
        for i in range(1500):
            repl._thinking_chunks.append(f"chunk-{i}")
        # Hard cap at 1000 — oldest 500 must be dropped.
        self.assertEqual(len(repl._thinking_chunks), 1000)
        self.assertNotIn("chunk-0", repl._thinking_chunks)
        self.assertIn("chunk-500", repl._thinking_chunks)
        self.assertIn("chunk-1499", repl._thinking_chunks)

    def test_thinking_chunks_order_preserved_for_join(self) -> None:
        repl = self._make_repl()
        repl._thinking_chunks.append("alpha")
        repl._thinking_chunks.append("beta")
        repl._thinking_chunks.append("gamma")
        self.assertEqual("".join(repl._thinking_chunks), "alphabetagamma")

    # -- expandable blocks --

    def test_expandable_blocks_bounded_by_maxlen(self) -> None:
        repl = self._make_repl()
        for i in range(50):
            repl._expandable_blocks.append((f"label-{i}", f"content-{i}"))
        self.assertEqual(len(repl._expandable_blocks), 20)
        # Oldest 30 must be dropped.
        self.assertNotIn(("label-0", "content-0"), repl._expandable_blocks)
        self.assertIn(("label-49", "content-49"), repl._expandable_blocks)

    # -- the cleanup helper itself --

    def test_clear_pending_turn_buffers_resets_queues(self) -> None:
        repl = self._make_repl()
        # Populate all three buffers.
        repl._enqueue_prompt("p1")
        repl._enqueue_prompt("p2")
        repl._thinking_chunks.append("think-1")
        repl._thinking_chunks.append("think-2")
        repl._expandable_blocks.append(("lbl", "content"))

        repl.clear_pending_turn_buffers()

        # _thinking_chunks is transient UI state and must be cleared so
        # the next turn's spinner starts fresh (this is the buffer that
        # drove the WSL2 3.8 GB OOM repro).
        self.assertEqual(len(repl._thinking_chunks), 0)
        # _expandable_blocks must NOT be cleared — see helper docstring
        # on the "stash the user just expanded from" rationale.
        self.assertEqual(len(repl._expandable_blocks), 1)

    def test_clear_pending_turn_buffers_preserves_queued_prompts(self) -> None:
        """Regression: the LiveStatus spinner enqueues user input into
        ``_queued_prompts`` *during* a turn. ``run()`` drains that queue
        at the *start* of the next iteration via ``_pop_queued_prompt``.
        Clearing the queue at the turn boundary (the d6b8ac0 regression)
        silently dropped whatever the user typed while the engine was
        running, breaking the "type while it's still thinking" affordance.
        The helper must therefore leave ``_queued_prompts`` alone — the
        ``deque(maxlen=100)`` is the actual memory cap.
        """
        repl = self._make_repl()
        repl._enqueue_prompt("typed-during-turn-1")
        repl._enqueue_prompt("typed-during-turn-2")

        repl.clear_pending_turn_buffers()

        self.assertEqual(len(repl._queued_prompts), 2)
        # FIFO order preserved so the next ``_pop_queued_prompt()`` returns
        # the earliest typed input first.
        self.assertEqual(repl._pop_queued_prompt(), ("typed-during-turn-1", "user"))
        self.assertEqual(repl._pop_queued_prompt(), ("typed-during-turn-2", "user"))
        self.assertIsNone(repl._pop_queued_prompt())

    def test_clear_pending_turn_buffers_idempotent(self) -> None:
        repl = self._make_repl()
        # Calling on empty buffers must not raise.
        repl.clear_pending_turn_buffers()
        repl.clear_pending_turn_buffers()
        self.assertEqual(len(repl._queued_prompts), 0)
        self.assertEqual(len(repl._thinking_chunks), 0)

    def test_clear_pending_turn_buffers_thread_safe_with_enqueue(self) -> None:
        """Concurrent enqueue from another thread while the main thread
        calls the helper must not raise. ``_queued_prompts`` is protected
        by its lock; ``_thinking_chunks`` is not, but the helper is
        documented as single-threaded at the turn boundary.
        """
        import threading

        repl = self._make_repl()
        errors: list[BaseException] = []

        def pumper() -> None:
            try:
                for i in range(200):
                    repl._enqueue_prompt(f"bg-{i}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=pumper, daemon=True)
        t.start()
        repl.clear_pending_turn_buffers()
        t.join()
        self.assertEqual(errors, [])

    # -- no-growth regression --

    def test_no_buffer_growth_across_n_turns(self) -> None:
        """Simulate N turns: the helper must keep ``_thinking_chunks``
        bounded by the cap (the buffer that drove the OOM repro) and
        must not let leftovers from earlier turns accumulate forever.
        ``_queued_prompts`` is a *separate* concern — see
        ``test_clear_pending_turn_buffers_preserves_queued_prompts``.
        """
        repl = self._make_repl()
        for turn in range(50):
            # Simulate 30 thinking chunks per turn.
            for i in range(30):
                repl._thinking_chunks.append(f"turn-{turn}-chunk-{i}")
            # And the live agent replies while queued prompts grow.
            self.assertLessEqual(len(repl._thinking_chunks), 1000)
            # Now the turn boundary fires.
            repl.clear_pending_turn_buffers()
            self.assertEqual(len(repl._thinking_chunks), 0)

    def test_user_queue_has_priority_over_cron_queue(self) -> None:
        """User prompts are consumed before cron prompts."""
        repl = self._make_repl()
        repl._enqueue_cron_prompt("cron-first")
        repl._enqueue_prompt("user-input")
        repl._enqueue_cron_prompt("cron-second")

        result = repl._pop_queued_prompt()
        self.assertEqual(result, ("user-input", "user"))

        result = repl._pop_queued_prompt()
        self.assertEqual(result, ("cron-first", "cron"))

        result = repl._pop_queued_prompt()
        self.assertEqual(result, ("cron-second", "cron"))

        self.assertIsNone(repl._pop_queued_prompt())

    def test_cron_prompt_consumed_when_user_queue_empty(self) -> None:
        """Cron prompts are consumed when no user prompts are queued."""
        repl = self._make_repl()
        repl._enqueue_cron_prompt("cron-task")

        result = repl._pop_queued_prompt()
        self.assertEqual(result, ("cron-task", "cron"))

    def test_mixed_user_and_cron_ordering(self) -> None:
        """Multiple user + cron prompts: all user first, then all cron."""
        repl = self._make_repl()
        repl._enqueue_cron_prompt("cron-a")
        repl._enqueue_prompt("user-1")
        repl._enqueue_cron_prompt("cron-b")
        repl._enqueue_prompt("user-2")

        results = []
        while True:
            r = repl._pop_queued_prompt()
            if r is None:
                break
            results.append(r)

        self.assertEqual(
            results,
            [
                ("user-1", "user"),
                ("user-2", "user"),
                ("cron-a", "cron"),
                ("cron-b", "cron"),
            ],
        )

    def test_queued_count_includes_both_queues(self) -> None:
        """_queued_count returns total of user + cron queues."""
        repl = self._make_repl()
        self.assertEqual(repl._queued_count(), 0)
        repl._enqueue_prompt("user-1")
        repl._enqueue_cron_prompt("cron-1")
        repl._enqueue_cron_prompt("cron-2")
        self.assertEqual(repl._queued_count(), 3)


if __name__ == "__main__":
    unittest.main()
