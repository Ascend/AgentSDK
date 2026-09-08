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

"""Regression tests for the REPL large-paste freeze (Theta(n^2) completion).

Root cause: prompt_toolkit reruns the completer after *every* text
insertion while ``complete_while_typing`` is true, and the merged
completer's sub-completers scan the whole ``text_before_cursor`` with
end-anchored regexes. A single-line paste of n characters therefore
drives Theta(n^2) event-loop work and froze the REPL for minutes once a
line reached ~58 KiB.

Fix: ``ClawcodexREPL`` (and ``ClawCodexExtREPL``) now pass a
``Condition`` filter as ``complete_while_typing`` that disables live
completion once the focused draft exceeds
``_LIVE_COMPLETION_MAX_DRAFT_CHARS``, bounding the per-paste completer
cost at O(capacity * n) while keeping per-keystroke completion for
normal typing. Explicit completion (Tab) and submission (Enter) are
untouched.

These tests pin:

* the pure capacity boundary of ``_live_completion_allowed``;
* the app-aware filter reading the *live* buffer length (via a stubbed
  ``get_app`` — no event loop, no wall clock);
* the filter's no-application fallback (prompt_toolkit returns a
  DummyApplication with an empty buffer when nothing runs);
* that the real REPL wires a ``Condition`` (not ``True``) into the
  ``PromptSession``;
* the algorithmic guarantee: a simulated n-character paste opens the
  gate at most ``capacity`` times — not once per character — and a
  real ``Buffer`` past the capacity stops scheduling completion work
  altogether.

NOTE: prompt_toolkit's ``Buffer.insert_text`` reaches ``get_app()``
through the reference cached in ``prompt_toolkit.buffer`` while our
filter imports it lazily from ``prompt_toolkit.application`` — the
harness patches both spellings with the same stub app.
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import Condition

# ``src.repl`` is the legacy proxy: importing it first bootstraps the
# lazy ``__getattr__`` chain that pulls ``clawcodex_ext.repl.core`` into
# ``sys.modules`` before we touch it (see tests/repl/test_repl_accept_tab.py).
import src.repl  # noqa: F401

from clawcodex_ext.repl.core import (
    _LIVE_COMPLETION_MAX_DRAFT_CHARS as _CAP,
    _live_completion_allowed,
    _live_completion_while_typing_filter,
)

_FILTER = Condition(_live_completion_while_typing_filter)


class _CountingCompleter(Completer):
    """Completer that records how often it is invoked."""

    def __init__(self) -> None:
        self.runs = 0

    def get_completions(self, document, complete_event):
        self.runs += 1
        return iter(())


class _AppStub:
    """Minimal stand-in for the running prompt_toolkit Application.

    The gate only reads ``current_buffer.text``; ``create_background_task``
    must absorb the scheduled completer coroutine without an event loop
    (``coro.close()`` — never awaited — also silences the
    "coroutine was never awaited" RuntimeWarning).
    """

    def __init__(self, buffer) -> None:
        self.current_buffer = buffer
        self.create_background_task = Mock(side_effect=lambda coro: coro.close())


def _patch_app(buffer) -> ExitStack:
    """Bind *buffer* as the active buffer for both ``get_app`` spellings.

    The contexts are already entered; usable as ``with _patch_app(buf):``.
    The bound stub is reachable as ``stack.stub`` for spy assertions.
    """
    stack = ExitStack()
    stack.stub = _AppStub(buffer)
    stack.enter_context(patch("prompt_toolkit.application.get_app", return_value=stack.stub))
    stack.enter_context(patch("prompt_toolkit.buffer.get_app", return_value=stack.stub))
    return stack


def _make_repl():
    """Construct a ``ClawcodexREPL`` with provider/session/config patched.

    Copied from ``tests/input/test_multiline_input.py``: the patch
    targets are ``clawcodex_ext.repl.core`` — the module where
    ``ClawcodexREPL.__init__`` physically executes — not the
    ``src.repl.core`` facade.
    """

    from src.repl.core import ClawcodexREPL

    mock_provider = Mock()
    mock_provider.model = "glm-4.5"

    with (
        patch(
            "clawcodex_ext.repl.core.get_provider_config",
            return_value={"api_key": "x", "default_model": "glm-4.5"},
        ),
        patch("clawcodex_ext.repl.core.Session.create"),
        patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class,
    ):
        mock_provider_class.return_value = mock_provider
        return ClawcodexREPL(provider_name="glm")


class _PasteSim:
    """Per-keystroke paste model over a *real* prompt_toolkit ``Buffer``.

    ``insert_text`` is the exact prompt_toolkit insertion path: when the
    gate is open it fires ``on_text_insert`` and (had we an event loop)
    schedules ``_async_completer`` — one merged-completer pass over the
    whole current draft per qualifying keystroke, which is the Theta(n^2)
    behaviour under test. We count the keystrokes for which the gate is
    open; the E2E measurements showed exactly one completion pass per
    qualifying keystroke.
    """

    def __init__(self, text: str) -> None:
        self.completer = _CountingCompleter()
        self.buffer = Buffer(
            completer=self.completer,
            complete_while_typing=_FILTER,
        )
        self.text = text

    def open_keystrokes(self) -> int:
        count = 0
        with _patch_app(self.buffer):
            for ch in self.text:
                self.buffer.insert_text(ch)
                if self.buffer.complete_while_typing():
                    count += 1
        return count


class TestPasteCompletionGuard(unittest.TestCase):
    """Pin the live-completion gate introduced by the paste-freeze fix."""

    # ----- pure boundary -----

    def test_capacity_boundary(self):
        self.assertTrue(_live_completion_allowed(0))
        self.assertTrue(_live_completion_allowed(_CAP - 1))
        self.assertFalse(_live_completion_allowed(_CAP))
        self.assertFalse(_live_completion_allowed(_CAP * 4))

    # ----- app-aware filter -----

    def test_filter_reads_live_buffer_length(self):
        buf = Buffer()
        buf.text = "x" * (_CAP * 4)
        with _patch_app(buf):
            self.assertFalse(_live_completion_while_typing_filter())
        buf.text = "short draft"
        with _patch_app(buf):
            self.assertTrue(_live_completion_while_typing_filter())

    def test_filter_falls_back_to_allowed_outside_application(self):
        # prompt_toolkit's ``get_app`` never raises: with no running app it
        # returns a DummyApplication whose buffer is empty, so the gate
        # stays open. Test fixtures never run a real Application, so the
        # unpatched call is deterministic.
        self.assertTrue(_live_completion_while_typing_filter())

    # ----- real REPL wiring -----

    def test_repl_wires_condition_not_constant_true(self):
        repl = _make_repl()
        gate = repl.prompt_session.complete_while_typing
        self.assertIsInstance(gate, Condition)
        # Outside an application the filter falls back to allowed, so
        # callers that invoke the gate without an app keep working.
        self.assertTrue(gate())

    def test_repl_gate_off_inside_long_draft_app(self):
        repl = _make_repl()
        gate = repl.prompt_session.complete_while_typing
        buf = Buffer()
        buf.text = "x" * (_CAP * 4)
        with _patch_app(buf):
            self.assertFalse(gate())

    # ----- algorithmic guarantee (regression core) -----

    def test_small_paste_keeps_per_keystroke_live_completion(self):
        # Normal typing must be unaffected: every keystroke below the
        # capacity still opens the live-completion gate.
        sim = _PasteSim("hello world" * 10)  # 110 chars
        self.assertEqual(sim.open_keystrokes(), len(sim.text))

    def test_large_paste_completion_runs_bounded_by_capacity_not_length(self):
        # A single-line paste of n >> capacity characters must open the
        # gate at most ~capacity times (the pre-fix behaviour kept it open
        # for all n keystrokes, i.e. one merged-completer pass over a
        # growing draft per character).
        n = _CAP * 4
        sim = _PasteSim("a" * n)
        open_keys = sim.open_keystrokes()
        self.assertGreaterEqual(open_keys, 1)
        self.assertLessEqual(open_keys, _CAP)  # bounded by the capacity ...
        self.assertLess(open_keys, n // 4)  # ... not by the draft length
        # The gate opens only while the running draft stays short: after
        # the first ~capacity characters no further pass is scheduled.
        self.assertEqual(open_keys, _CAP - 1)

    def test_insert_past_capacity_schedules_no_completion_work(self):
        # Real framework path: once the draft is past the capacity, an
        # insertion must not schedule any completion work — neither the
        # completer's ``get_completions`` nor ``create_background_task``.
        completer = _CountingCompleter()
        buf = Buffer(
            completer=completer,
            complete_while_typing=_FILTER,
        )
        buf.text = "a" * (_CAP + 10)
        with _patch_app(buf) as stack:
            # ``stack.stub`` is what the patched ``get_app`` spellings return.
            self.assertFalse(buf.complete_while_typing())
            buf.insert_text("a")
            self.assertFalse(stack.stub.create_background_task.called)
            self.assertEqual(completer.runs, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
