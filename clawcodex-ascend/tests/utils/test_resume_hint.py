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

"""Unit tests for the centralised resume hint helper.

Covers S-R1 (CCB ``printResumeHint()`` alignment) — gates on
non-empty session_id and a TTY stream, plus the process-wide idempotency
latch that prevents the atexit-callback double-print.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from clawcodex_ext.utils.resume_hint import (
    print_resume_hint,
    reset_resume_hint_for_test_only,
)


@pytest.fixture(autouse=True)
def _reset_latch():
    """Clear the process-wide hint latch before every test in this module.

    Prevents bleed between test cases: once one test prints, the latch
    stays True for the rest of the pytest run unless reset.
    """
    reset_resume_hint_for_test_only()
    yield
    reset_resume_hint_for_test_only()


class _FakeTTYStream:
    """A minimal stream that pretends to be a TTY."""

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.flushed = 0

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self.buffer.append(s)
        return len(s)

    def flush(self) -> None:
        self.flushed += 1


class _FakeNonTTYStream:
    """A stream that explicitly is NOT a TTY."""

    def __init__(self) -> None:
        self.buffer: list[str] = []

    def isatty(self) -> bool:
        return False

    def write(self, s: str) -> int:
        self.buffer.append(s)
        return len(s)


class _NoIsattyStream:
    """A stream without an ``isatty`` method (defensive probe)."""

    def __init__(self) -> None:
        self.buffer: list[str] = []

    def write(self, s: str) -> int:
        self.buffer.append(s)
        return len(s)


class _FlushRaisingStream:
    """A stream whose ``flush()`` raises — should be swallowed."""

    def __init__(self) -> None:
        self.buffer: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self.buffer.append(s)
        return len(s)

    def flush(self) -> None:
        raise OSError("disk full")


class TestPrintResumeHint:
    """Behavioural tests for ``print_resume_hint``."""

    def test_prints_hint_when_tty(self) -> None:
        stream = _FakeTTYStream()
        print_resume_hint("abc123", stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == ["\nResume this session with: clawcodex --resume abc123\n"]

    def test_skips_when_not_tty(self) -> None:
        stream = _FakeNonTTYStream()
        print_resume_hint("abc123", stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == []

    @pytest.mark.parametrize("empty_id", [None, "", "   ", "\t\n"])
    def test_skips_empty_session_id(self, empty_id: object) -> None:
        stream = _FakeTTYStream()
        print_resume_hint(empty_id, stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == []

    def test_uses_default_stdout_when_no_stream(self) -> None:
        fake = _FakeTTYStream()
        with patch.object(sys, "stdout", fake):
            print_resume_hint("session-xyz")
        assert fake.buffer == ["\nResume this session with: clawcodex --resume session-xyz\n"]

    def test_handles_stream_without_isatty(self) -> None:
        """Defensive: a stream that lacks ``isatty`` should not raise."""
        stream = _NoIsattyStream()
        # Without isatty, the helper should treat the stream as non-TTY
        # and skip. The probe itself must not raise.
        print_resume_hint("abc", stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == []

    def test_flushes_stream(self) -> None:
        stream = _FakeTTYStream()
        print_resume_hint("abc", stream=stream)  # type: ignore[arg-type]
        assert stream.flushed == 1

    def test_swallows_flush_exceptions(self) -> None:
        """A failing ``flush()`` must not propagate."""
        stream = _FlushRaisingStream()
        # Should not raise even though flush() raises OSError.
        print_resume_hint("abc", stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == ["\nResume this session with: clawcodex --resume abc\n"]

    def test_format_matches_ccb(self) -> None:
        """Lock the output format to the CCB ``printResumeHint()`` contract."""
        sid = "0123456789abcdef0123456789abcdef"
        stream = _FakeTTYStream()
        print_resume_hint(sid, stream=stream)  # type: ignore[arg-type]
        out = "".join(stream.buffer)
        assert out == f"\nResume this session with: clawcodex --resume {sid}\n"
        # Also assert the structural pieces are present verbatim.
        assert "Resume this session with:" in out
        assert "clawcodex --resume" in out
        assert sid in out
        # And that the leading/trailing newlines are still emitted.
        assert out.startswith("\n")
        assert out.endswith("\n")

    def test_strips_surrounding_whitespace_in_session_id(self) -> None:
        """``'  abc  '`` should be treated as ``'abc'`` and printed cleanly."""
        stream = _FakeTTYStream()
        print_resume_hint("  abc  ", stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == ["\nResume this session with: clawcodex --resume abc\n"]

    def test_integration_with_stringio_via_isatty_patch(self) -> None:
        """A plain ``io.StringIO`` is not a TTY, so the helper must skip."""
        buf = io.StringIO()
        # io.StringIO does not implement isatty in a way that returns True;
        # verify the helper does not raise on it.
        print_resume_hint("abc", stream=buf)
        assert buf.getvalue() == ""


class TestIdempotencyLatch:
    """The process-wide printed latch prevents S-R1 double-print.

    The atexit cleanup registered by
    ``clawcodex_ext/frontend/repl_extensions.py:_register_signal_session_save``
    and the inline REPL ``/exit`` print at ``repl/core.py:_print_resume_hint``
    both call this helper. Without a latch, the user would see the hint
    twice on a normal ``/exit``. The latch guarantees exactly-once.
    """

    def test_second_call_is_noop(self) -> None:
        stream = _FakeTTYStream()
        print_resume_hint("first-sid", stream=stream)  # type: ignore[arg-type]
        print_resume_hint("second-sid", stream=stream)  # type: ignore[arg-type]
        # Only the first call should have written.
        assert stream.buffer == ["\nResume this session with: clawcodex --resume first-sid\n"]

    def test_third_and_fourth_calls_still_noop(self) -> None:
        stream = _FakeTTYStream()
        for sid in ("alpha", "beta", "gamma", "delta"):
            print_resume_hint(sid, stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == ["\nResume this session with: clawcodex --resume alpha\n"]

    def test_latch_survives_across_streams(self) -> None:
        """The latch is process-wide, not per-stream.

        The first call on ``stream_a`` latches it; a later call on
        ``stream_b`` with a different sid still does nothing.
        """
        a = _FakeTTYStream()
        b = _FakeTTYStream()
        print_resume_hint("a-sid", stream=a)  # type: ignore[arg-type]
        print_resume_hint("b-sid", stream=b)  # type: ignore[arg-type]
        assert a.buffer == ["\nResume this session with: clawcodex --resume a-sid\n"]
        assert b.buffer == []

    def test_reset_clears_latch(self) -> None:
        stream1 = _FakeTTYStream()
        stream2 = _FakeTTYStream()
        print_resume_hint("before", stream=stream1)  # type: ignore[arg-type]
        assert stream1.buffer == ["\nResume this session with: clawcodex --resume before\n"]
        reset_resume_hint_for_test_only()
        print_resume_hint("after", stream=stream2)  # type: ignore[arg-type]
        assert stream2.buffer == ["\nResume this session with: clawcodex --resume after\n"]

    def test_latch_does_not_leak_when_first_call_skips(self) -> None:
        """If the first call is gated (non-TTY or empty sid), the latch
        must NOT flip — a later valid call should still print.

        This protects the S-R1 behaviour: a transient non-TTY probe
        during a test or pre-init phase must not silently disable the
        real hint at process end.
        """
        nontty = _FakeNonTTYStream()
        print_resume_hint("ignored", stream=nontty)  # type: ignore[arg-type]
        # Latch should still be False; the next valid call should print.
        stream = _FakeTTYStream()
        print_resume_hint("real-sid", stream=stream)  # type: ignore[arg-type]
        assert stream.buffer == ["\nResume this session with: clawcodex --resume real-sid\n"]
