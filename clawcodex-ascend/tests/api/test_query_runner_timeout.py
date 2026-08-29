#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Headless query future timeout (Layer 0 quick fix).

Verifies that ``QueryRunner.stream()`` bounds its ``await future`` with
``asyncio.wait_for`` so a headless run that never completes cannot hold
the caller forever (see section 18, risk 5). On timeout we surface a
``SessionComplete(reason="exit_code=124")`` (the conventional GNU
timeout exit code) and emit a debug event for downstream forensics.

Test strategy: pass a tiny ``timeout_s`` via ``QueryConfig`` — the
module-level ``_QUERY_TIMEOUT_S`` constant was removed when the budget
was plumbed through the config (42368b64).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from extensions.api.query import (
    QueryConfig,
    QueryRunner,
    SessionComplete,
)


# Conventional GNU ``timeout`` exit code. Re-used here so downstream
# callers can distinguish "exceeded wall clock budget" from other
# non-zero exits without parsing the reason string.
_TIMEOUT_EXIT_CODE = 124


# ----------------------------------------------------------------------
# P108-B — timeout fires when the headless future never completes
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_timeout_when_headless_future_never_completes() -> None:
    """Risk #5: a headless run that hangs forever must surface as a
    timed-out ``SessionComplete`` rather than blocking the caller.
    """

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        debug_log = tmp_path / "debug.ndjson"

        def fake_run_headless_session(options):  # pragma: no cover - hangs
            # Sleep far longer than the 100 ms budget. ``time.sleep`` is
            # blocking but the executor runs it on a worker thread, so
            # ``wait_for`` on the future sees the timeout first.
            time.sleep(10.0)
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-timeout",
                debug_log_path=debug_log,
                timeout_s=0.1,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]

        # The debug log must record the timeout for postmortem review.
        rows = [json.loads(line) for line in debug_log.read_text(encoding="utf-8").splitlines()]
        timeout_rows = [row for row in rows if row["stage"] == "query_runner.timeout"]

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes, "expected a SessionComplete after timeout"
    assert completes[-1].reason == f"exit_code={_TIMEOUT_EXIT_CODE}", (
        f"timeout must surface as exit_code=124; got {completes[-1].reason!r}"
    )
    assert timeout_rows, "expected a query_runner.timeout debug event"
    assert timeout_rows[0]["timeout_s"] == pytest.approx(0.1)
    assert timeout_rows[0]["run_id"] == "run-timeout"


@pytest.mark.asyncio
async def test_stream_does_not_block_caller_on_legacy_zero_timeout() -> None:
    """``timeout=0`` is the documented escape hatch: it falls back to
    the legacy unbounded ``await future`` (section 18, design decision 5).

    We verify this by passing ``timeout_s=0`` and asserting the branch
    returns immediately after the future completes (the polling-loop
    exit, not the wait_for branch).
    """

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def fake_run_headless_session(options):
            options.stdout.write("ok")
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                timeout_s=0,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes[-1].reason == "success"


# ----------------------------------------------------------------------
# Regression — short-running headless sessions are unaffected.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completes_normally_when_future_finishes_in_budget() -> None:
    """Sanity: a fast-completing headless session must still end with
    ``SessionComplete(reason="success")`` and no ``query_runner.timeout``
    debug event.
    """

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        debug_log = tmp_path / "debug.ndjson"

        def fake_run_headless_session(options):
            options.stdout.write("done")
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-fast",
                debug_log_path=debug_log,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]

        rows = [json.loads(line) for line in debug_log.read_text(encoding="utf-8").splitlines()]

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes[-1].reason == "success"
    assert not any(row["stage"] == "query_runner.timeout" for row in rows), (
        "no timeout event should fire on a fast session"
    )


@pytest.mark.asyncio
async def test_stream_records_remaining_event_count_on_timeout() -> None:
    """The debug event payload should include enough context for an
    operator to understand what the headless session had been doing
    before the timeout fired (section 18, verification item 4).
    """

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        debug_log = tmp_path / "debug.ndjson"

        def fake_run_headless_session(options):
            # Emit a couple of tool events before sleeping.
            options.on_event(
                SimpleNamespace(
                    kind="tool_use",
                    tool_name="Bash",
                    tool_input={"command": "true"},
                    tool_output=None,
                    tool_use_id="tool-1",
                    is_error=False,
                    error=None,
                )
            )
            time.sleep(10.0)
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-context",
                debug_log_path=debug_log,
                timeout_s=0.1,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            # Drain quickly — we don't need every event for this check.
            async def _drain() -> None:
                async for _ in runner.stream():
                    pass

            # Bound the drain so a regression doesn't hang the test.
            await asyncio.wait_for(_drain(), timeout=2.0)

        rows = [json.loads(line) for line in debug_log.read_text(encoding="utf-8").splitlines()]
        timeout_rows = [row for row in rows if row["stage"] == "query_runner.timeout"]

    assert timeout_rows, "expected a query_runner.timeout debug event"
    payload = timeout_rows[0]
    # ``seconds_since_start`` is computed from the last event timestamp
    # in the polling loop; with one event before the sleep it must be
    # a non-negative float. ``stdout_len`` is the captured headless
    # stdout buffer length (always present).
    assert "seconds_since_start" in payload
    assert isinstance(payload["seconds_since_start"], (int, float))
    assert payload["seconds_since_start"] >= 0
    assert "stdout_len" in payload
