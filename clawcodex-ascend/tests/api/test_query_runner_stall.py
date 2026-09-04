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

"""Stream-stall watchdog in the query event-drain loop.

The observed production failure: a provider accepts the request and then
never sends a single chunk. The per-provider idle watchdog (WI-5.2)
covers only the Anthropic SDK stream, so OpenAI-compatible paths burned
the entire wall-clock budget (20 minutes observed live, ~40 heartbeats
of ``seconds_since_last_event`` growing monotonically) doing nothing.

These tests pin the loop-level, provider-agnostic backstop:

- zero activity for ``stall_timeout_s`` → ``SessionComplete``
  with ``exit_code=125`` plus a ``query_runner.stall_detected`` debug
  event, within ~stall_timeout_s (not the full budget);
- any activity (stdout growth or tool events) keeps resetting the
  deadline — slow-but-alive runs are never killed;
- ``stall_timeout_s=0`` disables the watchdog (design decision #5).
"""

from __future__ import annotations

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

_STALL_EXIT_CODE = 125
_TIMEOUT_EXIT_CODE = 124


def _debug_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_stall_fires_on_zero_activity() -> None:
    """A headless run that produces nothing must end as exit_code=125
    after ~stall_timeout_s, not after the full wall-clock budget.
    """

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        debug_log = tmp_path / "debug.ndjson"

        def fake_run_headless_session(options):  # pragma: no cover - stalls
            time.sleep(2.0)  # no events, no stdout — a dead provider
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-stall",
                debug_log_path=debug_log,
                timeout_s=60.0,  # far away: the stall must fire first
                stall_timeout_s=0.3,
                stall_warn_s=0.1,
            )
        )

        started = time.monotonic()
        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]
        elapsed = time.monotonic() - started

        rows = _debug_rows(debug_log)
        stall_rows = [r for r in rows if r["stage"] == "query_runner.stall_detected"]
        warn_rows = [r for r in rows if r["stage"] == "query_runner.stall_suspected"]
        abort_rows = [r for r in rows if r["stage"] == "query_runner.abort_signalled"]

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes, "expected a SessionComplete after the stall"
    assert completes[-1].reason == f"exit_code={_STALL_EXIT_CODE}", (
        f"stall must surface as exit_code=125; got {completes[-1].reason!r}"
    )
    assert elapsed < 5.0, f"stall must fire in ~stall_timeout_s, took {elapsed:.1f}s"
    assert stall_rows, "expected a query_runner.stall_detected debug event"
    assert stall_rows[0]["run_id"] == "run-stall"
    assert stall_rows[0]["stall_timeout_s"] == pytest.approx(0.3)
    assert stall_rows[0]["tool_events"] == 0
    assert abort_rows, "stall must trip the abort controller for cooperative unwind"
    # WARN tier: the early diagnosis fires before the abort tier, and
    # exactly once for a single uninterrupted silence episode.
    assert len(warn_rows) == 1, f"expected one stall_suspected, got {len(warn_rows)}"
    assert warn_rows[0]["stall_warn_s"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_stdout_growth_keeps_resetting_the_deadline() -> None:
    """A slow-but-alive run (streamed text, no tool events) must never
    be killed by the watchdog.
    """

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def fake_run_headless_session(options):
            # Emit a chunk every 100 ms for ~0.8 s: each write is
            # activity, so a 300 ms stall deadline never expires even
            # though the total run exceeds it several times over.
            for _ in range(8):
                time.sleep(0.1)
                options.stdout.write("chunk ")
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-alive",
                debug_log_path=tmp_path / "debug.ndjson",
                timeout_s=60.0,
                stall_timeout_s=0.3,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes[-1].reason == "success", f"an active stream must complete normally; got {completes[-1].reason!r}"


@pytest.mark.asyncio
async def test_tool_events_count_as_activity() -> None:
    """Tool events (the other activity signal) also reset the deadline."""

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def fake_run_headless_session(options):
            for i in range(8):
                time.sleep(0.1)
                options.on_event(
                    SimpleNamespace(
                        kind="tool_use",
                        tool_name="Read",
                        tool_input={"path": f"f{i}"},
                        tool_use_id=f"tu-{i}",
                        is_error=False,
                        error=None,
                    )
                )
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-tools",
                debug_log_path=tmp_path / "debug.ndjson",
                timeout_s=60.0,
                stall_timeout_s=0.3,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes[-1].reason == "success"


@pytest.mark.asyncio
async def test_inflight_tool_pauses_stall_watchdog() -> None:
    """Silence during a tool call belongs to the per-tool watchdog."""

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        debug_log = tmp_path / "debug.ndjson"

        def fake_run_headless_session(options):
            options.on_event(
                SimpleNamespace(
                    kind="tool_use",
                    tool_name="Agent",
                    tool_input={"prompt": "long task"},
                    tool_use_id="tu-long",
                    is_error=False,
                    error=None,
                )
            )
            time.sleep(0.5)
            options.on_event(
                SimpleNamespace(
                    kind="tool_result",
                    tool_name="Agent",
                    tool_output="done",
                    tool_use_id="tu-long",
                    is_error=False,
                    error=None,
                )
            )
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-inflight",
                debug_log_path=debug_log,
                timeout_s=60.0,
                stall_timeout_s=0.2,
                stall_warn_s=0.1,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]

        rows = _debug_rows(debug_log)

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes[-1].reason == "success"
    assert not [row for row in rows if row["stage"] in {"query_runner.stall_suspected", "query_runner.stall_detected"}]


@pytest.mark.asyncio
async def test_zero_disables_the_watchdog() -> None:
    """``stall_timeout_s=0`` is the escape hatch: a silent run then
    falls through to the wall-clock budget (exit_code=124), proving the
    stall path stayed out of the way.
    """

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def fake_run_headless_session(options):  # pragma: no cover - stalls
            time.sleep(2.0)
            return 0

        runner = QueryRunner(
            QueryConfig(
                prompt="hello",
                workspace=tmp_path,
                run_id="run-disabled",
                debug_log_path=tmp_path / "debug.ndjson",
                timeout_s=0.3,
                stall_timeout_s=0.0,
            )
        )

        with patch(
            "extensions.capabilities.headless_runner.run_headless_session",
            fake_run_headless_session,
        ):
            events = [event async for event in runner.stream()]

    completes = [event for event in events if isinstance(event, SessionComplete)]
    assert completes[-1].reason == f"exit_code={_TIMEOUT_EXIT_CODE}", (
        f"with the watchdog disabled the wall-clock budget must be the one that fires; got {completes[-1].reason!r}"
    )
