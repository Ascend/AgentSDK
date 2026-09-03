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

"""Strict validation tests for the kairos service-layer dataclasses.

These cover the same validation rules that ``__post_init__`` enforces on
:class:`TickConfig`, :class:`BriefSummarySnapshot`, and
:class:`DailyLogEntry`. TickEvent is a plain payload dataclass with no
validation, so its coverage is in test_scheduler instead.
"""
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import time

import pytest

from src.services.kairos.models import (
    BriefSummarySnapshot,
    DailyLogEntry,
    TickConfig,
    TickEvent,
    format_local_timestamp,
)


# ---------------------------------------------------------------------------
# TickConfig
# ---------------------------------------------------------------------------


class TestTickConfig:
    def test_minimal_valid(self) -> None:
        cfg = TickConfig(id="main", interval_seconds=60.0)
        assert cfg.id == "main"
        assert cfg.interval_seconds == 60.0
        assert cfg.enabled is True
        assert cfg.jitter_fraction == 0.0
        assert cfg.name is None
        assert dict(cfg.metadata) == {}

    def test_id_must_be_nonempty_string(self) -> None:
        with pytest.raises(ValueError, match="id must be a non-empty string"):
            TickConfig(id="", interval_seconds=1.0)
        with pytest.raises(ValueError, match="id must be a non-empty string"):
            TickConfig(id=123, interval_seconds=1.0)  # type: ignore[arg-type]

    def test_id_rejects_invalid_characters(self) -> None:
        for bad in ["has space", "has/slash", "x" * 65, "weird!char"]:
            with pytest.raises(ValueError, match="invalid characters"):
                TickConfig(id=bad, interval_seconds=1.0)

    def test_interval_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds must be positive"):
            TickConfig(id="x", interval_seconds=0)
        with pytest.raises(ValueError, match="interval_seconds must be positive"):
            TickConfig(id="x", interval_seconds=-1.0)
        with pytest.raises(TypeError, match="interval_seconds must be a number"):
            TickConfig(id="x", interval_seconds="not a number")  # type: ignore[arg-type]

    def test_jitter_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="jitter must be non-negative"):
            TickConfig(id="x", interval_seconds=1.0, jitter_fraction=-0.1)
        with pytest.raises(ValueError, match="jitter is a fraction"):
            TickConfig(id="x", interval_seconds=1.0, jitter_fraction=1.5)
        with pytest.raises(TypeError, match="jitter must be a number"):
            TickConfig(id="x", interval_seconds=1.0, jitter_fraction="0.1")  # type: ignore[arg-type]
        # Boundaries are accepted.
        TickConfig(id="x", interval_seconds=1.0, jitter_fraction=0.0)
        TickConfig(id="x", interval_seconds=1.0, jitter_fraction=1.0)

    def test_name_must_be_string_or_none(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=1.0, name="My Tick")
        assert cfg.name == "My Tick"
        with pytest.raises(ValueError, match="name must be a string"):
            TickConfig(id="x", interval_seconds=1.0, name=123)  # type: ignore[arg-type]

    def test_metadata_keys_must_be_nonempty_strings(self) -> None:
        TickConfig(id="x", interval_seconds=1.0, metadata={"a": 1})
        with pytest.raises(ValueError, match="metadata keys must be non-empty strings"):
            TickConfig(id="x", interval_seconds=1.0, metadata={"": 1})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="metadata keys must be non-empty strings"):
            TickConfig(id="x", interval_seconds=1.0, metadata={1: "x"})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="metadata must be a mapping"):
            TickConfig(id="x", interval_seconds=1.0, metadata=["not a dict"])  # type: ignore[arg-type]

    def test_display_name_falls_back_to_id(self) -> None:
        cfg = TickConfig(id="main", interval_seconds=1.0)
        assert cfg.display_name == "main"
        cfg2 = TickConfig(id="main", interval_seconds=1.0, name="Main Loop")
        assert cfg2.display_name == "Main Loop"

    def test_is_frozen(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TickEvent (payload only)
# ---------------------------------------------------------------------------


class TestTickEvent:
    def test_payload_round_trip(self) -> None:
        ev = TickEvent(
            scheduler_id="main",
            tick_number=42,
            scheduled_at=100.0,
            actual_at=100.05,
            jitter_applied=0.05,
        )
        assert ev.scheduler_id == "main"
        assert ev.tick_number == 42
        assert ev.drift == pytest.approx(0.05)

    def test_drift_default_zero(self) -> None:
        ev = TickEvent(
            scheduler_id="main",
            tick_number=1,
            scheduled_at=0.0,
            actual_at=0.0,
        )
        assert ev.drift == 0.0


# ---------------------------------------------------------------------------
# BriefSummarySnapshot
# ---------------------------------------------------------------------------


class TestBriefSummarySnapshot:
    def test_minimal_valid(self) -> None:
        snap = BriefSummarySnapshot(agent_id="agent1", session_id="sess1")
        assert snap.agent_id == "agent1"
        assert snap.session_id == "sess1"
        assert snap.tick_number == 0
        assert snap.pending_tasks == ()
        assert snap.last_action is None
        assert dict(snap.metadata) == {}

    def test_agent_id_validated(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            BriefSummarySnapshot(agent_id="", session_id="sess1")
        with pytest.raises(ValueError, match="agent_id"):
            BriefSummarySnapshot(agent_id="bad space", session_id="sess1")

    def test_session_id_must_be_nonempty_string(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            BriefSummarySnapshot(agent_id="agent1", session_id="")
        with pytest.raises(ValueError, match="session_id"):
            BriefSummarySnapshot(agent_id="agent1", session_id=None)  # type: ignore[arg-type]

    def test_tick_number_must_be_nonneg_int(self) -> None:
        with pytest.raises(ValueError, match="tick_number"):
            BriefSummarySnapshot(agent_id="a", session_id="s", tick_number=-1)
        with pytest.raises(ValueError, match="tick_number"):
            BriefSummarySnapshot(agent_id="a", session_id="s", tick_number="1")  # type: ignore[arg-type]

    def test_pending_tasks_coerced_to_tuple(self) -> None:
        snap = BriefSummarySnapshot(
            agent_id="a",
            session_id="s",
            pending_tasks=["t1", "t2"],  # type: ignore[arg-type]
        )
        assert isinstance(snap.pending_tasks, tuple)
        assert snap.pending_tasks == ("t1", "t2")

    def test_last_action_must_be_string_when_set(self) -> None:
        with pytest.raises(ValueError, match="last_action"):
            BriefSummarySnapshot(
                agent_id="a",
                session_id="s",
                last_action=123,  # type: ignore[arg-type]
            )

    def test_metadata_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match="metadata must be a mapping"):
            BriefSummarySnapshot(
                agent_id="a",
                session_id="s",
                metadata=["not a dict"],  # type: ignore[arg-type]
            )

    def test_captured_at_defaults_to_now(self) -> None:
        before = time.time()
        snap = BriefSummarySnapshot(agent_id="a", session_id="s")
        after = time.time()
        assert before <= snap.captured_at <= after

    def test_captured_at_can_be_pinned(self) -> None:
        snap = BriefSummarySnapshot(agent_id="a", session_id="s", captured_at=1234567890.0)
        assert snap.captured_at == 1234567890.0


# ---------------------------------------------------------------------------
# DailyLogEntry
# ---------------------------------------------------------------------------


class TestDailyLogEntry:
    def test_minimal_valid(self) -> None:
        entry = DailyLogEntry(timestamp="2026-06-19T10:00:00", body="hello")
        out = entry.render()
        assert out.startswith("## 2026-06-19T10:00:00")
        assert "hello" in out

    def test_timestamp_must_be_nonempty_string(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            DailyLogEntry(timestamp="", body="x")
        with pytest.raises(ValueError, match="timestamp"):
            DailyLogEntry(timestamp=123, body="x")  # type: ignore[arg-type]

    def test_body_must_be_string(self) -> None:
        with pytest.raises(TypeError, match="body must be a string"):
            DailyLogEntry(timestamp="t", body=42)  # type: ignore[arg-type]

    def test_tags_coerced_and_validated(self) -> None:
        entry = DailyLogEntry(
            timestamp="t",
            body="x",
            tags=["a", "b"],  # type: ignore[arg-type]
        )
        assert isinstance(entry.tags, tuple)
        assert entry.tags == ("a", "b")
        with pytest.raises(ValueError, match="tags must be non-empty strings"):
            DailyLogEntry(timestamp="t", body="x", tags=["ok", ""])
        with pytest.raises(ValueError, match="tags must be non-empty strings"):
            DailyLogEntry(timestamp="t", body="x", tags=["ok", 1])  # type: ignore[arg-type]

    def test_render_strips_trailing_whitespace(self) -> None:
        entry = DailyLogEntry(timestamp="t", body="x\n\n")
        rendered = entry.render()
        # Trailing blank line from rstrip gone, then we add exactly one \n.
        assert rendered == "## t\n\nx\n"

    def test_render_with_tags(self) -> None:
        entry = DailyLogEntry(timestamp="t", body="x", tags=("a", "b"))
        rendered = entry.render()
        assert "#a #b" in rendered
        assert rendered.endswith("\n")


# ---------------------------------------------------------------------------
# format_local_timestamp
# ---------------------------------------------------------------------------


class TestFormatLocalTimestamp:
    def test_default_uses_time_time(self) -> None:
        before = time.time()
        out = format_local_timestamp()
        after = time.time()
        # Parse the ISO string and check the seconds component falls
        # within the wall-clock window. We compare only second precision
        # because datetime.fromisoformat returns a naive datetime that
        # ``timestamp()`` interprets as local time, which differs from
        # UTC ``time.time()`` by the local TZ offset.
        from datetime import datetime

        parsed = datetime.fromisoformat(out)
        assert before - 1 <= parsed.timestamp() <= after + 1

    def test_explicit_timestamp(self) -> None:
        # 2026-06-19T00:00:00 UTC. The local-time rendering will shift
        # this to a different calendar day depending on TZ, so the test
        # round-trips the seconds component rather than asserting a
        # specific year/month/day.
        from datetime import datetime, timezone

        target = datetime(2026, 6, 19, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        out = format_local_timestamp(target)
        # Round-trip through parse + epoch conversion. Allow 1-second
        # slack because the formatter truncates sub-second precision.
        parsed = datetime.fromisoformat(out).replace(tzinfo=None)
        assert abs(parsed.timestamp() - target) <= 1.0

    def test_seconds_precision(self) -> None:
        out = format_local_timestamp(1749696000.0)
        # No microseconds in output.
        assert "." not in out
