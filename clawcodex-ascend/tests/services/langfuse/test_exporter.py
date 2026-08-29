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

"""Tests for src/services/langfuse/exporter.py (P65-C)."""
# pylint: disable=no-name-in-module, redefined-outer-name

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from clawcodex_ext.services.analytics.events import AnalyticsEvent, EventType
from src.services.langfuse.exporter import (
    ExportResult,
    FORMAT_CHATML,
    FORMAT_JSONL,
    FORMAT_SFT,
    TrainingDataExporter,
    export_training_data,
)
from src.services.langfuse.sink import LangfuseSink


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sink_with_records() -> LangfuseSink:
    """A sink preloaded with a mix of events used by the tests below."""
    sink = LangfuseSink()
    # TURN_END with both prompt and completion → SFT-eligible.
    sink.emit(
        AnalyticsEvent(
            type=EventType.TURN_END,
            session_id="s1",
            model="claude-x",
            data={
                "prompt": "What is 2+2?",
                "completion": "4",
                "usage": {"input": 5, "output": 1},
                "latency_ms": 80.0,
            },
        )
    )
    # TURN_END with prompt only → SFT-ineligible (skipped).
    sink.emit(
        AnalyticsEvent(
            type=EventType.TURN_END,
            session_id="s1",
            model="claude-x",
            data={"prompt": "no completion"},
        )
    )
    # A non-turn record (kept in raw JSONL, dropped from SFT/ChatML).
    sink.emit(
        AnalyticsEvent(
            type=EventType.TOOL_USE,
            session_id="s1",
            data={"name": "Read", "input": {"path": "x.py"}},
        )
    )
    return sink


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def test_write_jsonl_emits_one_object_per_line(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    result = TrainingDataExporter(sink_with_records).write_jsonl(target)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    payloads = [json.loads(line) for line in lines]
    assert {p["type"] for p in payloads} == {"turn_end", "tool_use"}
    assert result.format == FORMAT_JSONL
    assert result.count == 3
    assert result.skipped == 0


def test_write_jsonl_preserves_unicode(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    TrainingDataExporter(sink_with_records).write_jsonl(target)
    content = target.read_text(encoding="utf-8")
    assert "What is 2+2?" in content


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


def test_write_sft_extracts_prompt_completion_pairs(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    target = tmp_path / "train.jsonl"
    result = TrainingDataExporter(sink_with_records).write_sft(target)
    lines = target.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    # Only the first TURN_END has both prompt and completion.
    assert len(payloads) == 1
    assert payloads[0]["prompt"] == "What is 2+2?"
    assert payloads[0]["completion"] == "4"
    assert payloads[0]["model"] == "claude-x"
    assert result.count == 1
    # The empty TURN_END is counted as skipped; the TOOL_USE is
    # silently filtered out (it's not a TURN_END to begin with).
    assert result.skipped == 1


def test_write_sft_skips_non_turn_records(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    target = tmp_path / "train.jsonl"
    TrainingDataExporter(sink_with_records).write_sft(target)
    payloads = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert all(p.get("prompt") and p.get("completion") for p in payloads)


# ---------------------------------------------------------------------------
# ChatML
# ---------------------------------------------------------------------------


def test_write_chatml_produces_messages_array(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    target = tmp_path / "train.jsonl"
    result = TrainingDataExporter(sink_with_records).write_chatml(target)
    payloads = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    assert payloads[0]["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert result.format == FORMAT_CHATML


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_export_dispatches_by_format(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    exporter = TrainingDataExporter(sink_with_records)
    jsonl_path = tmp_path / "out.jsonl"
    sft_path = tmp_path / "out.sft.jsonl"
    chatml_path = tmp_path / "out.chatml.jsonl"

    exporter.export(jsonl_path, format=FORMAT_JSONL)
    exporter.export(sft_path, format=FORMAT_SFT)
    exporter.export(chatml_path, format=FORMAT_CHATML)

    assert jsonl_path.exists()
    assert sft_path.exists()
    assert chatml_path.exists()


def test_export_raises_on_unknown_format(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown export format"):
        TrainingDataExporter(sink_with_records).export(tmp_path / "out.jsonl", format="bogus")


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_does_not_leave_tmp_files(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    TrainingDataExporter(sink_with_records).write_jsonl(target)
    # No .tmp_*.jsonl scratch file should remain.
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftover == []


def test_atomic_write_survives_failed_replace(
    sink_with_records: LangfuseSink, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``os.replace`` raises, the temp scratch must be cleaned up
    and the destination file must not exist.
    """
    import src.services.langfuse.exporter as exporter_module

    real_replace = exporter_module.os.replace

    def _broken_replace(src: Any, dst: Any) -> None:
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(exporter_module.os, "replace", _broken_replace)

    with pytest.raises(OSError, match="simulated crash"):
        TrainingDataExporter(sink_with_records).write_jsonl(tmp_path / "out.jsonl")
    # Destination must not exist (no torn write).
    assert not (tmp_path / "out.jsonl").exists()
    # No temp scratch should linger.
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftover == []
    # Restore the real replace so other tests aren't affected.
    monkeypatch.setattr(exporter_module.os, "replace", real_replace)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_export_training_data_helper_writes_file(sink_with_records: LangfuseSink, tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    result = export_training_data(sink_with_records, target)
    assert isinstance(result, ExportResult)
    assert target.exists()
    assert result.path == target


# ---------------------------------------------------------------------------
# iter_records
# ---------------------------------------------------------------------------


def test_iter_records_yields_every_buffer_entry(
    sink_with_records: LangfuseSink,
) -> None:
    records = list(TrainingDataExporter(sink_with_records).iter_records())
    assert len(records) == 3
    assert {r["type"] for r in records} == {"turn_end", "tool_use"}


def test_iter_records_is_isolated_from_buffer_mutation(
    sink_with_records: LangfuseSink,
) -> None:
    """Mutating the sink after iteration begins must not crash."""
    exporter = TrainingDataExporter(sink_with_records)
    iterator = exporter.iter_records()
    first = next(iterator)
    sink_with_records.clear_buffer()
    assert first["type"] in {"turn_end", "tool_use"}
