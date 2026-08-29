#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""P65-C — training-data exporter for LangfuseSink buffer.

The :class:`LangfuseSink` keeps a bounded in-memory copy of every
event it forwards to Langfuse (see
:data:`LangfuseSink._buffer`). This module reads that buffer and
serialises it for downstream SFT / DPO pipelines.

Why a separate exporter?
------------------------
Export is I/O-bound and may target any of several on-disk shapes
(JSONL for fine-tuning, ChatML for alignment work, raw JSONL
for archival). Pulling that logic out of the sink keeps the sink
focused on dispatch + buffering and lets each format evolve
independently. The exporter does not require a live Langfuse
client — it works against a snapshot of the buffer.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sink import LangfuseSink

logger = logging.getLogger(__name__)


# --- Format identifiers ---------------------------------------------------


FORMAT_JSONL: str = "jsonl"
FORMAT_SFT: str = "sft"
FORMAT_CHATML: str = "chatml"

_VALID_FORMATS: frozenset[str] = frozenset({FORMAT_JSONL, FORMAT_SFT, FORMAT_CHATML})


# --- Result type ----------------------------------------------------------


@dataclass(frozen=True)
class ExportResult:
    """Summary of an export run.

    ``path`` is the file the exporter wrote to. ``count`` is the
    number of records that ended up in the file. ``skipped`` is
    the number of records the exporter chose not to include (e.g.
    a TURN_END without a prompt + completion pair in SFT mode).
    """

    path: Path
    count: int
    skipped: int
    format: str

    def __str__(self) -> str:  # pragma: no cover — debugging aid
        return f"ExportResult(path={self.path!s}, count={self.count}, skipped={self.skipped}, format={self.format!r})"


# --- Helpers --------------------------------------------------------------


def _atomic_write_text(path: Path, lines: Iterable[str]) -> None:
    """Write ``lines`` to ``path`` via a temp+rename swap.

    Mirrors the atomic-primitive used elsewhere in the repo
    (``extensions/orchestrator/report_writer.py:98-122``,
    ``src/agent/report_store.py``). On any error the temp file is
    removed so the destination either contains the old content or
    nothing — never a torn write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix or ".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _record_to_sft(record: dict[str, Any]) -> dict[str, Any] | None:
    """Project a TURN_END-shaped record into a SFT pair.

    Returns ``None`` if the record lacks both ``input`` and
    ``output`` (the caller counts this as ``skipped``).
    """
    prompt = record.get("input")
    completion = record.get("output")
    if not prompt or not completion:
        return None

    pair: dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
    }
    if record.get("model"):
        pair["model"] = record["model"]
    if record.get("session_id"):
        pair["session_id"] = record["session_id"]
    if record.get("usage"):
        pair["usage"] = record["usage"]
    if record.get("latency_ms") is not None:
        pair["latency_ms"] = record["latency_ms"]
    return pair


def _record_to_chatml(record: dict[str, Any]) -> dict[str, Any] | None:
    """Project a TURN_END-shaped record into a ChatML message list.

    Returns ``None`` if the record has neither ``input`` nor
    ``output``. The exported shape is the one most fine-tuning
    libraries (Axolotl, LLaMA-Factory) accept out of the box.
    """
    prompt = record.get("input")
    completion = record.get("output")
    if not prompt or not completion:
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "session_id": record.get("session_id", ""),
        "model": record.get("model", ""),
    }


# --- Exporter -------------------------------------------------------------


class TrainingDataExporter:
    """Serialize a :class:`LangfuseSink` buffer to disk.

    Parameters
    ----------
    sink:
        The sink whose buffer will be exported. The exporter
        reads via :meth:`LangfuseSink.snapshot`, so it never holds
        the sink's internal lock for longer than the snapshot
        call. Pass a sink-like duck-type (any object with a
        ``snapshot() -> list[dict]`` method) for testing.
    """

    def __init__(self, sink: LangfuseSink) -> None:
        self._sink = sink

    # -- raw access ---------------------------------------------------------

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield every record currently in the sink's buffer."""
        yield from self._sink.snapshot()

    # -- format-specific writers -------------------------------------------

    def write_jsonl(self, path: Path | str) -> ExportResult:
        """Write one JSON object per line (raw, lossless)."""
        target = Path(path)
        lines = (json.dumps(r, ensure_ascii=False, default=str) for r in self.iter_records())
        records = list(self.iter_records())
        _atomic_write_text(target, lines)
        return ExportResult(
            path=target,
            count=len(records),
            skipped=0,
            format=FORMAT_JSONL,
        )

    def write_sft(self, path: Path | str) -> ExportResult:
        """Write SFT pairs ``{prompt, completion}`` derived from
        TURN_END records.

        Non-turn records are skipped. Records without both
        ``input`` and ``output`` are skipped (counted in
        ``skipped``).
        """
        target = Path(path)
        all_records = list(self.iter_records())

        written = 0
        skipped = 0
        lines: list[str] = []

        for record in all_records:
            if record.get("type") != "turn_end":
                continue
            pair = _record_to_sft(record)
            if pair is None:
                skipped += 1
                continue
            lines.append(json.dumps(pair, ensure_ascii=False, default=str))
            written += 1

        _atomic_write_text(target, lines)
        return ExportResult(
            path=target,
            count=written,
            skipped=skipped,
            format=FORMAT_SFT,
        )

    def write_chatml(self, path: Path | str) -> ExportResult:
        """Write ChatML ``{messages: [...]}`` records derived from
        TURN_END events.

        Skips non-turn records and turn records missing either
        side of the conversation (counted in ``skipped``).
        """
        target = Path(path)
        all_records = list(self.iter_records())

        written = 0
        skipped = 0
        lines: list[str] = []

        for record in all_records:
            if record.get("type") != "turn_end":
                continue
            chatml = _record_to_chatml(record)
            if chatml is None:
                skipped += 1
                continue
            lines.append(json.dumps(chatml, ensure_ascii=False, default=str))
            written += 1

        _atomic_write_text(target, lines)
        return ExportResult(
            path=target,
            count=written,
            skipped=skipped,
            format=FORMAT_CHATML,
        )

    # -- dispatch -----------------------------------------------------------

    def export(
        self,
        path: Path | str,
        format: str = FORMAT_JSONL,  # pylint: disable=redefined-builtin
    ) -> ExportResult:
        """Dispatch to the right writer based on ``format``.

        Raises :class:`ValueError` for unknown formats.
        """
        if format not in _VALID_FORMATS:
            raise ValueError(f"unknown export format {format!r}; valid options: {sorted(_VALID_FORMATS)}")

        if format == FORMAT_JSONL:
            return self.write_jsonl(path)
        if format == FORMAT_SFT:
            return self.write_sft(path)
        # format == FORMAT_CHATML
        return self.write_chatml(path)


# --- Module-level convenience --------------------------------------------


def export_training_data(
    sink: LangfuseSink,
    path: Path | str,
    format: str = FORMAT_JSONL,  # pylint: disable=redefined-builtin
) -> ExportResult:
    """One-shot helper: build a :class:`TrainingDataExporter` and run it.

    This is the function most callers will reach for. For
    repeated exports, instantiate :class:`TrainingDataExporter`
    directly so the sink is snapshot only once.
    """
    return TrainingDataExporter(sink).export(path, format=format)


__all__ = [
    "FORMAT_CHATML",
    "FORMAT_JSONL",
    "FORMAT_SFT",
    "ExportResult",
    "TrainingDataExporter",
    "export_training_data",
]
