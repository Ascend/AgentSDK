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

"""Asciicast recorder protocols copied here for package self-containment.

Originally this Protocol module lived at
``extensions/capabilities/recorder.py``. The visualizer package keeps
a local copy so it can run without importing the upstream
``extensions.capabilities`` package.

This is a *local copy*, not a re-export. The upstream module remains
the authoritative reference for ``extensions.recording``'s writer and
capture-handle implementations. Review this local copy against the
upstream module every six months, and propagate signature or semantic
changes to both modules in the same commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "AsciicastCapture",
    "AsciicastEvent",
    "AsciicastHeader",
    "RecordableSource",
]


# ---------------------------------------------------------------------------
# Wire-format dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AsciicastHeader:
    """The first line of an asciicast v2 ``.cast`` file.

    All fields except :attr:`version`, :attr:`width`, :attr:`height` are
    optional per the v2 spec. Callers usually leave ``env`` and
    ``theme`` empty; ``title`` and ``command`` are the most useful
    metadata for human readers browsing a directory of recordings.
    """

    width: int
    height: int
    version: int = 2
    timestamp: int | None = None
    duration: float | None = None
    idle_time_limit: float | None = None
    command: str | None = None
    title: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    theme: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Strip None values so the encoded header stays compact and
        # matches what asciinema players expect.
        data: dict[str, Any] = {"version": self.version}
        if self.timestamp is not None:
            data["timestamp"] = self.timestamp
        data["width"] = self.width
        data["height"] = self.height
        if self.duration is not None:
            data["duration"] = self.duration
        if self.idle_time_limit is not None:
            data["idle_time_limit"] = self.idle_time_limit
        if self.command is not None:
            data["command"] = self.command
        if self.title is not None:
            data["title"] = self.title
        if self.env:
            data["env"] = dict(self.env)
        if self.theme:
            data["theme"] = dict(self.theme)
        return data


@dataclass
class AsciicastEvent:
    """One event frame.

    ``t`` is *seconds since capture start* (monotonic). The writer
    computes the actual wall-clock time per event using the monotonic
    clock it started when the capture opened.

    ``kind`` follows the v2 vocabulary:
      * ``"o"`` — output written to the terminal (may contain ANSI
        escapes).
      * ``"i"`` — input read from the terminal.
      * ``"m"`` — navigation marker (the player treats the label as a
        breakpoint when ``pause-on-markers`` is set).
      * ``"r"`` — terminal resize; the writer formats ``data`` as
        ``"{cols}x{rows}"``.
    """

    t: float
    kind: Literal["o", "i", "m", "r"]
    data: str


# ---------------------------------------------------------------------------
# Capture handle — what adapters call into
# ---------------------------------------------------------------------------


@runtime_checkable
class AsciicastCapture(Protocol):
    """The handle an adapter receives at registration.

    Adapters call :meth:`emit` to push a frame, :meth:`marker` to drop a
    navigation breakpoint, :meth:`resize` to record a terminal resize,
    and :meth:`close` when their work for this capture is done (the
    writer may still be open for other adapters — close is per-source).
    """

    def emit(self, event: AsciicastEvent) -> None: ...
    def marker(self, label: str, text: str = "") -> None: ...
    def resize(self, cols: int, rows: int) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# RecordableSource — Protocol every contributing subsystem implements
# ---------------------------------------------------------------------------


@runtime_checkable
class RecordableSource(Protocol):
    """Anything that can contribute structured events to one capture.

    The source receives the :class:`AsciicastCapture` handle when
    :meth:`open` is called and is expected to register itself with the
    underlying subsystem (e.g. add a sink to
    :class:`CompositeProgressSink`). :meth:`close` reverses the wiring.

    A single capture may have multiple sources registered — each
    source's events go into the same ``.cast`` file.
    """

    source_id: str

    def open(self, capture: AsciicastCapture) -> None: ...
    def close(self) -> None: ...
