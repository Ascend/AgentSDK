# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Recorder capability contracts — Protocol-only surface for asciicast recording.

Concrete writers, adapters, and the CLI live in ``extensions/recording/``.
Primitives:

* :class:`AsciicastHeader` — JSON header written as line 1 of a v2 ``.cast`` file.
* :class:`AsciicastEvent` — one in-memory frame -> a ``[time, code, data]`` line.
* :class:`AsciicastCapture` — handle adapters receive at registration.
* :class:`RecordableSource` — Protocol subsystems implement to contribute events.

The Protocol layer lets tests and the CLI build stub adapters without
dragging in the writer, and lets the recording package evolve the file
format without breaking per-subsystem adapters.
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


@dataclass
class AsciicastHeader:
    """First line of an asciicast v2 ``.cast`` file.

    Only :attr:`version`, :attr:`width`, :attr:`height` are required;
    ``env`` / ``theme`` are usually left empty.
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
        # Strip None values so the encoded header stays compact.
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

    ``t`` is seconds since capture start (monotonic). ``kind`` follows
    the v2 vocabulary: ``"o"`` output, ``"i"`` input, ``"m"`` navigation
    marker, ``"r"`` resize.
    """

    t: float
    kind: Literal["o", "i", "m", "r"]
    data: str


@runtime_checkable
class AsciicastCapture(Protocol):
    """Registration handle adapters call into: emit, marker, resize, close."""

    def emit(self, event: AsciicastEvent) -> None: ...
    def marker(self, label: str, text: str = "") -> None: ...
    def resize(self, cols: int, rows: int) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class RecordableSource(Protocol):
    """Anything that can contribute structured events to one capture.

    Receives the :class:`AsciicastCapture` on :meth:`open` (registering
    itself with the underlying subsystem); :meth:`close` reverses the
    wiring. Multiple sources may share one capture.
    """

    source_id: str

    def open(self, capture: AsciicastCapture) -> None: ...
    def close(self) -> None: ...
