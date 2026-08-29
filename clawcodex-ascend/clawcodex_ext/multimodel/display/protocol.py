#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Display-neutral state and contracts for a multi-model turn.

The scheduler can use these types without importing Textual.  That keeps the
same result stream usable by the TUI and by non-interactive CLI output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from clawcodex_ext.capabilities.multimodel_protocol import MultiModelResult  # pylint: disable=no-name-in-module


class DisplayPhase(str, Enum):
    STREAMING = "streaming"
    SELECTION = "selection"
    ADOPTED = "adopted"
    CANCELLED = "cancelled"


@dataclass
class ModelDisplayState:
    """The renderable state for one configured model slot."""

    slot: str
    content: str = ""
    duration_ms: int | None = None
    tokens: dict[str, int] = field(default_factory=dict)
    status: str = "pending"
    error: str | None = None
    expanded: bool = False

    @property
    def progress_percent(self) -> int:
        return {"pending": 0, "streaming": 50, "complete": 100, "error": 100, "cancelled": 100}.get(self.status, 0)

    @classmethod
    def from_result(cls, result: MultiModelResult) -> "ModelDisplayState":
        return cls(
            slot=result.slot_name,
            content=result.response.content,
            duration_ms=result.duration_ms,
            tokens=dict(result.tokens),
            status="cancelled" if result.cancelled else ("error" if result.error else "complete"),
            error=result.error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "duration_ms": self.duration_ms,
            "tokens": dict(self.tokens),
            "content": self.content,
            "status": self.status,
            **({"error": self.error} if self.error else {}),
        }


@runtime_checkable
class MultiModelDisplayProtocol(Protocol):
    """Small common surface implemented by interactive displays."""

    phase: DisplayPhase

    def on_progress(self, slot: str, chunk: str, *, status: str = "streaming") -> None: ...

    def on_complete(self, result: MultiModelResult) -> None: ...

    def handle_key(self, key: str) -> str | None: ...
