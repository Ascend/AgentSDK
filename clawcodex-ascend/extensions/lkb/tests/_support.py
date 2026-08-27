#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Small deterministic helpers shared by the current Plan Graph tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

_CLOCK_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


class DeterministicClock:
    def __init__(self) -> None:
        self._seconds = 0

    def now(self) -> str:
        value = self.peek()
        self._seconds += 1
        return value

    def advance(self, seconds: int) -> None:
        self._seconds += max(0, seconds)

    def peek(self) -> str:
        return (
            (_CLOCK_EPOCH + timedelta(seconds=self._seconds)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )

    __call__ = now


class DeterministicIdFactory:
    def __init__(self, width: int = 3) -> None:
        self._width = width
        self._counters: dict[str, int] = {}

    def __call__(self, prefix: str = "T") -> str:
        value = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = value
        return f"{prefix}-{value:0{self._width}d}"

    def reset(self, prefix: str | None = None) -> None:
        if prefix is None:
            self._counters.clear()
        else:
            self._counters.pop(prefix, None)


class Failpoint:
    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}

    def register(self, name: str, handler: Any) -> None:
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        self._handlers.pop(name, None)

    def clear(self) -> None:
        self._handlers.clear()

    def hit(self, name: str) -> None:
        handler = self._handlers.get(name)
        if isinstance(handler, type) and issubclass(handler, BaseException):
            raise handler()
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            handler(name)

    def __contains__(self, name: str) -> bool:
        return name in self._handlers


__all__ = ["DeterministicClock", "DeterministicIdFactory", "Failpoint"]
