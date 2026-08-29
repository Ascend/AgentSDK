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
#

"""Remote task execution protocol — Layer 2 Protocol definitions.

Interface contract between daemon workers and external task dispatchers,
agnostic to any specific cloud provider or message queue. Implementations
live in ``extensions/daemon/workers/`` (worker side) and ``clawcodex_ext/``
(client side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# Data models


@dataclass(frozen=True)
class TaskRequest:
    """Task request from an external dispatcher; ``id`` is dispatcher-assigned."""

    id: str
    command: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """Task execution result."""

    task_id: str
    status: str
    output: str = ""
    error: str = ""
    exit_code: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# Transport protocols


@runtime_checkable
class TaskTransportServer(Protocol):
    """Worker-side task transport: listens for requests, sends results."""

    async def serve(self, cancel_event: Any | None = None) -> None:
        """Start the transport server; ``cancel_event`` triggers graceful shutdown."""
        ...

    async def shutdown(self) -> None:
        """Stop the transport server and release resources."""
        ...


@runtime_checkable
class TaskTransportClient(Protocol):
    """Dispatcher-side transport client: connects, sends, receives."""

    async def connect(self) -> None: ...

    async def send_request(self, request: TaskRequest) -> TaskResult:
        """Send a request and await the result."""
        ...

    async def close(self) -> None: ...


# Execution layer


@runtime_checkable
class TaskExecutor(Protocol):
    """Task executor — takes a TaskRequest, produces a TaskResult."""

    async def execute(self, request: TaskRequest) -> TaskResult:
        """Execute one task."""
        ...

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task; ``True`` if cancelled, ``False`` if unknown."""
        ...


# Worker contract


@runtime_checkable
class RemoteTaskWorker(Protocol):
    """Overall contract for a remote task execution worker."""

    kind: str

    async def run(self, env: dict[str, str]) -> int:
        """Run the worker main loop.

        Exit codes: ``0`` normal, ``78`` permanent error, other = transient error.
        """
        ...

    def health_check(self) -> dict[str, Any] | None:
        """Optional health snapshot or ``None``."""
        ...


__all__ = [
    "RemoteTaskWorker",
    "TaskExecutor",
    "TaskRequest",
    "TaskResult",
    "TaskTransportClient",
    "TaskTransportServer",
]
