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
"""Custom exceptions raised by the daemon subsystem."""

from __future__ import annotations


class DaemonError(Exception):
    """Base class for all daemon errors; catching it covers every subclass."""


class UnknownWorkerKindError(DaemonError, KeyError):
    """The requested worker kind is not registered."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"unknown worker kind: {kind!r}")
        self.kind = kind


class PermanentWorkerError(DaemonError):
    """Signal from a worker that it should not be restarted."""

    def __init__(self, message: str = "permanent worker failure") -> None:
        super().__init__(message)


class WorkerSpawnError(DaemonError):
    """The supervisor failed to spawn a worker subprocess."""


class DaemonAlreadyRunningError(DaemonError):
    """``daemon start`` refused because another instance owns the state file."""


class DaemonNotRunningError(DaemonError):
    """``daemon stop`` / ``daemon status`` called but no live daemon found."""


class InvalidDaemonConfigError(DaemonError, ValueError):
    """``DaemonConfig`` failed validation; also a :class:`ValueError`."""
