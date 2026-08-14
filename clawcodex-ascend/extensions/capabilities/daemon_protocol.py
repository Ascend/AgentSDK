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
"""Daemon subsystem Protocol — Worker contract.

The ``Worker`` Protocol defines the surface the supervisor uses to
interact with a daemon worker. Concrete implementations live in
``extensions/daemon/workers/`` and are registered through
``extensions.daemon.worker_registry.WorkerRegistry``.

Design notes: ``@runtime_checkable`` (``isinstance`` checks without a
base class); plain ``dict[str, str]`` env; ``health_check()`` optional.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

#: Exit code returned by :meth:`Worker.run` on permanent failure (parked,
#: supervisor must not restart the worker).
EXIT_PARKED: int = 78


@runtime_checkable
class Worker(Protocol):
    """Subprocess-side worker entry point.

    A worker is instantiated inside the worker subprocess launched by
    the supervisor, which passes a prepared environment (with
    ``CLAWCODEX_DAEMON_*`` variables set); ``run`` drives the work
    until it exits.

    Exit semantics: ``0`` normal completion; ``EXIT_PARKED`` (78) permanent
    failure (parked, no restarts); anything else transient failure (restart).
    """

    #: Logical worker kind — must match the registration key under
    #: :class:`extensions.daemon.worker_registry.WorkerRegistry`.
    kind: str

    async def run(self, env: dict[str, str]) -> int:
        """Worker main loop.

        Args:
            env: Merged environment for the worker subprocess.

        Returns:
            Exit code: ``0`` success, ``EXIT_PARKED`` (78) permanent
            failure, anything else transient failure.
        """

    def health_check(self) -> dict[str, Any] | None:
        """Optional health snapshot (PID, uptime, last error...) or ``None``."""


__all__ = ["EXIT_PARKED", "Worker"]
