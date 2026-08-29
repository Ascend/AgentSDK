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


@runtime_checkable
class Worker(Protocol):
    """Subprocess-side worker entry point.

    A worker is instantiated inside the worker subprocess launched by
    the supervisor, which passes a prepared environment (with
    ``CLAWCODEX_DAEMON_*`` variables set); ``run`` drives the work
    until it exits.

    Exit semantics: ``0`` normal completion; 78 permanent failure
    (parked, no restarts); anything else transient failure (restart).
    """

    #: Logical worker kind — must match the registration key under
    #: :class:`extensions.daemon.worker_registry.WorkerRegistry`.
    kind: str

    async def run(self, env: dict[str, str]) -> int:
        """Worker main loop.

        Args:
            env: Merged environment for the worker subprocess.

        Returns:
            Exit code: ``0`` success, ``78`` permanent failure,
            anything else transient failure.
        """
        ...

    def health_check(self) -> dict[str, Any] | None:
        """Optional health snapshot (PID, uptime, last error...) or ``None``."""
        ...


__all__ = ["Worker"]
