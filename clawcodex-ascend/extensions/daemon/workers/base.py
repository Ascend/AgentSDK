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
"""Common base class for daemon workers.

Workers are deliberately thin — they only need to expose ``kind``
and a coroutine ``run(env)`` (plus the optional ``health_check``).
The :class:`BaseWorker` here provides:

* a stable ``kind`` slot,
* a ``started_at`` timestamp for ``health_check`` snapshots,
* a helper for reading the supervisor-provided env vars so worker
  implementations don't have to remember the ``CLAWCODEX_DAEMON_*``
  variable names.

It is **not** a :class:`extensions.capabilities.daemon_protocol.Worker`
on its own — concrete workers subclass it and inherit the ``kind``
slot that the Protocol requires.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

from extensions.daemon.constants import (
    ENV_VAR_DAEMON_CAPACITY,
    ENV_VAR_DAEMON_DIR,
    ENV_VAR_DAEMON_NAME,
    ENV_VAR_DAEMON_PERMISSION_MODE,
    ENV_VAR_DAEMON_SANDBOX,
    ENV_VAR_DAEMON_SPAWN_MODE,
    ENV_VAR_DAEMON_TIMEOUT_MS,
)


class BaseWorker:
    """Common scaffolding for daemon worker implementations.

    Subclasses must set :attr:`kind` and implement :meth:`run`. They
    may override :meth:`health_check` if they have useful diagnostics
    to surface to the supervisor or RCS dashboard.
    """

    #: Subclasses MUST override.
    kind: str = ""

    def __init__(self) -> None:
        # Wall-clock timestamp recorded at construction; ``health_check``
        # uses this to compute uptime.
        self._started_at: float = time.time()

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    async def run(self, env: dict[str, str]) -> int:  # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__}.run not implemented")

    def health_check(self) -> dict[str, Any] | None:
        """Default health snapshot.

        Returns ``{"kind": <kind>, "uptime_s": <float>}``. Subclasses
        are free to override with richer details.
        """
        if not self.kind:
            return None
        return {
            "kind": self.kind,
            "uptime_s": round(time.time() - self._started_at, 3),
        }

    # ------------------------------------------------------------------
    # Environment helpers — kept as @classmethods so subclassers can
    # call them before super().__init__ if they need to.
    # ------------------------------------------------------------------

    @classmethod
    def read_daemon_env(cls, env: Mapping[str, str]) -> dict[str, Any]:
        """Read the canonical ``CLAWCODEX_DAEMON_*`` env vars into a dict.

        Strings are coerced to their typed values (``int`` for capacity,
        ``bool`` for sandbox, ``Path`` for the working directory).
        """
        try:
            capacity = int(env.get(ENV_VAR_DAEMON_CAPACITY, "4"))
        except ValueError:
            capacity = 4
        sandbox_raw = env.get(ENV_VAR_DAEMON_SANDBOX, "0")
        sandbox = sandbox_raw.strip().lower() in ("1", "true", "yes", "on")
        try:
            timeout_ms = int(env.get(ENV_VAR_DAEMON_TIMEOUT_MS, "30000"))
        except ValueError:
            timeout_ms = 30_000

        dir_ = Path(env.get(ENV_VAR_DAEMON_DIR) or os.getcwd())

        return {
            "name": env.get(ENV_VAR_DAEMON_NAME, ""),
            "dir": dir_,
            "spawn_mode": env.get(ENV_VAR_DAEMON_SPAWN_MODE, "same-dir"),
            "capacity": capacity,
            "permission_mode": env.get(ENV_VAR_DAEMON_PERMISSION_MODE) or None,
            "sandbox": sandbox,
            "timeout_ms": timeout_ms,
        }


__all__ = ["BaseWorker"]
