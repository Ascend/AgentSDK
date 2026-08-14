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
"""``cron`` daemon worker (placeholder).

The cron Worker is registered so the supervisor can spawn it, but the
concrete ``run()`` is intentionally a no-op until a full implementation
lands. Returning :data:`EXIT_CODE_PERMANENT` would be over-aggressive —
instead we sleep until cancelled, so the supervisor sees a healthy
long-running worker that it can restart cleanly.

The real implementation will be a follow-up patch in
``extensions/cron_system/``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from extensions.daemon.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class CronWorker(BaseWorker):
    """Stub cron worker — sleeps until cancelled."""

    kind = "cron"

    def __init__(self) -> None:
        super().__init__()
        self._cancel: asyncio.Event | None = None

    async def run(self, env: dict[str, str]) -> int:
        # Mirror the env read so misconfiguration is loud early.
        cfg = self.read_daemon_env(env)
        logger.info(
            "cron worker (stub) starting: name=%s dir=%s capacity=%s",
            cfg["name"],
            cfg["dir"],
            cfg["capacity"],
        )
        cancel = asyncio.Event()
        self._cancel = cancel
        try:
            await cancel.wait()
        except asyncio.CancelledError:
            return 0
        return 0

    def health_check(self) -> dict[str, Any] | None:
        snap = super().health_check() or {}
        snap["stub"] = True
        return snap


def build_cron_worker() -> CronWorker:
    return CronWorker()


__all__ = ["CronWorker", "build_cron_worker"]
