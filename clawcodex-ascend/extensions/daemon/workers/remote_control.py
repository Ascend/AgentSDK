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
"""``remoteControl`` daemon worker.

A thin wrapper around :class:`TaskServerWorker` — listens on a local
Unix Domain Socket, accepts JSON-lines task requests, and runs agent
tasks in subprocesses. No longer depends on the Anthropic Cloud bridge.

Back-compat
-----------
The ``remoteControl`` kind in :class:`WorkerRegistry` is unchanged, so
existing ``--workers remoteControl`` invocations keep working. The new
``task_server`` kind is an identical implementation for new deploys.

Migration path
--------------
``remoteControl`` will be deprecated in a future release; use
``task_server`` instead.
"""

from __future__ import annotations

import logging

from extensions.daemon.workers.task_worker import TaskServerWorker

logger = logging.getLogger(__name__)


class RemoteControlWorker(TaskServerWorker):
    """``remoteControl`` worker (a ``TaskServerWorker`` alias).

    Keeps ``kind = "remoteControl"`` for backward compatibility.
    """

    kind = "remoteControl"


def build_remote_control_worker() -> RemoteControlWorker:
    """Factory for ``WorkerRegistry.register("remoteControl", ...)``."""
    return RemoteControlWorker()


__all__ = ["RemoteControlWorker", "build_remote_control_worker"]
