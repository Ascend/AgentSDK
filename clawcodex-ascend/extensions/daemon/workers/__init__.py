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
"""Built-in daemon workers.

Importing this package **eagerly** registers the built-in worker
factories with :class:`extensions.daemon.worker_registry.WorkerRegistry`.

Callers that want a clean registry (e.g. tests that override the
``remoteControl`` factory) should call ``WorkerRegistry.reset()``
**before** importing this package.

Worker kinds
------------
* ``remoteControl`` — remote task executor listening on a Unix socket (legacy alias)
* ``task_server``   — remote task executor (same implementation as ``remoteControl``)
* ``cron``          — scheduled task dispatcher
"""

from __future__ import annotations

import logging

from extensions.daemon.worker_registry import WorkerRegistry

from extensions.daemon.workers.base import BaseWorker
from extensions.daemon.workers.cron import CronWorker, build_cron_worker
from extensions.daemon.workers.remote_control import RemoteControlWorker, build_remote_control_worker
from extensions.daemon.workers.task_worker import TaskServerWorker, build_task_server_worker

logger = logging.getLogger(__name__)

# Register built-in workers. Re-registration is allowed; this is the
# canonical place to declare default factories.
WorkerRegistry.register("remoteControl", build_remote_control_worker)
WorkerRegistry.register("task_server", build_task_server_worker)
WorkerRegistry.register("cron", build_cron_worker)

__all__ = [
    "BaseWorker",
    "CronWorker",
    "RemoteControlWorker",
    "TaskServerWorker",
    "build_cron_worker",
    "build_remote_control_worker",
    "build_task_server_worker",
]
