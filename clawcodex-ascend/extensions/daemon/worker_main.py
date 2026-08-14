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
"""Worker subprocess entry point.

Run with::

    python -m extensions.daemon.worker_main <kind>

The supervisor spawns workers through this module so the worker
process is a *clean* Python subprocess — no CLI dispatch overhead,
no TUI imports, no argparser. The worker reads its configuration
from the inherited environment (the supervisor injects
``CLAWCODEX_DAEMON_*`` vars).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Sequence

from extensions.daemon.errors import PermanentWorkerError, UnknownWorkerKindError
from extensions.daemon.worker_registry import WorkerRegistry

# Eager registration: importing this module should make every
# built-in worker factory discoverable.
import extensions.daemon.workers  # noqa: F401  (registers on import)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


async def _run_worker(kind: str) -> int:
    try:
        worker = WorkerRegistry.create(kind)
    except UnknownWorkerKindError:
        print(
            f"worker_main: unknown worker kind {kind!r}; known: {WorkerRegistry.known_kinds()}",
            file=sys.stderr,
        )
        return 78

    env = dict(__import__("os").environ)
    try:
        return await worker.run(env)
    except PermanentWorkerError as exc:
        logging.error("worker_main: permanent error: %s", exc)
        return 78
    except asyncio.CancelledError:
        return 0
    except Exception:
        logging.exception("worker_main: unhandled exception")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv
    if len(argv) < 2 or not argv[1]:
        print(
            "usage: python -m extensions.daemon.worker_main <kind>",
            file=sys.stderr,
        )
        return 78
    _configure_logging()
    kind = argv[1]
    return asyncio.run(_run_worker(kind))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
