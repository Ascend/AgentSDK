#!/usr/bin/env python3
# coding=utf-8

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


import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

DEFAULT_RELEASE_TIMEOUT = 5.0

STATUS_RELEASED = "released"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class ResourceEntry:
    """One resource registered in the ledger. """

    kind: str
    name: str
    disposer: Optional[Callable[[], None]] = None
    timeout: float = DEFAULT_RELEASE_TIMEOUT


class RolloutResourceLedger:
    """Registration-order, per-item isolated, idempotent resource ledger."""

    def __init__(self, log_enabled=False):
        """Initialize the ledger. """
        self._entries: List[ResourceEntry] = []
        self._released_ids = set()
        self._log_enabled = bool(log_enabled)

    def acquire(self, entry):
        """Register a resource right after it was created. """
        self._entries.append(entry)

    def release_all(self, reason="shutdown"):
        """Release every registered resource in registration order."""
        owned = 0
        released = 0
        failed = 0
        items = []
        for entry in self._entries:
            if entry.disposer is None or id(entry) in self._released_ids:
                items.append((entry.kind, entry.name, STATUS_SKIPPED))
                self._log_item(entry, STATUS_SKIPPED, 0.0, None)
                continue
            owned += 1
            status, elapsed, error = self._release_one(entry)
            if status == STATUS_RELEASED:
                released += 1
                self._released_ids.add(id(entry))
            else:
                failed += 1
            items.append((entry.kind, entry.name, status))
            self._log_item(entry, status, elapsed, error)
        return {"owned": owned, "released": released, "failed": failed, "items": items}

    def _release_one(self, entry):
        """Run one disposer with a per-item timeout. """
        errors = []

        def _run():
            try:
                entry.disposer()
            except BaseException as exp:
                errors.append(exp)

        started = time.time()
        thread = threading.Thread(
            target=_run,
            name=f"release-{entry.kind}-{entry.name}",
            daemon=True,
        )
        thread.start()
        thread.join(max(0.0, float(entry.timeout)))
        elapsed = time.time() - started
        if thread.is_alive():
            return STATUS_FAILED, elapsed, f"timeout>{float(entry.timeout)}s"
        if errors:
            return STATUS_FAILED, elapsed, type(errors[0]).__name__
        return STATUS_RELEASED, elapsed, None

    def _log_item(self, entry, status, elapsed, error):
        """Emit one ``manager.release.item`` line (``failed`` as WARNING)."""
        if not self._log_enabled:
            return
        message = (
            f"[AURA_ROLLOUT] manager.release.item kind={entry.kind} name={entry.name} "
            f"status={status} elapsed={elapsed:.3f}"
        )
        if status == STATUS_FAILED:
            logger.warning(f"{message} error={error}")
        else:
            logger.info(message)
