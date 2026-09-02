#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Telemetry shutdown-flush lifecycle hooks.

Registers a :func:`src.utils.graceful_shutdown.register_cleanup` callback
that flushes the telemetry recorder (aggregate + reporter emit) on process
exit — via SIGINT, SIGTERM, or normal ``atexit``.

This is a best-effort, fire-and-forget hook. If telemetry is disabled or
the recorder has not been initialized, the cleanup is a no-op.

The flush is run in a short-lived daemon thread so a slow HTTP request
to the remote Issue tracker never blocks the process exit. If the thread
does not finish before the interpreter shuts down, it is silently killed.

Usage
-----
Called lazily from ``src/init.py:init()``, after the exception hooks and
the nested-transcript resolver have been installed::

    from clawcodex_ext.telemetry_lifecycle import install_telemetry_shutdown_flush
    install_telemetry_shutdown_flush()

Idempotent — calling it more than once is safe (second call is a no-op).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_INSTALLED = False


def _telemetry_shutdown_flush() -> None:
    """Best-effort flush of the telemetry recorder on process exit.

    Only emits to reporters when today's summary contains crashes or
    errors — normal clean exits do NOT push an Issue.

    Runs in a daemon thread so the HTTP request never blocks exit.
    Swallows all exceptions so a misconfigured or broken telemetry
    subsystem never blocks the shutdown drain.
    """

    def _do_flush() -> None:
        try:
            from telemetry.recorder import get_recorder
            from telemetry.storage import LocalJsonlStorage, utc_date, utc_now
            from telemetry.aggregator import DailyAggregator

            recorder = get_recorder()
            if not getattr(recorder, "enabled", False):
                return
            if not recorder.config.reporting.reporting_enabled:
                return

            date = utc_date(utc_now())
            storage = LocalJsonlStorage(
                recorder.config.storage_dir,
                recorder.config.retention_days,
            )
            summary = DailyAggregator(storage).aggregate(date)
            if not summary:
                return

            # When auto_push_errors_only is False, flush every time (stats + errors).
            # When True (default), only flush when today's summary has crashes.
            auto_push_errors_only = getattr(recorder.config.reporting, "auto_push_errors_only", True)
            if auto_push_errors_only:
                crashes = summary.get("crashes", {}) or {}
                if crashes.get("total", 0) == 0:
                    return

            recorder.flush()
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug("telemetry: shutdown flush failed: %s", exc)

    t = threading.Thread(target=_do_flush, name="telemetry-shutdown-flush", daemon=False)
    t.start()
    t.join(timeout=20)


def install_telemetry_shutdown_flush() -> None:
    """Register the telemetry flush as a graceful-shutdown cleanup.

    Idempotent — the second call is a no-op. Safe to call from
    ``src/init.py:init()``, where ``src.utils.graceful_shutdown`` is
    already set up.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from src.utils.graceful_shutdown import register_cleanup

        register_cleanup(_telemetry_shutdown_flush)
        _INSTALLED = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry: failed to register shutdown flush: %s", exc)


def uninstall_telemetry_shutdown_flush() -> None:
    """Reset the installed flag. Primarily for tests.

    Does **not** unregister the cleanup from the graceful-shutdown
    registry (that module does not expose a ``remove`` API). The
    cleanup itself is inherently idempotent and safe to call even
    after ``uninstall``, so this is acceptable for test teardown.
    """
    global _INSTALLED
    _INSTALLED = False


__all__ = [
    "install_telemetry_shutdown_flush",
    "uninstall_telemetry_shutdown_flush",
]
