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

# pylint: disable=cyclic-import

"""Cron integration for auto-dream.

Wires the dreaming subsystem into the downstream cron system as a
**permanent** cron task alongside catch-up and morning check-in.

* :func:`install_dream_permanent_cron_task` — idempotent installer.
  Adds a ``permanent=True`` cron task that fires daily at 3 AM (the
 ``DreamConfig.min_hours=24`` gate is the second line of defence:
 the task may be installed, but ``execute_auto_dream`` won't run
 more than once per 24h).
* :func:`wire_dream_fire_handler` — wraps the scheduler's
  ``on_fire_task`` so the dream task is handled **locally** (calls
  :func:`execute_auto_dream`) instead of being routed to the model
  via the default outbox push. Other tasks fall through to the
  original handler unchanged.
* :func:`install_and_wire_dream` — convenience: install + wire in
  one call. Returns ``(task, created)`` from the install.

Why a *local* fire handler (not a model prompt)?

* Dream is a forked-agent operation — the model should never
 see the consolidation prompt. Routing it via the cron outbox would
 hand a 24h background task to whatever model the user has
 configured, including cheap / non-Anthropic providers, and would
 race the in-flight dream lock with whatever the model decides to
 do.
* Upstream mirrors the same shape: ``runForkedAgent`` is invoked
 from the cron fire path, not the model's prompt stream.

Usage at startup::

 from clawcodex_ext.cron_system.runtime import attach_cron_runtime # pylint: disable=no-name-in-module
 from clawcodex_ext.dreaming import init_auto_dream
 from clawcodex_ext.dreaming.cron_integration import (
 install_and_wire_dream,)

 init_auto_dream(registry=shared_registry) # service init
 scheduler = attach_cron_runtime(ctx, autostart=False)
 task, created = install_and_wire_dream(workspace_root, scheduler)
 scheduler.start # safe to fire tasks now
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


# Fixed well-known task id so the wire handler can match deterministically
# without scanning the prompt text.
DREAM_PERMANENT_TASK_ID: str = "dream"

# Daily at 03:00 — quiet hours, before the morning-checkin (typically 09:00).
DREAM_DEFAULT_CRON: str = "0 3 * * *"

# Marker prompt — never routed to the model (wire_dream_fire_handler
# intercepts before the outbox), but visible in ``cron list`` and the
# outbox payload for debugging.
DREAM_PERMANENT_PROMPT: str = "[auto-dream] memory consolidation (F-100) — local fire handler; not a model prompt"


def install_dream_permanent_cron_task(
    workspace_root: Any,
    *,
    cron_expr: str | None = None,
    prompt: str | None = None,
    task_id: str | None = None,
    jitter: Any = None,
) -> tuple[Any, bool]:
    """Install the dream permanent cron task (idempotent).

    Wraps :func:`clawcodex_ext.cron_system.tasks.write_permanent_task_if_missing`
    with the well-known dream defaults. We call the underlying writer
    directly (not :func:`clawcodex_ext.cron_system.runtime.install_permanent_cron_tasks`)
    because the latter catches and swallows :class:`PermissionError`,
    which we want to propagate so the caller can detect collisions
    with a hand-installed permanent task.

    Args:
        workspace_root: Workspace directory passed through to
            :func:`write_permanent_task_if_missing`.
        cron_expr: 5-field cron expression. Defaults to
            :data:`DREAM_DEFAULT_CRON` (3 AM daily).
        prompt: Prompt label. Defaults to :data:`DREAM_PERMANENT_PROMPT`.
        task_id: Task id. Defaults to :data:`DREAM_PERMANENT_TASK_ID`.
        jitter: Optional :class:`CronJitterConfig`.

    Returns:
        ``(task, created)`` from
        :func:`clawcodex_ext.cron_system.tasks.write_permanent_task_if_missing`.
        ``created=True`` on first install, ``False`` on re-install.

    Raises:
        PermissionError: If an existing permanent task has the same
            cron but a different prompt (or vice versa). The
            :func:`write_permanent_task_if_missing` invariant guards
            against silent overwrites.
    """
    from clawcodex_ext.cron_system.tasks import write_permanent_task_if_missing  # pylint: disable=no-name-in-module

    task, created = write_permanent_task_if_missing(
        workspace_root,
        cron=cron_expr or DREAM_DEFAULT_CRON,
        prompt=prompt or DREAM_PERMANENT_PROMPT,
        recurring=True,
        jitter=jitter,
        task_id=task_id or DREAM_PERMANENT_TASK_ID,
    )
    if created:
        _log.info(
            "dream permanent cron task installed: id=%s cron=%r",
            task.id,
            task.cron,
        )
    else:
        _log.debug("dream permanent cron task already present: id=%s", task.id)
    return task, created


def wire_dream_fire_handler(
    scheduler: Any,
    *,
    task_id: str | None = None,
    registry: Any = None,
) -> None:
    """Wrap the scheduler's ``on_fire_task`` to handle the dream task locally.

    On a fired task with ``task.id == task_id`` (default
    :data:`DREAM_PERMANENT_TASK_ID`), call
    :func:`clawcodex_ext.dreaming.execute_auto_dream` instead of the
    default outbox push. Other tasks fall through to the original
    handler unchanged.

    Args:
        scheduler: The :class:`CronScheduler` instance whose
            ``on_fire_task`` should be wrapped.
        task_id: Override the dream task id to intercept. Defaults to
            :data:`DREAM_PERMANENT_TASK_ID`.
        registry: Optional :class:`RuntimeTaskRegistry` passed to
            :func:`execute_auto_dream`. If omitted, uses the registry
            from the dream service closure (set up via
            :func:`init_auto_dream`) or creates a fresh one.

    **Concurrency**: must be called BEFORE ``scheduler.start()`` —
    the fire path reads ``on_fire_task`` from the scheduler thread
    without a lock, so a mid-fire replacement could race.
    """
    from clawcodex_ext.dreaming import execute_auto_dream

    target_id = task_id or DREAM_PERMANENT_TASK_ID
    original = scheduler.on_fire_task

    def _wrapped(task: Any, run: Any) -> None:
        if getattr(task, "id", None) == target_id:
            _log.debug(
                "dream permanent cron fired (task_id=%s, run_id=%s)",
                task.id,
                getattr(run, "id", "?"),
            )
            # ``run_coroutine_threadsafe``-style: the fire callback runs
            # on the scheduler's tick thread. ``execute_auto_dream`` is
            # an async coroutine; we need to schedule it on a running
            # event loop if one is available, else run it inline.
            coro = execute_auto_dream(force=False, registry=registry)
            try:
                import asyncio

                loop = asyncio.get_running_loop()
                # We're on the scheduler thread; offload to the loop.
                loop.create_task(coro)
            except RuntimeError:
                # No running loop on this thread — run inline.
                asyncio.run(coro)
            return
        if original is not None:
            original(task, run)

    scheduler.on_fire_task = _wrapped
    _log.debug(
        "dream fire handler wired (target_id=%r, has_original=%s)",
        target_id,
        original is not None,
    )


def install_and_wire_dream(
    workspace_root: Any,
    scheduler: Any,
    *,
    cron_expr: str | None = None,
    prompt: str | None = None,
    task_id: str | None = None,
    jitter: Any = None,
    registry: Any = None,
) -> tuple[Any, bool]:
    """Install the dream permanent cron task and wire the fire handler.

    Convenience wrapper for the common startup flow. Call BEFORE
    ``scheduler.start()`` to avoid the wire-after-start race.

    Returns:
        ``(task, created)`` from
        :func:`install_dream_permanent_cron_task`.
    """
    task, created = install_dream_permanent_cron_task(
        workspace_root,
        cron_expr=cron_expr,
        prompt=prompt,
        task_id=task_id,
        jitter=jitter,
    )
    wire_dream_fire_handler(scheduler, task_id=task.id, registry=registry)
    return task, created


__all__ = [
    "DREAM_DEFAULT_CRON",
    "DREAM_PERMANENT_PROMPT",
    "DREAM_PERMANENT_TASK_ID",
    "install_and_wire_dream",
    "install_dream_permanent_cron_task",
    "wire_dream_fire_handler",
]
