#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Auto-dream service main loop — F-100.

Mirrors ``typescript/src/services/autoDream/autoDream.ts``. The gate
order (cheapest first) is preserved:

1. **Enabled gate** — :func:`is_auto_dream_enabled` (or ``force`` for
   manual ``/dream``).
2. **Time gate** — hours since ``lastConsolidatedAt`` ≥ ``min_hours``.
3. **Scan throttle** — only re-scan sessions every
   ``SESSION_SCAN_INTERVAL_MS`` (10min default) to avoid a session
   scan on every tool turn.
4. **Session gate** — session count since last consolidation ≥
   ``min_sessions`` (excludes the current session).
5. **Lock gate** — try to acquire the consolidation lock; bail if
   another process holds it.

Once all gates pass, the service:

* Registers a :class:`DreamTask` on the shared :class:`RuntimeTaskRegistry`
* Stashes the prior mtime on the state for kill-time rollback
* Calls :func:`run_dream_consolidation` with a progress watcher that
  funnels assistant turns into :func:`add_dream_turn`
* On success, marks the task completed
* On failure (not aborted), marks the task failed + rewinds the lock
  mtime so the time gate re-opens promptly

State is closure-scoped inside :func:`init_auto_dream` (not
module-level) so tests can re-initialize for a fresh closure in
``setUp`` / ``beforeEach`` — matching the upstream pattern.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from clawcodex_ext.dreaming.config import (
    DreamConfig,
    get_dream_config,
    is_auto_dream_enabled,
)
from clawcodex_ext.dreaming.lock import (
    force_release_if_stale,
    list_sessions_touched_since,
    read_last_consolidated_at,
    record_consolidation,
    rollback_consolidation_lock,
    try_acquire_consolidation_lock,
)
from clawcodex_ext.dreaming.paths import (
    get_auto_mem_path,
    is_kairos_active,
    project_transcript_dir,
)
from clawcodex_ext.dreaming.prompt import build_consolidation_prompt
from clawcodex_ext.dreaming.runner import (
    DreamRunnerUnavailable,
    run_dream_consolidation,
)

_log = logging.getLogger(__name__)

# Scan throttle: when time-gate passes but session-gate doesn't, the
# lock mtime doesn't advance, so the time-gate keeps passing every
# turn. Mirrors upstream ``autoDream.ts::SESSION_SCAN_INTERVAL_MS``.
SESSION_SCAN_INTERVAL_MS = 10 * 60 * 1000


@dataclass
class _AutoDreamRunner:
    """Closure-scoped runner state. One instance per ``init_auto_dream``
    call so tests can spin up fresh closures cheaply.
    """

    config: DreamConfig
    last_session_scan_at: int = 0
    runner: Optional[Callable[..., Any]] = None  # assigned in init_auto_dream
    # Stash the registry the closure was initialized with so the
    # ``/dream`` skill and ``manual_dream`` can route task registration
    # back to the *same* registry the user wired up — without callers
    # having to thread the registry through every public entry point.
    registry: Any = None


_runner: _AutoDreamRunner | None = None


def init_auto_dream(
    config: DreamConfig | None = None,
    *,
    registry: Any = None,
) -> None:
    """Initialize the auto-dream service.

    Call once at startup, or per-test in ``beforeEach`` for a fresh
    closure. Subsequent calls replace the prior closure (the previous
    one is dropped — there is no overlap).

    Args:
        config: Override the active :class:`DreamConfig` (defaults to
            :func:`get_dream_config`).
        registry: The shared :class:`RuntimeTaskRegistry`. Required
            for task registration. If omitted, a fresh one is created
            per process so unit tests can run without bootstrapping
            the full app.
    """
    global _runner
    cfg = config or get_dream_config()

    if registry is None:
        from clawcodex_ext.task_registry import RuntimeTaskRegistry  # pylint: disable=no-name-in-module

        registry = RuntimeTaskRegistry()

    _runner = _AutoDreamRunner(config=cfg, runner=None, registry=registry)

    async def run(
        context: Any = None,
        current_session_id: str | None = None,
        *,
        force: bool = False,
    ) -> None:
        """One auto-dream pass. Idempotent — no-op when gates are closed."""
        await _execute(
            cfg=cfg,
            context=context,
            current_session_id=current_session_id,
            force=force,
            registry=registry,
        )

    _runner.runner = run


def get_active_registry() -> Any:
    """Return the registry the dream service was initialized with.

    Falls back to a fresh :class:`RuntimeTaskRegistry` when the
    service has not been initialized — useful for read-only paths
    like the ``/dream status`` skill, which should still work even
    if no one has called :func:`init_auto_dream` yet.
    """
    if _runner is not None and getattr(_runner, "registry", None) is not None:
        return _runner.registry
    from clawcodex_ext.task_registry import RuntimeTaskRegistry  # pylint: disable=no-name-in-module

    return RuntimeTaskRegistry()


# ---------------------------------------------------------------------------
# Gate + dispatch (testable entry point)
# ---------------------------------------------------------------------------


async def _execute(
    *,
    cfg: DreamConfig,
    context: Any,
    current_session_id: str | None,
    force: bool,
    registry: Any,
) -> None:
    """One auto-dream pass. Mirrors the gate chain from
    ``autoDream.ts::runner``.
    """
    # The task subsystem is migrated separately. Import it only when a
    # dream run executes so this module remains importable without the
    # removed legacy ``src`` namespace.
    from clawcodex_ext.tasks.dream.dream_task import (  # pylint: disable=no-name-in-module
        add_dream_turn,
        complete_dream_task,
        fail_dream_task,
        register_dream_task,
    )

    # --- Enabled gate ---
    if not force:
        if is_kairos_active():
            _log.debug("auto-dream skipped: KAIROS active (disk-skill path)")
            return
        if not is_auto_dream_enabled():
            _log.debug("auto-dream skipped: disabled by config / env")
            return

    # --- Time gate ---
    try:
        last_at = read_last_consolidated_at()
    except Exception as e:
        _log.debug("auto-dream: readLastConsolidatedAt failed: %s", e)
        return
    hours_since = (time.time() * 1000 - last_at) / 3_600_000
    if not force and hours_since < cfg.min_hours:
        _log.debug(
            "auto-dream: time gate closed (%.1fh < %.1fh)",
            hours_since,
            cfg.min_hours,
        )
        return

    # --- Scan throttle ---
    if _runner is not None:
        now_ms = int(time.time() * 1000)
        since_scan_ms = now_ms - _runner.last_session_scan_at
        if not force and since_scan_ms < SESSION_SCAN_INTERVAL_MS:
            _log.debug(
                "auto-dream: scan throttle (last scan %ds ago)",
                since_scan_ms // 1000,
            )
            return
        _runner.last_session_scan_at = now_ms

    # --- Session gate ---
    try:
        session_ids = list_sessions_touched_since(last_at)
    except Exception as e:
        _log.debug("auto-dream: listSessionsTouchedSince failed: %s", e)
        return
    if current_session_id is not None:
        session_ids = [s for s in session_ids if s != current_session_id]
    if not force and len(session_ids) < cfg.min_sessions:
        _log.debug(
            "auto-dream: session gate closed (%d < %d)",
            len(session_ids),
            cfg.min_sessions,
        )
        return

    # --- Lock gate (Phase B: TTL-aware) ---
    # Active stale-lock sweep *before* the in-band PID check: even
    # if the gate chain above blocks this run (e.g. session gate
    # closed), kicking a stale lock now keeps the next process
    # unblocked.
    try:
        force_release_if_stale()
    except Exception as e:
        _log.debug("auto-dream: force_release_if_stale failed: %s", e)
    try:
        prior_mtime = try_acquire_consolidation_lock()
    except Exception as e:
        _log.debug("auto-dream: lock acquire failed: %s", e)
        return
    if prior_mtime is None:
        _log.debug("auto-dream: lock held by another process")
        return

    _log.debug(
        "auto-dream firing — %.1fh since last, %d sessions to review",
        hours_since,
        len(session_ids),
    )

    # --- Register task + run ---
    task_id = register_dream_task(
        sessions_reviewing=len(session_ids),
        prior_mtime=prior_mtime,
        registry=registry,
    )

    def _on_message(
        *,
        text: str,
        tool_use_count: int,
        touched_paths: list[str],
    ) -> None:
        add_dream_turn(
            task_id,
            text=text,
            tool_use_count=tool_use_count,
            touched_paths=touched_paths,
            registry=registry,
        )

    try:
        memory_root = get_auto_mem_path()
        transcript_dir = project_transcript_dir()
        tool_constraints = (
            "\n\n**Tool constraints for this run:** Bash is restricted "
            "to read-only commands (`ls`, `find`, `grep`, `cat`, `stat`, "
            "`wc`, `head`, `tail`, and similar). Anything that writes, "
            "redirects to a file, or modifies state will be denied. Plan "
            "your exploration with this in mind — no need to probe.\n\n"
            f"Sessions since last consolidation ({len(session_ids)}):\n" + "\n".join(f"- {sid}" for sid in session_ids)
        )
        prompt = build_consolidation_prompt(
            memory_root=memory_root,
            transcript_dir=transcript_dir,
            extra=tool_constraints,
        )
        result = run_dream_consolidation(prompt, on_message=_on_message)
        complete_dream_task(task_id, registry)
        if result.summary:
            _log.info("auto-dream completed: %s", result.summary)
        else:
            _log.info(
                "auto-dream completed — files_touched=%d",
                len(result.files_touched),
            )
    except DreamRunnerUnavailable as e:
        if _is_aborted(task_id, registry):
            _log.debug("auto-dream aborted by user (no rewind)")
            return
        _log.warning("auto-dream runner unavailable: %s", e)
        fail_dream_task(task_id, registry)
        rollback_consolidation_lock(prior_mtime)
    except Exception as e:
        if _is_aborted(task_id, registry):
            _log.debug("auto-dream aborted by user (no rewind)")
            return
        _log.warning("auto-dream fork failed: %s", e)
        fail_dream_task(task_id, registry)
        rollback_consolidation_lock(prior_mtime)


def _is_aborted(task_id: str, registry: Any) -> bool:
    """Best-effort: was the task killed (status=='killed')?

    Used to suppress the post-failure rollback in the catch block —
    :func:`rollback_dream_lock_after_kill` already rewind the lock
    when the user kills the task, so we must not double-rewind.
    """
    try:
        state = registry.get(task_id)
    except Exception:
        return False
    if state is None:
        return False
    try:
        return state.status == "killed"
    except AttributeError:
        return False


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def execute_auto_dream(
    context: Any = None,
    current_session_id: str | None = None,
    *,
    force: bool = False,
    registry: Any = None,
) -> None:
    """Entry point from stopHooks / cron.

    No-op until :func:`init_auto_dream` has been called. Per-turn
    cost when enabled: one stat (lock mtime) + one scan if the time
    gate opened.

    If *registry* is omitted, falls back to the registry the service
    was initialized with (so :func:`manual_dream` and other callers
    can fire-and-forget without threading the registry through).
    """
    if _runner is None or _runner.runner is None:
        _log.debug("execute_auto_dream: not initialized, no-op")
        return
    if registry is None:
        registry = _runner.registry
        if registry is None:
            from clawcodex_ext.task_registry import RuntimeTaskRegistry  # pylint: disable=no-name-in-module

            registry = RuntimeTaskRegistry()
    await _runner.runner(
        context,
        current_session_id,
        force=force,
    )


def kill_dream_task(task_id: str, registry: Any) -> None:
    """External kill path used by the ``stop_task`` tool and the CLI.

    Two steps:

    1. Flip the task to ``killed`` and capture the prior mtime via
       :func:`rollback_dream_lock_after_kill` (atomic under the
       registry lock).
    2. Rewind the lock mtime so the next session can retry —
       same path as the fork-failure catch.
    """
    from clawcodex_ext.tasks.dream.dream_task import rollback_dream_lock_after_kill  # pylint: disable=no-name-in-module

    prior_mtime = rollback_dream_lock_after_kill(task_id, registry)
    if prior_mtime is not None:
        rollback_consolidation_lock(prior_mtime)


# ---------------------------------------------------------------------------
# Manual dream entry (used by the /dream skill in Phase C)
# ---------------------------------------------------------------------------


def manual_dream(context: Any = None, current_session_id: str | None = None) -> None:
    """Trigger a forced dream run (used by the ``/dream`` slash skill).

    Calls :func:`execute_auto_dream` with ``force=True``. Stamps the
    lock file optimistically (matching upstream ``recordConsolidation``
    in the manual /dream path) so a re-entrant /dream from another
    session doesn't double-fire while the prompt is in flight.
    """
    record_consolidation()
    # Async entry — fire and forget on the default loop. The runner is
    # responsible for any error handling.
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is None or not loop.is_running():
        asyncio.run(
            execute_auto_dream(
                context=context,
                current_session_id=current_session_id,
                force=True,
            )
        )
        return
    loop.create_task(
        execute_auto_dream(
            context=context,
            current_session_id=current_session_id,
            force=True,
        )
    )
