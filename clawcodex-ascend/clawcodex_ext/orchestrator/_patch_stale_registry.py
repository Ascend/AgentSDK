#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""Stale-registry auto-reload patch.

Why
---
``Orchestrator.__init__`` instantiates ``IssueRegistry`` once at startup
(``extensions/orchestrator/orchestrator.py:121``). The registry then serves
``self._registry.get(...)`` lookups throughout the daemon's lifetime.
However, operator actions executed via a separate CLI process —
``clawcodex-dev orchestrator issue retry --mode reset``, for example — write
``intent=retry`` / ``intent_source=cli`` / reset status directly to the
on-disk JSON, and the daemon's in-memory copy is never refreshed.

Symptoms
--------
- ``_resolve_intent`` reads stale ``intent=NONE`` from memory and merges to
  ``Intent.NONE``.
- ``_poll_and_dispatch`` then hits the ``is_terminal() / has_pr()`` skip
  branch (``orchestrator.py:594-599``) and logs
  ``Issue N already handled (registry), skipping`` indefinitely.
- The only fix today is a manual daemon restart, which defeats the purpose
  of a self-driving orchestrator.

What this patch does
--------------------
- Tracks ``self._path.stat().st_mtime_ns`` on ``IssueRegistry`` after every
  successful ``_load()`` and ``_save()``.
- Exposes ``IssueRegistry.reload_if_stale() -> bool`` which re-reads the
  file only when its mtime has advanced past the cached value.
- Wraps ``Orchestrator._poll_and_dispatch`` and
  ``Orchestrator._resolve_intent`` so each consults ``reload_if_stale()``
  before reading from the in-memory record map.

Cost
----
A single ``stat()`` on a <10 KB JSON file is microseconds; two of them per
60 s poll cycle are negligible.

Idempotency
-----------
Guarded by a module-level ``_INSTALLED`` flag, safe to call from multiple
entry points.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED: bool = False


def install_stale_registry_patch() -> None:
    """Monkey-patch ``IssueRegistry`` and ``Orchestrator`` for mtime reload.

    Safe to call multiple times; subsequent calls are no-ops.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from extensions.orchestrator import issue_registry as _ir_mod
    from extensions.orchestrator import orchestrator as _orch_mod

    _Registry = _ir_mod.IssueRegistry

    if not getattr(_Registry, "_stale_registry_patched", False):
        _patch_issue_registry(_Registry)
        _Registry._stale_registry_patched = True  # type: ignore[attr-defined]

    _Orchestrator = _orch_mod.Orchestrator

    if not getattr(_Orchestrator, "_stale_registry_patched", False):
        _patch_orchestrator(_Orchestrator)
        _Orchestrator._stale_registry_patched = True  # type: ignore[attr-defined]

    logger.debug("stale-registry reload patch installed")


# ---------------------------------------------------------------------------
# IssueRegistry patches
# ---------------------------------------------------------------------------


def _patch_issue_registry(_Registry: Any) -> None:
    """Add mtime tracking + ``reload_if_stale`` to ``IssueRegistry``."""

    _orig_init = _Registry.__init__
    _orig_load = _Registry._load
    _orig_save = _Registry._save

    def __init__(self: Any, storage_path: Path, **kwargs: Any) -> None:
        # Keep the upstream IssueRegistry initializer as the source of truth.
        # This patch only needs _mtime_ns to exist before _orig_init calls
        # self._load(), which is replaced below and updates the mtime cache.
        self._mtime_ns: int = 0
        _orig_init(self, Path(storage_path), **kwargs)

    def _load(self: Any) -> None:
        _orig_load(self)
        # Capture mtime after a successful load so subsequent
        # ``reload_if_stale`` calls compare against the right baseline.
        try:
            self._mtime_ns = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            self._mtime_ns = 0

    def _save(self: Any) -> None:
        _orig_save(self)
        # The daemon's own writes must NOT look like external writes on the
        # next poll, otherwise the next ``reload_if_stale`` would
        # unnecessarily re-parse the same file we just wrote.
        try:
            self._mtime_ns = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            self._mtime_ns = 0

    def reload_if_stale(self: Any) -> bool:
        """Reload the registry if the on-disk file is newer than memory.

        Returns ``True`` when a reload happened, ``False`` otherwise.
        """
        try:
            current_mtime_ns = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            return False
        if current_mtime_ns <= self._mtime_ns:
            return False
        previous_mtime_ns = self._mtime_ns
        self._load()
        logger.info(
            "IssueRegistry: external write detected (mtime_ns %d → %d); reloaded",
            previous_mtime_ns,
            current_mtime_ns,
        )
        return True

    _Registry.__init__ = __init__  # type: ignore[method-assign]
    _Registry._load = _load  # type: ignore[method-assign]
    _Registry._save = _save  # type: ignore[method-assign]
    _Registry.reload_if_stale = reload_if_stale  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Orchestrator patches
# ---------------------------------------------------------------------------


def _patch_orchestrator(_Orchestrator: Any) -> None:
    """Wrap poll + intent-resolve methods to reload the registry first."""

    _orig_poll = _Orchestrator._poll_and_dispatch
    _orig_resolve = _Orchestrator._resolve_intent

    async def _poll_and_dispatch(self: Any) -> None:
        # Pick up any external CLI writes that landed between polls.
        self._registry.reload_if_stale()
        await _orig_poll(self)

    async def _resolve_intent(self: Any, issue: Any) -> Any:
        # Defensive second reload — covers the case where CLI wrote between
        # poll-start and intent-resolution inside the same poll cycle
        # (relevant for the retry-queue path that consults the registry
        # via ``self._registry.get(...)`` at orchestrator.py:496, 511, 527,
        # 534, 557, 576).
        self._registry.reload_if_stale()
        return await _orig_resolve(self, issue)

    # Expose the originals on the wrapper objects so tests can swap them
    # without re-installing the patch (avoids the double-wrap trap).
    _poll_and_dispatch.__wrapped__ = _orig_poll  # type: ignore[attr-defined]
    _resolve_intent.__wrapped__ = _orig_resolve  # type: ignore[attr-defined]

    _Orchestrator._poll_and_dispatch = _poll_and_dispatch  # type: ignore[method-assign]
    _Orchestrator._resolve_intent = _resolve_intent  # type: ignore[method-assign]


__all__ = ["install_stale_registry_patch"]
