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

"""Coordinate global-index updates after ``launch_background_runner``.

The wrapper calls ``BgSessionManager.upsert_after_launch`` after the original
function writes its runner marker. When background sessions are disabled, the
upsert remains a no-op and no global index is written.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_installed: bool = False


def install_bg_session_index_hook() -> None:
    """Wrap ``launch_background_runner`` to update the global index.

    Installation is idempotent; failures are logged without blocking forks.
    """
    global _installed
    if _installed:
        return
    _installed = True

    try:
        import clawcodex_ext.agent.background_runner as br
        from clawcodex_ext.tasks.bg_session_manager import BgSessionManager
        from clawcodex_ext.tasks.bg_session_registry import BgSessionRegistry
    except Exception:  # noqa: BLE001 — never break agent init
        logger.debug("bg_session index hook skipped (import failed)", exc_info=True)
        return

    original = br.launch_background_runner
    if getattr(original, "_bg_session_wrapped", False):
        return

    def _wrapped(session, provider, tool_registry, tool_context, max_turns):  # type: ignore[no-untyped-def]
        pid = original(session, provider, tool_registry, tool_context, max_turns)
        # Best-effort indexing must not interrupt the fork path.
        try:
            ws = _resolve_workspace(tool_context)
            mgr = BgSessionManager(registry=BgSessionRegistry())
            mgr.upsert_after_launch(
                session.session_id,
                pid,
                workspace_root=ws,
                agent_name=getattr(session, "agent_name", None),
                description=getattr(session, "description", "") or "",
            )
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "bg_session upsert_after_launch failed for %s",
                getattr(session, "session_id", "?"),
                exc_info=True,
            )
        return pid

    _wrapped._bg_session_wrapped = True  # type: ignore[attr-defined]
    br.launch_background_runner = _wrapped  # type: ignore[assignment]
    # Keep the re-exported facade reference synchronized.
    try:
        import src.agent.background_runner as src_br

        if getattr(src_br.launch_background_runner, "_bg_session_wrapped", False) is False:
            src_br.launch_background_runner = _wrapped  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001 - installation must not block startup
        logger.debug("Could not install background-session index hook", exc_info=exc)


def _resolve_workspace(tool_context: Any) -> Path | None:
    for attr in ("workspace_root", "cwd", "working_dir"):
        val = getattr(tool_context, attr, None)
        if val is not None:
            return Path(val) if not isinstance(val, Path) else val
    return None


__all__ = ["install_bg_session_index_hook"]
