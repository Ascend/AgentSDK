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

"""F-94 P94-B/C — 协调 ``launch_background_runner`` 写全局 index。

按 CLAUDE.md 黄金法则 1（不侵入 ``src/``）与模式 B（猴补丁）：包装
``launch_background_runner``，在原函数写完 ``.background-runner.json``
marker 后追加 ``BgSessionManager.upsert_after_launch`` 写全局 index。

**关键契约**（f-94-bg-sessions.md §1.9 / 验收标准 1）：
``bg_sessions=off`` 时 ``upsert_after_launch`` 内部 no-op 返回 None，
退化为现有 marker 行为，不写 ``index.json``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_installed: bool = False


def install_bg_session_index_hook() -> None:
    """猴补丁 ``launch_background_runner`` — 写 marker 后 upsert 全局 index。

    幂等；失败仅记录日志，不阻断 background fork 主路径。
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
        # 尽力 upsert；任何异常都不影响 fork 主路径
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
    # 同步 src.agent.background_runner facade 的引用（re-export）
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
