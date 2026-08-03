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

"""F-94 P94-F — REPL/TUI 后台会话显示适配器。

提供 ``bg_sessions_panel``（footer 统计）与 ``format_bg_sessions_status``
（task list 分组渲染）。**纯展示层** — 不持有状态，每次调用从 registry
实时读取。

设计（f-94-bg-sessions.md §1.8 UI 规则）：

* TUI footer 显示当前 workspace 的 running BG sessions 数量；
* task list 中把 background shell、local agent 与 BG session 分组显示；
* completion notification 应包含 session_id 与恢复命令（由调用方在事件
  回调中拼接，本模块提供 ``format_completion_notification`` 辅助）。
"""

from __future__ import annotations
# pylint: disable=E0611

from pathlib import Path

from clawcodex_ext.tasks.bg_session import BgSession, is_bg_sessions_enabled
from clawcodex_ext.tasks.bg_session_registry import BgSessionRegistry


# ---------------------------------------------------------------------------
# Footer 统计
# ---------------------------------------------------------------------------


def footer_summary(
    registry: BgSessionRegistry,
    *,
    workspace_root: Path | None = None,
) -> str:
    """返回 footer 单行摘要，如 ``bg:2``。

    ``bg_sessions=off`` 时返回空串（footer 不显示）。
    """
    if not is_bg_sessions_enabled(registry.config):
        return ""
    sessions = registry.list(workspace_root=workspace_root)
    running = sum(1 for s in sessions if s.is_active())
    orphaned = sum(1 for s in sessions if s.status == "orphaned")
    if running == 0 and orphaned == 0:
        return ""
    parts = [f"bg:{running}"]
    if orphaned:
        parts.append(f"orphan:{orphaned}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Task list 分组渲染
# ---------------------------------------------------------------------------


def format_bg_sessions_status(
    registry: BgSessionRegistry,
    *,
    workspace_root: Path | None = None,
    include_completed: bool = False,
) -> str:
    """多行渲染 BG session 分组（供 task list / status 面板）。

    与 TaskList 的 background shell / local agent 分组并列显示。
    """
    if not is_bg_sessions_enabled(registry.config):
        return "(bg_sessions disabled)"
    sessions = registry.list(workspace_root=workspace_root)
    if not include_completed:
        sessions = [s for s in sessions if not s.is_terminal()]
    if not sessions:
        return "BG sessions: (none)"
    lines = ["BG sessions:"]
    for s in sessions:
        marker = _status_marker(s.status)
        lines.append(f"  {marker} {s.id}  pid={s.pid}  agent={s.agent_name or '-'}  ws={s.workspace_root}")
    return "\n".join(lines)


def _status_marker(status: str) -> str:
    return {
        "running": "▶",
        "starting": "…",
        "paused": "⏸",
        "orphaned": "⚠",
        "completed": "✓",
        "failed": "✗",
        "stopped": "■",
        "unknown": "?",
    }.get(status, "?")


# ---------------------------------------------------------------------------
# Completion notification
# ---------------------------------------------------------------------------


def format_completion_notification(session: BgSession) -> str:
    """后台 session 完成时的通知文本（含 session_id 与恢复命令）。

    供 task notification 队列消费方在 preamble 中注入。格式遵循
    f-94-bg-sessions.md §1.8 "completion notification 应包含 session_id
    与恢复命令"。
    """
    return (
        f"<bg-session-notification>\n"
        f"  session_id: {session.session_id}\n"
        f"  status: {session.status}\n"
        f"  workspace: {session.workspace_root}\n"
        f"  resume: clawcodex --resume {session.session_id}\n"
        f"</bg-session-notification>"
    )


# ---------------------------------------------------------------------------
# 便捷：从环境构造 registry 并扫描
# ---------------------------------------------------------------------------


def make_panel_registry() -> BgSessionRegistry:
    """构造一个用于面板展示的 registry（from_env，惰性 scan）。

    调用方负责在需要最新数据时 ``registry.scan()``。建议 REPL/TUI 启动时
    调用一次，并在 BG session 变化时（如 Ctrl+B 后）调用 scan 刷新。
    """
    return BgSessionRegistry()


__all__ = [
    "footer_summary",
    "format_bg_sessions_status",
    "format_completion_notification",
    "make_panel_registry",
]
