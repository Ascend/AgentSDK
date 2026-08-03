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

"""F-94 P94-A — BG_SESSIONS 数据模型与状态机。

定义 ``BgSession`` / ``BgSessionStatus`` / ``BgSessionEvent`` /
``BgSessionConfig`` 以及失败模式异常类。

状态机（见 f-94-bg-sessions.md §1.7）::

    starting
      ├─ marker written + pid alive ─▶ running
      ├─ launch failed ──────────────▶ failed
      └─ marker stale before pid ────▶ unknown

    running
      ├─ user attach/resume ─────────▶ paused
      ├─ completion marker ──────────▶ completed
      ├─ child exit non-zero ────────▶ failed
      ├─ pid gone, no completion ────▶ orphaned
      └─ user stop ──────────────────▶ stopped

    orphaned
      ├─ transcript completion ──────▶ completed
      ├─ cleanup removes marker ─────▶ stopped
      └─ user attach w/ transcript ──▶ paused

状态判断优先级（health.py 实现）：
1. 显式 marker ``status=completed|failed`` 优先；
2. ``status=running`` 时检查 PID 存活；
3. PID 不存活时检查 transcript completion marker；
4. transcript mtime 长时间不变但 PID 存活 → stale warning（不立即失败）；
5. 任何不确定状态 → ``unknown`` 或 ``orphaned``，**不静默删除**。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

# ---------------------------------------------------------------------------
# 路径与默认配置
# ---------------------------------------------------------------------------

#: 默认全局 index 路径。``~`` 在 ``BgSessionConfig`` 构造时展开。
#: 与 ``background_runner._sessions_dir()`` 保持 ``~/.clawcodex`` 根一致。
DEFAULT_INDEX_PATH: Path = Path("~/.clawcodex/bg_sessions/index.json")

#: 默认 sessions 目录（与 ``background_runner._sessions_dir`` 同源）。
DEFAULT_SESSIONS_DIR: Path = Path("~/.clawcodex/sessions")

#: marker 文件名（与 ``background_runner._runner_marker_path`` 同源）。
RUNNER_MARKER_NAME: str = ".background-runner.json"

#: transcript 中由 ``_run_agent_headless`` 写入的完成哨兵。
TRANSCRIPT_COMPLETION_SENTINEL: str = "__background_complete__"


# ---------------------------------------------------------------------------
# 状态字面量
# ---------------------------------------------------------------------------

BgSessionStatus: TypeAlias = Literal[
    "starting",
    "running",
    "paused",
    "completed",
    "failed",
    "stopped",
    "orphaned",
    "unknown",
]

#: 终态集合 — 不再发生状态迁移。
TERMINAL_BG_STATUSES: frozenset[BgSessionStatus] = frozenset({"completed", "failed", "stopped"})

#: 活跃态集合 — PID 应当存活。
ACTIVE_BG_STATUSES: frozenset[BgSessionStatus] = frozenset({"starting", "running", "paused"})

BgSessionEventType: TypeAlias = Literal[
    "created",
    "backgrounded",
    "attached",
    "resumed",
    "stopped",
    "completed",
    "failed",
    "orphaned",
    "cleaned",
]


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BgSession:
    """单个后台会话的快照视图。

    ``id`` 与 ``session_id`` 的关系：当 background 来自 Ctrl+B 时两者相同；
    当 background 来自 Agent tool / Team spawn 时 ``id`` 可为稳定的 bg id
    （F-99 DIRECT_CONNECT 共享 session_id 命名空间，通过 ``source=bg_session``
    区分）。
    """

    id: str
    session_id: str
    workspace_root: Path
    status: BgSessionStatus
    pid: int | None = None
    task_id: str | None = None
    team_id: str | None = None
    agent_name: str | None = None
    description: str = ""
    transcript_path: Path | None = None
    marker_path: Path | None = None
    output_file: Path | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    last_activity_at: str | None = None
    error: str | None = None

    def is_terminal(self) -> bool:
        """是否处于终态（不会再迁移）。"""
        return self.status in TERMINAL_BG_STATUSES

    def is_active(self) -> bool:
        """是否处于活跃态（PID 应当存活）。"""
        return self.status in ACTIVE_BG_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容字典（Path → str）。"""
        d = asdict(self)
        for key, val in list(d.items()):
            if isinstance(val, Path):
                d[key] = str(val)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BgSession":
        """从字典反序列化，容错处理 Path / 缺失字段。"""
        path_keys = (
            "workspace_root",
            "transcript_path",
            "marker_path",
            "output_file",
        )
        kwargs: dict[str, Any] = {}
        for key in (
            "id",
            "session_id",
            "workspace_root",
            "status",
            "pid",
            "task_id",
            "team_id",
            "agent_name",
            "description",
            "transcript_path",
            "marker_path",
            "output_file",
            "started_at",
            "updated_at",
            "completed_at",
            "last_activity_at",
            "error",
        ):
            if key not in data:
                continue
            val = data[key]
            if key in path_keys and val is not None and not isinstance(val, Path):
                val = Path(str(val))
            kwargs[key] = val
        # 必填字段兜底
        kwargs.setdefault("id", kwargs.get("session_id", ""))
        kwargs.setdefault("session_id", kwargs.get("id", ""))
        kwargs.setdefault("workspace_root", Path("."))
        kwargs.setdefault("status", "unknown")
        return cls(**kwargs)


@dataclass(frozen=True)
class BgSessionEvent:
    """后台会话事件日志条目。"""

    id: str
    bg_session_id: str
    event_type: BgSessionEventType
    actor: str
    message: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BgSessionEvent":
        return cls(
            id=str(data.get("id", "")),
            bg_session_id=str(data.get("bg_session_id", "")),
            event_type=data.get("event_type", "created"),  # type: ignore[arg-type]
            actor=str(data.get("actor", "")),
            message=str(data.get("message", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class BgSessionConfig:
    """F-94 BG_SESSIONS 运行时配置。

    ``enabled`` 由环境变量 ``CLAWCODEX_BG_SESSIONS`` 决定（``off`` / ``0``
    / ``false`` 关闭，其余开启）。关闭时不写全局 index，仅保留现有
    per-session marker 行为（验收标准 1）。
    """

    enabled: bool = False
    index_path: Path = DEFAULT_INDEX_PATH
    sessions_dir: Path = DEFAULT_SESSIONS_DIR
    stale_after_seconds: int = 600
    max_sessions: int = 200
    cleanup_completed_after_seconds: int = 86_400
    allow_agent_attach: bool = True
    allow_cross_workspace: bool = False

    @classmethod
    def from_env(cls) -> "BgSessionConfig":
        """从环境变量构造，``CLAWCODEX_BG_SESSIONS=off`` 关闭。"""
        raw = os.environ.get("CLAWCODEX_BG_SESSIONS", "").strip().lower()
        enabled = raw not in ("", "off", "0", "false", "no", "disabled")
        return cls(
            enabled=enabled,
            index_path=DEFAULT_INDEX_PATH.expanduser(),
            sessions_dir=DEFAULT_SESSIONS_DIR.expanduser(),
        )


def is_bg_sessions_enabled(config: BgSessionConfig | None = None) -> bool:
    """便捷谓词：BG_SESSIONS 是否开启。"""
    cfg = config if config is not None else BgSessionConfig.from_env()
    return cfg.enabled


def replace_session(session: BgSession, **changes: Any) -> BgSession:
    """``dataclasses.replace`` 的薄包装 — 返回更新部分字段后的新快照。

    ``BgSession`` 是 frozen dataclass，状态迁移统一通过本函数生成新实例，
    避免散落的 ``replace(...)`` 调用点。
    """
    return replace(session, **changes)


def marker_path_for(session_id: str, sessions_dir: Path) -> Path:
    """``session_id`` 对应的 ``.background-runner.json`` 路径。

    与 ``background_runner._runner_marker_path`` 同源协议。
    """
    return sessions_dir / session_id / RUNNER_MARKER_NAME


def transcript_path_for(session_id: str, sessions_dir: Path) -> Path:
    """``session_id`` 对应的 JSONL transcript 路径。

    与 ``SessionStorage`` 默认布局一致：``sessions/<id>/<id>.jsonl``。
    """
    return sessions_dir / session_id / f"{session_id}.jsonl"


# ---------------------------------------------------------------------------
# 失败模式异常（f-94-bg-sessions.md §1.10）
# ---------------------------------------------------------------------------


class BgSessionsDisabledError(RuntimeError):
    """``bg_sessions=off`` — Tool 应 fallback 到 TaskList。"""


class BgSessionNotFoundError(LookupError):
    """``bg_session_id`` 不存在。"""


class BgSessionAlreadyRunningError(RuntimeError):
    """同一 session 重复 background。"""


class BgSessionAttachError(RuntimeError):
    """transcript 缺失或格式损坏。"""


class BgSessionPermissionError(PermissionError):
    """跨 workspace/team 越权。"""


class BgSessionOrphanedError(RuntimeError):
    """PID gone + 无 completion marker。"""


class BgSessionStopError(RuntimeError):
    """kill 失败。"""


__all__ = [
    "ACTIVE_BG_STATUSES",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_SESSIONS_DIR",
    "RUNNER_MARKER_NAME",
    "TERMINAL_BG_STATUSES",
    "TRANSCRIPT_COMPLETION_SENTINEL",
    "BgSession",
    "BgSessionAlreadyRunningError",
    "BgSessionAttachError",
    "BgSessionConfig",
    "BgSessionEvent",
    "BgSessionEventType",
    "BgSessionNotFoundError",
    "BgSessionOrphanedError",
    "BgSessionPermissionError",
    "BgSessionStatus",
    "BgSessionStopError",
    "BgSessionsDisabledError",
    "is_bg_sessions_enabled",
    "marker_path_for",
    "replace_session",
    "transcript_path_for",
]
