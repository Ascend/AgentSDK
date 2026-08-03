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

"""F-94 P94-B — BG_SESSIONS 全局 registry。

跨进程后台会话索引。扫描 ``~/.clawcodex/sessions/*/.background-runner.json``
重建内存视图，并持久化缓存到 ``~/.clawcodex/bg_sessions/index.json``。

设计要点（f-94-bg-sessions.md §1.6 / §3 风险"index 与真实 session 目录不一致"）：

* **``scan()`` 为事实源** — index.json 仅是缓存，损坏时通过 scan 重建
  （验收标准 10）。
* **纯增量** — 不修改现有 ``.background-runner.json`` marker；marker 由
  ``background_runner._write_runner_marker`` / ``_update_runner_status``
  继续管理。
* **线程安全** — RLock 保护内存 dict；磁盘 I/O 在锁外（save 时持有快照副本）。
* **性能** — 100 个 session scan < 100ms（验收标准 8）：只 stat + 读 marker，
  不解析整个 transcript。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .bg_session import (
    BgSession,
    BgSessionConfig,
    BgSessionStatus,
    marker_path_for,
    replace_session,
    transcript_path_for,
)
from . import bg_session_health

logger = logging.getLogger(__name__)


class BgSessionRegistry:
    """跨进程后台会话索引。

    线程安全；磁盘 I/O 尽量在锁外。``scan()`` 是事实源，``save()`` 写缓存。
    """

    def __init__(self, *, config: BgSessionConfig | None = None) -> None:
        self._config = config if config is not None else BgSessionConfig.from_env()
        self._lock = threading.RLock()
        self._sessions: dict[str, BgSession] = {}

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def config(self) -> BgSessionConfig:
        return self._config

    @property
    def index_path(self) -> Path:
        return self._config.index_path

    @property
    def sessions_dir(self) -> Path:
        return self._config.sessions_dir

    # ------------------------------------------------------------------
    # 事实源 — scan
    # ------------------------------------------------------------------

    def scan(self) -> list[BgSession]:
        """扫描 ``sessions_dir/*/.background-runner.json`` 重建视图。

        对每个 marker 读取并经 ``bg_session_health.reconcile`` 多信号校正。
        返回校正后的列表，并就地替换内存视图。
        """
        sessions: list[BgSession] = []
        sessions_dir = self.sessions_dir
        if not sessions_dir.exists():
            with self._lock:
                self._sessions = {}
            return sessions

        try:
            entries = list(sessions_dir.iterdir())
        except OSError:
            logger.exception("Cannot list sessions dir %s", sessions_dir)
            return sessions

        for entry in entries:
            if not entry.is_dir():
                continue
            session_id = entry.name
            marker = marker_path_for(session_id, sessions_dir)
            if not marker.exists():
                continue
            sess = self._session_from_marker(session_id, marker, sessions_dir)
            if sess is not None:
                sessions.append(sess)

        # 应用 max_sessions 上限（保留最新 started_at）
        if len(sessions) > self._config.max_sessions:
            sessions.sort(key=lambda s: s.started_at or "", reverse=True)
            sessions = sessions[: self._config.max_sessions]

        with self._lock:
            self._sessions = {s.id: s for s in sessions}
        return sessions

    def _session_from_marker(
        self,
        session_id: str,
        marker_path: Path,
        sessions_dir: Path,
    ) -> BgSession | None:
        """从 marker JSON 构造 ``BgSession`` 并做健康校正。"""
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Corrupt marker %s; skipping", marker_path, exc_info=True)
            return None

        transcript = transcript_path_for(session_id, sessions_dir)
        sess = BgSession(
            id=session_id,
            session_id=session_id,
            workspace_root=Path(data.get("workspace_root", ".")),
            status=data.get("status", "unknown"),  # type: ignore[arg-type]
            pid=data.get("pid"),
            task_id=data.get("task_id"),
            team_id=data.get("team_id"),
            agent_name=data.get("agent_name"),
            description=data.get("description", ""),
            transcript_path=transcript if transcript.exists() else None,
            marker_path=marker_path,
            output_file=Path(data["output_file"]) if data.get("output_file") else None,
            started_at=data.get("started_at"),
            updated_at=data.get("updated_at"),
            completed_at=data.get("completed_at"),
            last_activity_at=data.get("last_activity_at"),
            error=data.get("error"),
        )
        return bg_session_health.reconcile(sess, stale_after_seconds=self._config.stale_after_seconds)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def list(self, *, workspace_root: Path | None = None) -> list[BgSession]:
        """列出会话；可选按 workspace 过滤。

        不触发 scan（用内存视图）；若从未 scan 则返回空。调用方负责在
        需要最新数据时显式 ``scan()``。
        """
        with self._lock:
            sessions = list(self._sessions.values())
        if workspace_root is not None:
            target = workspace_root.resolve()
            sessions = [s for s in sessions if _safe_resolve(s.workspace_root) == target]
        return sessions

    def get(self, bg_session_id: str) -> BgSession | None:
        with self._lock:
            return self._sessions.get(bg_session_id)

    # ------------------------------------------------------------------
    # 变更
    # ------------------------------------------------------------------

    def upsert(self, session: BgSession) -> None:
        """插入或更新单个会话。返回前不自动 save（调用方决定持久化时机）。"""
        with self._lock:
            self._sessions[session.id] = session

    def remove(self, bg_session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(bg_session_id, None) is not None

    def update_status(
        self,
        bg_session_id: str,
        status: BgSessionStatus,
        *,
        error: str | None = None,
    ) -> BgSession | None:
        """就地更新某会话状态（不触发 scan）。"""
        with self._lock:
            sess = self._sessions.get(bg_session_id)
            if sess is None:
                return None
            changes: dict[str, Any] = {"status": status}
            if error is not None:
                changes["error"] = error
            if status in ("completed", "failed", "stopped"):
                changes["completed_at"] = _now_iso()
            changes["updated_at"] = _now_iso()
            updated = replace_session(sess, **changes)
            self._sessions[bg_session_id] = updated
            return updated

    # ------------------------------------------------------------------
    # 持久化 — index.json 缓存
    # ------------------------------------------------------------------

    def save(self) -> Path | None:
        """持久化内存视图到 ``index_path``。

        ``enabled=False`` 时 **不写**（验收标准 1）。
        返回写入路径；未写返回 None。
        """
        if not self._config.enabled:
            return None
        with self._lock:
            snapshot = [s.to_dict() for s in self._sessions.values()]
        target = self.index_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {"version": 1, "updated_at": _now_iso(), "sessions": snapshot},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError:
            logger.exception("Failed to persist bg_sessions index %s", target)
            return None
        return target

    def load(self) -> list[BgSession]:
        """从 ``index_path`` 加载缓存（不触发 scan）。

        损坏时记录 audit 日志并返回空列表（验收标准 10 要求能重建；
        重建由 ``scan()`` 完成，本方法仅读缓存）。
        """
        target = self.index_path
        if not target.exists():
            return []
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("bg_sessions index %s corrupt; will be rebuilt by scan()", target)
            return []
        raw_sessions = data.get("sessions", []) if isinstance(data, dict) else []
        sessions = [BgSession.from_dict(d) for d in raw_sessions if isinstance(d, dict)]
        with self._lock:
            self._sessions = {s.id: s for s in sessions}
        return sessions

    def rebuild_and_save(self) -> list[BgSession]:
        """scan + save 组合 — index 损坏后的恢复入口。"""
        sessions = self.scan()
        self.save()
        return sessions


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


def _safe_resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p


__all__ = [
    "BgSessionRegistry",
]
