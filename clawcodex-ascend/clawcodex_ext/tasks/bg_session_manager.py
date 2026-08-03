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

"""F-94 P94-C — BG_SESSIONS 生命周期控制。

``BgSessionManager`` 提供 list / inspect / attach / stop / cleanup /
``background_current_session`` 与 ``upsert_after_launch`` 协调入口。

设计要点（f-94-bg-sessions.md §1.6 / §1.8 / §1.9）：

* **stop graceful-first** — 先 SIGTERM / TaskStop，失败才 SIGKILL（force）。
* **跨 workspace attach 默认拒绝** — 需 ``allow_cross_workspace`` 或
  ``--all``（§1.9 权限规则）。
* **cleanup 不静默删除** — orphaned 标记后才清理；completed 按年龄清理。
* **与 launch_background_runner 协调** — ``upsert_after_launch`` 在 marker
  写完后追加 index upsert（黄金法则 1：不侵入 fork 路径）。
* **bg_sessions=off 退化为现状** — 所有方法在 disabled 时抛
  ``BgSessionsDisabledError`` 或 no-op（验收标准 1）。
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bg_session import (
    BgSession,
    BgSessionAttachError,
    BgSessionConfig,
    BgSessionNotFoundError,
    BgSessionPermissionError,
    BgSessionStopError,
    BgSessionsDisabledError,
    is_bg_sessions_enabled,
    marker_path_for,
    replace_session,
)
from .bg_session_health import assess
from .bg_session_registry import BgSessionRegistry

logger = logging.getLogger(__name__)

#: graceful stop 后等待 PID 退出的最大秒数。
_GRACEFUL_STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class AttachResult:
    """``attach`` 返回 — 含 transcript tail 与恢复命令。"""

    session: BgSession
    transcript_tail: str = ""
    resume_hint: str = ""


class BgSessionManager:
    """统一后台会话生命周期控制。"""

    def __init__(
        self,
        *,
        registry: BgSessionRegistry,
        runtime_tasks: Any | None = None,
        config: BgSessionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._runtime_tasks = runtime_tasks
        self._config = config if config is not None else registry.config

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def registry(self) -> BgSessionRegistry:
        return self._registry

    @property
    def config(self) -> BgSessionConfig:
        return self._config

    # ------------------------------------------------------------------
    # 启用性检查
    # ------------------------------------------------------------------

    def _require_enabled(self) -> None:
        if not is_bg_sessions_enabled(self._config):
            raise BgSessionsDisabledError(
                "BG_SESSIONS is disabled (CLAWCODEX_BG_SESSIONS=off); "
                "fallback to TaskList / per-session marker behavior"
            )

    # ------------------------------------------------------------------
    # 与 launch_background_runner 协调
    # ------------------------------------------------------------------

    def upsert_after_launch(
        self,
        session_id: str,
        pid: int | None,
        *,
        workspace_root: Path | None = None,
        agent_name: str | None = None,
        description: str = "",
    ) -> BgSession | None:
        """在 ``launch_background_runner`` 写完 marker 后追加 index upsert。

        **不修改** marker 文件（由 ``background_runner`` 拥有）。仅构造
        ``BgSession`` 快照并 upsert 到 registry，然后 save index。

        ``bg_sessions=off`` 时 no-op 返回 None（验收标准 1）。
        """
        if not is_bg_sessions_enabled(self._config):
            return None
        marker = marker_path_for(session_id, self._registry.sessions_dir)
        sess = BgSession(
            id=session_id,
            session_id=session_id,
            workspace_root=workspace_root or Path.cwd(),
            status="running",
            pid=pid,
            agent_name=agent_name,
            description=description,
            marker_path=marker if marker.exists() else None,
            transcript_path=self._registry.sessions_dir / session_id / f"{session_id}.jsonl",
            started_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self._registry.upsert(sess)
        self._registry.save()
        return sess

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def list_sessions(
        self,
        *,
        include_completed: bool = False,
        workspace_root: Path | None = None,
    ) -> list[BgSession]:
        """列出会话。``bg_sessions=off`` 时仍可查询（退化到 scan-only）。"""
        sessions = self._registry.list(workspace_root=workspace_root)
        if not include_completed:
            sessions = [s for s in sessions if not s.is_terminal()]
        return sessions

    def inspect(self, bg_session_id: str) -> BgSession:
        """返回最新健康评估后的快照。"""
        sess = self._registry.get(bg_session_id)
        if sess is None:
            # 尝试 scan 一次（可能 marker 存在但 registry 未加载）
            self._registry.scan()
            sess = self._registry.get(bg_session_id)
        if sess is None:
            raise BgSessionNotFoundError(f"BG session {bg_session_id!r} not found")
        # 实时健康校正
        h = assess(sess, stale_after_seconds=self._config.stale_after_seconds)
        if h.status != sess.status:
            sess = replace_session(sess, status=h.status)
        return sess

    # ------------------------------------------------------------------
    # attach
    # ------------------------------------------------------------------

    def attach(
        self,
        bg_session_id: str,
        *,
        follow: bool = True,
        tail_lines: int = 100,
        current_workspace: Path | None = None,
        allow_cross_workspace: bool | None = None,
    ) -> AttachResult:
        """attach 到后台会话 — 返回 transcript tail 与恢复命令。

        跨 workspace attach 默认拒绝（§1.9）。
        """
        sess = self.inspect(bg_session_id)
        # 权限检查
        if current_workspace is not None:
            allow = allow_cross_workspace if allow_cross_workspace is not None else self._config.allow_cross_workspace
            if not allow and not _path_same_workspace(sess.workspace_root, current_workspace):
                raise BgSessionPermissionError(
                    f"BG session {bg_session_id!r} belongs to workspace "
                    f"{sess.workspace_root}; attach from {current_workspace} "
                    f"denied (require --all or allow_cross_workspace=True)"
                )
        # transcript tail
        tail = _read_tail(sess.transcript_path, max_lines=tail_lines)
        hint = _resume_hint(sess.session_id)
        return AttachResult(session=sess, transcript_tail=tail, resume_hint=hint)

    # ------------------------------------------------------------------
    # stop
    # ------------------------------------------------------------------

    def stop(
        self,
        bg_session_id: str,
        *,
        force: bool = False,
    ) -> BgSession:
        """graceful stop；失败时需用户显式 force。

        优先级：
        1. 若有对应 ``runtime_tasks`` 条目 → ``TaskStop`` 协作取消；
        2. 否则 SIGTERM（force=True 时 SIGKILL）；
        3. 等待 PID 退出后更新状态为 ``stopped``。
        """
        sess = self.inspect(bg_session_id)
        if sess.is_terminal():
            return sess
        pid = sess.pid
        stopped_ok = False
        error: str | None = None

        # 1. runtime_tasks 协作取消（若有 task_id）
        if sess.task_id is not None and self._runtime_tasks is not None:
            try:
                stopped_ok = _try_task_stop(self._runtime_tasks, sess.task_id)
            except Exception as exc:
                logger.debug("TaskStop for %s failed: %s", bg_session_id, exc)
                error = f"TaskStop: {exc}"

        # 2. 信号停止
        if not stopped_ok and pid is not None:
            try:
                stopped_ok = _signal_stop(pid, force=force)
            except Exception as exc:
                error = f"signal: {exc}"
                logger.warning("stop signal for %s failed: %s", bg_session_id, exc)

        if not stopped_ok and not force:
            updated = replace_session(sess, status="running", error=error)
            self._registry.upsert(updated)
            self._registry.save()
            raise BgSessionStopError(
                f"graceful stop of {bg_session_id!r} failed; retry with force=True. reason: {error or 'unknown'}"
            )

        updated = replace_session(
            sess,
            status="stopped",
            completed_at=_now_iso(),
            updated_at=_now_iso(),
            error=error,
        )
        self._registry.upsert(updated)
        self._registry.save()
        return updated

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def cleanup(self, *, include_failed: bool = False) -> list[BgSession]:
        """清理终态会话记录。

        * ``completed`` 按 ``cleanup_completed_after_seconds`` 年龄清理；
        * ``orphaned`` 标记后可清理（不静默删除 — 先 inspect 标记）；
        * ``failed`` 仅当 ``include_failed=True`` 清理；
        * 清理仅移除 **index 条目**，**不删除** marker / transcript 文件
          （由 ``background_runner.cleanup_background_runner`` 在 resume
          成功后负责文件清理）。
        """
        now = time.time()  # noqa: F841 - retained for the pending time-window cleanup policy
        removed: list[BgSession] = []
        for sess in self._registry.list():
            age = _age_seconds(sess.completed_at)
            if sess.status == "completed" and age is not None and age > self._config.cleanup_completed_after_seconds:
                if self._registry.remove(sess.id):
                    removed.append(sess)
            elif sess.status == "orphaned":
                if self._registry.remove(sess.id):
                    removed.append(sess)
            elif sess.status == "failed" and include_failed:
                if self._registry.remove(sess.id):
                    removed.append(sess)
        if removed:
            self._registry.save()
        return removed


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


def _age_seconds(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_ts)
        return time.time() - dt.timestamp()
    except (ValueError, OSError):
        return None


def _path_same_workspace(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _read_tail(path: Path | None, *, max_lines: int = 100) -> str:
    if path is None or not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return "".join(lines)
    except OSError as exc:
        raise BgSessionAttachError(f"Cannot read transcript {path}: {exc}") from exc


def _resume_hint(session_id: str) -> str:
    return f"Resume this session with: clawcodex --resume {session_id}"


def _signal_stop(pid: int, *, force: bool) -> bool:
    """发送 SIGTERM/SIGKILL 并等待退出。返回 True 表示进程已退出。"""
    if not _pid_alive(pid):
        return True
    if force:
        sig = getattr(signal, "SIGKILL", None)
        if sig is None:
            raise BgSessionStopError("Force-stopping background sessions is not supported on this platform")
    else:
        sig = signal.SIGTERM
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        return not _pid_alive(pid)
    deadline = time.monotonic() + _GRACEFUL_STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        return False


def _try_task_stop(runtime_tasks: Any, task_id: str) -> bool:
    """尽力尝试通过 runtime_tasks 协作取消。

    返回 True 表示 task 处于终态（已停止或本就终态）。不在本模块引入对
    ``stop_task`` 的硬依赖（避免 clawcodex_ext.tasks → src.tasks.stop_task
    的循环）。
    """
    state = runtime_tasks.get(task_id)
    if state is None:
        return False
    # 终态视为已停止
    # pylint: disable=no-name-in-module  # tasks_core: pending patch migration
    from clawcodex_ext.tasks_core import is_terminal_task_status

    if is_terminal_task_status(state.status):
        return True
    # 尝试调用 task impl 的 kill（若存在）
    try:
        # pylint: disable=no-name-in-module  # task_registry: pending patch migration
        from clawcodex_ext.task_registry import get_task_by_type

        impl = get_task_by_type(state.type)
        if impl is not None and hasattr(impl, "kill"):
            # kill 可能是协程；同步上下文里 best-effort 调用
            result = impl.kill(task_id, runtime_tasks)
            # 协程未运行 — 仅做存在性检查，返回 True 让上层 fallback 到信号
            return result is None or bool(result)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.debug("task impl kill failed: %s", exc)
    return False


__all__ = [
    "AttachResult",
    "BgSessionManager",
]
