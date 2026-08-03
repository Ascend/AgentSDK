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

"""F-94 P94-D — BG_SESSIONS 多信号 orphan 检测。

实现 f-94-bg-sessions.md §1.7 状态判断优先级：

1. 显式 marker ``status=completed|failed`` 优先；
2. ``status=running`` 时检查 PID 存活；
3. PID 不存活时检查 transcript completion marker；
4. transcript mtime 长时间不变但 PID 存活 → stale warning（不立即失败）；
5. 任何不确定状态 → ``unknown`` 或 ``orphaned``，**不静默删除**。

与 ``background_runner.get_background_runner_status`` 的关系：后者只做
单信号（PID）检测并就地改写 marker；本模块做**多信号**判定，返回**新的**
``BgSession`` 快照，不改写 marker —— 改写由 manager.cleanup 统一负责。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bg_session import (
    BgSession,
    BgSessionStatus,
    TRANSCRIPT_COMPLETION_SENTINEL,
    replace_session,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthAssessment:
    """单次健康评估的结果快照。

    ``status`` 是评估后的状态；``is_stale`` 为 True 时仅表示 transcript
    mtime 长期未变（warning），不改变 ``status``（仍是 running）。
    ``signals`` 记录各信号原始值，供 audit / debug。
    """

    status: BgSessionStatus
    is_stale: bool = False
    pid_alive: bool | None = None
    marker_status: str | None = None
    transcript_has_completion: bool | None = None
    transcript_mtime_age_s: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "is_stale": self.is_stale,
            "pid_alive": self.pid_alive,
            "marker_status": self.marker_status,
            "transcript_has_completion": self.transcript_has_completion,
            "transcript_mtime_age_s": self.transcript_mtime_age_s,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 信号原语
# ---------------------------------------------------------------------------


def _pid_alive(pid: int | None) -> bool:
    """``os.kill(pid, 0)`` 存活检查 — 与 background_runner 同源。"""
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        return False


def _read_marker(marker_path: Path | None) -> dict[str, Any] | None:
    """读取 ``.background-runner.json``；损坏/缺失返回 None。"""
    if marker_path is None or not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _transcript_has_completion(transcript_path: Path | None) -> bool | None:
    """检查 transcript 是否含完成哨兵。

    返回 ``None`` 表示无法判定（文件缺失/读取失败），调用方应视为
    "不确定"而非"未完成"。
    """
    if transcript_path is None or not transcript_path.exists():
        return None
    try:
        # tail-style 读取：完成哨兵在文件末尾附近，避免全文件扫描。
        with transcript_path.open("rb") as f:
            try:
                f.seek(-8192, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="replace")
        return TRANSCRIPT_COMPLETION_SENTINEL in tail
    except Exception:
        return None


def _transcript_mtime_age_s(transcript_path: Path | None) -> float | None:
    """transcript 距上次 mtime 的秒数；缺失返回 None。"""
    if transcript_path is None or not transcript_path.exists():
        return None
    try:
        return time.time() - transcript_path.stat().st_mtime
    except OSError:
        return None


# ---------------------------------------------------------------------------
# 主判定
# ---------------------------------------------------------------------------


def assess(
    session: BgSession,
    *,
    stale_after_seconds: int = 600,
) -> HealthAssessment:
    """对单个 ``BgSession`` 做多信号健康评估。

    遵循 §1.7 优先级链。**纯函数**：不改写 marker、不删除文件。
    """
    marker = _read_marker(session.marker_path)
    marker_status = marker.get("status") if marker else None
    pid = session.pid if session.pid is not None else (int(marker["pid"]) if marker and "pid" in marker else None)
    pid_ok = _pid_alive(pid)
    transcript_done = _transcript_has_completion(session.transcript_path)
    mtime_age = _transcript_mtime_age_s(session.transcript_path)

    # 优先级 1：显式 marker 终态
    if marker_status in ("completed", "failed"):
        return HealthAssessment(
            status=marker_status,
            pid_alive=pid_ok,
            marker_status=marker_status,
            transcript_has_completion=transcript_done,
            transcript_mtime_age_s=mtime_age,
            reason="marker terminal status",
        )

    # 优先级 2：marker/record 说 running，检查 PID
    if session.status == "running" or marker_status == "running":
        if pid_ok:
            # 优先级 4：PID 存活但 transcript 长期未动 → stale warning
            is_stale = mtime_age is not None and mtime_age > stale_after_seconds
            return HealthAssessment(
                status="running",
                is_stale=is_stale,
                pid_alive=True,
                marker_status=marker_status,
                transcript_has_completion=transcript_done,
                transcript_mtime_age_s=mtime_age,
                reason="stale" if is_stale else "pid alive",
            )
        # PID 不存活
        # 优先级 3：检查 transcript completion
        if transcript_done is True:
            return HealthAssessment(
                status="completed",
                pid_alive=False,
                marker_status=marker_status,
                transcript_has_completion=True,
                transcript_mtime_age_s=mtime_age,
                reason="pid gone, transcript completed",
            )
        if transcript_done is False:
            return HealthAssessment(
                status="orphaned",
                pid_alive=False,
                marker_status=marker_status,
                transcript_has_completion=False,
                transcript_mtime_age_s=mtime_age,
                reason="pid gone, no completion marker",
            )
        # transcript 无法判定 → orphaned（不确定偏向保守）
        return HealthAssessment(
            status="orphaned",
            pid_alive=False,
            marker_status=marker_status,
            transcript_has_completion=None,
            transcript_mtime_age_s=mtime_age,
            reason="pid gone, transcript unreadable",
        )

    # starting：marker 已写但 PID 尚未可见 / 已死
    if session.status == "starting":
        if pid_ok:
            return HealthAssessment(
                status="running",
                pid_alive=True,
                marker_status=marker_status,
                transcript_has_completion=transcript_done,
                transcript_mtime_age_s=mtime_age,
                reason="starting → running (pid alive)",
            )
        if marker is None:
            return HealthAssessment(
                status="unknown",
                pid_alive=False,
                marker_status=None,
                transcript_has_completion=transcript_done,
                transcript_mtime_age_s=mtime_age,
                reason="starting, no marker yet",
            )
        return HealthAssessment(
            status="failed",
            pid_alive=False,
            marker_status=marker_status,
            transcript_has_completion=transcript_done,
            transcript_mtime_age_s=mtime_age,
            reason="starting, marker present but pid dead",
        )

    # paused / orphaned / unknown / terminal：保持原状态，仅刷新信号
    return HealthAssessment(
        status=session.status,
        pid_alive=pid_ok,
        marker_status=marker_status,
        transcript_has_completion=transcript_done,
        transcript_mtime_age_s=mtime_age,
        reason="preserve existing status",
    )


def reconcile(
    session: BgSession,
    *,
    stale_after_seconds: int = 600,
) -> BgSession:
    """评估并返回更新后的 ``BgSession``（纯函数，不改 marker）。"""
    h = assess(session, stale_after_seconds=stale_after_seconds)
    return replace_session(
        session,
        status=h.status,
        error=h.reason if h.status in ("orphaned", "failed", "unknown") else session.error,
    )


__all__ = [
    "HealthAssessment",
    "assess",
    "reconcile",
]
