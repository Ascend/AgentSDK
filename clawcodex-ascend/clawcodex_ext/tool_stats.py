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

"""工具/Skill 跨会话调用统计。

追加写 JSONL 到 ``~/.clawcodex/tool_stats.jsonl``，统一 schema 记录每次
工具或 Skill 调用的耗时、成功/失败状态。不依赖 audit 路径，独立运行。

用法::

    from clawcodex_ext.tool_stats import record_tool, record_skill

    # 工具执行完成后
    record_tool("Read", dur_ms=12.3, ok=True)

    # Skill 执行完成后
    record_skill("code_review", dur_ms=3200.0, ok=True, params={"target": "main.py"})
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────────────────────
_DEFAULT_STATS_PATH = Path.home() / ".clawcodex" / "tool_stats.jsonl"

# ── 缓冲写 ────────────────────────────────────────────────────────────
_lock = threading.Lock()
_buffer: list[str] = []
_BUFFER_FLUSH_SIZE = 20  # 攒够 20 行或 5 秒后落盘
_last_flush: float = time.monotonic()
_stats_path: Path = _DEFAULT_STATS_PATH


def configure(path: str | Path | None = None) -> None:
    """Override the default output path for tests or custom storage.

    Reconfiguration is a sink boundary: buffered rows must be flushed to the
    previous path instead of being redirected into the new file.
    """
    global _last_flush, _stats_path
    if path is None:
        return
    target = Path(path)
    with _lock:
        if target == _stats_path:
            return
        _do_flush()
        if _buffer:
            logger.warning(
                "tool_stats: dropping %d buffered row(s) after the previous sink could not be flushed",
                len(_buffer),
            )
            _buffer.clear()
        _stats_path = target
        _stats_path.parent.mkdir(parents=True, exist_ok=True)
        _last_flush = time.monotonic()


def record_tool(
    tool_name: str,
    dur_ms: float,
    ok: bool,
    *,
    error: str | None = None,
    agent_id: str = "main",
) -> None:
    """记录一次工具调用。"""
    _record(
        agent_id=agent_id,
        kind="tool",
        name=tool_name,
        dur_ms=dur_ms,
        ok=ok,
        error=error,
    )


def record_skill(
    skill_name: str,
    dur_ms: float,
    ok: bool,
    *,
    error: str | None = None,
    params: dict[str, Any] | None = None,
    skill_version: str | None = None,
    agent_id: str = "main",
) -> None:
    """记录一次 Skill 调用。"""
    _record(
        agent_id=agent_id,
        kind="skill",
        name=skill_name,
        dur_ms=dur_ms,
        ok=ok,
        error=error,
        extra={"params": params, "skill_version": skill_version},
    )


# ── 内部实现 ──────────────────────────────────────────────────────────


def _record(
    agent_id: str,
    kind: str,
    name: str,
    dur_ms: float,
    ok: bool,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """统一的记录入口。

    组装 JSON 行并追加到缓冲区。写操作在 ``threading.Lock`` 保护下执行，
    无需调用方关心并发安全。
    """
    entry: dict[str, Any] = {
        "agent_id": agent_id,
        "kind": kind,
        "ts": time.time(),
        "dur_ms": round(dur_ms, 1),
        "ok": ok,
    }
    if kind == "tool":
        entry["tool"] = name
    else:
        entry["skill"] = name

    if error is not None:
        entry["error"] = error
    if extra:
        entry.update({k: v for k, v in extra.items() if v is not None})

    _write_buffered(entry)


def _write_buffered(entry: dict[str, Any]) -> None:
    """批量追加写 JSONL，攒满 ``_BUFFER_FLUSH_SIZE`` 行或距上次 flush
    超过 5 秒时落盘。

    线程安全：使用模块级 ``threading.Lock``。
    """
    global _last_flush
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _lock:
        _buffer.append(line)
        now = time.monotonic()
        if len(_buffer) >= _BUFFER_FLUSH_SIZE or (now - _last_flush) >= 5.0:
            _do_flush()
            _last_flush = now


def _do_flush() -> None:
    """将缓冲区内容写入文件（在锁内调用）。"""
    if not _buffer:
        return
    try:
        _stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_stats_path, "a", encoding="utf-8") as f:
            f.write("\n".join(_buffer) + "\n")
        _buffer.clear()
    except OSError as e:
        logger.warning("tool_stats write failed: %s", e)


def flush() -> None:
    """强制落盘缓冲区（应用退出前调用）。"""
    with _lock:
        _do_flush()


# 进程退出时自动落盘
atexit.register(flush)


# ── 查询 ──────────────────────────────────────────────────────────────


def get_stats(
    kind: str | None = None,
    agent_id: str | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """读取并返回 ``tool_stats.jsonl`` 中的记录。

    参数:
        kind: ``"tool"`` 或 ``"skill"``，为 ``None`` 时返回全部。
        agent_id: 按 agent 过滤，为 ``None`` 时不限制。
        limit: 返回最大条数，0 表示全量。

    返回:
        解析后的 dict 列表，按时间戳降序排列。
    """
    flush()  # 先落盘确保最新数据
    path = _stats_path
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind is not None and row.get("kind") != kind:
                    continue
                if agent_id is not None and row.get("agent_id") != agent_id:
                    continue
                rows.append(row)
    except OSError as e:
        logger.warning("tool_stats read failed: %s", e)
        return []

    rows.sort(key=lambda r: r.get("ts", 0.0), reverse=True)
    if limit > 0:
        rows = rows[:limit]
    return rows


def get_summary(
    kind: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """返回聚合摘要。

    返回结构::

        {
            "total_calls": int,
            "by_name": { "ToolName": count, ... },
            "by_name_ok": { "ToolName": ok_count, ... },
            "avg_duration_ms": float,
            "error_rate": float,    # 0.0 ~ 1.0
        }
    """
    rows = get_stats(kind=kind, agent_id=agent_id)
    if not rows:
        return {
            "total_calls": 0,
            "by_name": {},
            "by_name_ok": {},
            "avg_duration_ms": 0.0,
            "error_rate": 0.0,
        }

    total = len(rows)
    by_name: dict[str, int] = {}
    by_name_ok: dict[str, int] = {}
    total_dur = 0.0
    error_count = 0

    for row in rows:
        name = row.get("tool") or row.get("skill") or "unknown"
        by_name[name] = by_name.get(name, 0) + 1
        if row.get("ok"):
            by_name_ok[name] = by_name_ok.get(name, 0) + 1
        else:
            error_count += 1
        total_dur += row.get("dur_ms", 0.0)

    return {
        "total_calls": total,
        "by_name": dict(sorted(by_name.items(), key=lambda x: -x[1])),
        "by_name_ok": dict(sorted(by_name_ok.items(), key=lambda x: -x[1])),
        "avg_duration_ms": round(total_dur / total, 1) if total else 0.0,
        "error_rate": round(error_count / total, 3) if total else 0.0,
    }
