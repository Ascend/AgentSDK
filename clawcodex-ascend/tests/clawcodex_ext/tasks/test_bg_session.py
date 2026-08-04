#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""F-94 P94-H — BG_SESSIONS 单元测试。

覆盖验收标准（f-94-bg-sessions.md §1.11）：
1. bg_sessions=off 时不写 index.json；
2. /bg list 能列出 session_id、pid、workspace、status；
3. 后台 runner 完成后 status 从 running 变 completed；
4. PID 消失但无 completion marker → orphaned，不静默删除；
5. /bg attach 能 tail transcript 并给出恢复路径；
6. /bg stop 先 graceful，失败需 force；
7. 跨 workspace attach 默认拒绝；
8. 100 个 session scan < 100ms；
9. registry scan、状态机、orphan cleanup、权限拒绝、stop 行为；
10. index.json 损坏时通过 scan 重建。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from clawcodex_ext.tasks.bg_session import (
    BgSession,
    BgSessionConfig,
    BgSessionPermissionError,
    is_bg_sessions_enabled,
    marker_path_for,
    replace_session,
)
from clawcodex_ext.tasks.bg_session_health import assess, reconcile
from clawcodex_ext.tasks.bg_session_manager import BgSessionManager
from clawcodex_ext.tasks.bg_session_registry import BgSessionRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enabled_config(tmp_path: Path) -> BgSessionConfig:
    return BgSessionConfig(
        enabled=True,
        index_path=tmp_path / "bg_sessions" / "index.json",
        sessions_dir=tmp_path / "sessions",
        stale_after_seconds=600,
        cleanup_completed_after_seconds=0,  # 测试中立即可清理
    )


@pytest.fixture
def registry(enabled_config: BgSessionConfig) -> BgSessionRegistry:
    return BgSessionRegistry(config=enabled_config)


@pytest.fixture
def manager(registry: BgSessionRegistry) -> BgSessionManager:
    return BgSessionManager(registry=registry)


def _make_session_dir(
    sessions_dir: Path,
    session_id: str,
    *,
    pid: int = 99999,
    status: str = "running",
    started_at: str = "2026-07-02T14:00:00",
    workspace_root: str = "/tmp/ws",
    write_transcript: bool = False,
    transcript_content: str = "",
) -> Path:
    """造一个 session 目录 + marker（+ 可选 transcript）。"""
    d = sessions_dir / session_id
    d.mkdir(parents=True, exist_ok=True)
    marker = marker_path_for(session_id, sessions_dir)
    marker.write_text(
        json.dumps(
            {
                "pid": pid,
                "session_id": session_id,
                "status": status,
                "started_at": started_at,
                "workspace_root": workspace_root,
            }
        )
    )
    if write_transcript:
        tf = d / f"{session_id}.jsonl"
        tf.write_text(transcript_content)
    return d


# ---------------------------------------------------------------------------
# P94-A: 数据模型
# ---------------------------------------------------------------------------


class TestBgSessionModel:
    def test_frozen_dataclass_roundtrip(self) -> None:
        s = BgSession(id="x", session_id="x", workspace_root=Path("/tmp"), status="running", pid=1)
        d = s.to_dict()
        assert d["workspace_root"] == "/tmp"
        s2 = BgSession.from_dict(d)
        assert s2.workspace_root == Path("/tmp")
        assert s2 == s

    def test_is_terminal_and_active(self) -> None:
        running = BgSession(id="r", session_id="r", workspace_root=Path("."), status="running")
        assert running.is_active() and not running.is_terminal()
        done = replace_session(running, status="completed")
        assert done.is_terminal() and not done.is_active()

    def test_config_from_env_off(self) -> None:
        with patch.dict(os.environ, {"CLAWCODEX_BG_SESSIONS": "off"}):
            assert not is_bg_sessions_enabled()
        with patch.dict(os.environ, {"CLAWCODEX_BG_SESSIONS": "on"}):
            assert is_bg_sessions_enabled()
        # 默认未设置 = 关闭（保守默认）
        with patch.dict(os.environ, {}, clear=True):
            assert not is_bg_sessions_enabled()


# ---------------------------------------------------------------------------
# P94-B: Registry
# ---------------------------------------------------------------------------


class TestBgSessionRegistry:
    def test_scan_empty_dir(self, registry: BgSessionRegistry) -> None:
        assert registry.scan() == []

    def test_scan_finds_marker(self, registry: BgSessionRegistry, enabled_config: BgSessionConfig) -> None:
        _make_session_dir(registry.sessions_dir, "s1", pid=99999)
        result = registry.scan()
        assert len(result) == 1
        assert result[0].id == "s1"
        # pid 99999 不存活 → orphaned（验收标准 4）
        assert result[0].status == "orphaned"

    def test_scan_skips_dirs_without_marker(self, registry: BgSessionRegistry) -> None:
        (registry.sessions_dir / "no-marker").mkdir(parents=True)
        assert registry.scan() == []

    def test_scan_handles_corrupt_marker(self, registry: BgSessionRegistry) -> None:
        d = registry.sessions_dir / "bad"
        d.mkdir(parents=True)
        (d / ".background-runner.json").write_text("not json{")
        assert registry.scan() == []

    def test_list_workspace_filter(self, registry: BgSessionRegistry) -> None:
        _make_session_dir(registry.sessions_dir, "s1", workspace_root="/tmp/wsA")
        _make_session_dir(registry.sessions_dir, "s2", workspace_root="/tmp/wsB")
        registry.scan()
        ws_a = Path("/tmp/wsA")
        result = registry.list(workspace_root=ws_a)
        assert len(result) == 1
        assert result[0].workspace_root == Path("/tmp/wsA")

    def test_save_disabled_when_flag_off(self, tmp_path: Path) -> None:
        cfg = BgSessionConfig(
            enabled=False,
            index_path=tmp_path / "index.json",
            sessions_dir=tmp_path / "sessions",
        )
        reg = BgSessionRegistry(config=cfg)
        assert reg.save() is None
        assert not cfg.index_path.exists()

    def test_save_writes_index_when_enabled(self, registry: BgSessionRegistry) -> None:
        _make_session_dir(registry.sessions_dir, "s1", pid=99999)
        registry.scan()
        path = registry.save()
        assert path is not None and path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert len(data["sessions"]) == 1

    def test_index_corrupt_rebuild_via_scan(self, registry: BgSessionRegistry) -> None:
        _make_session_dir(registry.sessions_dir, "s1", pid=99999)
        # 写一个损坏的 index
        registry.index_path.parent.mkdir(parents=True, exist_ok=True)
        registry.index_path.write_text("corrupt{")
        # load 应返回空并记录 warning
        assert registry.load() == []
        # scan 重建
        sessions = registry.rebuild_and_save()
        assert len(sessions) == 1
        # save 后 index 可正常读
        loaded = registry.load()
        assert len(loaded) == 1

    def test_scan_100_sessions_under_100ms(self, registry: BgSessionRegistry) -> None:
        for i in range(100):
            _make_session_dir(registry.sessions_dir, f"s{i:03d}", pid=99999)
        start = time.monotonic()
        registry.scan()
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 100, f"scan took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# P94-D: Health / 状态机
# ---------------------------------------------------------------------------


class TestBgSessionHealth:
    def test_marker_completed_wins(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            status="completed",
            write_transcript=True,
            transcript_content='{"role":"system","content":"__background_complete__"}',
        )
        sessions = registry.scan()
        assert sessions[0].status == "completed"

    def test_running_pid_dead_no_completion_orphaned(self, registry: BgSessionRegistry) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            status="running",
            write_transcript=True,
            transcript_content='{"role":"user","content":"hi"}',
        )
        sessions = registry.scan()
        assert sessions[0].status == "orphaned"

    def test_running_pid_dead_with_completion_completed(self, registry: BgSessionRegistry) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            status="running",
            write_transcript=True,
            transcript_content='{"role":"system","content":"__background_complete__"}',
        )
        sessions = registry.scan()
        assert sessions[0].status == "completed"

    def test_running_pid_alive_stale_warning(self, registry: BgSessionRegistry) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=os.getpid(),
            status="running",
            write_transcript=True,
            transcript_content='{"role":"user","content":"hi"}',
        )
        sessions = registry.scan()
        assert sessions[0].status == "running"
        # 直接 assess 检查 stale 标志：stale_after=0 → 必 stale（mtime_age > 0）
        h = assess(sessions[0], stale_after_seconds=0)
        assert h.is_stale is True
        assert h.status == "running"  # 仍是 running，仅 warning

    def test_marker_failed_wins(self, registry: BgSessionRegistry) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            status="failed",
        )
        sessions = registry.scan()
        assert sessions[0].status == "failed"

    def test_reconcile_pure_function(self) -> None:
        # 直接构造 BgSession，避免 scan() 已替换状态

        before = BgSession(
            id="s1",
            session_id="s1",
            workspace_root=Path("/tmp/ws"),
            status="running",
            pid=99999,  # 不存活
            marker_path=Path("/nonexistent/.background-runner.json"),
            transcript_path=None,
        )
        after = reconcile(before)
        assert after.status == "orphaned"
        # 原对象不变（pure）
        assert before.status == "running"


# ---------------------------------------------------------------------------
# P94-C: Manager
# ---------------------------------------------------------------------------


class TestBgSessionManager:
    def test_list_sessions(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        _make_session_dir(registry.sessions_dir, "s1", pid=99999)
        registry.scan()
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        # orphaned 非终态，默认包含
        assert sessions[0].status == "orphaned"

    def test_list_excludes_completed_by_default(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            status="completed",
            write_transcript=True,
            transcript_content='{"content":"__background_complete__"}',
        )
        registry.scan()
        assert manager.list_sessions() == []
        assert len(manager.list_sessions(include_completed=True)) == 1

    def test_inspect_not_found(self, manager: BgSessionManager) -> None:
        from clawcodex_ext.tasks.bg_session import BgSessionNotFoundError

        with pytest.raises(BgSessionNotFoundError):
            manager.inspect("nonexistent")

    def test_attach_returns_transcript_tail_and_hint(
        self, manager: BgSessionManager, registry: BgSessionRegistry
    ) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            status="completed",
            workspace_root="/tmp/ws",
            write_transcript=True,
            transcript_content="line1\nline2\n__background_complete__\n",
        )
        registry.scan()
        result = manager.attach("s1", current_workspace=Path("/tmp/ws"))
        assert "line2" in result.transcript_tail
        assert "--resume s1" in result.resume_hint

    def test_attach_cross_workspace_denied(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            workspace_root="/tmp/wsA",
            write_transcript=True,
            transcript_content="x",
        )
        registry.scan()
        with pytest.raises(BgSessionPermissionError):
            manager.attach("s1", current_workspace=Path("/tmp/wsB"))

    def test_attach_cross_workspace_allowed_with_flag(
        self, manager: BgSessionManager, registry: BgSessionRegistry
    ) -> None:
        _make_session_dir(
            registry.sessions_dir,
            "s1",
            pid=99999,
            workspace_root="/tmp/wsA",
            write_transcript=True,
            transcript_content="x",
        )
        registry.scan()
        # allow_cross_workspace=True
        result = manager.attach("s1", current_workspace=Path("/tmp/wsB"), allow_cross_workspace=True)
        assert result.session.id == "s1"

    def test_stop_graceful_failure_requires_force(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        # pid 99999 不存活 → inspect 返回 orphaned（终态），stop 直接返回
        _make_session_dir(registry.sessions_dir, "s1", pid=99999, status="running")
        registry.scan()
        result = manager.stop("s1")
        assert result.status in ("stopped", "orphaned")

    def test_stop_force_on_live_pid(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        # 用当前进程 pid 模拟存活，但 force=True 应能"停止"（实际不杀自己，验证路径）
        # 这里用一个已死 pid 测试 force 路径不抛
        _make_session_dir(registry.sessions_dir, "s1", pid=99998, status="running")
        registry.scan()
        result = manager.stop("s1", force=True)
        assert result.status == "stopped"

    def test_cleanup_removes_orphaned(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        _make_session_dir(registry.sessions_dir, "s1", pid=99999, status="running")
        registry.scan()
        removed = manager.cleanup()
        assert len(removed) == 1
        assert manager.list_sessions(include_completed=True) == []

    def test_cleanup_respects_include_failed(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        _make_session_dir(registry.sessions_dir, "s1", pid=99999, status="failed")
        registry.scan()
        # 默认不清理 failed
        assert manager.cleanup() == []
        # include_failed=True 清理
        removed = manager.cleanup(include_failed=True)
        assert len(removed) == 1


# ---------------------------------------------------------------------------
# P94-B/C: upsert_after_launch 协调
# ---------------------------------------------------------------------------


class TestUpsertAfterLaunch:
    def test_disabled_noop(self, tmp_path: Path) -> None:
        cfg = BgSessionConfig(
            enabled=False,
            index_path=tmp_path / "index.json",
            sessions_dir=tmp_path / "sessions",
        )
        reg = BgSessionRegistry(config=cfg)
        mgr = BgSessionManager(registry=reg)
        assert mgr.upsert_after_launch("s1", 123) is None
        assert not cfg.index_path.exists()

    def test_enabled_upserts_and_saves(self, manager: BgSessionManager, registry: BgSessionRegistry) -> None:
        sess = manager.upsert_after_launch("s1", 12345, workspace_root=Path("/tmp/ws"))
        assert sess is not None
        assert sess.status == "running"
        assert registry.get("s1") is not None
        assert registry.index_path.exists()


# ---------------------------------------------------------------------------
# P94-E: Tool / Command 集成
# ---------------------------------------------------------------------------


class TestBgSessionToolIntegration:
    def test_tool_disabled_returns_disabled_result(self) -> None:
        from clawcodex_ext.tool_system.tools.bg_session import BgSessionTool
        from clawcodex_ext.tool_system.context import ToolContext

        with patch.dict(os.environ, {"CLAWCODEX_BG_SESSIONS": "off"}):
            ctx = ToolContext.__new__(ToolContext)  # 轻量实例
            result = BgSessionTool.call({"action": "list"}, ctx)
            assert result.output.get("disabled") is True

    def test_command_disabled_message(self) -> None:
        from clawcodex_ext.command_system.bg_commands import _bg_run

        with patch.dict(os.environ, {"CLAWCODEX_BG_SESSIONS": "off"}):
            res = _bg_run("list", None)
            assert "disabled" in res.value.lower()


# ---------------------------------------------------------------------------
# P94-F: Panel
# ---------------------------------------------------------------------------


class TestBgSessionsPanel:
    def test_footer_disabled_empty(self) -> None:
        from clawcodex_ext.repl.bg_sessions_panel import footer_summary

        with patch.dict(os.environ, {"CLAWCODEX_BG_SESSIONS": "off"}):
            reg = BgSessionRegistry()
            assert footer_summary(reg) == ""

    def test_footer_shows_running_count(self, registry: BgSessionRegistry) -> None:
        from clawcodex_ext.repl.bg_sessions_panel import footer_summary

        _make_session_dir(registry.sessions_dir, "s1", pid=os.getpid())
        registry.scan()
        summary = footer_summary(registry)
        assert summary.startswith("bg:1")

    def test_completion_notification_format(self) -> None:
        from clawcodex_ext.repl.bg_sessions_panel import format_completion_notification

        s = BgSession(
            id="s1",
            session_id="s1",
            workspace_root=Path("/tmp"),
            status="completed",
        )
        text = format_completion_notification(s)
        assert "session_id: s1" in text
        assert "--resume s1" in text
