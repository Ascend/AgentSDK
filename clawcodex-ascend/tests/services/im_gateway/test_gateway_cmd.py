#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for the Gateway daemon lifecycle (PID/lock/stale socket/health)
and CLI routing for the flattened `gateway` command.
"""
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from clawcodex_ext.cli.gateway_cmd.commands import run_gateway_command
from extensions.im_gateway.server import (
    DaemonPaths,
    GatewayDaemon,
    _channel_status_ready,
    acquire_lock,
    cleanup_stale,
    is_pid_alive,
    read_health,
    read_pid,
    write_health,
    write_pid,
)


def test_historical_default_state_dir_import_is_preserved() -> None:
    from clawcodex_ext.services.im_gateway.config import DEFAULT_STATE_DIR as canonical
    from extensions.im_gateway.server import DEFAULT_STATE_DIR as historical

    assert historical == canonical


def test_status_not_running(tmp_path) -> None:
    daemon = GatewayDaemon(DaemonPaths.for_state_dir(tmp_path))
    rc = daemon.status()
    assert rc == 0  # not-running is a valid status


def test_stale_socket_cleanup(tmp_path) -> None:
    paths = DaemonPaths.for_state_dir(tmp_path)
    # stale PID pointing at a dead process + a leftover socket
    paths.pid_file.write_text("999999\n", encoding="utf-8")
    paths.sock_file.write_text("", encoding="utf-8")
    assert cleanup_stale(paths) is True
    assert not paths.pid_file.exists()
    assert not paths.sock_file.exists()


def test_stale_socket_kept_when_pid_alive(tmp_path) -> None:
    paths = DaemonPaths.for_state_dir(tmp_path)
    paths.pid_file.write_text(f"{__import__('os').getpid()}\n", encoding="utf-8")
    paths.sock_file.write_text("", encoding="utf-8")
    # current process is alive → not stale
    assert cleanup_stale(paths) is False
    assert paths.pid_file.exists()


def test_acquire_lock_single_instance(tmp_path) -> None:
    paths = DaemonPaths.for_state_dir(tmp_path)
    fd1 = acquire_lock(paths)
    assert fd1 is not None
    fd2 = acquire_lock(paths)
    assert fd2 is None  # already locked
    os.close(fd1)


def test_is_pid_alive() -> None:
    assert is_pid_alive(os.getpid()) is True
    assert is_pid_alive(999999) is False
    assert is_pid_alive(0) is False


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("connected", True),
        ("logged_in", True),
        ("websocket:connected", True),
        ("websocket:disconnected", False),
        ("websocket:reconnecting", False),
        ("websocket:retrying", False),
    ],
)
def test_channel_status_ready_uses_exact_allowlist(status, expected) -> None:
    assert _channel_status_ready(status) is expected


@pytest.mark.parametrize(
    "payload",
    [
        "null",
        "[]",
        '"invalid"',
        '{"started_at": "invalid"}',
        '{"channels": ["wechat", 1]}',
        '{"channel_status": {"wechat": 1}}',
    ],
)
def test_read_health_rejects_invalid_schema(tmp_path, payload: str) -> None:
    paths = DaemonPaths.for_state_dir(tmp_path)
    paths.health_file.write_text(payload, encoding="utf-8")
    assert read_health(paths) is None


def test_write_health_is_atomic_and_cleans_temp_after_replace_failure(tmp_path, monkeypatch) -> None:
    from extensions.im_gateway import server as srv

    paths = DaemonPaths.for_state_dir(tmp_path)
    write_health(paths)
    initial = paths.health_file.read_text(encoding="utf-8")

    def fail_replace(_source, _target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(srv.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_health(paths, channels=["next"])

    assert paths.health_file.read_text(encoding="utf-8") == initial
    assert not list(paths.state_dir.glob(".health.json.*.tmp"))


def test_concurrent_health_writers_publish_one_complete_document(tmp_path) -> None:
    paths = DaemonPaths.for_state_dir(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: write_health(paths, channels=[str(index)]), range(20)))

    health = read_health(paths)
    assert health is not None
    assert health["channels"] in ([str(index)] for index in range(20))
    assert not list(paths.state_dir.glob(".health.json.*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not available on Windows")
def test_daemon_state_artifacts_are_private(tmp_path) -> None:
    from extensions.im_gateway import server as srv

    paths = DaemonPaths.for_state_dir(tmp_path / "state")
    write_pid(paths, 123)
    write_health(paths)
    handler = srv._PrivateRotatingFileHandler(paths.log_file, maxBytes=1024, backupCount=1)
    handler.close()
    fd = acquire_lock(paths)
    assert fd is not None
    os.close(fd)

    assert paths.state_dir.stat().st_mode & 0o777 == 0o700
    for path in (paths.pid_file, paths.health_file, paths.log_file, paths.lock_file):
        assert path.stat().st_mode & 0o777 == 0o600


def test_daemon_fails_closed_without_posix_flock(tmp_path, monkeypatch, capsys) -> None:
    import asyncio

    from extensions.im_gateway import server as srv

    paths = DaemonPaths.for_state_dir(tmp_path)
    monkeypatch.setattr(srv, "HAS_FLOCK", False)

    assert acquire_lock(paths) is None
    assert asyncio.run(srv.serve(paths)) == 1
    assert "requires POSIX flock support" in capsys.readouterr().err


@pytest.mark.integration
def test_daemon_start_status_stop_smoke(tmp_path) -> None:
    """Start the real daemon subprocess, verify PID/socket/health, stop it.

    Marked integration (needs subprocess + POSIX UDS). Run in WSL.
    """
    daemon = GatewayDaemon(DaemonPaths.for_state_dir(tmp_path))
    try:
        rc = daemon.start()
        assert rc == 0, f"daemon failed to start; see {daemon.paths.log_file}"
        pid = read_pid(daemon.paths)
        assert pid is not None and is_pid_alive(pid)
        assert daemon.paths.sock_file.exists()
        # health file written
        health = daemon.paths.health_file.read_text(encoding="utf-8")
        assert '"running": true' in health
    finally:
        daemon.stop()
    # after stop, cleaned up
    remaining_pid = read_pid(daemon.paths)
    assert remaining_pid is None or not is_pid_alive(remaining_pid)


# -- flattened gateway routing ------------------------------------------------


def test_gateway_no_args_prints_usage(capsys) -> None:
    """`gateway` (no args) prints usage and returns 0."""
    rc = run_gateway_command([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage:" in out


def test_gateway_help_prints_usage(capsys) -> None:
    """`gateway help` prints usage and returns 0."""
    rc = run_gateway_command(["help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage:" in out


def test_gateway_unknown_subcommand_errors(capsys) -> None:
    """`gateway <unknown>` reports an error."""
    rc = run_gateway_command(["bogus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown gateway subcommand" in err


def test_gateway_server_start_errors(capsys) -> None:
    """`gateway server start` is no longer valid — 'server' is an unknown verb."""
    rc = run_gateway_command(["server", "start"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown gateway subcommand" in err


def test_gateway_channels_status_errors(capsys) -> None:
    """`gateway channels status` is no longer valid — 'channels' is an unknown verb."""
    rc = run_gateway_command(["channels", "status"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown gateway subcommand" in err


def test_serve_writes_pid_before_gateway_start(tmp_path, monkeypatch) -> None:
    """PID file must be written BEFORE channel adapters start.

    A hanging / crashing adapter start used to block ``await gateway.start()``
    so ``write_pid`` was never reached — the daemon became invisible to
    ``stop()``/``restart()`` and left an orphan holding the flock. Asserting
    the PID exists even when ``gateway.start`` raises proves the ordering.
    """
    import asyncio

    from extensions.im_gateway import server as srv

    paths = DaemonPaths.for_state_dir(tmp_path)
    seen_pid_at_start: list[int | None] = []

    class _CrashingGateway:
        def __init__(self, *a, **kw) -> None:
            pass

        async def start(self) -> None:
            # If write_pid ran before us, the PID file already exists.
            seen_pid_at_start.append(read_pid(paths))
            raise RuntimeError("adapter start crashed")

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(srv, "MessageGateway", _CrashingGateway)
    # load_config needs a config file; write a minimal one.
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    (paths.state_dir / "channels.yaml").write_text("enabled: true\nchannels: []\n", encoding="utf-8")

    # gateway.start raises → serve propagates the RuntimeError.
    with pytest.raises(RuntimeError, match="adapter start crashed"):
        asyncio.run(srv.serve(paths, log_level=40))  # CRITICAL = quiet

    # The PID file was already written when gateway.start ran (proving
    # write_pid precedes adapter start), then cleaned up on the failure path.
    assert seen_pid_at_start == [__import__("os").getpid()]
    assert seen_pid_at_start[0] is not None
    assert not paths.pid_file.exists()


def test_gateway_start_reports_retrying_channel_as_degraded_success(tmp_path, monkeypatch, capsys) -> None:
    from extensions.im_gateway import server as srv

    paths = DaemonPaths.for_state_dir(tmp_path)
    paths.log_file.write_text("", encoding="utf-8")

    class _FakeProc:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(srv.subprocess, "Popen", lambda *a, **kw: _FakeProc())
    read_pid_values = iter([None, 12345])
    monkeypatch.setattr(srv, "read_pid", lambda _paths: next(read_pid_values, 12345))
    monkeypatch.setattr(srv, "is_pid_alive", lambda pid: pid == 12345)
    monkeypatch.setattr(
        srv,
        "read_health",
        lambda _paths: {
            "started_at": time.time(),
            "channels": ["feishu"],
            "channel_status": {"feishu": "websocket:retrying"},
        },
    )
    monkeypatch.setattr(srv, "startup_health_wait_seconds", lambda _paths: 0.1)

    rc = GatewayDaemon(paths).start()
    captured = capsys.readouterr()

    assert rc == 0
    assert "Gateway daemon started" in captured.out
    assert "channel feishu: websocket:retrying" in captured.err
    assert "retrying in background" in captured.err
    assert "NOT connected" not in captured.err
    assert "messages may be dropped" not in captured.err


def test_startup_health_wait_seconds_includes_feishu_sdk_import_buffer(tmp_path) -> None:
    from extensions.im_gateway import server as srv

    paths = DaemonPaths.for_state_dir(tmp_path)
    (paths.state_dir / "channels.yaml").write_text(
        "\n".join(
            [
                "enabled: true",
                "channels:",
                "  - type: feishu",
                '    webhook_url: ""',
                "    name: feishu",
                "    enabled: true",
                "    extra:",
                "      connection_mode: websocket",
                "      app_id: cli_app",
                "      app_secret: secret",
                "      websocket:",
                "        startup_connect_timeout_seconds: 7.5",
            ]
        ),
        encoding="utf-8",
    )

    assert srv.startup_health_wait_seconds(paths) == pytest.approx(157.5)


def test_gateway_start_with_name_errors(capsys) -> None:
    """`gateway start <name>` is invalid — start takes no channel name."""
    rc = run_gateway_command(["start", "wechat"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "start takes no channel name" in err.lower()


def test_gateway_stop_with_name_errors(capsys) -> None:
    """`gateway stop <name>` is invalid — stop takes no channel name."""
    rc = run_gateway_command(["stop", "wechat"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "stop takes no channel name" in err.lower()


def test_gateway_setup_uses_state_dir_then_restarts_daemon(tmp_path, monkeypatch) -> None:
    """A successful setup writes the selected config and applies it via restart."""
    wizard_calls: list[str | None] = []
    restart_calls: list[bool] = []

    def _fake_wizard(path: str | None = None, *, input_fn=None) -> int:
        wizard_calls.append(path)
        return 0

    class _FakeDaemon:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def restart(self, verbose=False):
            restart_calls.append(verbose)
            return 0

    from clawcodex_ext.cli.channels_cmd import commands as ch

    monkeypatch.setattr(ch, "run_wizard", _fake_wizard)
    monkeypatch.setattr("extensions.im_gateway.server.GatewayDaemon", _FakeDaemon)
    rc = run_gateway_command(["setup", "--state-dir", str(tmp_path), "--verbose"])
    assert rc == 0
    assert wizard_calls == [str(tmp_path / "channels.yaml")]
    assert restart_calls == [True]


def test_gateway_setup_failure_does_not_restart(monkeypatch) -> None:
    restart_calls: list[bool] = []

    class _FakeDaemon:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def restart(self, verbose=False):
            restart_calls.append(verbose)
            return 0

    from clawcodex_ext.cli.channels_cmd import commands as ch

    monkeypatch.setattr(ch, "run_wizard", lambda _path: 1)
    monkeypatch.setattr("extensions.im_gateway.server.GatewayDaemon", _FakeDaemon)

    assert run_gateway_command(["setup"]) == 1
    assert restart_calls == []


def test_gateway_restart_channel(monkeypatch) -> None:
    """`gateway restart <name>` calls restart_channel."""
    calls: list[tuple] = []

    def _fake_restart(name: str, *, state_dir: str | None = None) -> int:
        calls.append((name, state_dir))
        return 0

    from clawcodex_ext.cli.channels_cmd import commands as ch

    monkeypatch.setattr(ch, "restart_channel", _fake_restart)
    rc = run_gateway_command(["restart", "wechat"])
    assert rc == 0
    assert calls == [("wechat", None)]


def test_gateway_restart_daemon(monkeypatch) -> None:
    """`gateway restart` (no name) calls daemon.restart."""
    calls: list[bool] = []

    class _FakeDaemon:
        def __init__(self, *a, **kw):
            pass

        def restart(self, verbose=False):
            calls.append(verbose)
            return 0

    monkeypatch.setattr("extensions.im_gateway.server.GatewayDaemon", _FakeDaemon)
    rc = run_gateway_command(["restart"])
    assert rc == 0
    assert calls == [False]


def test_gateway_status_channel(monkeypatch, capsys) -> None:
    """`gateway status <name>` calls format_status for that name."""
    calls: list[tuple] = []

    def _fake_format_status(path=None, name=None, *, state_dir=None) -> str:
        calls.append((path, name, state_dir))
        return f"STATUS:{name}"

    from clawcodex_ext.cli.channels_cmd import commands as ch

    monkeypatch.setattr(ch, "format_status", _fake_format_status)
    rc = run_gateway_command(["status", "wechat"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "STATUS:wechat" in out
    assert calls == [(None, "wechat", None)]


def test_gateway_status_unified(monkeypatch, capsys) -> None:
    """Bare `gateway status` prints daemon status THEN all-channels status."""
    daemon_status_calls: list[int] = []
    format_status_calls: list[tuple] = []

    class _FakeDaemon:
        def __init__(self, *a, **kw):
            pass

        def status(self):
            daemon_status_calls.append(1)
            print("DAEMON: running", end="")
            return 0

    def _fake_format_status(path=None, name=None, *, state_dir=None) -> str:
        format_status_calls.append((path, name, state_dir))
        return "CHANNELS: all"

    monkeypatch.setattr("extensions.im_gateway.server.GatewayDaemon", _FakeDaemon)
    from clawcodex_ext.cli.channels_cmd import commands as ch

    monkeypatch.setattr(ch, "format_status", _fake_format_status)

    rc = run_gateway_command(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DAEMON: running" in out
    assert "CHANNELS: all" in out
    assert len(daemon_status_calls) == 1
    assert format_status_calls == [(None, None, None)]


def test_gateway_disconnect_channel(monkeypatch) -> None:
    """`gateway disconnect <name>` calls _disconnect_gateway_connection."""
    calls: list[tuple] = []

    def _fake_disconnect(name: str, *, state_dir: str | None = None) -> int:
        calls.append((name, state_dir))
        return 0

    from clawcodex_ext.cli.channels_cmd import commands as ch

    monkeypatch.setattr(ch, "_disconnect_gateway_connection", _fake_disconnect)
    rc = run_gateway_command(["disconnect", "wechat"])
    assert rc == 0
    assert calls == [("wechat", None)]


def test_gateway_login_channel(monkeypatch) -> None:
    """`gateway login <name>` calls wechat_login."""
    calls: list[tuple] = []

    def _fake_login(name: str, *, state_dir: str | None = None) -> int:
        calls.append((name, state_dir))
        return 0

    from clawcodex_ext.cli.channels_cmd import commands as ch

    monkeypatch.setattr(ch, "wechat_login", _fake_login)
    rc = run_gateway_command(["login", "wechat"])
    assert rc == 0
    assert calls == [("wechat", None)]


def test_gateway_wizard_is_unknown(capsys) -> None:
    """`gateway wizard` is no longer valid — only `setup` runs the wizard."""
    rc = run_gateway_command(["wizard"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown gateway subcommand" in err
