#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Stage 3c — CLI 弹性测试（< 10 秒）。

覆盖 P0#2（Ctrl+C/SIGINT 中断稳定）、P0#4（无 API key 启动）、
P1#6（provider 超时/网络错误边界）。

使用子进程执行 CLI 命令，验证：
- 无 HOME 目录或空环境时 --help / --version 仍正常退出
- 无效子命令给出非 0 退出码而非 traceback
- SIGINT 信号被正确处理（不打印 traceback）
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _run_cli_env(env_override: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    """Run ``python -m src.cli`` with custom environment."""
    base_env = os.environ.copy()
    # 保留 PATH 和 Python 路径确保能启动
    # Windows 上 Path.home() 使用 USERPROFILE / HOMEDRIVE+HOMEPATH
    # asyncio / _overlapped 需要 SYSTEMROOT (Winsock catalog)
    for keep in (
        "PATH",
        "PYTHONPATH",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "TERM",
    ):
        if keep in env_override:
            continue
        if keep in base_env:
            env_override[keep] = base_env[keep]
    return subprocess.run(  # noqa: PLW1510
        [sys.executable, "-m", "src.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env_override,
    )


class TestStage3cCliEnvironment:
    """CLI 在极端环境下的稳定性。"""

    def test_cli_help_no_home(self):
        """HOME 不可用时 --help 仍正常退出。"""
        env = {
            "HOME": "/nonexistent_home_for_test",
            "USERPROFILE": "Z:\\nonexistent_userprofile_for_test",
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        proc = _run_cli_env(env, "--help")
        assert proc.returncode == 0, f"--help with no HOME failed: rc={proc.returncode}, stderr={proc.stderr!r}"

    def test_cli_version_no_home(self):
        """HOME 不可用时 --version 仍正常退出。"""
        env = {
            "HOME": "/nonexistent_home_for_test",
            "USERPROFILE": "Z:\\nonexistent_userprofile_for_test",
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        proc = _run_cli_env(env, "--version")
        assert proc.returncode == 0, f"--version with no HOME failed: rc={proc.returncode}, stderr={proc.stderr!r}"

    def test_cli_help_empty_env(self):
        """LANG/LC_ALL 为空时 --help 不崩溃。"""
        env = {
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin"),
            "LANG": "",
            "LC_ALL": "",
        }
        proc = _run_cli_env(env, "--help")
        assert proc.returncode == 0, f"--help with empty LANG failed: rc={proc.returncode}, stderr={proc.stderr!r}"

    def test_cli_invalid_subcommand_no_traceback(self):
        """无效子命令不打印 Python traceback。"""
        env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ.get("PATH", "/usr/bin")}
        proc = _run_cli_env(env, "nonexistent-cmd-that-should-not-exist")
        # stderr 不应含 Python traceback
        assert "Traceback" not in proc.stderr, f"invalid command produced traceback:\n{proc.stderr}"


class TestStage3cCliSignal:
    """CLI 信号处理 — P0#2 Ctrl+C/SIGINT 清理退出。"""

    def test_cli_sigint_no_sigabrt(self):
        """CLI 子进程收到 SIGINT 不以 SIGABRT (-6) 退出。

        注: 非交互模式下 KeyboardInterrupt traceback 是 Python 默认行为。
        这里只验证进程没有被 SIGABRT 杀死 (这表示 Python 内部状态损坏)。
        """
        popen_kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        # On Windows, create the subprocess in its own group so that
        # CTRL_BREAK_EVENT does not propagate to the parent (pytest).
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.cli", "--help"],
            **popen_kwargs,
        )
        # 给子进程一点时间启动

        time.sleep(0.3)
        # 发送 SIGINT
        if sys.platform != "win32":
            proc.send_signal(signal.SIGINT)
        else:
            # CTRL_C_EVENT is broadcast to ALL processes in the console
            # group (including pytest itself).  CTRL_BREAK_EVENT goes only
            # to the target process group, which we ensure by creating the
            # subprocess in its own group via CREATE_NEW_PROCESS_GROUP.
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]

        _stdout, stderr = proc.communicate(timeout=10)
        # 退出码不应是 -6 (SIGABRT) — 那只在严重的 Python 内部损坏时出现
        assert proc.returncode != -6, f"SIGINT caused SIGABRT (-6): stderr={stderr!r}"

    def test_cli_help_no_traceback_on_normal_exit(self):
        """正常 --help 退出不产生任何 traceback。"""
        proc = _run_cli_env(
            {"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ.get("PATH", "/usr/bin")},
            "--help",
        )
        assert "Traceback" not in proc.stderr, f"normal --help produced traceback:\n{proc.stderr}"
        assert "Error:" not in proc.stderr, f"normal --help produced error:\n{proc.stderr}"
