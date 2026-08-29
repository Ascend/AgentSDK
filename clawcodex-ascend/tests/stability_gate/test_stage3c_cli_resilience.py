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

"""P0 P1 Tests for stage3c cli resilience."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _run_cli_env(env_override: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    """Run ``python -m src.cli`` with custom environment."""
    base_env = os.environ.copy()
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
    """Tests for TestStage3cCliEnvironment."""

    def test_cli_help_no_home(self):
        """Verify cli help no home."""
        env = {
            "HOME": "/nonexistent_home_for_test",
            "USERPROFILE": "Z:\\nonexistent_userprofile_for_test",
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        proc = _run_cli_env(env, "--help")
        assert proc.returncode == 0, f"--help with no HOME failed: rc={proc.returncode}, stderr={proc.stderr!r}"

    def test_cli_version_no_home(self):
        """Verify cli version no home."""
        env = {
            "HOME": "/nonexistent_home_for_test",
            "USERPROFILE": "Z:\\nonexistent_userprofile_for_test",
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        proc = _run_cli_env(env, "--version")
        assert proc.returncode == 0, f"--version with no HOME failed: rc={proc.returncode}, stderr={proc.stderr!r}"

    def test_cli_help_empty_env(self):
        """Verify cli help empty env."""
        env = {
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin"),
            "LANG": "",
            "LC_ALL": "",
        }
        proc = _run_cli_env(env, "--help")
        assert proc.returncode == 0, f"--help with empty LANG failed: rc={proc.returncode}, stderr={proc.stderr!r}"

    def test_cli_invalid_subcommand_no_traceback(self):
        """Verify cli invalid subcommand no traceback."""
        env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ.get("PATH", "/usr/bin")}
        proc = _run_cli_env(env, "nonexistent-cmd-that-should-not-exist")
        assert "Traceback" not in proc.stderr, f"invalid command produced traceback:\n{proc.stderr}"


class TestStage3cCliSignal:
    """P0 Tests for TestStage3cCliSignal."""

    def test_cli_sigint_no_sigabrt(self):
        """Verify cli sigint no sigabrt."""
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
        time.sleep(0.3)
        if sys.platform != "win32":
            proc.send_signal(signal.SIGINT)
        else:
            # CTRL_C_EVENT is broadcast to ALL processes in the console
            # group (including pytest itself).  CTRL_BREAK_EVENT goes only
            # to the target process group, which we ensure by creating the
            # subprocess in its own group via CREATE_NEW_PROCESS_GROUP.
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]

        _stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode != -6, f"SIGINT caused SIGABRT (-6): stderr={stderr!r}"

    def test_cli_help_no_traceback_on_normal_exit(self):
        """Verify cli help no traceback on normal exit."""
        proc = _run_cli_env(
            {"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ.get("PATH", "/usr/bin")},
            "--help",
        )
        assert "Traceback" not in proc.stderr, f"normal --help produced traceback:\n{proc.stderr}"
        assert "Error:" not in proc.stderr, f"normal --help produced error:\n{proc.stderr}"
