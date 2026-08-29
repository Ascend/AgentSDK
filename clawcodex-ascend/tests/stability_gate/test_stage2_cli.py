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

"""Tests for stage2 cli."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``python -m src.cli`` with *args in a subprocess."""
    return subprocess.run(  # noqa: PLW1510
        [sys.executable, "-m", "src.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestStage2CliSmoke:
    """Tests for TestStage2CliSmoke."""

    def test_cli_help_exits_0(self):
        proc = _run_cli("--help")
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        assert "usage:" in proc.stdout.lower() or "usage:" in proc.stderr.lower()

    def test_cli_help_contains_subcommands(self):
        proc = _run_cli("--help")
        output = proc.stdout + proc.stderr
        for keyword in ("provider", "model", "schedule", "print"):
            assert keyword in output, f"Expected {keyword!r} in --help output"

    def test_cli_version_exits_0(self):
        proc = _run_cli("--version")
        assert proc.returncode == 0
        assert len(proc.stdout) > 0 or len(proc.stderr) > 0

    def test_cli_provider_list_exits_0(self):
        proc = _run_cli("provider", "list")
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        output = proc.stdout + proc.stderr
        for name in ("anthropic", "openai"):
            assert name.lower().replace("-", "") in output.lower().replace("-", ""), (
                f"Expected {name!r} in provider list output"
            )

    def test_cli_model_list_exits_0(self):
        proc = _run_cli("model", "list")
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        assert len(proc.stdout.strip()) > 0

    def test_cli_telemetry_status_exits_0(self):
        proc = _run_cli("telemetry", "status")
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        assert "Telemetry status" in proc.stdout

    @pytest.mark.parametrize(
        "flag,desc",
        [
            ("--dangerously-skip-permissions", "bypass permissions flag"),
            ("--verbose", "verbose mode flag"),
        ],
    )
    def test_cli_common_flags_parse(self, flag, desc):
        if flag == "--permission-mode":
            proc = _run_cli(flag, "plan", "--help")
        else:
            proc = _run_cli(flag, "--help")
        assert proc.returncode == 0, f"{desc}: stderr={proc.stderr!r}"

    def test_cli_help_does_not_load_heavy_modules(self):
        """Verify cli help does not load heavy modules."""
        import time

        start = time.monotonic()
        proc = _run_cli("--help")
        elapsed = time.monotonic() - start
        assert proc.returncode == 0
        assert elapsed < 5.0, f"--help took {elapsed:.2f}s, expected < 5s"

    def test_cli_print_mode_initializes_without_crash(self):
        """Verify cli print mode initializes without crash."""
        import subprocess as _sp

        proc = _sp.Popen(
            [sys.executable, "-m", "src.cli", "-p", "hello"],
            stdout=_sp.PIPE,
            stderr=_sp.STDOUT,
            text=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=12)
            output = stdout
            assert "Traceback (most recent call last)" not in output, f"CLI crashed with unhandled exception:\n{output}"
            assert "SystemError" not in output, f"CLI crashed with SystemError:\n{output}"
        except _sp.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate(timeout=5)
            output = stdout
            assert "Traceback (most recent call last)" not in output, (
                f"CLI crashed with unhandled exception (partial output):\n{output}"
            )
            assert "SystemError" not in output, f"CLI crashed with SystemError (partial output):\n{output}"
            assert "model" in output.lower(), f"Expected model-related output in print mode:\n{output}"
