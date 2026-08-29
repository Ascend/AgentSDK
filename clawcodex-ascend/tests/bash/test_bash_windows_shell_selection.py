#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

from __future__ import annotations


# pylint: disable=E0611
from pathlib import Path
from unittest import mock

from clawcodex_ext.utils.shell_resolver import resolve_shell
from src.tool_system.context import ToolContext
from src.tool_system.tools.bash.bash_tool import _BashRunResult, _bash_call


def test_bash_call_auto_uses_powershell_on_windows(tmp_path: Path) -> None:
    """Regression: Windows auto shell must not execute foreground commands via WSL bash."""
    ctx = ToolContext(workspace_root=tmp_path)
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> _BashRunResult:
        seen["argv"] = argv
        return _BashRunResult(returncode=0, stdout="ok\n", stderr="")

    with (
        mock.patch("clawcodex_ext.utils.shell_resolver._sys.platform", "win32"),
        mock.patch(
            "clawcodex_ext.utils.shell_resolver.find_powershell_path",
            return_value=r"C:\Program Files\PowerShell\7\pwsh.exe",
        ),
        mock.patch(
            "src.tool_system.tools.bash.bash_tool._run_bash_with_abort",
            side_effect=fake_run,
        ),
    ):
        result = _bash_call({"command": "Write-Output ok"}, ctx)

    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[0].endswith("pwsh.exe")
    assert "-Command" in argv
    assert "Get-Location" in argv[-1]
    assert result.output["stdout"] == "ok\n"


def test_bash_call_explicit_bash_still_uses_bash(tmp_path: Path) -> None:
    """Explicit shell='bash' preserves the existing bash execution path."""
    ctx = ToolContext(workspace_root=tmp_path)
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> _BashRunResult:
        seen["argv"] = argv
        return _BashRunResult(returncode=0, stdout="ok\n", stderr="")

    with mock.patch(
        "src.tool_system.tools.bash.bash_tool._run_bash_with_abort",
        side_effect=fake_run,
    ):
        result = _bash_call({"command": "echo ok", "shell": "bash"}, ctx)

    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[:2] == ["bash", "-lc"]
    assert "pwd >" in argv[-1]
    assert result.output["stdout"] == "ok\n"


def test_resolve_shell_powershell_fallback_on_posix() -> None:
    """Explicit shell='powershell' on POSIX without pwsh falls back to bash."""
    with mock.patch(
        "clawcodex_ext.utils.shell_resolver.find_powershell_path",
        return_value=None,
    ):
        kind, factory = resolve_shell("powershell")

    assert kind == "bash"
    assert factory("echo ok") == ["bash", "-lc", "echo ok"]
