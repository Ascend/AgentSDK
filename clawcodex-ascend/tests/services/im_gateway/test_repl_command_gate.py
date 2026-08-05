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

"""Tests for repl_command_gate: IM 侧 REPL 斜杠命令白名单门禁。

覆盖：
- 白名单内每个命令（含别名）放行 → (True, "")
- 带参数的命令放行
- 非白名单命令拒绝 → (False, reason)，reason 回显被拒绝的命令
- 非斜杠输入放行
- 大小写不敏感
"""

from __future__ import annotations

import pytest

from clawcodex_ext.services.im_gateway.repl_command_gate import (
    ORCHESTRATOR_ALLOWED_COMMANDS,
    REPL_ALLOWED_COMMANDS,
    check_orchestrator_command,
    check_repl_command,
)


# -- 白名单内命令放行 --------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    sorted(REPL_ALLOWED_COMMANDS),
    ids=lambda c: f"allowed:{c}",
)
def test_allowed_command_passes(cmd: str) -> None:
    allowed, reason = check_repl_command(cmd)
    assert allowed is True
    assert reason == ""


def test_allowed_command_with_args_passes() -> None:
    """带参数的命令按命令名前缀判定，应放行。"""
    for text in ["/goal finish the task", "/clear all", "/help me", "/stop now"]:
        allowed, reason = check_repl_command(text)
        assert allowed is True, f"expected {text!r} to be allowed"
        assert reason == ""


# -- 非白名单命令拒绝 --------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "/exit",
        "/quit",
        "/q",
        "/login",
        "/permissions",
        "/permission",
        "/model",
        "/provider",
        "/init",
        "/compact",
        "/save",
        "/load",
        "/resume",
        "/cron-run",
        "/cron-fire",
        "/cron-delete",
        "/tool",
        "/memory",
        "/rewind",
        "/advisor",
        "/telemetry",
        "/vim",
        "/tui",
        "/unknown-cmd-xyz",
    ],
    ids=lambda c: f"blocked:{c}",
)
def test_blocked_command_rejected(cmd: str) -> None:
    allowed, reason = check_repl_command(cmd)
    assert allowed is False
    # reason 必须回显被拒绝的命令
    assert cmd.lower() in reason
    assert "not in the command allowlist" in reason


def test_blocked_command_reason_echoes_command_token() -> None:
    """拒绝消息必须包含被拒绝的命令 token（回显）。"""
    allowed, reason = check_repl_command("/exit")
    assert allowed is False
    assert "`/exit`" in reason


def test_blocked_command_with_args_rejected() -> None:
    """带参数的非白名单命令也应拒绝，reason 回显命令 token。"""
    allowed, reason = check_repl_command("/model gpt-4")
    assert allowed is False
    assert "`/model`" in reason


# -- 非斜杠输入放行 ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["hello", "普通文本消息", "  /  ", "/", "", "   ", "not a command"],
    ids=lambda t: f"passthrough:{t!r}",
)
def test_non_slash_input_passes(text: str) -> None:
    allowed, reason = check_repl_command(text)
    assert allowed is True
    assert reason == ""


# -- 大小写不敏感 ------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    ["/STOP", "/Clear", "/RESET", "/NEW", "/GOAL", "/Help", "/COST", "/DOCTOR"],
)
def test_case_insensitive(cmd: str) -> None:
    allowed, reason = check_repl_command(cmd)
    assert allowed is True, f"expected {cmd!r} (case-insensitive) to be allowed"
    assert reason == ""


def test_case_insensitive_blocked() -> None:
    """非白名单命令的大写形式也应拒绝。"""
    allowed, reason = check_repl_command("/EXIT")
    assert allowed is False
    assert "/exit" in reason  # reason 中回显的 token 是小写化后的


def test_repl_command_uses_configured_allowlist() -> None:
    allowed, reason = check_repl_command("/model gpt-5", allowed_commands={"/model"})
    assert allowed is True
    assert reason == ""

    allowed, reason = check_repl_command("/clear", allowed_commands=set())
    assert allowed is False
    assert "`/clear`" in reason


# -- Orchestrator 白名单 -----------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    sorted(ORCHESTRATOR_ALLOWED_COMMANDS),
    ids=lambda c: f"orch-allowed:{c}",
)
def test_orchestrator_allowed_commands_pass(cmd: str) -> None:
    allowed, reason = check_orchestrator_command(cmd)
    assert allowed is True
    assert reason == ""


def test_orchestrator_allowed_command_with_args_passes() -> None:
    for text in [
        "/issue list --status running",
        "/issue show --id AGENTSDK-15",
        '/issue inject --id AGENTSDK-15 "address review comments"',
        "/issue feedback --id AGENTSDK-15 --approve",
        '/issue review --id AGENTSDK-15 --reject --feedback "needs tests"',
        "/issue retry --id AGENTSDK-15 --mode reset",
        "/server status --workflow ./workflow.md",
        "/issue rebase --id AGENTSDK-15",
    ]:
        allowed, reason = check_orchestrator_command(text)
        assert allowed is True, f"expected {text!r} to be allowed"
        assert reason == ""


@pytest.mark.parametrize(
    ("cmd", "reason"),
    [
        ("/server stop", "Command /server stop is not supported."),
        ("/issue takeover", "Command /issue takeover is not supported."),
        ("/dashboard", "Command /dashboard is not supported."),
        ("/workflow init", "Command /workflow init is not supported."),
        ("/unknown-cmd-xyz", "Command /unknown-cmd-xyz is not supported."),
    ],
    ids=str,
)
def test_orchestrator_blocked_commands_rejected(cmd: str, reason: str) -> None:
    allowed, actual = check_orchestrator_command(cmd)
    assert allowed is False
    assert actual == reason


def test_orchestrator_plain_text_passes() -> None:
    allowed, reason = check_orchestrator_command("普通文本 follow-up")
    assert allowed is True
    assert reason == ""


def test_orchestrator_command_uses_configured_allowlist() -> None:
    allowed, reason = check_orchestrator_command(
        "/issue takeover --id AGENTSDK-15",
        allowed_commands={"/issue takeover"},
    )
    assert allowed is True
    assert reason == ""

    allowed, reason = check_orchestrator_command(
        "/server status",
        allowed_commands=set(),
    )
    assert allowed is False
    assert reason == "Command /server status is not supported."
