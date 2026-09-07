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

# pylint: disable=no-name-in-module
"""Core fast-path CLI subcommands — login/config/mcp/daemon/doctor/orchestrator."""

from __future__ import annotations

from clawcodex_ext.cli.subcommand_registry import register


@register("login")  # nosec B105
def run_login_command(args: list[str]) -> int:
    """Handle ``clawcodex login``."""
    from src.cli import handle_login

    return handle_login()


@register("config")  # nosec B105
def run_config_command(args: list[str]) -> int:
    """Handle ``clawcodex config``."""
    from src.cli import show_config

    return show_config()


@register("mcp")  # nosec B105
def run_mcp_command(args: list[str]) -> int:
    """Handle ``clawcodex mcp`` subcommands."""
    from src.entrypoints.mcp import run_mcp_subcommand

    return run_mcp_subcommand(args)


@register("daemon", telemetry_mode="daemon")  # nosec B105
def run_daemon_command(args: list[str]) -> int:
    """Handle ``clawcodex daemon`` subcommands."""
    from src.entrypoints.daemon import run_daemon_subcommand

    return run_daemon_subcommand(args)


@register("doctor")  # nosec B105
def run_doctor_command(args: list[str]) -> int:
    """Handle ``clawcodex doctor``."""
    from src.entrypoints.doctor import run_doctor

    return run_doctor()


@register("orchestrator", telemetry_mode="daemon")  # nosec B105
def run_orchestrator_command(args: list[str]) -> int:
    """Handle ``clawcodex orchestrator`` subcommands."""
    from src.entrypoints.orchestrator import run_orchestrator_subcommand

    return run_orchestrator_subcommand(args)
