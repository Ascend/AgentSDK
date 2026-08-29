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

"""TUI frontend plugin for the downstream registry."""

from __future__ import annotations
# pylint: disable=E0611

from clawcodex_ext.frontend.protocol import FrontendPlugin
from clawcodex_ext.frontend.registry import register_frontend


@register_frontend
class TUIFrontend(FrontendPlugin):
    name = "tui"
    display_name = "Textual TUI"

    def run(self, ctx, argv: list[str]) -> int:
        from src.entrypoints.tui import TUIOptions

        from clawcodex_ext.tui.entrypoint import run_tui

        options = TUIOptions(
            provider_name=ctx.provider_name,
            model=ctx.options.model,
            max_turns=ctx.options.max_turns,
            allowed_tools=ctx.options.allowed_tools,
            disallowed_tools=ctx.options.disallowed_tools,
            stream=True,
            permission_mode=ctx.options.permission_mode,
            is_bypass_permissions_mode_available=ctx.options.is_bypass_permissions_mode_available,
            workspace_root=ctx.workspace_root,
            append_system_prompt=ctx.options.append_system_prompt,
        )
        return run_tui(
            options,
            provider=ctx.provider,
            session=ctx.session,
            tool_registry=ctx.tool_registry,
            tool_context=ctx.tool_context,
            runtime_context=ctx,
            resume_session_id=ctx.options.resume_session_id,
            resume_browse=ctx.options.resume_browse,
        )
