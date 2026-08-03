# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""REPL frontend plugin for the downstream registry."""

from __future__ import annotations

from clawcodex_ext.frontend.protocol import FrontendPlugin
from clawcodex_ext.frontend.registry import register_frontend


@register_frontend
class REPLFrontend(FrontendPlugin):
    name = "repl"
    display_name = "Interactive REPL"

    def run(self, ctx, argv: list[str]) -> int:
        from clawcodex_ext.repl.app import ClawCodexExtREPL

        from clawcodex_ext.frontend.repl_extensions import install_repl_extensions

        # Resolve session ID: --resume <id> or --resume (browse) via REPL browser
        session_id = ctx.options.resume_session_id
        if ctx.options.resume_browse and not session_id:
            from clawcodex_ext.repl.color_scheme import build_oklch_console
            from clawcodex_ext.repl.session_browser import browse_sessions_interactive

            console = build_oklch_console()
            console.print("[dim]No session ID specified. Opening session browser...[/dim]")
            picked = browse_sessions_interactive(console=console)
            if picked:
                session_id = picked
            else:
                console.print("[dim]No session selected. Starting a new session.[/dim]")

        repl = ClawCodexExtREPL(
            provider_name=ctx.provider_name,
            stream=ctx.options.stream,
            permission_mode=ctx.options.permission_mode,
            is_bypass_permissions_mode_available=ctx.options.is_bypass_permissions_mode_available,
            resume_session_id=session_id,
            provider=ctx.provider,
            session=ctx.session,
            tool_registry=ctx.tool_registry,
            tool_context=ctx.tool_context,
            workspace_root=ctx.workspace_root,
            runtime_context=ctx,
            append_system_prompt=ctx.options.append_system_prompt,
        )
        install_repl_extensions(repl, ctx)
        try:
            from extensions.recording.repl_source import install_repl_capture

            install_repl_capture(repl, ctx)
        except Exception:  # nosec B110
            # Recorder extension is optional; a missing/broken recording
            # package must not prevent the REPL from starting.
            pass
        repl.run()
        return 0
