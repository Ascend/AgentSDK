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

"""Headless (print-mode) frontend plugin for the downstream registry."""

from __future__ import annotations

from clawcodex_ext.frontend.protocol import FrontendPlugin
from clawcodex_ext.frontend.registry import register_frontend


@register_frontend
class HeadlessFrontend(FrontendPlugin):
    name = "headless"
    display_name = "Headless / Print Mode"

    def run(self, ctx, argv: list[str]) -> int:
        from src.entrypoints.headless import HeadlessOptions, run_headless

        # forward resume / fork / resume-session-at from the
        # RuntimeContext that ``dispatch.py`` already built. The pre-
        # resolved session on ``ctx.session`` is passed as
        # ``external_session`` so ``run_headless`` skips its own
        # ``Session.create()`` (C1 — eliminates the double code path).
        # When the user passes neither ``--resume`` nor
        # ``--fork-session``, ``ctx.session`` is ``None`` and the
        # legacy ``Session.create()`` branch in ``run_headless``
        # kicks in — preserving single-shot behaviour for callers
        # that never set up RuntimeContext.
        options = HeadlessOptions(
            prompt=getattr(ctx.options, "prompt", None),
            output_format=getattr(ctx.options, "output_format", "text"),
            input_format=getattr(ctx.options, "input_format", "text"),
            provider_name=ctx.provider_name,
            provider_instance=ctx.provider,
            model=ctx.options.model,
            fallback_model=ctx.options.fallback_model,
            effort=ctx.options.effort,
            max_turns=ctx.options.max_turns,
            max_turns_explicit=getattr(ctx.options, "max_turns_explicit", False),
            permission_mode=ctx.options.permission_mode,
            is_bypass_permissions_mode_available=ctx.options.is_bypass_permissions_mode_available,
            skip_permissions=ctx.options.skip_permissions,
            allowed_tools=ctx.options.allowed_tools,
            disallowed_tools=ctx.options.disallowed_tools,
            include_partial_messages=getattr(ctx.options, "include_partial_messages", False),
            verbose=ctx.options.verbose,
            workspace_root=ctx.workspace_root,
            append_system_prompt=ctx.options.append_system_prompt,
            startup_agent=ctx.options.startup_agent,
            bundle_context=getattr(ctx.tool_context, "bundle_context", None),
            mcp_clients=getattr(ctx.tool_context, "mcp_clients", {}),
            mcp_manager_loop=getattr(ctx.tool_context, "mcp_manager_loop", None),
            resume_session_id=getattr(ctx.options, "resume_session_id", None),
            fork_session_id=getattr(ctx.options, "fork_session_id", None),
            resume_session_at=getattr(ctx.options, "resume_session_at", None),
            external_session=getattr(ctx, "session", None),
            record=getattr(ctx.options, "record", None),
            record_width=getattr(ctx.options, "record_width", None),
            record_height=getattr(ctx.options, "record_height", None),
        )
        # C14: release the TailFollower that ``RuntimeContext.build``
        # obtained from ``resume_session_with_tail``. Headless never
        # iterates it — without an explicit release the follower keeps a
        # reference to the transcript path and asyncio event state for
        # the lifetime of the RuntimeContext. Best-effort; failures are
        # swallowed by ``close`` itself.
        try:
            return run_headless(options)
        finally:
            close = getattr(ctx, "close", None)
            if callable(close):
                close()
