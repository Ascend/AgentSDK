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

# pylint: disable=cyclic-import,no-name-in-module
"""Bind live runtime objects onto a :class:`ToolContext`.

Entry points may either construct a fresh context or receive one from a
``RuntimeContext``/SDK caller.  In both cases tools must see the registry,
session id, and provider that own the current run.  Keeping the assignment in
one small helper prevents injected contexts from retaining stale references.
"""

from __future__ import annotations

from typing import Any


def bind_tool_context_runtime(
    tool_context: Any,
    *,
    tool_registry: Any | None = None,
    session: Any | None = None,
    provider: Any | None = None,
) -> Any:
    """Attach available live runtime objects to ``tool_context``.

    ``None`` means that a runtime object is not available yet, so the
    corresponding context field is left untouched.  Supplied values always
    replace existing fields; this is important for injected contexts and
    provider/session switches.
    """

    if tool_context is None:
        return None

    if tool_registry is not None:
        tool_context.tool_registry = tool_registry

    if session is not None:
        session_id = getattr(session, "session_id", None)
        if session_id is not None:
            tool_context.session_id = str(session_id)

    if provider is not None:
        tool_context._active_provider = provider

    # Shared by REPL, TUI, headless, and SDK construction paths.
    from clawcodex_ext.configuration import apply_configuration_snapshot
    from src.bootstrap.state import get_session_trust_accepted

    tool_context.workspace_trusted = bool(tool_context.workspace_trusted or get_session_trust_accepted())
    apply_configuration_snapshot(tool_context)
    return tool_context


__all__ = ["bind_tool_context_runtime"]
