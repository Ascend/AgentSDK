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


"""Direct execution of a single @agent- mention in the REPL."""

from __future__ import annotations
# pylint: disable=E0611

import json
import logging
from typing import Any

from clawcodex_ext.agent.constants import AGENT_TOOL_NAME
from clawcodex_ext.command_system.input_processing import strip_agent_mentions
from clawcodex_ext.tool_system.protocol import ToolCall

logger = logging.getLogger(__name__)


def should_run_mentioned_agent_directly(
    agent_attachments: list[dict[str, str]],
    at_attachments: list[dict[str, Any]],
) -> bool:
    """True when the user message is a single @agent- mention (no @file paths)."""
    return len(agent_attachments) == 1 and not at_attachments


def run_mentioned_agent_direct(
    repl: Any,
    *,
    agent_type: str,
    user_input: str,
) -> bool:
    """Invoke the named sub-agent synchronously; return True when handled."""
    prompt = strip_agent_mentions(user_input)
    if not prompt:
        prompt = "Follow your agent definition and respond to the user's request."

    tool_context = repl.tool_context
    try:
        result = repl.tool_registry.dispatch(
            ToolCall(
                name=AGENT_TOOL_NAME,
                input={
                    "subagent_type": agent_type,
                    "prompt": prompt,
                    "description": prompt[:80],
                },
            ),
            tool_context,
        )
    except Exception as exc:
        logger.exception("direct mentioned-agent dispatch failed")
        repl.console.print(f"[error]Failed to invoke @{agent_type}: {exc}[/error]")
        return True

    output = getattr(result, "output", result)
    if isinstance(output, dict) and output.get("status") == "error":
        err = output.get("error") or output.get("message") or "unknown error"
        repl.console.print(f"[error]@{agent_type}: {err}[/error]")
        return True

    content_text = _extract_agent_result_text(output)
    repl.console.print(f"\n[bold]● Agent (@{agent_type})[/bold]")
    if content_text:
        repl.console.print(content_text)
    else:
        repl.console.print("[dim](no text output)[/dim]")

    repl.session.conversation.add_user_message(user_input)
    if content_text:
        repl.session.conversation.add_assistant_message(content_text)
    try:
        repl.session.save_transcript()
    except Exception:  # nosec B110
        pass  # Session persistence is best-effort at this recovery or shutdown boundary.
    return True


def _extract_agent_result_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if not isinstance(output, dict):
        return str(output).strip()

    blocks = output.get("content")
    if isinstance(blocks, list):
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n\n".join(parts)

    for key in ("result", "text", "output"):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    try:
        return json.dumps(output, ensure_ascii=False, indent=2)
    except TypeError:
        return str(output)
