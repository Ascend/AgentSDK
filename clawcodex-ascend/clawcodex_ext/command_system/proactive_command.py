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

from __future__ import annotations

from clawcodex_ext.command_system.types import CommandContext, LocalCommand, LocalCommandResult
from clawcodex_ext.services.proactive import TickEmitter, get_default_controller
from clawcodex_ext.services.proactive.constants import DEFAULT_FOCUS_LEVEL, FOCUS_LEVELS

_EMITTERS_BY_CONTEXT_ID: dict[int, TickEmitter] = {}


def is_proactive_feature_enabled() -> bool:
    try:
        from clawcodex_ext.feature_gate import get_registry  # pylint: disable=no-name-in-module

        reg = get_registry()
        return reg.is_enabled("PROACTIVE") or reg.is_enabled("KAIROS")
    except Exception:
        return False


def _parse_focus(parts: list[str]) -> str | None:
    for part in parts:
        if part.startswith("focus="):
            return part.split("=", 1)[1].strip()
        if part in FOCUS_LEVELS:
            return part
    return None


def _invalid_focus_result(focus: str) -> LocalCommandResult | None:
    if focus in FOCUS_LEVELS:
        return None
    return LocalCommandResult(
        type="text",
        value=(f"Invalid proactive focus {focus!r}. Expected one of: {', '.join(FOCUS_LEVELS)}."),
    )


def _get_or_create_emitter(context: CommandContext) -> TickEmitter | None:
    tool_context = getattr(context, "tool_context", None)
    outbox = getattr(tool_context, "outbox", None)
    if not hasattr(outbox, "append"):
        return None
    context_id = id(tool_context)
    emitter = _EMITTERS_BY_CONTEXT_ID.get(context_id)
    if emitter is not None:
        return emitter
    emitter = TickEmitter(controller=get_default_controller(), outbox=outbox)
    _EMITTERS_BY_CONTEXT_ID[context_id] = emitter
    return emitter


def _pop_emitter(context: CommandContext) -> TickEmitter | None:
    tool_context = getattr(context, "tool_context", None)
    if tool_context is None:
        return None
    return _EMITTERS_BY_CONTEXT_ID.pop(id(tool_context), None)


def proactive_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    parts = (args or "").split()
    action = parts[0].lower() if parts else "on"
    focus = _parse_focus(parts[1:] if parts else [])
    ctrl = get_default_controller()

    if action in ("on", "enable", "start"):
        selected_focus = focus or DEFAULT_FOCUS_LEVEL
        invalid = _invalid_focus_result(selected_focus)
        if invalid is not None:
            return invalid
        ctrl.activate("slash_command", focus=selected_focus)  # type: ignore[arg-type]
        emitter = _get_or_create_emitter(context)
        if emitter is not None:
            emitter.start()
        return LocalCommandResult(
            type="prompt",
            value=(
                "<system-reminder>\n"
                f"Proactive mode enabled with focus={selected_focus}. "
                "Future <tick> messages are autonomous wake-up signals; continue useful work when the next step is clear.\n"
                "</system-reminder>"
            ),
            display_text=f"Proactive mode enabled (focus={selected_focus}).",
        )
    if action in ("off", "disable", "stop"):
        emitter = _pop_emitter(context)
        if emitter is not None:
            emitter.stop()
        ctrl.deactivate()
        return LocalCommandResult(type="text", value="Proactive mode disabled.")
    if action == "pause":
        emitter = _get_or_create_emitter(context)
        if emitter is not None:
            emitter.pause()
        else:
            ctrl.pause()
        return LocalCommandResult(type="text", value="Proactive mode paused.")
    if action in ("resume", "resume-blocked"):
        emitter = _get_or_create_emitter(context)
        if emitter is not None:
            emitter.resume()
        else:
            ctrl.resume()
        return LocalCommandResult(type="text", value="Proactive mode resumed.")
    if action == "focus":
        selected_focus = focus or (parts[1] if len(parts) > 1 else "")
        if not selected_focus:
            return LocalCommandResult(
                type="text",
                value=f"Usage: /proactive focus <{'|'.join(FOCUS_LEVELS)}>",
            )
        invalid = _invalid_focus_result(selected_focus)
        if invalid is not None:
            return invalid
        ctrl.set_focus(selected_focus)  # type: ignore[arg-type]
        return LocalCommandResult(type="text", value=f"Proactive focus set to {selected_focus}.")
    if action == "status":
        state = ctrl.state
        return LocalCommandResult(
            type="text",
            value=(f"Proactive: {state.phase} focus={state.focus} ticks={state.tick_count}"),
        )
    return LocalCommandResult(
        type="text",
        value="Usage: /proactive [on [focus=<full|medium|minimal>] | off | pause | resume | focus <level> | status]",
    )


PROACTIVE_COMMAND = LocalCommand(
    name="proactive",
    description="Enable or control proactive tick-driven mode",
    argument_hint="[on|off|pause|resume|focus|status] [focus=<full|medium|minimal>]",
    supports_non_interactive=True,
    is_enabled=is_proactive_feature_enabled,
)
PROACTIVE_COMMAND.set_call(proactive_command_call)


__all__ = ["PROACTIVE_COMMAND", "is_proactive_feature_enabled", "proactive_command_call"]
