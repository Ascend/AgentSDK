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

"""P102-D Agent-loop hook registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

LoopHookPhase = Literal[
    "pre_llm",
    "post_llm",
    "pre_tool",
    "post_tool",
    "on_turn_start",
    "on_turn_end",
]


@dataclass
class LoopHook:
    """Metadata for one agent-loop hook."""

    name: str
    fn: Callable[..., Any]
    phase: LoopHookPhase
    priority: int = 0


# Global mapping from phase to hooks ordered by priority.
_REGISTRY: dict[LoopHookPhase, list[LoopHook]] = {
    "pre_llm": [],
    "post_llm": [],
    "pre_tool": [],
    "post_tool": [],
    "on_turn_start": [],
    "on_turn_end": [],
}


# -- Public API -----------------------------------------------------------


def register_loop_hook(
    name: str,
    fn: Callable[..., Any],
    phase: LoopHookPhase,
    priority: int = 0,
) -> None:
    """Register an agent-loop hook for a phase."""
    unregister_loop_hook(name, phase)
    hook = LoopHook(name=name, fn=fn, phase=phase, priority=priority)
    _REGISTRY[phase].append(hook)
    _REGISTRY[phase].sort(key=lambda h: h.priority)
    logger.debug("Registered loop hook %r at phase %r (priority=%d)", name, phase, priority)


def unregister_loop_hook(name: str, phase: LoopHookPhase) -> None:
    """Unregister an agent-loop hook from a phase."""
    before = len(_REGISTRY[phase])
    _REGISTRY[phase] = [h for h in _REGISTRY[phase] if h.name != name]
    after = len(_REGISTRY[phase])
    if after < before:
        logger.debug("Unregistered loop hook %r from phase %r", name, phase)


def call_hooks(phase: LoopHookPhase, *args: Any, **kwargs: Any) -> tuple[Any, ...]:  # noqa: ANN401
    """Call registered hooks for a phase in priority order."""
    hooks = _REGISTRY.get(phase, [])
    current_args: tuple[Any, ...] = args
    for hook in hooks:
        try:
            result = hook.fn(*current_args, **kwargs)
        except Exception:
            logger.exception("Loop hook %r at phase %r raised an error", hook.name, phase)
            continue
        if result is not None:
            if isinstance(result, tuple):
                current_args = result
            else:
                current_args = (result,)
    return current_args


def list_hooks(phase: LoopHookPhase | None = None) -> list[LoopHook]:
    """Return a read-only list of registered loop hooks."""
    if phase is not None:
        return list(_REGISTRY.get(phase, []))
    return [h for hooks in _REGISTRY.values() for h in hooks]


def clear_hooks(phase: LoopHookPhase | None = None) -> None:
    """Clear the hook registry for test isolation."""
    if phase is not None:
        _REGISTRY[phase].clear()
    else:
        for p in _REGISTRY:  # pylint: disable=consider-using-dict-items
            _REGISTRY[p].clear()
