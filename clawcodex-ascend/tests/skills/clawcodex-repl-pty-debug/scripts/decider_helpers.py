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

"""Small predicates for ``pty_adaptive_driver.py`` decider files.

Deciders should normally make prompt and completion decisions from the current
controller response. ``screen`` is cumulative terminal state and may contain
stale prompts from earlier turns.
"""

from __future__ import annotations

import json
import re
from typing import Any


_EXIT_CODE_RE = re.compile(
    r'"exit_code"\s*:\s*(-?\d+)|\bexit_code\b\s*[:=]\s*(-?\d+)',
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def current_delta(response: dict[str, Any]) -> str:
    """Return only the current response delta."""
    return _text(response.get("delta"))


def cumulative_screen(response: dict[str, Any]) -> str:
    """Return the cumulative screen text; use for context, not active prompts."""
    return _text(response.get("screen"))


def current_signals(response: dict[str, Any]) -> set[str]:
    raw = response.get("signals")
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw}
    if raw is None:
        return set()
    return {str(raw)}


def has_current_text(
    response: dict[str, Any],
    needle: str,
    *,
    case_sensitive: bool = True,
) -> bool:
    haystack = current_delta(response)
    if case_sensitive:
        return needle in haystack
    return needle.lower() in haystack.lower()


def has_cumulative_text(
    response: dict[str, Any],
    needle: str,
    *,
    case_sensitive: bool = True,
) -> bool:
    haystack = cumulative_screen(response)
    if case_sensitive:
        return needle in haystack
    return needle.lower() in haystack.lower()


def has_current_permission_prompt(response: dict[str, Any]) -> bool:
    """True only for an active permission prompt in the current response."""
    if response.get("kind") == "permission_prompt":
        return True
    if response.get("state") == "awaiting_permission" and "permission_prompt" in current_signals(response):
        return True
    delta = current_delta(response)
    if "Permission Required" not in delta:
        return False
    tail = delta.rsplit("Permission Required", 1)[-1]
    return not any(marker in tail for marker in ("Tool result:", "Tool error:", "⎿", "Goodbye!", "\n❯"))


def _json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start == -1:
            return values
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = start + end


def bash_exit_code(response: dict[str, Any], *, allow_screen: bool = False) -> int | None:
    """Return the latest Bash ``exit_code`` visible in current delta.

    Set ``allow_screen=True`` only when auditing a completed transcript where
    the current delta is unavailable or intentionally truncated.
    """
    text = current_delta(response)
    if allow_screen and not text:
        text = cumulative_screen(response)
    latest: int | None = None
    for value in _json_objects(text):
        if isinstance(value, dict) and "exit_code" in value:
            try:
                latest = int(value["exit_code"])
            except (TypeError, ValueError):
                pass  # Invalid structured values fall through to text parsing.
    for match in _EXIT_CODE_RE.finditer(text):
        for group in match.groups():
            if group is not None:
                latest = int(group)
                break
    return latest


def bash_succeeded(response: dict[str, Any], *, allow_screen: bool = False) -> bool:
    return bash_exit_code(response, allow_screen=allow_screen) == 0


def decision_basis(response: dict[str, Any]) -> dict[str, Any]:
    """Compact fields worth writing beside a decider request."""
    delta = current_delta(response)
    screen = cumulative_screen(response)
    basis: dict[str, Any] = {
        "current_delta_chars": len(delta),
        "screen_chars": len(screen),
        "current_permission_prompt": has_current_permission_prompt(response),
        "screen_mentions_permission": "Permission Required" in screen,
        "bash_exit_code": bash_exit_code(response),
    }
    if delta:
        basis["current_delta_tail"] = delta[-500:]
    return basis
