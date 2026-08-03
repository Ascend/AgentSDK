#!/usr/bin/env python3
# coding=utf-8

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

"""Normalize and display the ghost-suggestion accept key.

The REPL uses prompt_toolkit (which registers bindings as ``"c-e"``,
``"tab"``, ``"c-j"`` …) and the TUI uses Textual (which dispatches
``on_key`` with ``event.key`` strings like ``"ctrl+e"``, ``"tab"``,
``"ctrl+j"``). They share a single user-configurable key but speak two
dialects, so this module is the single place that translates between
them and renders a human-readable form for the ghost-text hint.

The canonical on-disk form is prompt_toolkit's (e.g. ``"c-e"``). That
keeps the on-disk setting close to the upstream
``AutoSuggestFromHistory`` accept key and matches what most REPL users
have seen in documentation.
"""

from __future__ import annotations

# Keys that don't take a "c-" prefix and need no transformation when
# going prompt_toolkit -> Textual. Anything not in this map that starts
# with "c-" is treated as Ctrl+<suffix> in both dialects.
_PLAIN_KEYS: frozenset[str] = frozenset(
    {
        "tab",
        "enter",
        "return",
        "escape",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "delete",
        "backspace",
        "space",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
    }
)


def to_prompt_toolkit_key(key: str) -> str:
    """Return *key* in prompt_toolkit's ``KeyBindings.add`` dialect.

    Accepts the canonical form (``"c-e"``, ``"tab"``, ``"ctrl+e"``) and
    returns the prompt_toolkit spelling. Unknown inputs pass through
    lower-cased — bindings that don't exist will simply never fire.
    """
    if not key:
        return "c-e"
    k = key.strip().lower()
    if k.startswith("ctrl+") and len(k) > 5:
        return "c-" + k[5:]
    return k


def to_textual_key(key: str) -> str:
    """Return *key* in Textual's ``event.key`` dialect.

    Translates ``"c-e"`` to ``"ctrl+e"`` and leaves plain keys
    (``"tab"``, ``"enter"`` …) untouched. This is the value that
    matches inside an ``on_key`` switch.
    """
    if not key:
        return "ctrl+e"
    k = key.strip().lower()
    if k.startswith("c-") and len(k) == 3:
        return "ctrl+" + k[2:]
    if k in _PLAIN_KEYS:
        return k
    return k


def display_key(key: str) -> str:
    """Render *key* as the human-readable label for the ghost-text hint.

    Examples: ``"c-e"`` -> ``"CTRL + E"``, ``"tab"`` -> ``"TAB"``,
    ``"f2"`` -> ``"F2"``, ``"c-j"`` -> ``"CTRL + J"``. Mirrors the
    TS reference's hint casing so existing users see no visual change
    after upgrading.
    """
    if not key:
        return "CTRL + E"
    k = key.strip().lower()
    if k.startswith("c-") and len(k) == 3:
        return "CTRL + " + k[2:].upper()
    if k.startswith("ctrl+") and len(k) > 5:
        return "CTRL + " + k[5:].upper()
    if k in _PLAIN_KEYS:
        return k.upper()
    return k.upper()
