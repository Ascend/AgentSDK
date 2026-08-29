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

# pylint: disable=cell-var-from-loop

"""Shared interactive input helpers for the CLI wizard and REPL menus.

This module provides:
- :func:`arrow_select`: a module-level arrow-key navigable menu built on
  ``prompt_toolkit``, extracted from
  :meth:`clawcodex_ext.repl.core.ClawcodexREPL._run_arrow_menu` so the CLI
  wizard (which has no REPL instance) can reuse the same up/down/Enter/1-9/ESC
  interaction as the Permission Required prompt.
- :func:`prompt_with_escape`: a text prompt that binds ESC, returning
  ``(value, escaped)`` so callers can distinguish "Enter with empty input"
  from "ESC to go back".
- :class:`InteractiveInput`: a unified façade exposing ``select`` / ``prompt``
  / ``confirm``. When constructed with an injected ``input_fn`` (used by tests
  and the ``prompt_toolkit``-less fallback) it degrades to a numbered line
  reader where an empty line is treated as ESC (``None``).
"""

from __future__ import annotations

import sys
from typing import Callable

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    _HAS_PROMPT_TOOLKIT = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_PROMPT_TOOLKIT = False

InputFn = Callable[[str], str]


def _clear_rendered_lines(line_count: int) -> None:
    """Clear the previous ``line_count`` rendered lines with ANSI codes.

    This cleans up ``arrow_select`` after a ``full_screen=False`` menu exits,
    preventing the parent menu from accumulating duplicate renderings. It is
    not called when prompt_toolkit is unavailable.

    ``line_count <= 0`` is a no-op. Positive values move the cursor up N
    lines (``\033[NA``) and clear to the bottom of the screen (``\033[J``).
    """
    if line_count <= 0:
        return
    sys.stdout.write(f"\033[{line_count}A\033[J")
    sys.stdout.flush()


def arrow_select(
    options: list[tuple[str, str]],
    *,
    title: str = "",
    allow_other: bool = False,
    multi_select: bool = False,
    run_wrapper: Callable[[Callable[[], object]], object] | None = None,
) -> int | list[int] | None:
    """Show an arrow-key navigable menu.

    Args:
        options: List of (label, description) pairs.
        title: Optional title shown above the menu.
        allow_other: If True, add an "Other" option at the end.
        multi_select: If True, allow Space-toggle multi-selection.
        run_wrapper: Optional callable that wraps ``app.run()`` — used by the
            REPL to pause its live status spinner while the menu is active.

    Returns:
        Single-select: 0-based index or ``None`` for ESC/Ctrl-C cancel.
        Multi-select: list of 0-based indices, or ``None`` for cancel.

    Guard: returns ``None`` when ``prompt_toolkit`` is unavailable or there
    are no options — callers must fall back to a line-based reader.
    """
    if not _HAS_PROMPT_TOOLKIT:
        return None

    total = len(options) + (1 if allow_other else 0)
    if total == 0:
        return None

    cursor = [0]
    selected: set[int] | None = set() if multi_select else None
    rendered_lines = [0]  # Number of rendered lines to clear when the menu exits.

    def get_menu_fragments() -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        if title:
            fragments.append(("[bold]", f"\n{title}\n\n"))
        for i, (label, desc) in enumerate(options):
            is_cursor = i == cursor[0]
            is_sel = multi_select and i in (selected or set())
            prefix = "▸" if is_cursor else " "
            check = "✓" if is_sel else " "
            item_style = "class:arrow-cursor" if is_cursor else ""
            fragments.append((item_style, f"  {prefix} {check} {i + 1}. {label}"))
            if desc:
                fragments.append(("class:dim", f"    {desc}"))
            fragments.append(("", "\n"))
        if allow_other:
            i = len(options)
            is_cursor = i == cursor[0]
            prefix = "▸" if is_cursor else " "
            item_style = "class:arrow-cursor" if is_cursor else ""
            fragments.append((item_style, f"  {prefix}   {i + 1}. Other"))
            fragments.append(("class:dim", "  (provide custom text)"))
            fragments.append(("", "\n"))
        if multi_select:
            hint = "  ↑↓ navigate · Space toggle · Enter confirm · 1-9 quick select · Esc cancel"
        else:
            hint = "  ↑↓ navigate · Enter select · 1-9 quick select · Esc cancel"
        fragments.append(("class:dim", f"\n{hint}"))
        # Count rendered lines for cleanup. The final hint has no trailing
        # newline but still occupies the cursor line when prompt_toolkit exits.
        rendered_lines[0] = sum(text.count("\n") for _style, text in fragments) + 1
        return fragments

    kb = KeyBindings()

    @kb.add("up")
    def _move_up(event):  # noqa: ANN001
        cursor[0] = max(0, cursor[0] - 1)
        event.app.invalidate()

    @kb.add("down")
    def _move_down(event):  # noqa: ANN001
        cursor[0] = min(total - 1, cursor[0] + 1)
        event.app.invalidate()

    @kb.add("enter")
    def _handle_enter(event):  # noqa: ANN001
        if multi_select:
            sel_list = sorted(selected) if selected else [0]
            event.app.exit(result=sel_list)
        else:
            event.app.exit(result=cursor[0])

    @kb.add("space")
    def _handle_space(event):  # noqa: ANN001
        if multi_select:
            if cursor[0] < len(options):
                s = selected
                if s is not None:
                    if cursor[0] in s:
                        s.discard(cursor[0])
                    else:
                        s.add(cursor[0])
                    event.app.invalidate()
        else:
            event.app.exit(result=cursor[0])

    @kb.add("escape")
    def _handle_escape(event):  # noqa: ANN001
        event.app.exit(result=None)

    @kb.add("c-c")
    def _handle_ctrl_c(event):  # noqa: ANN001
        event.app.exit(result=None)

    for digit in range(1, min(10, total + 1)):  # pylint: disable=cell-var-from-loop

        @kb.add(str(digit))
        def _handle_digit(event, idx=digit):  # noqa: ANN001
            actual = idx - 1
            if multi_select:
                if actual < len(options):
                    s = selected
                    if s is not None:
                        if actual in s:
                            s.discard(actual)
                        else:
                            s.add(actual)
                        event.app.invalidate()
            else:
                event.app.exit(result=actual)

    pt_style = Style.from_dict({"arrow-cursor": "bold", "dim": "fg:gray"})

    app = Application(
        layout=Layout(
            Window(FormattedTextControl(get_menu_fragments)),
        ),
        key_bindings=kb,
        style=pt_style,
        full_screen=False,
        mouse_support=False,
    )
    if run_wrapper is not None:
        result = run_wrapper(app.run)
    else:
        result = app.run()

    # Clear this menu so its parent can render again in the same location.
    _clear_rendered_lines(rendered_lines[0])

    if result is None:
        return None
    if multi_select and selected:
        sel_list = sorted(selected)
        return sel_list if sel_list else [0]
    if multi_select and not selected:
        return [0]
    return result


def prompt_with_escape(prompt_str: str) -> tuple[str, bool]:
    """Read one line of text; ESC is detectable.

    Returns ``(value, escaped)``:
    - ``escaped=False``: user pressed Enter; ``value`` is the stripped input.
    - ``escaped=True``: user pressed ESC; ``value`` is ``''``.

    Guard: when ``prompt_toolkit`` is unavailable, falls back to ``input()``.
    An empty line is reported as ``escaped=True`` so callers treat it as
    "go back" — mirroring the injected ``input_fn`` contract.
    """
    if not _HAS_PROMPT_TOOLKIT:
        line = input(prompt_str)
        stripped = line.strip()
        if stripped == "":
            return "", True
        return stripped, False

    from prompt_toolkit import PromptSession

    kb = KeyBindings()
    state = {"escaped": False}

    @kb.add("escape")
    def _escape(event):  # noqa: ANN001
        state["escaped"] = True
        event.app.exit(result="")

    session = PromptSession(key_bindings=kb)
    line = session.prompt(prompt_str)
    if state["escaped"]:
        return "", True
    return str(line).strip(), False


class InteractiveInput:
    """Unified interactive façade for the gateway setup wizard.

    Wraps :func:`arrow_select` and :func:`prompt_with_escape` with a testable
    seam: when ``input_fn`` is provided (tests, or the ``prompt_toolkit``-less
    fallback), ``select`` prints a numbered list and reads a line, ``prompt``
    calls ``input_fn`` directly, and an empty line always means ESC (``None``).
    """

    def __init__(self, input_fn: InputFn | None = None) -> None:
        self._input_fn = input_fn

    @property
    def input_fn(self) -> InputFn:
        """Return the line reader used by legacy callable-based helpers."""
        return self._input_fn if self._input_fn is not None else input

    def select(
        self,
        options: list[tuple[str, str]],
        *,
        title: str = "",
    ) -> int | None:
        """Choose one option. Returns 0-based index, or ``None`` for ESC."""
        if self._input_fn is not None or not _HAS_PROMPT_TOOLKIT:
            return self._select_via_input_fn(options, title=title)
        result = arrow_select(options, title=title)
        return result

    def prompt(self, prompt_str: str) -> str | None:
        """Read one line. Returns the stripped value, or ``None`` for ESC."""
        if self._input_fn is not None or not _HAS_PROMPT_TOOLKIT:
            return self._prompt_via_input_fn(prompt_str)
        value, escaped = prompt_with_escape(prompt_str)
        if escaped:
            return None
        return value

    def confirm(self, prompt_str: str) -> bool | None:
        """Yes/No confirm. Returns ``True``/``False``, or ``None`` for ESC."""
        if self._input_fn is not None or not _HAS_PROMPT_TOOLKIT:
            return self._confirm_via_input_fn(prompt_str)
        idx = arrow_select(
            [("Yes", ""), ("No", "")],
            title=prompt_str,
        )
        if idx is None:
            return None
        return idx == 0

    # -- injected / fallback paths ------------------------------------------

    def _select_via_input_fn(
        self,
        options: list[tuple[str, str]],
        *,
        title: str,
    ) -> int | None:
        if title:
            print(f"\n{title}")
        for i, (label, desc) in enumerate(options, 1):
            line = f"  {i}) {label}"
            if desc:
                line += f"   {desc}"
            print(line)
        while True:
            raw = self._read("选择 (1-{} / 留空=返回): ".format(len(options)))
            if raw is None:
                return None
            try:
                idx = int(raw) - 1
            except ValueError:
                print("无效输入")
                continue
            if 0 <= idx < len(options):
                return idx
            print("序号超出范围")

    def _prompt_via_input_fn(self, prompt_str: str) -> str | None:
        value = self._read(prompt_str)
        if value is None:
            return None
        return value

    def _confirm_via_input_fn(self, prompt_str: str) -> bool | None:
        while True:
            raw = self._read(prompt_str)
            if raw is None:
                return None
            lowered = raw.lower()
            if lowered in ("y", "yes"):
                return True
            if lowered in ("n", "no"):
                return False
            print("请输入 y/n（留空=返回）")

    def _read(self, prompt_str: str) -> str | None:
        """Read one line via the injected input_fn; empty => None (ESC)."""
        assert self._input_fn is not None or not _HAS_PROMPT_TOOLKIT
        reader = self._input_fn if self._input_fn is not None else input
        line = reader(prompt_str)
        stripped = line.strip() if isinstance(line, str) else ""
        if stripped == "":
            return None
        return stripped
