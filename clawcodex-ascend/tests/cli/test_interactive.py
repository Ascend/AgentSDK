#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for the shared interactive input module (arrow menu + ESC-aware prompt).

These tests exercise the ``input_fn`` injection path of :class:`InteractiveInput`,
which mirrors the production ``prompt_toolkit`` behavior with a scripted line
reader. The ``prompt_toolkit``-driven path is covered by manual verification
(see the plan's verification steps) because CI cannot simulate real key events.
"""

from __future__ import annotations

from clawcodex_ext.cli._interactive import InteractiveInput


class _ScriptedInput:
    """Yields scripted responses, recording the prompts shown to the user."""

    def __init__(self, replies: list[str]) -> None:
        self._iter = iter(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self._iter)


# -- InteractiveInput.select (injected input_fn path) -----------------------


def test_select_returns_zero_based_index_for_chosen_option() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["2"]))
    idx = ui.select([("feishu", ""), ("wechat", "")], title="频道选择")
    assert idx == 1


def test_select_first_option_returns_zero() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["1"]))
    idx = ui.select([("feishu", ""), ("wechat", "")])
    assert idx == 0


def test_select_empty_line_returns_none_meaning_esc() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput([""]))
    idx = ui.select([("feishu", ""), ("wechat", "")])
    assert idx is None


def test_select_invalid_then_valid_re_prompts() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["x", "9", "1"]))
    idx = ui.select([("feishu", ""), ("wechat", "")])
    assert idx == 0


def test_select_out_of_range_re_prompts() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["3", "1"]))
    idx = ui.select([("feishu", ""), ("wechat", "")])
    assert idx == 0


def test_select_shows_numbered_options_with_title() -> None:
    script = _ScriptedInput(["1"])
    ui = InteractiveInput(input_fn=script)
    ui.select([("feishu", ""), ("wechat", "")], title="频道选择")
    # The title and numbered options are printed (not prompted) — capture stdout instead
    # Here we assert the prompt shown to input_fn contains the select hint
    assert any("选择" in p for p in script.prompts)


# -- InteractiveInput.prompt (injected input_fn path) -----------------------


def test_prompt_returns_stripped_value() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["  hello  "]))
    assert ui.prompt("name: ") == "hello"


def test_prompt_empty_line_returns_none_meaning_esc() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput([""]))
    assert ui.prompt("name: ") is None


def test_prompt_records_prompt_string() -> None:
    script = _ScriptedInput(["value"])
    ui = InteractiveInput(input_fn=script)
    ui.prompt("webhook URL: ")
    assert script.prompts == ["webhook URL: "]


# -- InteractiveInput.confirm (injected input_fn path) ----------------------


def test_confirm_yes_returns_true() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["y"]))
    assert ui.confirm("确认? (y/n): ") is True


def test_confirm_yes_full_word_returns_true() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["yes"]))
    assert ui.confirm("确认? ") is True


def test_confirm_no_returns_false() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["n"]))
    assert ui.confirm("确认? ") is False


def test_confirm_empty_line_returns_none_meaning_esc() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput([""]))
    assert ui.confirm("确认? ") is None


def test_confirm_invalid_then_yes() -> None:
    ui = InteractiveInput(input_fn=_ScriptedInput(["maybe", "y"]))
    assert ui.confirm("确认? ") is True


# -- arrow_select module function: guard behaviour --------------------------


def test_arrow_select_returns_none_when_prompt_toolkit_missing(monkeypatch) -> None:
    from clawcodex_ext.cli import _interactive

    monkeypatch.setattr(_interactive, "_HAS_PROMPT_TOOLKIT", False)
    assert _interactive.arrow_select([("a", ""), ("b", "")]) is None


def test_arrow_select_returns_none_for_empty_options(monkeypatch) -> None:
    from clawcodex_ext.cli import _interactive

    # Even when prompt_toolkit is available, no options => None
    monkeypatch.setattr(_interactive, "_HAS_PROMPT_TOOLKIT", True)
    assert _interactive.arrow_select([]) is None


# -- Integration: InteractiveInput falls back when prompt_toolkit missing ----


def test_select_falls_back_to_input_fn_when_prompt_toolkit_missing(monkeypatch) -> None:
    from clawcodex_ext.cli import _interactive

    monkeypatch.setattr(_interactive, "_HAS_PROMPT_TOOLKIT", False)
    ui = _interactive.InteractiveInput(input_fn=_ScriptedInput(["1"]))
    assert ui.select([("a", ""), ("b", "")]) == 0


def test_prompt_falls_back_to_input_fn_when_prompt_toolkit_missing(monkeypatch) -> None:
    from clawcodex_ext.cli import _interactive

    monkeypatch.setattr(_interactive, "_HAS_PROMPT_TOOLKIT", False)
    ui = _interactive.InteractiveInput(input_fn=_ScriptedInput(["value"]))
    assert ui.prompt("x: ") == "value"


def test_clear_rendered_lines_writes_ansi_escape(capsys) -> None:
    """For N > 0, move up N lines and clear to the bottom of the screen."""
    from clawcodex_ext.cli import _interactive

    _interactive._clear_rendered_lines(3)
    captured = capsys.readouterr()
    assert captured.out == "\033[3A\033[J"


def test_clear_rendered_lines_zero_is_noop(capsys) -> None:
    """For N = 0, write no escape sequence."""
    from clawcodex_ext.cli import _interactive

    _interactive._clear_rendered_lines(0)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_clear_rendered_lines_negative_is_noop(capsys) -> None:
    """For N < 0, write no escape sequence."""
    from clawcodex_ext.cli import _interactive

    _interactive._clear_rendered_lines(-1)
    captured = capsys.readouterr()
    assert captured.out == ""
