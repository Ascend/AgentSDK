#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Unit tests for :mod:`clawcodex_ext.utils.key_format`.

The two consumers — prompt_toolkit REPL and Textual TUI — speak
different dialects for the same logical key. The helper must keep
them in sync and render a stable display form so the ghost-text hint
matches the binding.
"""

from __future__ import annotations

from clawcodex_ext.utils.key_format import (
    display_key,
    to_prompt_toolkit_key,
    to_textual_key,
)


class TestToPromptToolkitKey:
    def test_canonical_c_e_passes_through(self):
        assert to_prompt_toolkit_key("c-e") == "c-e"

    def test_ctrl_plus_collapsed_to_c_dash(self):
        assert to_prompt_toolkit_key("ctrl+e") == "c-e"

    def test_uppercase_normalised(self):
        assert to_prompt_toolkit_key("C-E") == "c-e"
        assert to_prompt_toolkit_key("CTRL+E") == "c-e"

    def test_plain_tab_passes_through(self):
        assert to_prompt_toolkit_key("tab") == "tab"

    def test_empty_falls_back_to_c_e(self):
        assert to_prompt_toolkit_key("") == "c-e"

    def test_whitespace_stripped(self):
        assert to_prompt_toolkit_key("  c-j  ") == "c-j"


class TestToTextualKey:
    def test_c_dash_becomes_ctrl_plus(self):
        assert to_textual_key("c-e") == "ctrl+e"

    def test_ctrl_plus_passes_through(self):
        assert to_textual_key("ctrl+e") == "ctrl+e"

    def test_plain_tab_unchanged(self):
        assert to_textual_key("tab") == "tab"

    def test_empty_falls_back_to_ctrl_e(self):
        assert to_textual_key("") == "ctrl+e"

    def test_function_key(self):
        assert to_textual_key("f2") == "f2"


class TestDisplayKey:
    def test_default_c_e_renders_classic_hint(self):
        assert display_key("c-e") == "CTRL + E"

    def test_tab_renders_as_tab(self):
        assert display_key("tab") == "TAB"

    def test_ctrl_plus_form_normalised(self):
        assert display_key("ctrl+e") == "CTRL + E"

    def test_function_keys_preserved(self):
        assert display_key("f2") == "F2"

    def test_uppercase_input_lowered_first(self):
        assert display_key("C-J") == "CTRL + J"

    def test_empty_falls_back_to_classic_hint(self):
        assert display_key("") == "CTRL + E"


class TestRoundTrip:
    """The two consumers must agree on the same logical key."""

    def test_c_e_round_trip(self):
        # c-e -> prompt_toolkit (c-e) and Textual (ctrl+e) — both
        # downstream consumers can dispatch and render the hint.
        assert to_prompt_toolkit_key("c-e") == "c-e"
        assert to_textual_key("c-e") == "ctrl+e"
        assert display_key("c-e") == "CTRL + E"

    def test_tab_round_trip(self):
        assert to_prompt_toolkit_key("tab") == "tab"
        assert to_textual_key("tab") == "tab"
        assert display_key("tab") == "TAB"
