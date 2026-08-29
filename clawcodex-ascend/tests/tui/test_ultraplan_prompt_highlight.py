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

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from clawcodex_ext.tui.widgets.prompt_input import PromptInput


def _prompt() -> PromptInput:
    return PromptInput(words_provider=lambda: [])


def test_prompt_input_shows_ultraplan_trigger_preview() -> None:
    prompt = _prompt()
    prompt._refresh_ultraplan_trigger_preview("/ultraplan refactor")  # type: ignore[attr-defined]
    assert not prompt._ultraplan_trigger_preview.has_class("-hidden")  # type: ignore[attr-defined]
    assert prompt._ultraplan_trigger_preview.renderable.plain == "ultraplan: /ultraplan refactor"  # type: ignore[attr-defined]


def test_prompt_input_hides_preview_for_middle_trigger() -> None:
    prompt = _prompt()
    prompt._refresh_ultraplan_trigger_preview("echo /ultraplan")  # type: ignore[attr-defined]
    assert prompt._ultraplan_trigger_preview.has_class("-hidden")  # type: ignore[attr-defined]
