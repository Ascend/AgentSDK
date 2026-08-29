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

from types import SimpleNamespace

from clawcodex_ext.intent_forecast.messages import ForecastResult, format_forecast_for_display
from clawcodex_ext.tui.app import ClawCodexTUI


class _Transcript:
    def __init__(self, *, is_mounted: bool) -> None:
        self.is_mounted = is_mounted
        self.rows: list[tuple[str, str, str | None]] = []
        self.scrolls = 0

    def append_system(self, text: str, *, style: str = "muted", render: str | None = None) -> None:
        self.rows.append((text, style, render))

    def scroll_end(self, *, animate: bool = False) -> None:
        self.scrolls += 1


def test_tui_buffers_forecast_until_transcript_is_mounted() -> None:
    app = ClawCodexTUI.__new__(ClawCodexTUI)
    transcript = _Transcript(is_mounted=False)
    app._repl_screen = SimpleNamespace(transcript=transcript)
    app._pending_system_messages = []
    app.call_after_refresh = lambda callback: None  # type: ignore[method-assign]

    text = format_forecast_for_display(ForecastResult(generated=False, suggestions=[]))
    app._append_repl_system_message(text, style="light", render="markdown")

    assert transcript.rows == []
    assert app._pending_system_messages == [(text, "light", "markdown")]

    transcript.is_mounted = True
    app._flush_pending_system_messages()

    assert transcript.rows == [(text, "light", "markdown")]
    assert app._pending_system_messages == []


def test_tui_scrolls_after_appending_mounted_forecast() -> None:
    app = ClawCodexTUI.__new__(ClawCodexTUI)
    transcript = _Transcript(is_mounted=True)
    app._repl_screen = SimpleNamespace(transcript=transcript)
    app._pending_system_messages = []
    callbacks = []
    app.call_after_refresh = callbacks.append  # type: ignore[method-assign]

    text = format_forecast_for_display(ForecastResult(generated=False, suggestions=[]))
    app._append_repl_system_message(text, style="light", render="markdown")

    assert transcript.rows == [(text, "light", "markdown")]
    assert len(callbacks) == 1
    callbacks[0]()
    assert transcript.scrolls == 1
