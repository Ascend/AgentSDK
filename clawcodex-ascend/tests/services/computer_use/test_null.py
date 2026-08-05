#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

from __future__ import annotations

import pytest

from src.services.computer_use import (
    DryRunRecorder,
    MouseButton,
    NullClipboardManager,
    NullInputSimulator,
    NullScreenshotProvider,
    NullWindowManager,
    ScreenRegion,
    WindowRef,
    build_null_suite,
)


def test_null_screenshot_returns_png_and_records() -> None:
    recorder = DryRunRecorder()
    provider = NullScreenshotProvider(recorder)

    full = provider.capture_fullscreen()
    assert full.startswith(b"\x89PNG")
    assert provider.is_dry_run is True
    assert recorder.screenshots()[0][0] == "fullscreen"

    region = ScreenRegion(x=0, y=0, width=320, height=200)
    assert provider.capture_region(region).startswith(b"\x89PNG")
    assert recorder.screenshots()[1][2]["width"] == 320

    assert provider.capture_window(WindowRef(title="editor")) is None


def test_null_input_records_actions_and_validates_coordinates() -> None:
    recorder = DryRunRecorder()
    sim = NullInputSimulator(recorder)

    sim.move_mouse(10, 20)
    sim.click(MouseButton.RIGHT, x=100, y=200)
    sim.double_click()
    sim.type_text("hello world")
    sim.press_key("enter")
    sim.scroll(dx=1, dy=-1)
    sim.drag(0, 0, 50, 50)

    assert sim.is_dry_run is True
    kinds = [a.kind for a in recorder.actions()]
    assert kinds == [
        "move_mouse",
        "click",
        "double_click",
        "type_text",
        "press_key",
        "scroll",
        "drag",
    ]


def test_null_input_rejects_invalid_coordinates() -> None:
    sim = NullInputSimulator()
    with pytest.raises(TypeError):
        sim.move_mouse("10", 20)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sim.move_mouse(-1, 0)
    with pytest.raises(ValueError):
        sim.move_mouse(0, 99999)


def test_null_input_rejects_empty_key() -> None:
    sim = NullInputSimulator()
    with pytest.raises(ValueError):
        sim.press_key("")
    with pytest.raises(ValueError):
        sim.press_key("   ")


def test_null_input_rejects_non_string_text() -> None:
    sim = NullInputSimulator()
    with pytest.raises(TypeError):
        sim.type_text(123)  # type: ignore[arg-type]


def test_null_clipboard_round_trip() -> None:
    recorder = DryRunRecorder()
    manager = NullClipboardManager(recorder)

    assert manager.get_text() == ""
    manager.set_text("hello")
    assert manager.get_text() == "hello"
    assert any(a.kind == "clipboard_set" for a in recorder.actions())


def test_null_window_manager_records_and_returns_empty_list() -> None:
    recorder = DryRunRecorder()
    manager = NullWindowManager(recorder)

    assert manager.list_windows() == []
    assert manager.focus_window(WindowRef(title="x")) is False
    assert manager.close_window(WindowRef(title="x")) is False
    assert {a.kind for a in recorder.actions()} == {"focus_window", "close_window"}


def test_build_null_suite_shares_recorder() -> None:
    suite = build_null_suite()
    recorder = suite["recorder"]
    suite["input"].click()
    suite["clipboard"].set_text("hi")
    suite["window"].focus_window(WindowRef(title="x"))
    assert recorder.action_count == 3
