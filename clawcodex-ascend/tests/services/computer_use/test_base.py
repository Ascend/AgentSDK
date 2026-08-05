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

import inspect

from src.services.computer_use import (
    ClipboardManager,
    InputSimulator,
    ScreenshotProvider,
    WindowManager,
)


def _abstract_methods(cls: type) -> set[str]:
    return set(getattr(cls, "__abstractmethods__", set()))


def test_screenshot_provider_has_three_abstract_methods() -> None:
    methods = _abstract_methods(ScreenshotProvider)
    assert methods == {"capture_fullscreen", "capture_region", "capture_window", "is_dry_run"}


def test_input_simulator_has_required_abstract_methods() -> None:
    methods = _abstract_methods(InputSimulator)
    expected = {
        "move_mouse",
        "click",
        "double_click",
        "type_text",
        "press_key",
        "scroll",
        "drag",
        "is_dry_run",
    }
    assert methods == expected


def test_clipboard_manager_signatures() -> None:
    methods = _abstract_methods(ClipboardManager)
    assert methods == {"get_text", "set_text", "is_dry_run"}


def test_window_manager_signatures() -> None:
    methods = _abstract_methods(WindowManager)
    assert methods == {"list_windows", "focus_window", "close_window", "is_dry_run"}


def test_methods_are_async_or_sync_consistently() -> None:
    # None of the Computer Use providers should be coroutines; everything is
    # synchronous on purpose so a future async Tool layer can wrap it.
    for cls in (ScreenshotProvider, InputSimulator, ClipboardManager, WindowManager):
        for method_name in _abstract_methods(cls):
            method = getattr(cls, method_name)
            assert not inspect.iscoroutinefunction(method), f"{cls.__name__}.{method_name} must be sync"
