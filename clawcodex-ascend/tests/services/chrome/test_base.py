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

"""Tests for src/services/chrome/base.py — ABC contract."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import inspect

import pytest

from clawcodex_ext.services.chrome.base import ChromeController, ChromeError


def test_chrome_controller_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ChromeController()  # type: ignore[abstract]


def test_chrome_controller_lists_all_abstract_methods() -> None:
    """The spec names the operation surface; the ABC must declare
    exactly those methods as abstract.
    """
    expected_abstract = {
        "start",
        "stop",
        "navigate",
        "click",
        "type_text",
        "select_option",
        "hover",
        "scroll",
        "screenshot",
        "eval_js",
        "get_visible_text",
        "get_html",
        "start_recording",
        "stop_recording",
    }
    actual = set(ChromeController.__abstractmethods__)
    assert actual == expected_abstract


def test_every_abstract_method_is_async() -> None:
    for name in ChromeController.__abstractmethods__:
        method = getattr(ChromeController, name)
        assert inspect.iscoroutinefunction(method), f"{name} must be async"


def test_concrete_properties_have_default_implementations() -> None:
    """`is_recording`, `current_url`, and `health()` are concrete;
    they should NOT be abstract.
    """
    non_abstract = set(ChromeController.__abstractmethods__)
    assert "is_recording" not in non_abstract
    assert "current_url" not in non_abstract
    assert "health" not in non_abstract


def test_chrome_error_is_runtime_error_subclass() -> None:
    err = ChromeError("boom")
    assert isinstance(err, RuntimeError)
    assert str(err) == "boom"


def test_subclass_must_implement_all_abstract_methods() -> None:
    """A subclass that misses one abstract method cannot be instantiated."""

    class Incomplete(ChromeController):
        async def start(self, headless: bool = True) -> None:
            return None

        # The rest are missing on purpose.

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_complete_subclass_is_instantiable() -> None:
    """A subclass implementing every abstract method should be instantiable."""

    class Complete(ChromeController):
        async def start(self, headless: bool = True) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def navigate(self, url: str):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, url=url)

        async def click(self, selector: str):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data=selector)

        async def type_text(self, selector: str, text: str, *, clear_first: bool = True):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data=text)

        async def select_option(self, selector: str, value: str):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data=value)

        async def hover(self, selector: str):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data=selector)

        async def scroll(self, *, dx: int = 0, dy: int = 1):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data=f"dx={dx},dy={dy}")

        async def screenshot(self, selector: str | None = None, *, full_page: bool = True):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data=b"")

        async def eval_js(self, script: str):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data="null")

        async def get_visible_text(self):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data="")

        async def get_html(self):
            from clawcodex_ext.services.chrome.models import ChromeActionResult

            return ChromeActionResult(success=True, data="")

        async def start_recording(self, output_path: str, *, fps: int = 1) -> None:
            return None

        async def stop_recording(self) -> str:
            return ""

    instance = Complete()
    assert isinstance(instance, ChromeController)
    assert instance.is_recording is False
    assert instance.current_url == ""
    health = instance.health()
    assert health["is_live"] is False
    assert health["is_recording"] is False
