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
    InputAction,
    MouseButton,
    ScreenRegion,
    ScrollDirection,
    WindowRef,
)


def test_screen_region_defaults() -> None:
    region = ScreenRegion()
    assert region.to_dict() == {"x": 0, "y": 0, "width": 1920, "height": 1080}


def test_screen_region_round_trip() -> None:
    region = ScreenRegion(x=10, y=20, width=800, height=600)
    assert ScreenRegion.from_dict(region.to_dict()) == region


def test_screen_region_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        ScreenRegion(width=0)
    with pytest.raises(ValueError):
        ScreenRegion(height=-1)


def test_screen_region_rejects_negative_origin() -> None:
    with pytest.raises(ValueError):
        ScreenRegion(x=-1)


def test_screen_region_rejects_oversized_extent() -> None:
    with pytest.raises(ValueError):
        ScreenRegion(x=32767, width=100)


def test_window_ref_requires_non_empty_title() -> None:
    with pytest.raises(ValueError):
        WindowRef(title="")
    with pytest.raises(ValueError):
        WindowRef(title="   ")


def test_window_ref_round_trip() -> None:
    ref = WindowRef(title="editor", pid=1234, window_id="0xdead")
    assert WindowRef.from_dict(ref.to_dict()) == ref


def test_window_ref_from_dict_rejects_blank_title() -> None:
    with pytest.raises(ValueError):
        WindowRef.from_dict({"title": ""})


def test_input_action_rejects_non_dict_args() -> None:
    with pytest.raises(ValueError):
        InputAction(kind="click", args="not-a-dict")  # type: ignore[arg-type]


def test_input_action_rejects_empty_kind() -> None:
    with pytest.raises(ValueError):
        InputAction(kind="", args={})


def test_mouse_button_values_are_stable() -> None:
    assert {b.value for b in MouseButton} == {"left", "middle", "right"}


def test_scroll_direction_values_are_stable() -> None:
    assert {d.value for d in ScrollDirection} == {"up", "down", "left", "right"}
