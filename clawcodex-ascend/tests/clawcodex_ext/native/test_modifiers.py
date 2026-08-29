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

"""modifier-key detection tests without real keyboard hardware."""

from __future__ import annotations

import pytest
from clawcodex_ext.native import load_or_fallback
from clawcodex_ext.native.modifiers import (
    ModifiersFallback,
    ModifiersModule,
    ModifierState,
)


def test_modifiers_registered():
    from clawcodex_ext.native import NativeModuleRegistry

    assert NativeModuleRegistry.is_registered("modifiers")


def test_modifier_state_defaults():
    s = ModifierState()
    assert (s.shift, s.ctrl, s.alt, s.meta) == (False, False, False, False)
    assert s.any_pressed() is False


def test_modifier_state_any_pressed():
    s = ModifierState(ctrl=True)
    assert s.any_pressed() is True


def test_modifier_state_equality():
    assert ModifierState(shift=True) == ModifierState(shift=True)
    assert ModifierState(shift=True) != ModifierState(ctrl=True)


def test_modifiers_fallback_returns_all_false():
    fb = ModifiersFallback()
    assert fb.is_available() is False
    assert fb.get_version() == "fallback-noop"
    state = fb.current_state()
    assert state == ModifierState()


def test_modifiers_load_or_fallback_returns_object():
    inst = load_or_fallback("modifiers")
    assert inst is not None
    assert isinstance(inst, (ModifiersModule, ModifiersFallback))


def test_modifiers_current_state_raises_when_unavailable():
    mod = ModifiersModule()
    mod._backend = None
    from clawcodex_ext.native import NativeModuleError

    with pytest.raises(NativeModuleError):
        mod.current_state()


def test_modifiers_module_backend_detection():
    """Construction selects no backend, pynput, or evdev for the environment."""
    mod = ModifiersModule()
    assert mod._backend in (None, "pynput", "evdev")
    assert mod.is_available() == (mod._backend is not None)
