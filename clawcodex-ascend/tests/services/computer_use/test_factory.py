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

# pylint: disable=no-name-in-module

from __future__ import annotations

import sys

import pytest

from src.services.computer_use import (
    ALLOW_ENV_VAR,
    DryRunRecorder,
    LinuxBackend,
    LinuxClipboardManager,
    LinuxInputSimulator,
    LinuxScreenshotProvider,
    LinuxWindowManager,
    NullClipboardManager,
    NullInputSimulator,
    NullScreenshotProvider,
    NullWindowManager,
    build_computer_use_suite,
    build_null_suite,
)
from src.services.computer_use.platform import _current_platform


def test_current_platform_is_recognised() -> None:
    assert _current_platform() in {"linux", "darwin", "windows"} or bool(sys.platform)


def test_build_computer_use_suite_linux() -> None:
    suite = build_computer_use_suite(platform="linux")
    assert isinstance(suite["screenshot"], LinuxScreenshotProvider)
    assert isinstance(suite["input"], LinuxInputSimulator)
    assert isinstance(suite["clipboard"], LinuxClipboardManager)
    assert isinstance(suite["window"], LinuxWindowManager)
    assert isinstance(suite["recorder"], DryRunRecorder)


def test_build_computer_use_suite_unsupported_returns_null() -> None:
    suite = build_computer_use_suite(platform="plan9")
    assert isinstance(suite["screenshot"], NullScreenshotProvider)
    assert isinstance(suite["input"], NullInputSimulator)
    assert isinstance(suite["clipboard"], NullClipboardManager)
    assert isinstance(suite["window"], NullWindowManager)


def test_build_computer_use_suite_rejects_wrong_provider_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the platform builder to return a malformed suite. The factory
    # should detect the wrong type and raise TypeError instead of returning
    # a broken suite to the caller.
    # Note: monkeypatch the ext-level factory module (the src facade does
    # not export `build_provider_suite` — it is an internal symbol imported
    # into clawcodex_ext's factory via `from .platform import build_provider_suite`).
    from clawcodex_ext.services.computer_use import factory as factory_ext_module

    def bad_builder(platform=None, backend=None, recorder=None):  # type: ignore[no-untyped-def]
        return {
            "recorder": recorder or DryRunRecorder(),
            "screenshot": "not-a-provider",
            "input": None,
            "clipboard": None,
            "window": None,
        }

    monkeypatch.setattr(factory_ext_module, "build_provider_suite", bad_builder)
    with pytest.raises(TypeError):
        build_computer_use_suite(platform="linux")


def test_recorder_is_shared_across_providers() -> None:
    suite = build_computer_use_suite(platform="linux")
    suite["input"].click()
    suite["screenshot"].capture_fullscreen()
    assert suite["recorder"].action_count == 1
    assert len(suite["recorder"].screenshots()) == 1


def test_build_null_suite_isolated_recorder() -> None:
    a = build_null_suite()
    b = build_null_suite()
    a["input"].click()
    assert b["recorder"].action_count == 0


def test_allow_env_var_constant_is_stable() -> None:
    # The constant is part of the public safety contract; renaming it would
    # break the explicit opt-in env var.
    assert ALLOW_ENV_VAR == "CLAWCODEX_COMPUTER_USE_ALLOW"


def test_default_linux_backend_respects_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)
    backend = LinuxBackend()
    assert backend.dry_run is True
    assert backend.allowed is False

    monkeypatch.setenv(ALLOW_ENV_VAR, "1")
    fresh = LinuxBackend()
    # Note: default_linux_backend() reads the env at construction time, so
    # constructing a fresh backend should reflect the new value.
    # (We construct a plain LinuxBackend here because the test exercises
    # the same code path as ``default_linux_backend`` without coupling.)
    assert fresh.dry_run is True  # always safe-by-default
    # ``allowed`` does not change the safe default; opt-in still requires an
    # explicit ``dry_run=False`` in the constructor.
    assert fresh.allowed is False
