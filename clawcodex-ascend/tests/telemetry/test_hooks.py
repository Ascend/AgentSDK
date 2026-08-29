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

"""Tests for the global exception hooks."""

from __future__ import annotations

import sys

import pytest

from telemetry import hooks, recorder
from telemetry.config import TelemetryConfig
from telemetry.recorder import (
    _TelemetryRecorderImpl,
    override_recorder,
    reset_recorder_for_tests,
)
from telemetry.redaction import RedactionConfig, Redactor
from telemetry.storage import utc_date, utc_now


@pytest.fixture(autouse=True)
def _reset_recorder_and_hooks():
    reset_recorder_for_tests()
    hooks.uninstall_exception_hooks()
    yield
    hooks.uninstall_exception_hooks()
    reset_recorder_for_tests()


def test_install_uninstall_is_idempotent():
    hooks.install_exception_hooks()
    first = sys.excepthook
    hooks.install_exception_hooks()  # no double-wrap
    assert sys.excepthook is first
    hooks.uninstall_exception_hooks()
    hooks.uninstall_exception_hooks()  # no-op


def test_excepthook_invokes_recorder(tmp_path):
    # Wire a real recorder so the hook actually emits.
    storage = recorder.LocalJsonlStorage(tmp_path / "telemetry", 7)
    impl = _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder.CompositeReporter(),
    )
    override_recorder(impl)

    captured = {}

    def _boom(exc_type, exc_value, exc_tb):
        captured["called"] = True
        captured["exc"] = exc_value

    # Set the user's hook FIRST, then install so the wrapper chains
    # to ``_boom`` on the install path.
    sys.excepthook = _boom
    hooks.install_exception_hooks()

    try:
        raise RuntimeError("hook-test")
    except RuntimeError as exc:
        # The wrapped excepthook is invoked manually because pytest
        # catches RuntimeError raised inside the test body itself.
        sys.excepthook(type(exc), exc, exc.__traceback__)

    assert captured.get("called") is True
    today = utc_date(utc_now())
    crashes = storage.read_day("crashes", today)
    assert any(row["fields"]["error_class"] == "RuntimeError" for row in crashes)
