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

import importlib
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "scripts" / "ci" / "pytest_targets.py").is_file(),
    reason="standalone ClawCodex CI helpers are not part of the AgentSDK runtime migration",
)


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend("scripts/ci")
    return importlib.import_module("pytest_targets")


def test_changed_pytest_files_filter_and_normalize(monkeypatch):
    pytest_targets = _load_module(monkeypatch)

    selected = pytest_targets.changed_pytest_files(
        [
            "src/query/engine.py",
            "\ufefftests/api/test_api_retry.py",
            "tests/api/helpers.py",
            r"tests\bridge\test_bridge_api.py",
            "tests/agent/test_agent_loop.py",
            "tests/misc/example_test.py",
        ],
        exclude_prefixes=("tests/agent/",),
    )

    assert selected == [
        "tests/api/test_api_retry.py",
        "tests/bridge/test_bridge_api.py",
        "tests/misc/example_test.py",
    ]


def test_targets_for_preset_preserves_smoke_and_appends_unique_changed_tests(monkeypatch):
    pytest_targets = _load_module(monkeypatch)

    targets = pytest_targets.targets_for_preset(
        "core",
        [
            "tests/fast/test_fast_mode.py",
            "tests/api/test_api_retry.py",
            "tests/api/test_api_retry.py",
            "tests/orchestrator/test_orchestrator_dashboard.py",
        ],
        exclude_prefixes=("tests/orchestrator/",),
    )

    assert targets[: len(pytest_targets.CORE_PYTEST)] == list(pytest_targets.CORE_PYTEST)
    assert "tests/fast/test_fast_mode.py" not in targets
    assert "tests/api/test_api_retry.py" in targets
    assert "tests/orchestrator/test_orchestrator_dashboard.py" not in targets


def test_stability_gate_preset_covers_directory_without_duplicate_changed_files(monkeypatch):
    pytest_targets = _load_module(monkeypatch)

    targets = pytest_targets.targets_for_preset(
        "stability-gate",
        [
            "tests/stability_gate/test_stage1_imports.py",
            "tests/api/test_api_retry.py",
        ],
        include_prefixes=("tests/stability_gate/",),
    )

    assert targets == ["tests/stability_gate"]
