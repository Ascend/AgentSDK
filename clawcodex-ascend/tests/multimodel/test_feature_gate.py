#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from clawcodex_ext.feature_gate import get_registry
from clawcodex_ext.multimodel.cli import run_multimodel_command
from clawcodex_ext.multimodel.config import GroupConfig, SlotConfig
from clawcodex_ext.multimodel.factory import build_router


def test_multimodel_is_registered_disabled_by_default() -> None:
    flag = get_registry().get_flag("MULTIMODEL")
    assert flag is not None
    assert flag.default is False


def test_cli_and_runtime_factory_reject_when_gate_is_off(monkeypatch) -> None:
    registry = get_registry()
    registry.clear_override("MULTIMODEL")
    monkeypatch.setenv("CLAWCODEX_FEATURE_MULTIMODEL", "0")
    assert run_multimodel_command(["status"]) == 2
    group = GroupConfig("parallel", (SlotConfig("one", "openai", "gpt-4o"),))
    try:
        build_router(group, lambda *_args: object())
    except RuntimeError as exc:
        assert "disabled" in str(exc)
    else:  # pragma: no cover - assertion readability
        raise AssertionError("factory accepted a disabled multi-model feature")
