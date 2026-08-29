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

from pathlib import Path

from clawcodex_ext.services.proactive import reset_default_controller_for_tests
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.tools.sleep import _sleep_call
from extensions.remote_api.core import _response_metadata
from extensions.remote_api.state_reporter import current_automation_state


def test_sleep_tool_enters_proactive_sleep(tmp_path: Path) -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test")
    context = ToolContext(workspace_root=tmp_path)

    result = _sleep_call({"seconds": 0.2}, context)

    assert result.output["proactive"] is True
    assert ctrl.state.phase == "sleeping"


def test_remote_metadata_reports_automation_state() -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test", focus="minimal")

    state = current_automation_state()
    metadata = _response_metadata()

    assert state["phase"] == "active"
    assert metadata["automation_state"]["focus"] == "minimal"
