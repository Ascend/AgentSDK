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

from clawcodex_ext.query.query import _mark_proactive_context_blocked_on_error
from clawcodex_ext.services.proactive import (
    get_default_controller,
    reset_default_controller_for_tests,
)


def test_query_error_blocks_active_proactive_context() -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test")

    _mark_proactive_context_blocked_on_error(RuntimeError("api failed"))

    assert get_default_controller().state.phase == "blocked"


def test_query_error_does_not_activate_inactive_proactive_context() -> None:
    ctrl = reset_default_controller_for_tests()

    _mark_proactive_context_blocked_on_error(RuntimeError("api failed"))

    assert ctrl.state.phase == "inactive"
