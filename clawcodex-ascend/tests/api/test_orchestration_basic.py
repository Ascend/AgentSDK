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

"""Smoke test for extensions.api.orchestration — import check.

OrchestrationSubsystem has module-level imports from extensions.orchestrator.config.schema
(Part A.2, not yet migrated). This test verifies the module is syntactically valid
and skips when the dependency is missing.
"""

from __future__ import annotations

import pytest


def test_orchestration_import() -> None:
    try:
        from extensions.api.orchestration import OrchestrationSubsystem

        assert OrchestrationSubsystem is not None
    except ImportError:
        pytest.skip("config.schema not yet migrated (Part A.2)")
