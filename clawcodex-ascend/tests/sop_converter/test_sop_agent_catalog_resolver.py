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

"""Tests for the agent catalog home-directory constants.

The legacy path resolver was removed as dead code; these tests lock the
env-var name contract that :mod:`resource_catalog` depends on.
"""

from __future__ import annotations

import unittest

from extensions.sop_converter.core.agent_catalog_resolver import (
    HOME_ONLY_ENV,
    HOME_ROOT_ENV,
)


class TestAgentCatalogConstants(unittest.TestCase):
    def test_home_root_env_name(self) -> None:
        self.assertEqual(HOME_ROOT_ENV, "CLAWCODEX_HOME")

    def test_home_only_env_name(self) -> None:
        self.assertEqual(HOME_ONLY_ENV, "CLAWCODEX_CATALOG_HOME_ONLY")


if __name__ == "__main__":
    unittest.main()
