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

"""Agent catalog home-directory constants.

The legacy agent-catalog path resolver (``resolve_catalog_path`` /
``CatalogLocation``) was removed: the F-55 agent catalog is deprecated and
runtime persistence has migrated to :mod:`resource_catalog`. Only the
home-directory environment variables still consumed by
:mod:`resource_catalog` are kept here.
"""

HOME_ROOT_ENV = "CLAWCODEX_HOME"
HOME_ONLY_ENV = "CLAWCODEX_CATALOG_HOME_ONLY"

__all__ = [
    "HOME_ONLY_ENV",
    "HOME_ROOT_ENV",
]
