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

# Backward-compatibility stub — re-exports from runtime/composite_tools
"""Shim for pre-DECOUPLE import path ``extensions.sop_converter.composite_tools``.

Canonical implementation lives in
``extensions.sop_converter.runtime.composite_tools``.
"""

# pylint: disable=wildcard-import,unused-wildcard-import
from extensions.sop_converter.runtime.composite_tools import *  # noqa: F401, F403
from extensions.sop_converter.runtime.composite_tools import (  # noqa: F401
    _SKIP_PLACEHOLDER_COMPOSITE_TOOLS,
    _composite_to_agent_tool_spec,
    save_spec,
)
from extensions.sop_converter.runtime.composite_tools import __all__ as _runtime_all

__all__ = list(_runtime_all) + [
    "_SKIP_PLACEHOLDER_COMPOSITE_TOOLS",
    "_composite_to_agent_tool_spec",
    "save_spec",
]
