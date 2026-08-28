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

# Backward-compatibility stub — re-exports from runtime/agent_builder.py
from extensions.sop_converter.runtime.agent_builder import (
    AgentBuildResult,
    AgentBuilder,
    write_agent_markdown,
    AgentPersistenceSpec,
    persist_converted_agent,
)

__all__ = [
    "AgentBuildResult",
    "AgentBuilder",
    "write_agent_markdown",
    "AgentPersistenceSpec",
    "persist_converted_agent",
]

from extensions.sop_converter.runtime import agent_builder as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _impl
