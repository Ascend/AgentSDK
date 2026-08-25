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

"""Data parsers for the Multi-Session Visualizer.

NOTE: Several parser modules (transcript_parser, tool_events_parser) have
circular imports with the builders package.  To avoid triggering those at
package-load time, only safe (non-circular) imports are listed here.
Use direct submodule imports for the others, e.g.::

    from .orchestrator_state_parser import OrchestratorStateParser

# Stats file parser kept separate to avoid circular imports.
from .stats_parser import StatsFileParser
    from .transcript_parser import TranscriptParser
    from .tool_events_parser import ToolEventsParser
"""

from .session_parser import SessionMetadataParser
from .multi_agent_parser import MultiAgentParser

# OrchestratorStateParser has no circular deps — safe to expose here.
from .orchestrator_state_parser import OrchestratorStateParser


__all__ = [
    "SessionMetadataParser",
    "MultiAgentParser",
    "OrchestratorStateParser",
]
