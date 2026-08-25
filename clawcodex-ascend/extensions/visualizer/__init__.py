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

"""Local Session Visualizer.

A standalone web application for visualizing agent execution sessions
via Gantt charts, timelines, and performance analytics.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The asciicast dashboard source adapter used to live here and
# was reverse-registered into ``extensions.agent_dashboard`` on import.
# The adapter moved to ``extensions.recording.visualizer_dashboard_source``
# where its real consumer (``extensions.recording._factories._visualizer_factory``)
# lives. Recording's factory loader imports it lazily so a partial
# checkout that lacks the recording extension cannot break the
# visualizer import path. No module-level registration happens here.
