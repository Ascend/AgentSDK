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

"""In-tree orchestrator runtime: protocols and clawcodex adapters.

Ships as a monorepo package, not a separate PyPI distribution. When
``ORCHESTRATOR_USE_RUNTIME=1``, callers may take the adapter path;
otherwise they keep using ``extensions.orchestrator`` directly.
"""

from __future__ import annotations

__version__ = "0.1.0a0"

# Package marker only until clawcodex_compat is the default import path.
# See ``extensions.orchestrator_runtime.adapters.clawcodex_compat``.

__all__ = ["__version__"]
