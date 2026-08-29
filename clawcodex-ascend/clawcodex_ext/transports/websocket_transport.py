#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Compatibility facade — see :mod:`extensions.ports.transports.websocket_v1`.

P3-out-2: this facade previously routed through
``src.transports.websocket_transport``, which transitively loaded the
``src.transports`` package ``__init__`` and risked a circular import
via ``extensions.ports.transports.hybrid_v1``. Routing directly at
the ``extensions/`` module avoids the package ``__init__`` side
effects; the legacy ``src.transports.*`` path is still a thin
forwarding seam and keeps working for upstream callers.
"""

from extensions.ports.transports.websocket_v1 import *  # noqa: F401,F403  # pylint: disable=wildcard-import,unused-wildcard-import
