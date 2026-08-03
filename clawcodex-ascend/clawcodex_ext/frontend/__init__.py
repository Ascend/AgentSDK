# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Downstream frontend extensions — plugin-based frontend registry."""

from clawcodex_ext.frontend.protocol import Frontend, FrontendPlugin
from clawcodex_ext.frontend.registry import get_frontend, list_frontends, register_frontend

# Import all plugins to trigger @register_frontend decorator
from clawcodex_ext.frontend import headless  # noqa: F401
from clawcodex_ext.frontend import repl  # noqa: F401
from clawcodex_ext.frontend import tui  # noqa: F401

__all__ = [
    "Frontend",
    "FrontendPlugin",
    "get_frontend",
    "list_frontends",
    "register_frontend",
]
