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

"""Per-hook shell selection — Chapter 12 round 2.

**Deprecated**: Import from ``clawcodex_ext.utils.shell_resolver`` instead.
This module is kept as a re-export shim for existing consumers.
"""

from __future__ import annotations

from clawcodex_ext.utils.shell_resolver import (
    DEFAULT_HOOK_SHELL,
    SHELL_TYPES,
    ShellType,
    build_powershell_args,
    find_powershell_path,
)

__all__ = [
    "DEFAULT_HOOK_SHELL",
    "SHELL_TYPES",
    "ShellType",
    "build_powershell_args",
    "find_powershell_path",
]
