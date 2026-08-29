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

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class EnvExpansionResult:
    expanded: str
    missing_vars: list[str] = field(default_factory=list)


def expand_env_vars_in_string(value: str) -> EnvExpansionResult:
    missing_vars: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        var_content = match.group(1)
        parts = var_content.split(":-", 1)
        var_name = parts[0]
        default_value = parts[1] if len(parts) > 1 else None

        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        if default_value is not None:
            return default_value

        missing_vars.append(var_name)
        return match.group(0)

    expanded = re.sub(r"\$\{([^}]+)\}", _replace, value)
    return EnvExpansionResult(expanded=expanded, missing_vars=missing_vars)
