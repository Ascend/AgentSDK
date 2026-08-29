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

import re
import shlex


def quote(s: str) -> str:
    if not s:
        return "''"
    if re.match(r"^[a-zA-Z0-9._/=:@%^,+-]+$", s):
        return s
    return shlex.quote(s)


def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return [command]


def is_glob_pattern(s: str) -> bool:
    return bool(re.search(r"(?<!\\)[*?[\]]", s))


def expand_home(path: str) -> str:
    if path.startswith("~/") or path == "~":
        import os

        return os.path.expanduser(path)
    return path
