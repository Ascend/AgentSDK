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

"""Shared path utilities for tools (relative paths, suggestions)."""

from __future__ import annotations

import os
from pathlib import Path


def to_relative_path(absolute: str, cwd: str | Path) -> str:
    """Convert an absolute path to a relative one if shorter."""
    try:
        rel = os.path.relpath(absolute, str(cwd))
    except ValueError:
        return absolute
    if len(rel) < len(absolute):
        return rel
    return absolute


def suggest_path_under_cwd(path: str, cwd: str | Path) -> str | None:
    """Suggest a corrected path if the given path looks like a relative path
    that should be under cwd.
    """
    name = os.path.basename(path)
    if not name:
        return None
    candidate = os.path.join(str(cwd), name)
    if os.path.exists(candidate):
        return candidate
    return None
