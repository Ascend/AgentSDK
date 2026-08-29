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

"""Detect long standalone or leading ``sleep N`` patterns to avoid.

Catches ``sleep 6``, ``sleep 6 && check``, ``sleep 6; check`` -- but permits
the 1-5 second waits recommended by the Bash tool prompt.  Sleep inside
pipelines, subshells, or scripts is left alone.
"""

from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||[;])\s*")
_SLEEP_RE = re.compile(r"^sleep\s+(\d+)\s*$")


def detect_blocked_sleep_pattern(command: str) -> str | None:
    """Return a description of the blocked sleep pattern, or ``None`` if OK."""
    parts = _SPLIT_RE.split(command.strip())
    if not parts:
        return None
    first = parts[0].strip()
    m = _SLEEP_RE.match(first)
    if not m:
        return None
    secs = int(m.group(1))
    if secs <= 5:
        return None

    rest = " ".join(p.strip() for p in parts[1:] if p.strip())
    if rest:
        return f"sleep {secs} followed by: {rest}"
    return f"standalone sleep {secs}"
