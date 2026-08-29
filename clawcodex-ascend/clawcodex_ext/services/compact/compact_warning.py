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

"""
Compact warning suppression state.

Port of ``typescript/src/services/compact/compactWarningState.ts``.

Tracks whether the "context left until autocompact" warning should be
suppressed.  We suppress immediately after successful compaction since
accurate token counts are unavailable until the next API response.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_suppressed: bool = False


def suppress_compact_warning() -> None:
    """Suppress the compact warning.  Call after successful compaction."""
    global _suppressed
    with _lock:
        _suppressed = True


def clear_compact_warning_suppression() -> None:
    """Clear suppression.  Called at the start of a new compact attempt."""
    global _suppressed
    with _lock:
        _suppressed = False


def is_compact_warning_suppressed() -> bool:
    """Return whether the compact warning is currently suppressed."""
    with _lock:
        return _suppressed
