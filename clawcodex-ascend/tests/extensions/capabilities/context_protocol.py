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
#
"""ContextBuilder Protocol — interface for context/prompt construction.

Concrete implementation: src/context_system/builder.py (build_context_prompt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

__all__ = ["ContextBuilderProtocol"]


class ContextBuilderProtocol(Protocol):
    """Protocol for building tool execution context.

    Provides: build_context_prompt(workspace_root, *, cwd) -> str.
    """

    def build_context_prompt(
        self,
        workspace_root: str | Path,
        *,
        cwd: str | Path | None = None,
    ) -> str: ...  # pragma: no cover
