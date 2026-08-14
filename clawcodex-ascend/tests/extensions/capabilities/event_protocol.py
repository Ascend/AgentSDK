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
"""Event Protocol — interface for tool event emission.

Decouples src/api/query.py from the upstream concrete implementation
(src/tool_system/agent_loop.ToolEvent).
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["ToolEventProtocol"]


class ToolEventProtocol(Protocol):
    """Protocol for tool-use / tool-result / tool-error events.

    Concrete: src/tool_system/agent_loop.ToolEvent.
    """

    @property
    def kind(self) -> str: ...

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_input(self) -> dict[str, Any] | None: ...

    @property
    def tool_output(self) -> Any | None: ...

    @property
    def tool_use_id(self) -> str | None: ...

    @property
    def is_error(self) -> bool: ...

    @property
    def error(self) -> str | None: ...
