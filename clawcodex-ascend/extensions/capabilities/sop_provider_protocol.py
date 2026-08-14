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
"""SOPAssistantProvider Protocol — lightweight chat boundary for SOP grouping.

Trims ``clawcodex_ext.providers.base.BaseProvider`` to the capability
``skill_grouper``'s LLM_SEMANTIC path uses: ``provider.chat(messages) ->
response.content``. Messages accept ``dict[str, Any]`` or ``ChatMessage``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["SOPAssistantProviderProtocol"]


# Mirrors ``clawcodex_ext.providers.base.MessageInput``; typed loosely to
# keep the Protocol duck-typed without dragging the provider package in.
SOPProviderMessage = Any


@runtime_checkable
class SOPAssistantProviderProtocol(Protocol):
    """Single-shot chat boundary used by ``skill_grouper``; MUST return str."""

    def chat(self, messages: list[SOPProviderMessage]) -> str: ...
