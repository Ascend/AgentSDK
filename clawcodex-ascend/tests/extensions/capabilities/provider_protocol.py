#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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
#

"""LLMProvider Protocol — interface for LLM provider abstraction.

Abstraction boundary between Layer 1 (upstream) and Layer 3 (features);
concrete implementation: src/providers/base.py (BaseProvider).
"""

from __future__ import annotations

from typing import Protocol, Generator

__all__ = ["LLMProviderProtocol"]


class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider abstraction.

    Provides: chat() and streaming chat_stream().
    """

    def chat(
        self,
        messages: "list[MessageInput]",  # noqa: F821
        tools: "list[dict[str, object]] | None" = None,
        **kwargs: object,
    ) -> "ChatResponse": ...  # pragma: no cover  # noqa: F821

    def chat_stream(
        self,
        messages: "list[MessageInput]",  # noqa: F821
        tools: "list[dict[str, object]] | None" = None,
        **kwargs: object,
    ) -> Generator[str, None, None]: ...  # pragma: no cover
