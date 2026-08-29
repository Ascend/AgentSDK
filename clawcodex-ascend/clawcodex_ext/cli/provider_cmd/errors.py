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

"""Provider command errors."""

from __future__ import annotations


class ProviderCommandError(Exception):
    """Base class for provider command failures."""


class UnknownProviderError(ProviderCommandError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"Unknown provider: {provider}")


class NotConfiguredError(ProviderCommandError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"Provider is not configured: {provider}")


class UnsupportedScopeError(ProviderCommandError):
    def __init__(self, scope: str) -> None:
        super().__init__(f"Unsupported scope: {scope}. Only user scope is supported.")
