#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Model command errors."""

from __future__ import annotations


class ModelCommandError(Exception):
    """Base class for model command failures."""


class UnknownModelError(ModelCommandError):
    def __init__(self, model: str, provider: str | None = None) -> None:
        suffix = f" for provider {provider}" if provider else ""
        super().__init__(f"Unknown model: {model}{suffix}")


class ProviderMismatchError(ModelCommandError):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__(f"Model {model} is not available for provider {provider}")


class AmbiguousModelError(ModelCommandError):
    def __init__(self, model: str, providers: list[str]) -> None:
        super().__init__(f"Model {model} is available for multiple providers: {', '.join(providers)}. Use --provider.")


class UnsupportedScopeError(ModelCommandError):
    def __init__(self, scope: str) -> None:
        super().__init__(f"Unsupported scope: {scope}. Only user scope is supported.")
