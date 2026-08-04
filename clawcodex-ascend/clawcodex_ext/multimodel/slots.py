#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Runtime description of a provider participating in a model group."""

from __future__ import annotations

from dataclasses import dataclass

from clawcodex_ext.providers.base import BaseProvider


@dataclass(frozen=True)
class ProviderSlot:
    """A provider and the policy used when the router invokes it."""

    name: str
    provider: BaseProvider
    model: str | None = None
    weight: float = 1.0
    timeout_ms: int = 120_000
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("provider slot name must not be empty")
        if self.weight <= 0:
            raise ValueError("provider slot weight must be greater than zero")
        if self.timeout_ms <= 0:
            raise ValueError("provider slot timeout_ms must be greater than zero")
