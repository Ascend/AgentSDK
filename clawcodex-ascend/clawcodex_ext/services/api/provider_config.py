#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
# This file is derived from Clawd Codex (https://github.com/agentforce314/clawcodex),
# which is licensed under the MIT License.
# Copyright (c) 2026 Clawd Codex Team
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderOverride:
    model: str
    base_url: str
    api_key: str


def _normalize(key: str) -> str:
    return key.lower().replace("-", "").replace("_", "")


def resolve_agent_provider(
    name: str | None,
    subagent_type: str | None,
    settings: dict[str, Any] | None,
) -> ProviderOverride | None:
    if not settings:
        return None

    routing = settings.get("agentRouting")
    models = settings.get("agentModels")
    if not routing or not models:
        return None

    normalized_routing: dict[str, str] = {}
    for key, value in routing.items():
        nk = _normalize(key)
        if nk not in normalized_routing:
            normalized_routing[nk] = value

    candidates = [c for c in [name, subagent_type, "default"] if c]
    model_name: str | None = None

    for candidate in candidates:
        match = normalized_routing.get(_normalize(candidate))
        if match:
            model_name = match
            break

    if not model_name:
        return None

    model_config = models.get(model_name)
    if not model_config:
        return None

    return ProviderOverride(
        model=model_name,
        base_url=model_config.get("base_url", ""),
        api_key=model_config.get("api_key", ""),
    )
