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

"""Configuration owned by per-stage provider/model routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


def _model_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(stage).strip().casefold(): str(model).strip()
        for stage, model in raw.items()
        if str(stage).strip() and str(model or "").strip()
    }


def normalize_stage_overrides(
    raw: Any,
    *,
    value_resolver: Callable[[str | None], str | None] | None = None,
) -> dict[str, dict[str, str]]:
    """Normalize the canonical ``agent.stages`` YAML mapping."""
    if not isinstance(raw, Mapping):
        return {}
    resolve = value_resolver or (lambda value: value)
    routes: dict[str, dict[str, str]] = {}
    for raw_stage, raw_route in raw.items():
        if not isinstance(raw_route, Mapping):
            continue
        stage = str(raw_stage).strip().casefold()
        if not stage:
            continue
        route: dict[str, str] = {}
        for name in ("provider", "model"):
            raw_value = raw_route.get(name)
            value = resolve(str(raw_value) if raw_value is not None else None)
            normalized = str(value or "").strip()
            if normalized:
                route[name] = normalized
        if route:
            routes[stage] = route
    return routes


@dataclass(slots=True)
class ProviderRoutingConfig:
    """All provider/model selection fields formerly spread across schema.py."""

    router_model: str = "deepseek-v4-flash"
    pipeline_stage_models: dict[str, str] = field(default_factory=dict)
    debate_judge_model: str | None = None
    debate_proposer_models: dict[str, str] = field(default_factory=dict)
    stage_overrides: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        agent_raw: Mapping[str, Any],
        modes_raw: Mapping[str, Any],
        *,
        value_resolver: Callable[[str | None], str | None] | None = None,
    ) -> "ProviderRoutingConfig":
        router = modes_raw.get("router") or {}
        pipeline = modes_raw.get("pipeline") or {}
        debate = modes_raw.get("debate") or {}
        router_model = str(router.get("model", "deepseek-v4-flash")).strip()
        judge_model = str(debate.get("judge_model") or "").strip() or None
        return cls(
            router_model=router_model or "deepseek-v4-flash",
            pipeline_stage_models=_model_map(pipeline.get("stage_models")),
            debate_judge_model=judge_model,
            debate_proposer_models=_model_map(debate.get("proposer_models")),
            stage_overrides=normalize_stage_overrides(
                agent_raw.get("stages"),
                value_resolver=value_resolver,
            ),
        )


__all__ = ["ProviderRoutingConfig", "normalize_stage_overrides"]
