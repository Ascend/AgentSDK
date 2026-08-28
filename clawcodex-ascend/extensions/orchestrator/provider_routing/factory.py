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

"""Build immutable stage routing from a parsed workflow configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from extensions.capabilities.provider_protocol import LLMProviderProtocol
else:
    LLMProviderProtocol = Any
from extensions.orchestrator.contracts.provider_routing import StageModel, StageProvider

from .config import ProviderRoutingConfig
from .router import ProviderReference, StaticProviderRouter


def _clean_map(values: Mapping[str, Any] | None) -> dict[str, str]:
    """Normalize a loose YAML-derived mapping and discard blank entries."""
    if not values:
        return {}
    result: dict[str, str] = {}
    for raw_stage, raw_value in values.items():
        stage = str(raw_stage).strip().casefold()
        value = str(raw_value).strip() if raw_value is not None else ""
        if stage and value:
            result[stage] = value
    return result


def build_provider_router(
    workflow: Any,
    *,
    provider_overrides: Mapping[str, LLMProviderProtocol | str] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    explicit_overrides_win: bool = False,
) -> StaticProviderRouter:
    """Create the one routing snapshot used for an execution-mode run.

    ``provider_overrides`` and ``model_overrides`` carry legacy mode-local
    settings such as ``modes.pipeline.stage_models``. The canonical
    ``agent.stages`` entries are applied last, so the unified configuration
    wins when both old and new forms configure the same stage.
    """
    agent = workflow.agent
    modes = getattr(workflow, "modes", None)
    routing: ProviderRoutingConfig = (
        getattr(agent, "provider_routing", None)
        or getattr(modes, "provider_routing", None)
        or ProviderRoutingConfig(stage_overrides=getattr(agent, "stage_overrides", {}) or {})
    )
    explicit_providers: dict[str, LLMProviderProtocol] = {}
    for stage, provider in (provider_overrides or {}).items():
        canonical = str(stage).strip().casefold()
        if canonical:
            explicit_providers[canonical] = ProviderReference(provider) if isinstance(provider, str) else provider
    explicit_models = _clean_map(model_overrides)
    providers = dict(explicit_providers)
    models = dict(explicit_models)

    for raw_stage, raw_route in routing.stage_overrides.items():
        if not isinstance(raw_route, Mapping):
            continue
        stage = str(raw_stage).strip().casefold()
        if not stage:
            continue
        provider = str(raw_route.get("provider") or "").strip()
        model = str(raw_route.get("model") or "").strip()
        if provider:
            providers[stage] = ProviderReference(provider)
        if model:
            models[stage] = model

    if explicit_overrides_win:
        providers.update(explicit_providers)
        models.update(explicit_models)

    return StaticProviderRouter(
        default_provider=ProviderReference(str(getattr(agent, "provider", "") or "")),
        default_model=str(getattr(agent, "model", None) or ""),
        stage_providers=(StageProvider(stage, provider) for stage, provider in providers.items()),
        stage_models=(StageModel(stage, model) for stage, model in models.items()),
    )


def stage_id_from_run_kind(run_kind: Any) -> str:
    """Convert runtime labels such as ``pipeline:tester:retry1`` to ``tester``."""
    normalized = str(run_kind or "single").strip().casefold()
    parts = [part for part in normalized.split(":") if part]
    if parts and parts[0] in {"pipeline", "debate"}:
        return parts[1] if len(parts) > 1 else parts[0]
    return parts[0] if parts else "single"


__all__ = ["build_provider_router", "stage_id_from_run_kind"]
