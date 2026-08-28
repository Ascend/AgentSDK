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

"""Immutable implementation of per-stage provider/model routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from extensions.capabilities.provider_protocol import LLMProviderProtocol
else:
    LLMProviderProtocol = Any

from extensions.orchestrator.contracts.provider_routing import (
    StageModel,
    StageProvider,
    _require_stage_id,
)


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    """Resolved provider/model pair for a single execution stage."""

    provider: LLMProviderProtocol
    model: str


@dataclass(frozen=True, slots=True)
class ProviderReference:
    """Named Provider object used before the concrete Layer-3 provider is built.

    The reference satisfies ``LLMProviderProtocol`` and keeps a stable name for
    the current QueryConfig adapter. If a concrete delegate is supplied, chat
    calls are forwarded to it.
    """

    provider_name: str
    delegate: LLMProviderProtocol | None = None

    def __post_init__(self) -> None:
        normalized = self.provider_name.strip()
        if not normalized:
            raise ValueError("provider_name must not be empty")
        object.__setattr__(self, "provider_name", normalized)

    def chat(self, messages: Any, tools: Any = None, **kwargs: object) -> Any:
        if self.delegate is None:
            raise RuntimeError("provider reference has no concrete delegate")
        return self.delegate.chat(messages, tools, **kwargs)

    def chat_stream(self, messages: Any, tools: Any = None, **kwargs: object):
        if self.delegate is None:
            raise RuntimeError("provider reference has no concrete delegate")
        yield from self.delegate.chat_stream(messages, tools, **kwargs)


def provider_name(provider: LLMProviderProtocol | str) -> str:
    """Adapt a Provider object to the stable identifier required by QueryConfig."""
    if isinstance(provider, str):
        normalized = provider.strip()
    else:
        normalized = str(getattr(provider, "provider_name", None) or getattr(provider, "name", None) or "").strip()
    if not normalized:
        raise ValueError("provider object must expose provider_name or name")
    return normalized


class StaticProviderRouter:
    """Resolve stage overrides with immutable, workflow-level fallbacks.

    The router never mutates ``workflow.agent`` or ``AgentRunner``. Each mode
    asks for a provider/model pair and passes that selection into its run,
    which makes concurrent issue execution independent and deterministic.
    """

    def __init__(
        self,
        default_provider: LLMProviderProtocol,
        default_model: str,
        *,
        stage_providers: Iterable[StageProvider] = (),
        stage_models: Iterable[StageModel] = (),
    ) -> None:
        if default_provider is None or isinstance(default_provider, str):
            raise ValueError("default_provider must be a Provider object")
        normalized_default_model = str(default_model or "").strip()

        self._default_provider = default_provider
        self._default_model = normalized_default_model
        self._stage_providers = self._index_unique(stage_providers, "provider")
        self._stage_models = self._index_unique(stage_models, "model")

    @staticmethod
    def _index_unique(bindings: Iterable[object], value_attribute: str) -> dict[str, object]:
        indexed: dict[str, object] = {}
        for binding in bindings:
            stage_id = getattr(binding, "stage_id")
            if stage_id in indexed:
                raise ValueError(f"duplicate route for stage_id {stage_id!r}")
            indexed[stage_id] = getattr(binding, value_attribute)
        return indexed

    def provider_for_stage(self, stage_id: str) -> LLMProviderProtocol:
        """Return a stage override or the workflow's default provider."""
        canonical = _require_stage_id(stage_id)
        return self._stage_providers.get(canonical, self._default_provider)  # type: ignore[return-value]

    def model_for_stage(self, stage_id: str) -> str:
        """Return a stage override or the workflow's default model."""
        canonical = _require_stage_id(stage_id)
        return self._stage_models.get(canonical, self._default_model)  # type: ignore[return-value]

    def route_for_stage(self, stage_id: str) -> ProviderRoute:
        """Resolve both values into one immutable hand-off object."""
        return ProviderRoute(
            provider=self.provider_for_stage(stage_id),
            model=self.model_for_stage(stage_id),
        )


__all__ = [
    "ProviderReference",
    "ProviderRoute",
    "StaticProviderRouter",
    "provider_name",
]
