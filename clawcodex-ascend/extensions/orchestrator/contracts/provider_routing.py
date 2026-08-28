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

"""Contracts for selecting a provider and model for one execution stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from extensions.capabilities.provider_protocol import LLMProviderProtocol
else:
    LLMProviderProtocol = Any


def _require_stage_id(stage_id: str) -> str:
    """Return the canonical stage identifier or reject an empty value."""
    canonical = stage_id.strip().casefold()
    if not canonical:
        raise ValueError("stage_id must not be empty")
    return canonical


@dataclass(frozen=True, slots=True)
class StageProvider:
    """Bind one canonical stage identifier to a provider object."""

    stage_id: str
    provider: LLMProviderProtocol

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _require_stage_id(self.stage_id))
        if self.provider is None or isinstance(self.provider, str):
            raise ValueError("provider must be a Provider object")


@dataclass(frozen=True, slots=True)
class StageModel:
    """Bind one canonical stage identifier to a non-empty model name."""

    stage_id: str
    model: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _require_stage_id(self.stage_id))
        model = self.model.strip()
        if not model:
            raise ValueError("model must not be empty")
        object.__setattr__(self, "model", model)


@runtime_checkable
class ProviderRouter(Protocol):
    """Read-only provider/model lookup used by orchestrator mode runners."""

    def provider_for_stage(self, stage_id: str) -> LLMProviderProtocol:
        """Return the provider selected for ``stage_id``."""
        raise NotImplementedError

    def model_for_stage(self, stage_id: str) -> str:
        """Return the model selected for ``stage_id``."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RoutingSnapshot:
    """Serializable provider/model selection captured for one agent run."""

    provider_name: str
    model: str


__all__ = ["ProviderRouter", "RoutingSnapshot", "StageModel", "StageProvider"]
