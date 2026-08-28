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

"""Per-stage provider/model routing for orchestrator execution modes."""

from extensions.orchestrator.contracts.provider_routing import (
    ProviderRouter,
    RoutingSnapshot,
    StageModel,
    StageProvider,
)

from .config import ProviderRoutingConfig, normalize_stage_overrides
from .factory import build_provider_router, stage_id_from_run_kind
from .router import ProviderReference, ProviderRoute, StaticProviderRouter, provider_name

__all__ = [
    "ProviderReference",
    "ProviderRoute",
    "ProviderRouter",
    "ProviderRoutingConfig",
    "RoutingSnapshot",
    "StageModel",
    "StageProvider",
    "StaticProviderRouter",
    "build_provider_router",
    "normalize_stage_overrides",
    "provider_name",
    "stage_id_from_run_kind",
]
