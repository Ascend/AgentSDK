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

"""Unit tests for the A.10 per-stage provider/model router."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from extensions.orchestrator.contracts.provider_routing import (
    ProviderRouter,
    StageModel,
    StageProvider,
)
from extensions.orchestrator.provider_routing import (
    ProviderReference,
    StaticProviderRouter,
    build_provider_router,
    normalize_stage_overrides,
    provider_name,
    stage_id_from_run_kind,
)
from extensions.orchestrator.config.schema import WorkflowConfig


def test_router_uses_workflow_defaults_for_unconfigured_stage() -> None:
    default = ProviderReference("deepseek")
    router = StaticProviderRouter(default, "sonnet")

    route = router.route_for_stage("tester")

    assert route.provider is default
    assert route.model == "sonnet"
    assert isinstance(router, ProviderRouter)


def test_router_resolves_provider_and_model_overrides_independently() -> None:
    default = ProviderReference("deepseek")
    analyzer = ProviderReference("openrouter")
    router = StaticProviderRouter(
        default,
        "sonnet",
        stage_providers=(StageProvider("Analyzer", analyzer),),
        stage_models=(StageModel(" analyzer ", " opus "), StageModel("tester", "haiku")),
    )

    analyzer_route = router.route_for_stage("ANALYZER")
    tester_route = router.route_for_stage("tester")

    assert analyzer_route.provider is analyzer
    assert analyzer_route.model == "opus"
    assert tester_route.provider is default
    assert tester_route.model == "haiku"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: StageProvider("", ProviderReference("deepseek")), "stage_id"),
        (lambda: StageProvider("stage", ""), "provider"),
        (lambda: StageModel("stage", "  "), "model"),
        (lambda: StaticProviderRouter("", "sonnet"), "default_provider"),
    ],
)
def test_invalid_route_configuration_fails_fast(factory, message: str) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match=message):
        factory()


def test_duplicate_stage_override_is_rejected() -> None:
    provider = ProviderReference("deepseek")

    with pytest.raises(ValueError, match="duplicate route"):
        StaticProviderRouter(
            provider,
            "sonnet",
            stage_models=(StageModel("tester", "haiku"), StageModel("TESTER", "opus")),
        )


def test_stage_bindings_and_resolved_route_are_immutable() -> None:
    provider = ProviderReference("deepseek")
    binding = StageModel("tester", "haiku")
    route = StaticProviderRouter(provider, "sonnet", stage_models=(binding,)).route_for_stage("tester")

    with pytest.raises(FrozenInstanceError):
        binding.model = "opus"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        route.model = "opus"  # type: ignore[misc]


def test_blank_lookup_stage_is_rejected() -> None:
    router = StaticProviderRouter(ProviderReference("deepseek"), "sonnet")

    with pytest.raises(ValueError, match="stage_id"):
        router.model_for_stage("  ")


def test_router_allows_provider_default_model() -> None:
    router = StaticProviderRouter(ProviderReference("deepseek"), "")

    assert router.model_for_stage("tester") == ""


def test_router_is_built_from_workflow_and_canonical_stage_overrides_win() -> None:
    workflow = SimpleNamespace(
        agent=SimpleNamespace(
            provider="deepseek",
            model="default-model",
            stage_overrides={
                "analyzer": {"provider": "openrouter", "model": "canonical-model"},
            },
        )
    )

    router = build_provider_router(
        workflow,
        model_overrides={"analyzer": "legacy-model", "tester": "tester-model"},
    )

    assert provider_name(router.route_for_stage("analyzer").provider) == "openrouter"
    assert router.route_for_stage("analyzer").model == "canonical-model"
    assert router.route_for_stage("tester").model == "tester-model"


def test_stage_override_normalization_and_run_kind_mapping() -> None:
    routes = normalize_stage_overrides(
        {
            " Analyzer ": {"provider": " deepseek ", "model": " opus "},
            "ignored": "not-a-mapping",
            "blank": {"model": ""},
        }
    )

    assert routes == {"analyzer": {"provider": "deepseek", "model": "opus"}}
    assert stage_id_from_run_kind("pipeline:Analyzer:retry1") == "analyzer"
    assert stage_id_from_run_kind("debate:judge") == "judge"
    assert stage_id_from_run_kind("review_followup") == "review_followup"


def test_workflow_yaml_shape_reaches_router(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYZER_MODEL", "env-analyzer-model")
    workflow = WorkflowConfig.from_dict(
        {
            "agent": {
                "provider": "deepseek",
                "model": "default-model",
                "stages": {
                    "analyzer": {
                        "provider": "openrouter",
                        "model": "$ANALYZER_MODEL",
                    }
                },
            },
            "modes": {
                "pipeline": {
                    "stage_models": {
                        "analyzer": "legacy-analyzer",
                        "tester": "tester-model",
                    }
                }
            },
        }
    )

    router = build_provider_router(
        workflow,
        model_overrides=workflow.modes.pipeline_stage_models,
    )

    assert provider_name(router.route_for_stage("analyzer").provider) == "openrouter"
    assert router.route_for_stage("analyzer").model == "env-analyzer-model"
    assert router.route_for_stage("tester").model == "tester-model"
