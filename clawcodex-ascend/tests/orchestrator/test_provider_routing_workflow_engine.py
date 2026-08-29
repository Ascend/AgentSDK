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

"""Structural and A.6 integration tests for the complete A.10 boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from extensions.orchestrator.config.schema import AgentConfig, ModesConfig, WorkflowConfig
from extensions.orchestrator.provider_routing import provider_name
from extensions.orchestrator.workflow_engine.stage_runner import StageRunner
from extensions.orchestrator.workflow_engine.workflow_state import StageKind, StageNode


def _workflow() -> WorkflowConfig:
    return WorkflowConfig.from_dict(
        {
            "agent": {
                "provider": "deepseek",
                "model": "default-model",
                "stages": {"analysis": {"provider": "openrouter", "model": "opus"}},
            },
            "modes": {
                "router": {"model": "router-model"},
                "pipeline": {"stage_models": {"tester": "tester-model"}},
                "debate": {
                    "judge_model": "judge-model",
                    "proposer_models": {"security": "security-model"},
                },
            },
        }
    )


def test_schema_fields_are_owned_by_provider_routing_config() -> None:
    workflow = _workflow()
    routing = workflow.agent.provider_routing

    assert routing is workflow.modes.provider_routing
    assert routing.router_model == "router-model"
    assert routing.pipeline_stage_models == {"tester": "tester-model"}
    assert routing.debate_judge_model == "judge-model"
    assert routing.debate_proposer_models == {"security": "security-model"}
    assert routing.stage_overrides["analysis"]["provider"] == "openrouter"
    assert "stage_overrides" not in AgentConfig.__dataclass_fields__
    for field_name in (
        "router_model",
        "pipeline_stage_models",
        "debate_judge_model",
        "debate_proposer_models",
    ):
        assert field_name not in ModesConfig.__dataclass_fields__


def test_workflow_engine_stage_uses_provider_router() -> None:
    workflow = _workflow()
    runner = StageRunner(object(), workflow)
    stage = StageNode(
        id=1,
        name="analyze",
        kind=StageKind.AGENT,
        phase="analysis",
        agent_config={"provider": "zai", "model": "stage-model"},
    )

    hooks = runner._routing_hooks(stage)

    assert provider_name(hooks["provider_override"]) == "zai"
    assert hooks["model_override"] == "stage-model"


def test_agent_session_owns_one_routing_snapshot_object() -> None:
    source_path = Path(__file__).parents[2] / "extensions" / "orchestrator" / "agent_session.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    agent_session = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AgentSession")
    annotations = {
        node.target.id
        for node in agent_session.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert "_routing_snapshot" in annotations
    assert "_snapshot_provider" not in annotations
    assert "_snapshot_model" not in annotations
