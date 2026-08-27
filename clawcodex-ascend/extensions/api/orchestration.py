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

"""Public Python API for autonomous orchestration.

Top-level entry point for the orchestration subsystem.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..orchestrator.agent_runner import AgentRunner
    from ..orchestrator.orchestrator import Orchestrator

from ..orchestrator.config.schema import WorkflowConfig  # pylint: disable=import-error
from ..orchestrator.status_dashboard import StatusDashboard
from ..orchestrator.tracker import (  # pylint: disable=import-error
    TrackerAdapter,
    create_tracker_adapter,
    repository_clone_url_for_tracker,
)
from ..orchestrator.workspace import WorkspaceConfig, WorkspaceManager  # pylint: disable=import-error

logger = logging.getLogger(__name__)


class OrchestrationSubsystem:
    """Autonomous mode orchestration engine.

    Replaces Symphony's GenServer-based orchestrator with a Python-native
    async implementation, leveraging ClawCodex's existing infrastructure.
    """

    workflow: WorkflowConfig
    workspace_manager: WorkspaceManager
    tracker_adapter: TrackerAdapter
    agent_runner: "AgentRunner"
    status_dashboard: StatusDashboard
    _orchestrator: "Orchestrator | None" = None

    def __init__(self, workflow_config: WorkflowConfig, *, workflow_yaml_path: str | None = None) -> None:
        from ..orchestrator.agent_runner import AgentRunner

        self.workflow = workflow_config
        resolved_yaml = workflow_yaml_path
        if not resolved_yaml:
            try:
                from extensions.sop_converter.bundle_workflow import discover_workflow_yaml

                discovered = discover_workflow_yaml(Path(workflow_config.workspace.root))
                if discovered is not None:
                    resolved_yaml = str(discovered)
                    logger.info("Auto-discovered workflow.yaml: %s", resolved_yaml)
            except ImportError:
                pass

        self._workflow_yaml_path = resolved_yaml
        self._bundle_dir: Path | None = None
        if resolved_yaml:
            bundle_dir = Path(resolved_yaml).resolve().parent
            self._bundle_dir = bundle_dir
            try:
                from extensions.sop_converter.bundle_agents import register_bundle_agents
                from extensions.sop_converter.bundle_skills import register_bundle_skills

                agents = register_bundle_agents(bundle_dir)
                if agents:
                    logger.info("Loaded %d stage agents from bundle %s", len(agents), bundle_dir.name)
                register_bundle_skills(bundle_dir, Path(workflow_config.workspace.root))
            except ImportError as exc:
                logger.debug("Bundle workflow bootstrap skipped: %s", exc)
        self.workspace_manager = WorkspaceManager(
            WorkspaceConfig(
                root=Path(workflow_config.workspace.root),
                hooks=workflow_config.workspace.hooks,
                repo_clone_url=workflow_config.workspace.repo_clone_url
                or repository_clone_url_for_tracker(workflow_config.tracker),
                upstream_clone_url=workflow_config.workspace.upstream_clone_url,
                clone_depth=workflow_config.workspace.clone_depth,
                checkout_issue_branch=(workflow_config.workspace.checkout_issue_branch),
                git_username=workflow_config.workspace.git_username,
                git_token=workflow_config.workspace.git_token,
                gitignore_patterns=workflow_config.workspace.gitignore_patterns,
                strategy=workflow_config.workspace.strategy,
                base_branch=workflow_config.workspace.base_branch,
                integration_branch=workflow_config.workspace.integration_branch,
                require_clean_start=workflow_config.workspace.require_clean_start,
                require_clean_between_issues=(workflow_config.workspace.require_clean_between_issues),
                preserve_on_terminal=workflow_config.workspace.preserve_on_terminal,
                sequential_lock=workflow_config.workspace.sequential_lock,
            )
        )
        self.tracker_adapter = create_tracker_adapter(workflow_config.tracker)
        self.agent_runner = AgentRunner(
            agent_config=workflow_config.agent,
            sandbox_config=workflow_config.sandbox,
            workspace_cfg=workflow_config.workspace,
        )
        # Build per-stage AgentRunners for multi-model stage overrides.
        # Each stage inherits all settings from the main agent config;
        # only provider and/or model are replaced.
        self.stage_runners: dict[str, AgentRunner] = {}
        for stage_name, override in workflow_config.agent.stage_overrides.items():
            from dataclasses import replace

            stage_config = replace(
                workflow_config.agent,
                provider=override.get("provider", workflow_config.agent.provider),
                model=override.get("model", workflow_config.agent.model),
            )
            self.stage_runners[stage_name] = AgentRunner(
                agent_config=stage_config,
                sandbox_config=workflow_config.sandbox,
                workspace_cfg=workflow_config.workspace,
            )
            logger.info(
                "stage runner [%s]: provider=%s model=%s",
                stage_name,
                stage_config.provider,
                stage_config.model,
            )
        self.status_dashboard = StatusDashboard()
        self._orchestrator = None

    async def run(self) -> None:
        """Start polling and issue execution. Runs until cancelled."""
        from ..orchestrator.orchestrator import Orchestrator

        self._orchestrator = Orchestrator(
            workflow=self.workflow,
            tracker=self.tracker_adapter,
            workspace=self.workspace_manager,
            agent_runner=self.agent_runner,
            status_dashboard=self.status_dashboard,
            stage_runners=self.stage_runners,
            workflow_yaml_path=self._workflow_yaml_path,
        )
        await self._orchestrator.run()

    async def shutdown(self) -> None:
        """Graceful shutdown — stop polling, clean up workspaces."""
        if self._orchestrator:
            await self._orchestrator.shutdown()
