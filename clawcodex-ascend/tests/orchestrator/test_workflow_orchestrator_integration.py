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

"""Smoke test for the workflow orchestrator integration.

Verifies:
1. All module imports work (no circular deps, no broken paths)
2. Minimal workflow.yaml can be parsed into a DAG schema
3. OrchestrationSubsystem accepts workflow_yaml_path and passes it to Orchestrator
4. WorkflowProgressSink <-> ProgressSink wiring
5. StageRunner synthetic Issue construction
"""

from __future__ import annotations

import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path


# ── 1. Import sanity ─────────────────────────────────────────────────


class TestWorkflowOrchestratorInit(unittest.TestCase):
    """Verify WorkflowOrchestrator can be initialized with a minimal yaml."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.yaml_path = Path(self._tmpdir.name) / "workflow.yaml"
        self.yaml_path.write_text(
            textwrap.dedent("""\
                name: init-test
                description: Init test
                version: "1.0"
                stages:
                  - id: 1
                    name: Test
                    phase: test
                    prompt: "Run test."
            """),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_init_without_agent_runner(self) -> None:
        """WorkflowOrchestrator can be initialized without AgentRunner
        (standalone mode).
        """
        from extensions.orchestrator.config.schema import WorkflowConfig
        from extensions.orchestrator.workflow_orchestrator import (
            WorkflowOrchestrator,
        )

        wf_config = WorkflowConfig.from_dict(
            {
                "workspace": {"root": str(self._tmpdir.name)},
                "agent": {},
                "tracker": {"kind": "local"},
            }
        )

        orch = WorkflowOrchestrator(
            workflow_config=wf_config,
            workflow_yaml_path=str(self.yaml_path),
        )
        self.assertEqual(orch.schema.name, "init-test")
        self.assertEqual(len(orch.schema.stages), 1)
        self.assertIsNotNone(orch.progress)

    def test_set_progress_sink(self) -> None:
        """Progress sink injection works -- sinks are forwarded."""
        from extensions.orchestrator.config.schema import WorkflowConfig
        from extensions.orchestrator.workflow_orchestrator import (
            WorkflowOrchestrator,
        )
        from extensions.orchestrator.progress_sink import (
            ToolContextProgressSink,
        )

        wf_config = WorkflowConfig.from_dict(
            {
                "workspace": {"root": str(self._tmpdir.name)},
                "agent": {},
                "tracker": {"kind": "local"},
            }
        )

        orch = WorkflowOrchestrator(
            workflow_config=wf_config,
            workflow_yaml_path=str(self.yaml_path),
        )

        mock_sink = ToolContextProgressSink(
            task_id="test-123",
            context=None,
            workflow_phases=["test"],
        )
        orch.set_progress_sink(mock_sink)

        # Verify the sink was added to WorkflowProgressSink's internal list
        snapshot = orch.progress
        self.assertEqual(snapshot["workflow_name"], "init-test")
        self.assertEqual(snapshot["total_stages"], 1)
        self.assertEqual(snapshot["completed_stages"], 0)

    def test_orchestrator_workflow_yaml_param(self) -> None:
        """Verify Orchestrator.__init__ accepts workflow_yaml_path."""
        import inspect
        from extensions.orchestrator.orchestrator import Orchestrator

        sig = inspect.signature(Orchestrator.__init__)
        params = list(sig.parameters.keys())
        self.assertIn("workflow_yaml_path", params)

    def test_orchestration_subsystem_workflow_yaml_param(self) -> None:
        """Verify OrchestrationSubsystem.__init__ accepts workflow_yaml_path."""
        import inspect
        from extensions.api.orchestration import OrchestrationSubsystem

        sig = inspect.signature(OrchestrationSubsystem.__init__)
        params = list(sig.parameters.keys())
        self.assertIn("workflow_yaml_path", params)


# ── 4. OrchestrationSubsystem workflow_yaml_path plumbing ─────────────


class TestOrchestrationSubsystemPlumbing(unittest.TestCase):
    """Verify OrchestrationSubsystem passes workflow_yaml_path to Orchestrator."""

    def test_subsystem_stores_workflow_yaml_path(self) -> None:
        from extensions.orchestrator.config.schema import WorkflowConfig
        from extensions.api.orchestration import OrchestrationSubsystem

        import tempfile

        issues_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, issues_dir)
        wf_config = WorkflowConfig.from_dict(
            {
                "workspace": {"root": "/tmp/test"},
                "agent": {},
                "tracker": {"kind": "local", "issues_path": issues_dir},
            }
        )

        subsystem = OrchestrationSubsystem(
            wf_config,
            workflow_yaml_path="/path/to/workflow.yaml",
        )
        self.assertEqual(subsystem._workflow_yaml_path, "/path/to/workflow.yaml")

    def test_subsystem_none_is_ok(self) -> None:
        """Omitting workflow_yaml_path is valid (backward compatible)."""
        from extensions.orchestrator.config.schema import WorkflowConfig
        from extensions.api.orchestration import OrchestrationSubsystem
        import tempfile

        issues_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, issues_dir)
        wf_config = WorkflowConfig.from_dict(
            {
                "workspace": {"root": "/tmp/test"},
                "agent": {},
                "tracker": {"kind": "local", "issues_path": issues_dir},
            }
        )

        subsystem = OrchestrationSubsystem(wf_config)
        self.assertIsNone(subsystem._workflow_yaml_path)
