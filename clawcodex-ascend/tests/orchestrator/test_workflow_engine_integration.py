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

"""Smoke test for the declarative workflow engine integration.

Verifies:
1. All module imports work (no circular deps, no broken paths)
2. Minimal workflow.yaml can be parsed into a DAG schema
3. OrchestrationSubsystem accepts workflow_yaml_path and passes it to Orchestrator
4. WorkflowProgressSink <-> ProgressSink wiring
5. StageRunner synthetic Issue construction
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from extensions.orchestrator.workflow_engine.checkpoint import CheckpointManager
from extensions.orchestrator.workflow_engine.engine import DeclarativeWorkflowEngine, EngineConfig, WorkflowResumer
from extensions.orchestrator.workflow_engine.workflow_state import WorkflowState


class TestWorkflowEngineImports(unittest.TestCase):
    """Verify all modules in the new workflow engine tree import cleanly."""

    def test_core_imports(self) -> None:
        from extensions.orchestrator.workflow_engine import (
            WorkflowSchema,
            WorkflowState,
        )

        self.assertIsNotNone(WorkflowSchema)
        self.assertIsNotNone(WorkflowState)

    def test_engine_import(self) -> None:
        from extensions.orchestrator.workflow_engine.engine import (
            DeclarativeWorkflowEngine,
        )

        self.assertIsNotNone(DeclarativeWorkflowEngine)

    def test_stage_runner_import(self) -> None:
        from extensions.orchestrator.workflow_engine.stage_runner import (
            StageRunner,
        )

        self.assertIsNotNone(StageRunner)

    def test_observability_imports(self) -> None:
        from extensions.orchestrator.workflow_engine.observability import (
            WorkflowObservability,
            WorkflowProgressSink,
        )

        self.assertIsNotNone(WorkflowObservability)
        self.assertIsNotNone(WorkflowProgressSink)

    def test_validators_import(self) -> None:
        from extensions.orchestrator.workflow_engine.validators import (
            ContractValidator,
            ValidationResult,
        )

        self.assertIsNotNone(ContractValidator)
        self.assertIsNotNone(ValidationResult)

    def test_checkpoint_import(self) -> None:
        from extensions.orchestrator.workflow_engine.checkpoint import (
            CheckpointManager,
        )

        self.assertIsNotNone(CheckpointManager)

    def test_workflow_orchestrator_import(self) -> None:
        from extensions.orchestrator.workflow_orchestrator import (
            WorkflowOrchestrator,
        )

        self.assertIsNotNone(WorkflowOrchestrator)

    """Verify a minimal workflow.yaml can be parsed into a DAG schema."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.yaml_path = Path(self._tmpdir.name) / "workflow.yaml"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_yaml(self, content: str) -> Path:
        self.yaml_path.write_text(content, encoding="utf-8")
        return self.yaml_path

    def test_minimal_workflow_yaml(self) -> None:
        """A minimal 2-stage linear workflow parses correctly."""
        content = textwrap.dedent("""\
            name: smoke-test
            description: A minimal smoke test workflow
            version: "1.0"
            stages:
              - id: 1
                name: Analyze
                phase: analyze
                prompt: "Analyze the issue."
              - id: 2
                name: Implement
                phase: implement
                depends_on: [1]
                prompt: "Implement the fix."
        """)
        self._write_yaml(content)

        from extensions.orchestrator.workflow_engine import WorkflowSchema

        schema = WorkflowSchema.from_yaml(self.yaml_path)
        self.assertEqual(schema.name, "smoke-test")
        self.assertEqual(len(schema.stages), 2)
        self.assertEqual(schema.stages[0].name, "Analyze")
        self.assertEqual(schema.stages[1].depends_on, [1])

    def test_dag_topological_order(self) -> None:
        """A 3-stage DAG with a diamond dependency produces correct order."""
        content = textwrap.dedent("""\
            name: dag-test
            description: Diamond dependency DAG
            version: "1.0"
            stages:
              - id: 1
                name: Setup
                phase: setup
              - id: 2
                name: Branch A
                phase: branch_a
                depends_on: [1]
              - id: 3
                name: Branch B
                phase: branch_b
                depends_on: [1]
              - id: 4
                name: Merge
                phase: merge
                depends_on: [2, 3]
        """)
        self._write_yaml(content)

        from extensions.orchestrator.workflow_engine import WorkflowSchema

        schema = WorkflowSchema.from_yaml(self.yaml_path)
        order = schema.build_dag_order()
        stage_ids = order

        self.assertEqual(stage_ids[0], 1)
        self.assertEqual(stage_ids[-1], 4)
        # Setup must come before all others
        setup_idx = stage_ids.index(1)
        for sid in [2, 3, 4]:
            self.assertGreater(stage_ids.index(sid), setup_idx)


# ── 3. WorkflowOrchestrator initialization ───────────────────────────


class TestWorkflowResumer:
    """WorkflowResumer resumes engine execution from a checkpoint."""

    @pytest.fixture
    def tmp_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    @pytest.mark.asyncio
    async def test_resume_restores_cost_tracker(self, tmp_run_dir: str) -> None:
        """Accumulated cost should be restored into the CostTracker on resume."""
        from extensions.orchestrator.workflow_engine.engine import (
            WorkflowSchema,
        )

        called = {"times": 0}

        async def fake_execute(*args, **kwargs):  # type: ignore[no-redef]
            called["times"] += 1
            return MagicMock()

        schema = WorkflowSchema(name="wf", version="1.0", stages=[])
        engine = DeclarativeWorkflowEngine(schema, config=EngineConfig())
        engine.execute = fake_execute  # type: ignore[method-assign]

        mgr = CheckpointManager(tmp_run_dir)
        state = WorkflowState(workflow_name="wf", workflow_version="1.0")
        state.current_stage = 0
        state.cost_accumulated_usd = 12.34
        mgr.save(state)

        resumer = WorkflowResumer(mgr)
        await resumer.resume(engine)

        assert engine.cost_tracker.total_usd == 12.34
        assert called["times"] == 1


# ── ArtifactResolver ───────────────────────────────────────────────
