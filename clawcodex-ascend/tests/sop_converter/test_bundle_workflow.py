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

"""Tests for bundle workflow discovery and manifest metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.bundle_agents import register_bundle_agents
from extensions.sop_converter.bundle_manifest import read_bundle_manifest, write_bundle_manifest
from extensions.sop_converter.bundle_workflow import (
    discover_workflow_yaml,
    workflow_artifacts_enabled,
)


class TestBundleWorkflow(unittest.TestCase):
    def test_workflow_artifacts_enabled(self) -> None:
        self.assertTrue(workflow_artifacts_enabled(has_mapped_stages=True, workflow_mode="sdk"))
        self.assertTrue(workflow_artifacts_enabled(has_mapped_stages=False, workflow_mode="fwa"))
        self.assertFalse(workflow_artifacts_enabled(has_mapped_stages=False, workflow_mode="sdk"))

    def test_discover_workflow_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            bundle = ws / ".clawcodex" / "AutoResearchClaw"
            bundle.mkdir(parents=True)
            (bundle / "workflow.yaml").write_text("name: test\nstages: []\n", encoding="utf-8")
            write_bundle_manifest(
                bundle,
                sdk_source_dir=ws,
                workflow_yaml="workflow.yaml",
                workflow_mode="fwa",
            )
            found = discover_workflow_yaml(ws)
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.parent.name, "AutoResearchClaw")

    def test_register_bundle_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            agents_dir = bundle / ".claude" / "agents"
            agents_dir.mkdir(parents=True)
            (agents_dir / "topic-init-agent.md").write_text(
                "---\n"
                "name: topic-init-agent\n"
                "description: Topic init stage\n"
                "tools:\n"
                "  - Read\n"
                "---\n\n"
                "Run topic init.\n",
                encoding="utf-8",
            )
            names = register_bundle_agents(bundle)
            self.assertIn("topic-init-agent", names)

    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            path = write_bundle_manifest(
                bundle,
                sdk_source_dir=bundle,
                workflow_yaml="workflow.yaml",
                bridge_script="bridge/run.py",
                workflow_mode="fwa",
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["workflow_mode"], "fwa")
            manifest = read_bundle_manifest(bundle)
            assert manifest is not None
            self.assertEqual(manifest.workflow_yaml, "workflow.yaml")
