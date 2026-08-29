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

"""Tests for the bridge MCP adapter (AgentToolSpec registration)."""

from __future__ import annotations

import json
from pathlib import Path

from clawcodex_ext.agent.tool_authoring.persistence import _dict_to_spec
from clawcodex_ext.agent.tool_authoring.validators import validate_spec
from extensions.sop_converter.workflow_mode.bridge.mcp_adapter import (
    bridge_tool_name,
    register_bridge_tool,
)


def _minimal_bridge_script(path: Path) -> None:
    path.write_text(
        '"""Minimal bridge for tests."""\ndef main():\n    return 0\n',
        encoding="utf-8",
    )


class TestBridgeToolName:
    def test_kebab_from_mixed_case(self):
        assert bridge_tool_name("JiuwenAgent") == "jiuwenagent-execute-stage"

    def test_kebab_from_underscores(self):
        assert bridge_tool_name("foo_bar") == "foo-bar-execute-stage"


class TestRegisterBridgeTool:
    def test_register_builds_valid_spec(self, tmp_path: Path):
        script = tmp_path / "bridge" / "demo_bridge.py"
        script.parent.mkdir(parents=True)
        _minimal_bridge_script(script)
        bundle = tmp_path / "bundle"

        name = register_bridge_tool(
            bridge_tool_name("fwa-test"),
            script,
            persist=True,
            bundle_dir=bundle,
        )

        assert name == "fwa-test-execute-stage"
        spec_path = bundle / "agent-tools" / f"{name}.json"
        assert spec_path.is_file()
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        assert data["call_type"] == "bash"
        assert "{stage_id}" in data["call_impl"]
        assert "{project_dir}" in data["call_impl"]
        assert "stage_id" in data["input_schema"]["properties"]
        assert "run_dir" in data["input_schema"]["properties"]
        assert data["input_schema"]["required"] == ["stage_id"]
        assert data["source"] == "sop-converter"

        validate_spec(_dict_to_spec(data))

    def test_missing_script_returns_none(self, tmp_path: Path):
        assert register_bridge_tool("demo-execute-stage", tmp_path / "missing.py") is None

    def test_invalid_tool_name_returns_none(self, tmp_path: Path):
        script = tmp_path / "demo_bridge.py"
        _minimal_bridge_script(script)
        assert register_bridge_tool("INVALID NAME!", script, bundle_dir=tmp_path / "b") is None

    def test_copies_script_into_bundle_scripts_dir(self, tmp_path: Path):
        script = tmp_path / "proj_bridge.py"
        _minimal_bridge_script(script)
        bundle = tmp_path / "out"

        register_bridge_tool("proj-execute-stage", script, bundle_dir=bundle)

        copied = bundle / "agent-tools" / "scripts" / "proj_bridge.py"
        assert copied.is_file()
