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

"""Tests for composite macro tool registration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clawcodex_ext.agent.tool_authoring.persistence import bundle_tool_dir, save_spec
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
import extensions.sop_converter.composite_tools as composite_tools
from extensions.sop_converter.composite_tools import (
    CompositeStage,
    CompositeToolSpec,
    emit_composite_workflow_yaml,
    register_composite_tools,
)
from extensions.sop_converter.runtime.composite_tools.builtin import builtin_composite_tools
from extensions.sop_converter.runtime.composite_tools.builtin import lifecycle_tools_for_skill
from extensions.sop_converter.tool_retrieval import load_tool_retrieval_index


def _echo_composite_spec(*, call_impl: str | None = None) -> CompositeToolSpec:
    return CompositeToolSpec(
        name="test_echo_composite",
        description="Echo JSON args for registration smoke test.",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
        stages=[
            CompositeStage(
                name="echo",
                description="Echo the message.",
                agent_ref="demo-agent",
            ),
        ],
        call_type="bash",
        call_impl=call_impl,
    )


class TestCompositeTools(unittest.TestCase):
    def test_builtin_composite_tools(self) -> None:
        names = {spec.name for spec in builtin_composite_tools()}
        self.assertEqual(
            names,
            {
                "agent_teams",
                "pipeline_execute",
                "code_review",
                "invoke_existing_agent",
                "resume_resource",
            },
        )
        invoke = next(s for s in builtin_composite_tools() if s.name == "invoke_existing_agent")
        self.assertIsNotNone(invoke.call_impl)
        self.assertIsNotNone(invoke.workflow_spec)
        self.assertEqual(invoke.call_type, "workflow")
        self.assertEqual(
            invoke.call_impl,
            {"catalog_id": "builtin:invoke-existing-agent"},
        )
        self.assertIsNotNone(invoke.output_schema)
        self.assertNotIn("invoke-existing-agent", invoke.aliases)
        self.assertIn("call-agent-by-id", invoke.aliases)
        self.assertIn("agent_ref", invoke.input_schema["properties"])
        self.assertEqual(
            invoke.input_schema["anyOf"],
            [{"required": ["agent_ref"]}, {"required": ["agent_id"]}],
        )

    def test_composite_to_agent_tool_spec_executable(self) -> None:
        spec = _echo_composite_spec(call_impl='python3 "/tmp/echo_composite.py" \'{"message":"hi"}\'')
        agent_spec = composite_tools._composite_to_agent_tool_spec(
            spec,
            bundle_dir=Path("demo_bundle"),
        )
        self.assertEqual(agent_spec.name, "test-echo-composite")
        self.assertIn("message", agent_spec.input_schema["properties"])
        self.assertEqual(agent_spec.call_impl, spec.call_impl)
        self.assertEqual(agent_spec.call_type, "bash")
        self.assertEqual(agent_spec.bundle_id, "demo_bundle")

    def test_composite_to_agent_tool_spec_preserves_workflow_call_type(self) -> None:
        invoke = next(s for s in builtin_composite_tools() if s.name == "invoke_existing_agent")
        agent_spec = composite_tools._composite_to_agent_tool_spec(invoke)
        self.assertEqual(agent_spec.call_type, "workflow")
        self.assertEqual(
            agent_spec.call_impl,
            {"catalog_id": "builtin:invoke-existing-agent"},
        )

    def test_register_preserves_workflow_call_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "demo_bundle"
            bundle.mkdir()
            registered = register_composite_tools(persist=True, bundle_dir=bundle)
            self.assertEqual(
                registered.get("invoke_existing_agent"),
                "invoke-existing-agent",
            )
            saved = json.loads((bundle / "agent-tools" / "invoke-existing-agent.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["call_type"], "workflow")
            self.assertEqual(
                saved["call_impl"],
                {"catalog_id": "builtin:invoke-existing-agent"},
            )

    def test_register_skips_placeholders_but_keeps_executable_macros(self) -> None:
        with patch("extensions.sop_converter.runtime.composite_tools.save_spec") as save_spec:
            registered = register_composite_tools(persist=True)
        self.assertIn("invoke_existing_agent", registered)
        self.assertEqual(registered["invoke_existing_agent"], "invoke-existing-agent")
        self.assertIn("resume_resource", registered)
        self.assertEqual(registered["resume_resource"], "resume-resource")
        saved_names = {call.args[0].name for call in save_spec.call_args_list}
        self.assertEqual(
            saved_names,
            {"invoke-existing-agent", "resume-resource"},
        )

    def test_register_executable_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "demo_bundle"
            bundle.mkdir()
            with (
                patch(
                    "extensions.sop_converter.runtime.composite_tools._SKIP_PLACEHOLDER_COMPOSITE_TOOLS",
                    False,
                ),
                patch("extensions.sop_converter.runtime.composite_tools.save_spec") as save_spec,
            ):
                registered = register_composite_tools(
                    persist=True,
                    bundle_dir=bundle,
                )
            self.assertIn("invoke_existing_agent", registered)
            self.assertEqual(registered["invoke_existing_agent"], "invoke-existing-agent")
            saved_names = {call.args[0].name for call in save_spec.call_args_list}
            self.assertIn("invoke-existing-agent", saved_names)

    def test_register_builtin_compiles_retrieval_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "demo_bundle"
            bundle.mkdir()
            tool_dir = bundle_tool_dir(bundle)
            for name in (
                "pkg-llmagent-invoke",
                "pkg-send-to-agent",
            ):
                save_spec(
                    AgentToolSpec(
                        name=name,
                        description="atomic",
                        input_schema={"type": "object", "properties": {}},
                        call_type="bash",
                        call_impl="echo ok",
                        source="sop-converter",
                        bundle_id=bundle.name,
                    ),
                    tool_dir=tool_dir,
                )
            registered = register_composite_tools(persist=True, bundle_dir=bundle)
            self.assertEqual(registered["invoke_existing_agent"], "invoke-existing-agent")
            retrieval = load_tool_retrieval_index(bundle)
            coverage = retrieval.coverage_for_macro("invoke-existing-agent")
            self.assertEqual(coverage.intent_key, "agent.invoke_existing")  # type: ignore[union-attr]
            self.assertEqual(
                set(
                    retrieval.covered_names(
                        "invoke-existing-agent",
                        ["pkg-llmagent-invoke", "pkg-send-to-agent"],
                    )
                ),
                {"pkg-llmagent-invoke", "pkg-send-to-agent"},
            )

    def test_promotes_agent_macro_when_type_inference_has_no_lifecycle_graph(self) -> None:
        promoted = lifecycle_tools_for_skill(
            [
                "openjiuwen-core-application-llm-agent-create-llm-agent",
                "openjiuwen-core-application-llm-agent-llmagent-invoke",
            ],
            None,
            {
                "invoke_existing_agent": "invoke-existing-agent",
                "resume_resource": "resume-resource",
            },
        )
        self.assertEqual(
            promoted,
            ["invoke-existing-agent", "resume-resource"],
        )

    def test_emit_workflow_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = _echo_composite_spec()
            out = Path(tmp) / "out"
            out.mkdir()
            path = emit_composite_workflow_yaml(spec, out, project_name="demo")
            assert path is not None
            text = path.read_text(encoding="utf-8")
            self.assertIn("Composite tool: test_echo_composite", text)
            self.assertIn("echo", text)
            self.assertIn("demo-agent", text)


if __name__ == "__main__":
    unittest.main()
