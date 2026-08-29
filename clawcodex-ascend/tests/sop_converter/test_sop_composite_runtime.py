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

"""Tests for the executable composite workflow runtime."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from extensions.sop_converter.composite_runtime import (
    CompositeWorkflowError,
    CompositeWorkflowRunner,
    CompositeWorkflowSpec,
    CompositeWorkflowStep,
    normalize_workflow_output,
)
from clawcodex_ext.tool_system.tools.tool_search_matching import rank_tool_matches


@dataclass
class _SearchTool:
    name: str
    aliases: tuple[str, ...] = ()
    search_hint: str = ""
    input_schema: dict | None = None

    def prompt(self) -> str:
        return self.name


class TestCompositeWorkflowRunner(unittest.TestCase):
    def test_runs_linear_python_steps_and_resolves_bindings(self) -> None:
        spec = CompositeWorkflowSpec(
            name="demo",
            description="demo workflow",
            inputs={"message": {"type": "string", "required": True}},
            steps=(
                CompositeWorkflowStep(
                    id="prepare",
                    kind="python",
                    callable_ref="test:prepare",
                    args={"value": "$input.message"},
                ),
                CompositeWorkflowStep(
                    id="render",
                    kind="python",
                    callable_ref="test:render",
                    args={"value": "$steps.prepare.output.value"},
                ),
            ),
            outputs={"output": "$steps.render.output.text"},
            trusted=True,
        )
        runner = CompositeWorkflowRunner(
            python_callables={
                "test:prepare": lambda value: {"value": value.upper()},
                "test:render": lambda value: {"text": f"<{value}>"},
            }
        )

        result = runner.run(spec, {"message": "ping"})

        self.assertFalse(result.is_error)
        self.assertEqual(result.output["output"], "<PING>")
        self.assertEqual([step.status for step in result.trace], ["success", "success"])

    def test_reports_missing_binding_before_running_steps(self) -> None:
        spec = CompositeWorkflowSpec(
            name="demo",
            description="demo workflow",
            inputs={"agent_id": {"type": "string", "required": True}},
            steps=(),
            outputs={},
            trusted=True,
        )

        result = CompositeWorkflowRunner().run(spec, {})

        self.assertTrue(result.is_error)
        self.assertEqual(result.error_code, "workflow_binding_missing")

    def test_preserves_domain_error_code_and_step_trace(self) -> None:
        def fail() -> None:
            raise CompositeWorkflowError("resource_catalog_missing", "record missing")

        spec = CompositeWorkflowSpec(
            name="demo",
            description="demo workflow",
            inputs={},
            steps=(
                CompositeWorkflowStep(
                    id="load_agent_record",
                    kind="catalog",
                    callable_ref="test:fail",
                ),
            ),
            outputs={},
            trusted=True,
        )
        result = CompositeWorkflowRunner(python_callables={"test:fail": fail}).run(spec, {})

        self.assertTrue(result.is_error)
        self.assertEqual(result.error_code, "resource_catalog_missing")
        self.assertEqual(result.trace[-1].step_id, "load_agent_record")
        self.assertEqual(result.trace[-1].status, "error")

    def test_runs_portable_tool_chain_with_normalized_outputs(self) -> None:
        spec = CompositeWorkflowSpec(
            name="portable",
            description="portable tool workflow",
            inputs={"message": {"type": "string", "required": True}},
            steps=(
                CompositeWorkflowStep(
                    id="prepare",
                    kind="tool",
                    callable_ref="prepare-tool",
                    args={"message": "$input.message"},
                    output_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                ),
                CompositeWorkflowStep(
                    id="render",
                    kind="tool",
                    callable_ref="render-tool",
                    args={"value": "$steps.prepare.output.value"},
                ),
            ),
            outputs={"text": "$steps.render.output.text"},
        )

        def run_tool(name: str, args: dict) -> object:
            if name == "prepare-tool":
                return {"value": args["message"].upper()}
            return f"<{args['value']}>"

        result = CompositeWorkflowRunner(tool_runner=run_tool).run(
            spec,
            {"message": "ping"},
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.output, {"text": "<PING>"})

    def test_private_output_is_visible_only_to_trusted_steps(self) -> None:
        opaque = object()
        spec = CompositeWorkflowSpec(
            name="trusted",
            description="trusted private workflow",
            inputs={},
            steps=(
                CompositeWorkflowStep(
                    id="materialize",
                    kind="python",
                    callable_ref="test:materialize",
                    visibility="private",
                ),
                CompositeWorkflowStep(
                    id="invoke",
                    kind="python",
                    callable_ref="test:invoke",
                    args={"agent": "$private.materialize.output.agent"},
                ),
            ),
            outputs={"text": "$steps.invoke.output.text"},
            trusted=True,
        )
        runner = CompositeWorkflowRunner(
            python_callables={
                "test:materialize": lambda: {"agent": opaque},
                "test:invoke": lambda agent: {"text": "ok" if agent is opaque else "bad"},
            }
        )

        result = runner.run(spec, {})

        self.assertFalse(result.is_error)
        self.assertEqual(result.output, {"text": "ok"})
        self.assertNotIn("agent", str(result.output))

    def test_untrusted_workflow_rejects_private_and_python_steps(self) -> None:
        spec = CompositeWorkflowSpec(
            name="untrusted",
            description="untrusted workflow",
            inputs={},
            steps=(
                CompositeWorkflowStep(
                    id="hidden",
                    kind="python",
                    callable_ref="test:hidden",
                    visibility="private",
                ),
            ),
            outputs={},
        )
        result = CompositeWorkflowRunner(python_callables={"test:hidden": lambda: object()}).run(spec, {})

        self.assertTrue(result.is_error)
        self.assertEqual(result.error_code, "workflow_step_failed")

    def test_normalizes_json_text_plain_text_and_scalar(self) -> None:
        self.assertEqual(normalize_workflow_output('{"value":"ok"}'), {"value": "ok"})
        self.assertEqual(
            normalize_workflow_output("ping"),
            {"text": "ping", "value": "ping"},
        )
        self.assertEqual(normalize_workflow_output([1, 2]), {"value": [1, 2]})

    def test_existing_agent_intent_prefers_f57_macro(self) -> None:
        tools = [
            _SearchTool("openjiuwen-core-application-llm-agent-llmagent-invoke"),
            _SearchTool(
                "invoke-existing-agent",
                aliases=("run-existing-agent", "call-agent-by-id"),
            ),
        ]

        matches = rank_tool_matches(
            "invoke existing agent by agent_id",
            tools,
            max_results=5,
        )

        self.assertEqual(matches, ["invoke-existing-agent"])

    def test_named_bot_intent_prefers_f57_macro(self) -> None:
        tools = [
            _SearchTool("openjiuwen-core-application-llm-agent-llmagent-invoke"),
            _SearchTool("invoke-existing-agent"),
        ]

        matches = rank_tool_matches("用 verify-bot 回复 ping", tools, max_results=5)

        self.assertEqual(matches, ["invoke-existing-agent"])

    def test_generic_send_message_search_prefers_f57_macro(self) -> None:
        tools = [
            _SearchTool("openjiuwen-core-controller-legacy-basecontroller-send-to-agent"),
            _SearchTool("openjiuwen-core-application-llm-agent-llmagent-invoke"),
            _SearchTool("invoke-existing-agent"),
        ]

        matches = rank_tool_matches(
            "invoke llm agent send message to agent",
            tools,
            max_results=5,
        )

        self.assertEqual(matches, ["invoke-existing-agent"])


if __name__ == "__main__":
    unittest.main()
