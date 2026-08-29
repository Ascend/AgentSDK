#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Phase B — compiler / trace-to-macro / promote."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import ToolContext, ToolUseOptions
from clawcodex_ext.types.content_blocks import ToolResultBlock, ToolUseBlock
from clawcodex_ext.types.messages import AssistantMessage, UserMessage
from extensions.sop_converter.runtime.macros.compiler import (
    MacroDraft,
    compile_macro_definition,
)
from extensions.sop_converter.runtime.macros.errors import MacroConvertError
from extensions.sop_converter.runtime.macros.promote import promote_session_macro_to_bundle
from extensions.sop_converter.runtime.macros.register_tool import (
    PROMOTE_MACRO_WORKFLOW_TOOL_NAME,
    REGISTER_MACRO_FROM_TRACE_TOOL_NAME,
    PromoteMacroWorkflowTool,
    RegisterMacroFromTraceTool,
)
from extensions.sop_converter.runtime.macros.session import (
    SessionMacroOverlay,
    is_session_macro_tool,
    mark_session_macro_tool,
    register_session_macro,
)
from extensions.sop_converter.runtime.macros.trace import (
    TraceToolStep,
    extract_successful_tool_steps,
    trace_steps_to_definition_dict,
)
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import Tool


def _minimal_definition(**overrides) -> dict:
    data = {
        "version": 1,
        "name": "session-echo-demo",
        "description": "A session macro for tests",
        "scope": "session",
        "enabled": True,
        "workflow": {
            "inputs": {"query": {"type": "string", "required": True}},
            "steps": [
                {
                    "id": "run",
                    "kind": "tool",
                    "callable_ref": "echo-tool",
                    "args": {"text": "$input.query"},
                }
            ],
            "outputs": {"result": "$steps.run.output"},
        },
        "routing": {
            "phrases": ["run session demo"],
            "keywords": ["demo"],
            "selection": "prefer",
            "priority": 100,
            "target_tool": "session-echo-demo",
        },
        "provenance": {"kind": "session_nl"},
    }
    data.update(overrides)
    return data


def _create_tool_from_spec(spec: AgentToolSpec) -> Tool:
    return mark_session_macro_tool(
        build_tool(
            name=spec.name,
            input_schema=dict(spec.input_schema or {"type": "object", "properties": {}}),
            call=lambda _i, _c: {"ok": True},
            prompt=spec.description or spec.name,
        )
    )


def _ctx(*, allow: bool = True, confirm=None, messages=None, bundle_path=None):
    options = ToolUseOptions(
        tools=[
            build_tool(
                name="echo-tool",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                call=lambda i, _c: {"text": i.get("text")},
                prompt="echo",
            )
        ]
    )
    ctx = ToolContext(
        workspace_root=".",
        session_id="sess-1",
        session_macro_overlay=SessionMacroOverlay(),
        allow_session_macro_registration=allow,
        confirm_session_macro_plan=confirm if confirm is not None else (lambda _p: True),
        options=options,
        messages=list(messages or []),
    )
    if bundle_path is not None:
        ctx.bundle_context = SimpleNamespace(
            bundle_path=Path(bundle_path),
            tool_names=["echo-tool"],
        )
    return ctx


class TestMacroCompiler(unittest.TestCase):
    def test_compile_accepts_valid_definition(self) -> None:
        validated = compile_macro_definition(
            _minimal_definition(),
            tool_index={"echo-tool"},
        )
        self.assertEqual(validated.definition.name, "session-echo-demo")
        self.assertEqual(validated.tool_spec.call_type, "workflow")

    def test_compile_rejects_unknown_field(self) -> None:
        data = _minimal_definition()
        data["extra"] = True
        with self.assertRaises(MacroConvertError) as raised:
            compile_macro_definition(data, tool_index={"echo-tool"})
        self.assertTrue(raised.exception.error_code)

    def test_compile_rejects_exclusive(self) -> None:
        data = _minimal_definition()
        data["routing"] = {
            **data["routing"],
            "selection": "exclusive",
        }
        with self.assertRaises(MacroConvertError) as raised:
            compile_macro_definition(data, tool_index={"echo-tool"})
        self.assertEqual(raised.exception.error_code, "macro_selection_forbidden")

    def test_macro_draft_dataclass(self) -> None:
        draft = MacroDraft(proposed_name="foo", provenance="session_trace")
        self.assertEqual(draft.requested_scope, "session")


class TestTraceExtract(unittest.TestCase):
    def test_extract_latest_successful_run(self) -> None:
        messages = [
            AssistantMessage(
                content=[
                    ToolUseBlock(id="1", name="echo-tool", input={"text": "a"}),
                ]
            ),
            UserMessage(
                content=[ToolResultBlock(tool_use_id="1", content="a", is_error=False)],
                toolUseResult={"text": "a"},
            ),
            AssistantMessage(
                content=[
                    ToolUseBlock(id="2", name="echo-tool", input={"text": "b"}),
                ]
            ),
            UserMessage(
                content=[ToolResultBlock(tool_use_id="2", content="x", is_error=True)],
            ),
            AssistantMessage(
                content=[
                    ToolUseBlock(id="3", name="echo-tool", input={"text": "c"}),
                    ToolUseBlock(id="4", name="echo-tool", input={"text": "c"}),
                ]
            ),
            UserMessage(
                content=[
                    ToolResultBlock(tool_use_id="3", content="c", is_error=False),
                    ToolResultBlock(tool_use_id="4", content="c", is_error=False),
                ],
                toolUseResult={"text": "c"},
            ),
        ]
        steps = extract_successful_tool_steps(messages, max_steps=16)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].input["text"], "c")

    def test_empty_trace_errors(self) -> None:
        with self.assertRaises(MacroConvertError) as raised:
            trace_steps_to_definition_dict([], name="empty-macro")
        self.assertEqual(raised.exception.error_code, "macro_trace_empty")

    def test_binding_when_next_input_equals_prev_output(self) -> None:
        steps = [
            TraceToolStep(tool_name="echo-tool", input={"text": "hello"}, output="hello"),
            TraceToolStep(tool_name="echo-tool", input={"text": "hello"}, output="hello"),
        ]
        definition = trace_steps_to_definition_dict(steps, name="bind-demo")
        self.assertEqual(
            definition["workflow"]["steps"][1]["args"]["text"],
            "$steps.step1.output",
        )
        self.assertEqual(definition["provenance"]["kind"], "session_trace")


class TestRegisterFromTrace(unittest.TestCase):
    def test_from_trace_deny_confirm_zero_write(self) -> None:
        messages = [
            AssistantMessage(content=[ToolUseBlock(id="1", name="echo-tool", input={"text": "hi"})]),
            UserMessage(
                content=[ToolResultBlock(tool_use_id="1", content="hi", is_error=False)],
                toolUseResult="hi",
            ),
        ]
        ctx = _ctx(confirm=lambda _p: False, messages=messages)
        result = RegisterMacroFromTraceTool.call(
            {"name": "from-trace-demo", "replace": False},
            ctx,
        )
        self.assertTrue(result.is_error)
        self.assertEqual(result.name, REGISTER_MACRO_FROM_TRACE_TOOL_NAME)
        self.assertEqual(result.output.get("error_code"), "macro_registration_denied")
        self.assertIsNone(ctx.session_macro_overlay.read())

    def test_from_trace_yes_registers(self) -> None:
        messages = [
            AssistantMessage(content=[ToolUseBlock(id="1", name="echo-tool", input={"text": "hi"})]),
            UserMessage(
                content=[ToolResultBlock(tool_use_id="1", content="hi", is_error=False)],
                toolUseResult="hi",
            ),
        ]
        ctx = _ctx(messages=messages)
        result = RegisterMacroFromTraceTool.call(
            {"name": "from-trace-demo", "replace": False},
            ctx,
        )
        self.assertFalse(result.is_error)
        self.assertTrue(result.output.get("registered"))
        snap = ctx.session_macro_overlay.read()
        self.assertIsNotNone(snap)
        self.assertIn("from-trace-demo", snap.definitions)
        self.assertTrue(any(is_session_macro_tool(t) for t in ctx.options.tools or []))


class TestPromote(unittest.TestCase):
    def test_promote_requires_bundle(self) -> None:
        ctx = _ctx()
        register_session_macro(
            ctx,
            _minimal_definition(),
            replace=False,
            tool_index={"echo-tool"},
            workflow_tool_names=set(),
            protected_builtin_exclusive_targets=set(),
            create_tool=_create_tool_from_spec,
        )
        with self.assertRaises(MacroConvertError) as raised:
            promote_session_macro_to_bundle(ctx, "session-echo-demo", replace=False)
        self.assertEqual(raised.exception.error_code, "macro_promote_no_bundle")

    def test_promote_deny_no_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(bundle_path=tmp, confirm=lambda _p: True)
            register_session_macro(
                ctx,
                _minimal_definition(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
            ctx.confirm_session_macro_plan = lambda _p: False
            result = PromoteMacroWorkflowTool.call(
                {"name": "session-echo-demo", "replace": False},
                ctx,
            )
            self.assertTrue(result.is_error)
            self.assertEqual(result.name, PROMOTE_MACRO_WORKFLOW_TOOL_NAME)
            self.assertEqual(result.output.get("error_code"), "macro_registration_denied")
            macros = list(Path(tmp).joinpath(".clawcodex", "macros").glob("*.yaml"))
            self.assertEqual(macros, [])

    def test_promote_yes_writes_yaml_and_keeps_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(bundle_path=tmp)
            register_session_macro(
                ctx,
                _minimal_definition(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
            result = PromoteMacroWorkflowTool.call(
                {"name": "session-echo-demo", "replace": False},
                ctx,
            )
            self.assertFalse(result.is_error, result.output)
            self.assertTrue(result.output.get("promoted"))
            path = Path(result.output["path"])
            self.assertTrue(path.is_file())
            self.assertTrue(result.output.get("session_retained"))
            snap = ctx.session_macro_overlay.read()
            self.assertIsNotNone(snap)
            self.assertIn("session-echo-demo", snap.definitions)

            from extensions.sop_converter.runtime.macros.catalog import resolve_macro

            resolved = resolve_macro({"catalog_id": "bundle:session-echo-demo"})
            self.assertEqual(resolved.name, "session-echo-demo")

    def test_capability_denied_for_trace_and_promote(self) -> None:
        ctx = _ctx(allow=False)
        r1 = RegisterMacroFromTraceTool.call({"name": "x"}, ctx)
        self.assertEqual(r1.output.get("error_code"), "macro_capability_denied")
        r2 = PromoteMacroWorkflowTool.call({"name": "x"}, ctx)
        self.assertEqual(r2.output.get("error_code"), "macro_capability_denied")


if __name__ == "__main__":
    unittest.main()
