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

"""Phase-2 integration tests for ``call_type=workflow``."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from clawcodex_ext.agent.tool_authoring.factory import create_and_validate
from clawcodex_ext.agent.tool_authoring.persistence import load_spec, save_spec
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolCall, ToolResult
from clawcodex_ext.tool_system.registry import ToolRegistry
from extensions.sop_converter.core.agent_catalog import AgentCatalogEntry
from extensions.sop_converter.bundle_context import BundleContext
from extensions.sop_converter.composite_runtime import (
    CompositeWorkflowSpec,
    CompositeWorkflowStep,
)
from extensions.sop_converter.composite_tools import _composite_to_agent_tool_spec
from extensions.sop_converter.runtime.composite_tools.builtin import builtin_composite_tools
from extensions.sop_converter.runtime.macros import register_macro
from extensions.sop_converter.resource_catalog import (
    ResourceCatalog,
    agent_entry_to_resource_record,
    resolve_resource_catalog_path,
)


def _context(tmp: Path, registry: ToolRegistry, *, bundle: Path | None = None) -> ToolContext:
    context = ToolContext(workspace_root=tmp)
    context.tool_registry = registry
    if bundle is not None:
        context.bundle_context = BundleContext(
            bundle_path=bundle,
            bundle_name=bundle.name,
            skill_names=frozenset(),
            skill_dirs=(),
            tool_names=frozenset(tool.name for tool in registry.list_tools()),
        )
    return context


class TestWorkflowToolAuthoring(unittest.TestCase):
    def test_bundle_catalog_rejects_trusted_private_workflow(self) -> None:
        workflow = CompositeWorkflowSpec(
            name="private",
            description="private",
            inputs={},
            steps=(
                CompositeWorkflowStep(
                    id="private",
                    kind="python",
                    callable_ref="x:y",
                    visibility="private",
                ),
            ),
            outputs={},
            trusted=True,
        )
        with self.assertRaisesRegex(ValueError, "only builtin macros may be trusted"):
            register_macro("bundle:private", workflow, replace=True)

    def test_workflow_spec_round_trips_through_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tool_dir = Path(tmp_text)
            spec = AgentToolSpec(
                name="portable-demo",
                description="portable demo",
                input_schema={"type": "object", "properties": {}},
                call_type="workflow",
                call_impl={"catalog_id": "bundle:portable-demo"},
                output_schema={"type": "object", "properties": {"value": {}}},
                bundle_id="bundle",
            )
            save_spec(spec, tool_dir=tool_dir)

            loaded = load_spec("portable-demo", tool_dir=tool_dir)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.call_type, "workflow")
            self.assertEqual(loaded.call_impl, spec.call_impl)
            self.assertEqual(loaded.output_schema, spec.output_schema)

    def test_portable_workflow_dispatches_registered_tool_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            registry = ToolRegistry()
            registry.register(
                build_tool(
                    name="prepare-tool",
                    input_schema={
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                    call=lambda payload, _context: ToolResult(
                        name="prepare-tool",
                        output={"value": payload["message"].upper()},
                    ),
                )
            )
            registry.register(
                build_tool(
                    name="render-tool",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                    call=lambda payload, _context: ToolResult(
                        name="render-tool",
                        output=f"<{payload['value']}>",
                    ),
                )
            )
            workflow = CompositeWorkflowSpec(
                name="portable-demo",
                description="portable demo",
                inputs={"message": {"type": "string", "required": True}},
                steps=(
                    CompositeWorkflowStep(
                        id="prepare",
                        kind="tool",
                        callable_ref="prepare-tool",
                        args={"message": "$input.message"},
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
            register_macro("bundle:portable-demo", workflow, replace=True)
            macro = create_and_validate(
                AgentToolSpec(
                    name="portable-demo",
                    description="portable demo",
                    input_schema={
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                    call_type="workflow",
                    call_impl={"catalog_id": "bundle:portable-demo"},
                    output_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "trace": {"type": "array"},
                        },
                        "required": ["text", "trace"],
                    },
                )
            )
            registry.register(macro)
            context = _context(tmp, registry)

            result = registry.dispatch(
                ToolCall(name="portable-demo", input={"message": "ping"}),
                context,
            )

            self.assertFalse(result.is_error)
            self.assertEqual(result.output["text"], "<PING>")
            self.assertEqual(
                [step["step_id"] for step in result.output["trace"]],
                ["prepare", "render"],
            )

    def test_workflow_stack_rejects_recursive_macro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            workflow = CompositeWorkflowSpec(
                name="recursive",
                description="recursive",
                inputs={},
                steps=(
                    CompositeWorkflowStep(
                        id="again",
                        kind="tool",
                        callable_ref="recursive-workflow",
                    ),
                ),
                outputs={},
            )
            register_macro("bundle:recursive", workflow, replace=True)
            registry = ToolRegistry()
            registry.register(
                create_and_validate(
                    AgentToolSpec(
                        name="recursive-workflow",
                        description="recursive",
                        input_schema={"type": "object", "properties": {}},
                        call_type="workflow",
                        call_impl={"catalog_id": "bundle:recursive"},
                    )
                )
            )
            context = _context(tmp, registry)

            result = registry.dispatch(
                ToolCall(name="recursive-workflow", input={}),
                context,
            )

            self.assertTrue(result.is_error)
            self.assertEqual(result.output["error_code"], "workflow_cycle_detected")
            self.assertEqual(context.workflow_stack, [])

    def test_workflow_lazily_activates_persisted_child_macro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            bundle = tmp / "bundle"
            bundle.mkdir()
            child = CompositeWorkflowSpec(
                name="lazy-child",
                description="lazy child",
                inputs={},
                steps=(),
                outputs={"value": "loaded"},
            )
            parent = CompositeWorkflowSpec(
                name="lazy-parent",
                description="lazy parent",
                inputs={},
                steps=(
                    CompositeWorkflowStep(
                        id="child",
                        kind="tool",
                        callable_ref="lazy-child",
                    ),
                ),
                outputs={"value": "$steps.child.output.value"},
            )
            register_macro("bundle:lazy-child", child, replace=True)
            register_macro("bundle:lazy-parent", parent, replace=True)
            save_spec(
                AgentToolSpec(
                    name="lazy-child",
                    description="lazy child",
                    input_schema={"type": "object", "properties": {}},
                    call_type="workflow",
                    call_impl={"catalog_id": "bundle:lazy-child"},
                    source="composite-tool",
                    bundle_id=bundle.name,
                ),
                tool_dir=bundle / "agent-tools",
            )
            registry = ToolRegistry()
            registry.register(
                create_and_validate(
                    AgentToolSpec(
                        name="lazy-parent",
                        description="lazy parent",
                        input_schema={"type": "object", "properties": {}},
                        call_type="workflow",
                        call_impl={"catalog_id": "bundle:lazy-parent"},
                        source="composite-tool",
                        bundle_id=bundle.name,
                    )
                )
            )
            context = _context(tmp, registry, bundle=bundle)

            result = registry.dispatch(
                ToolCall(name="lazy-parent", input={}),
                context,
            )

            self.assertFalse(result.is_error, result.output)
            self.assertEqual(result.output["value"], "loaded")
            self.assertIsNotNone(registry.get("lazy-child"))

    def test_invoke_existing_agent_uses_bundle_context_and_json_safe_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            bundle = tmp / "bundle"
            bundle.mkdir()
            sdk = tmp / "workflow_sdk"
            sdk.mkdir()
            (sdk / "__init__.py").write_text("", encoding="utf-8")
            (sdk / "agent.py").write_text(
                textwrap.dedent(
                    """
                    class DemoAgent:
                        def invoke(self, query=""):
                            return {"echo": query}
                    """
                ).strip(),
                encoding="utf-8",
            )
            entry = AgentCatalogEntry(
                agent_id="verify-bot",
                sdk_source_dir=str(tmp),
                dsl={"name": "verify-bot"},
                class_name="DemoAgent",
                module_name="workflow_sdk.agent",
            )
            catalog = ResourceCatalog()
            catalog.upsert(agent_entry_to_resource_record(entry, bundle_id=bundle.name))
            catalog.save(resolve_resource_catalog_path(bundle).path)

            invoke_spec = next(
                item for item in builtin_composite_tools(bundle_dir=bundle) if item.name == "invoke_existing_agent"
            )
            registry = ToolRegistry()
            invoke_tool = create_and_validate(_composite_to_agent_tool_spec(invoke_spec, bundle_dir=bundle))
            registry.register(invoke_tool)
            context = _context(tmp, registry, bundle=bundle)

            try:
                result = registry.dispatch(
                    ToolCall(
                        name="invoke-existing-agent",
                        input={"agent_ref": "verify-bot", "query": "ping"},
                    ),
                    context,
                )
            finally:
                sys.modules.pop("workflow_sdk.agent", None)
                sys.modules.pop("workflow_sdk", None)

            self.assertFalse(result.is_error, result.output)
            self.assertEqual(result.output["agent_id"], "verify-bot")
            self.assertEqual(result.output["text"], "ping")
            self.assertEqual(result.output["raw"], {"echo": "ping"})
            json.dumps(result.output)

    def test_portable_create_then_invoke_uses_same_catalog_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            bundle = tmp / "bundle"
            bundle.mkdir()
            sdk = tmp / "create_invoke_sdk"
            sdk.mkdir()
            (sdk / "__init__.py").write_text("", encoding="utf-8")
            (sdk / "agent.py").write_text(
                textwrap.dedent(
                    """
                    class DemoAgent:
                        def invoke(self, query=""):
                            return {"echo": query}
                    """
                ).strip(),
                encoding="utf-8",
            )
            registry = ToolRegistry()

            def create_agent(payload: dict, context: ToolContext) -> ToolResult:
                active_bundle = context.bundle_context
                entry = AgentCatalogEntry(
                    agent_id=payload["agent_id"],
                    sdk_source_dir=str(tmp),
                    dsl={"name": payload["agent_id"]},
                    class_name="DemoAgent",
                    module_name="create_invoke_sdk.agent",
                )
                catalog = ResourceCatalog()
                catalog.upsert(
                    agent_entry_to_resource_record(
                        entry,
                        bundle_id=active_bundle.bundle_name,
                    )
                )
                location = resolve_resource_catalog_path(active_bundle.bundle_path)
                catalog.save(location.path)
                return ToolResult(
                    name="create-agent-tool",
                    output={
                        "agent_id": payload["agent_id"],
                        "created_persisted": True,
                        "resource_catalog_path": str(location.path),
                    },
                )

            registry.register(
                build_tool(
                    name="create-agent-tool",
                    input_schema={
                        "type": "object",
                        "properties": {"agent_id": {"type": "string"}},
                        "required": ["agent_id"],
                    },
                    call=create_agent,
                )
            )
            invoke_spec = next(
                item for item in builtin_composite_tools(bundle_dir=bundle) if item.name == "invoke_existing_agent"
            )
            registry.register(create_and_validate(_composite_to_agent_tool_spec(invoke_spec, bundle_dir=bundle)))
            workflow = CompositeWorkflowSpec(
                name="create-then-invoke",
                description="create and invoke without a model-visible path",
                inputs={
                    "agent_id": {"type": "string", "required": True},
                    "query": {"type": "string", "required": True},
                },
                steps=(
                    CompositeWorkflowStep(
                        id="create",
                        kind="tool",
                        callable_ref="create-agent-tool",
                        args={"agent_id": "$input.agent_id"},
                        output_schema={
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string"},
                                "created_persisted": {"const": True},
                                "resource_catalog_path": {"type": "string"},
                            },
                            "required": [
                                "agent_id",
                                "created_persisted",
                                "resource_catalog_path",
                            ],
                        },
                    ),
                    CompositeWorkflowStep(
                        id="invoke",
                        kind="tool",
                        callable_ref="invoke-existing-agent",
                        args={
                            "agent_ref": "$steps.create.output.agent_id",
                            "query": "$input.query",
                        },
                    ),
                ),
                outputs={
                    "agent_id": "$steps.create.output.agent_id",
                    "text": "$steps.invoke.output.text",
                },
            )
            register_macro("bundle:create-then-invoke", workflow, replace=True)
            registry.register(
                create_and_validate(
                    AgentToolSpec(
                        name="create-then-invoke",
                        description="create then invoke",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string"},
                                "query": {"type": "string"},
                            },
                            "required": ["agent_id", "query"],
                        },
                        call_type="workflow",
                        call_impl={"catalog_id": "bundle:create-then-invoke"},
                    )
                )
            )
            context = _context(tmp, registry, bundle=bundle)

            try:
                result = registry.dispatch(
                    ToolCall(
                        name="create-then-invoke",
                        input={"agent_id": "verify-bot", "query": "ping"},
                    ),
                    context,
                )
            finally:
                sys.modules.pop("create_invoke_sdk.agent", None)
                sys.modules.pop("create_invoke_sdk", None)

            self.assertFalse(result.is_error, result.output)
            self.assertEqual(result.output["agent_id"], "verify-bot")
            self.assertEqual(result.output["text"], "ping")
            self.assertEqual(context.workflow_stack, [])

    def test_invoke_existing_agent_propagates_catalog_version_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            bundle = tmp / "bundle"
            catalog_path = bundle / ".clawcodex" / "resource-catalog.json"
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text(
                json.dumps({"version": 999, "records": {}}),
                encoding="utf-8",
            )
            invoke_spec = next(
                item for item in builtin_composite_tools(bundle_dir=bundle) if item.name == "invoke_existing_agent"
            )
            registry = ToolRegistry(
                [create_and_validate(_composite_to_agent_tool_spec(invoke_spec, bundle_dir=bundle))]
            )
            context = _context(tmp, registry, bundle=bundle)

            result = registry.dispatch(
                ToolCall(
                    name="invoke-existing-agent",
                    input={"agent_id": "verify-bot", "query": "ping"},
                ),
                context,
            )

            self.assertTrue(result.is_error)
            self.assertEqual(result.output["error_code"], "resource_version_unsupported")
            self.assertEqual(result.output["step_id"], "load_agent_record")

    def test_invoke_existing_agent_propagates_missing_secret_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            bundle = tmp / "bundle"
            bundle.mkdir()
            record = agent_entry_to_resource_record(
                AgentCatalogEntry(
                    agent_id="verify-bot",
                    sdk_source_dir=str(tmp),
                    dsl={"name": "verify-bot"},
                    class_name="DemoAgent",
                    module_name="missing_secret_sdk.agent",
                    init_kwargs={"api_key": "env:TEST_PHASE2_MISSING_KEY"},
                    metadata={"env_vars": ["TEST_PHASE2_MISSING_KEY"]},
                ),
                bundle_id=bundle.name,
            )
            catalog = ResourceCatalog()
            catalog.upsert(record)
            catalog.save(resolve_resource_catalog_path(bundle).path)
            invoke_spec = next(
                item for item in builtin_composite_tools(bundle_dir=bundle) if item.name == "invoke_existing_agent"
            )
            registry = ToolRegistry(
                [create_and_validate(_composite_to_agent_tool_spec(invoke_spec, bundle_dir=bundle))]
            )
            context = _context(tmp, registry, bundle=bundle)

            result = registry.dispatch(
                ToolCall(
                    name="invoke-existing-agent",
                    input={"agent_id": "verify-bot", "query": "ping"},
                ),
                context,
            )

            self.assertTrue(result.is_error)
            self.assertEqual(result.output["error_code"], "resource_secret_missing")
            self.assertEqual(result.output["step_id"], "materialize_agent")
            self.assertIn("TEST_PHASE2_MISSING_KEY", result.output["error"])


if __name__ == "__main__":
    unittest.main()
