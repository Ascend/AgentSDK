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

"""Extensibility matrix (E1-E5) for resource types."""

from __future__ import annotations

import json
import shlex
import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.core.heuristics.lifecycle import (
    derive_resource_type,
    infer_lifecycle_kind,
    invoke_lifecycle_id_param,
)
from extensions.sop_converter.source_parser import ParamSpec, SourceOperation


def _op(**kwargs: object) -> SourceOperation:
    defaults: dict[str, object] = {
        "name": "invoke",
        "description": "",
        "parameters": [],
        "return_type": None,
        "class_name": None,
        "requires_interactive_input": False,
        "interactive_prompts": [],
    }
    defaults.update(kwargs)
    return SourceOperation(**defaults)  # type: ignore[arg-type]


class TestE1TypeContractNotParamName(unittest.TestCase):
    def test_resource_type_survives_generic_return_wrapper(self) -> None:
        create = _op(
            name="create_llm_agent",
            return_type="Dict[str, AgentConfig]",
        )
        self.assertEqual(derive_resource_type(create), "agentconfig")

    def test_agent_config_param_classifies_as_invoke_via_type(self) -> None:
        create = _op(name="create_llm_agent", return_type="AgentConfig")
        known = frozenset({derive_resource_type(create)})
        invoke = _op(
            name="invoke",
            parameters=[
                ParamSpec(name="agent_config", type_hint="AgentConfig", required=True),
                ParamSpec(name="query", type_hint="str", required=True),
            ],
        )

        self.assertEqual(infer_lifecycle_kind(invoke, known_create_types=known), "invoke")
        self.assertEqual(invoke_lifecycle_id_param(invoke), "agent_config")


class TestE2LegacyIdParamStillWorks(unittest.TestCase):
    def test_agent_id_param_still_invoke(self) -> None:
        invoke = _op(
            name="invoke_agent",
            parameters=[
                ParamSpec(name="agent_id", type_hint="str", required=True),
                ParamSpec(name="query", type_hint="str", required=True),
            ],
        )
        self.assertEqual(infer_lifecycle_kind(invoke), "invoke")


class TestE3NoTypeFallsBackToHeuristic(unittest.TestCase):
    def test_untyped_non_id_stays_none(self) -> None:
        invoke = _op(
            name="invoke",
            parameters=[ParamSpec(name="payload", type_hint=None, required=True)],
        )
        self.assertEqual(
            infer_lifecycle_kind(invoke, known_create_types=frozenset()),
            "none",
        )


class TestResourceRefSchemaInjection(unittest.TestCase):
    def test_invoke_schema_gets_resource_ref(self) -> None:
        from extensions.sop_converter.core.heuristics.lifecycle import (
            inject_resource_ref_schema,
        )

        schema = {
            "type": "object",
            "properties": {
                "agent_config": {"type": "object"},
                "query": {"type": "string"},
            },
            "required": ["agent_config", "query"],
        }
        out = inject_resource_ref_schema(
            schema,
            resource_type="agentconfig",
            create_tool_name="create-llm-agent",
            consume_param="agent_config",
        )

        self.assertEqual(out["properties"]["resource_ref"]["type"], "string")
        self.assertIn(
            "create-llm-agent",
            out["properties"]["resource_ref"]["description"],
        )
        self.assertEqual(out["properties"]["resource_type"]["default"], "agentconfig")
        self.assertIn("resource_ref", out["required"])
        self.assertNotIn("agent_config", out["required"])
        self.assertIn("agent_config", out["properties"])
        self.assertNotIn("resource_ref", schema["properties"])


class TestDynamicHandleField(unittest.TestCase):
    def test_create_metadata_does_not_default_foreign_type_to_agent_id(self) -> None:
        from extensions.sop_converter.core.heuristics.lifecycle import lifecycle_metadata_payload

        op = _op(name="create_team", return_type="TeamSession")
        metadata = lifecycle_metadata_payload(op, source_dir="/tmp/sdk")

        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata["resource_type"], "teamsession")
        self.assertNotEqual(metadata["handle_field"], "agent_id")


class TestFallbackPrefersResourceRef(unittest.TestCase):
    def test_stable_handle_from_resource_ref_arg(self) -> None:
        from extensions.sop_converter.tool_registry_bridge import (
            resolve_catalog_handle_from_args,
        )

        handle = resolve_catalog_handle_from_args(
            {"resource_ref": "verify-bot", "agent_id": "legacy", "query": "ping"},
            {"id_arg": "agent_id", "resource_type": "agentconfig"},
        )
        self.assertEqual(handle, "verify-bot")


class TestResourceHandlerRegistry(unittest.TestCase):
    def test_agent_is_registered_as_first_row(self) -> None:
        from extensions.sop_converter.resource_handlers import (
            ensure_builtin_handlers,
            get_resource_handler,
        )

        ensure_builtin_handlers()
        handler = get_resource_handler("agent")
        self.assertIsNotNone(handler)
        assert handler is not None
        self.assertTrue(callable(handler.materialize))
        self.assertTrue(callable(handler.invoke))

    def test_unknown_type_returns_none(self) -> None:
        from extensions.sop_converter.resource_handlers import (
            ensure_builtin_handlers,
            get_resource_handler,
        )

        ensure_builtin_handlers()
        self.assertIsNone(get_resource_handler("not-a-real-type-xyz"))


class TestE4SecondResourceTypeViaRegistry(unittest.TestCase):
    def test_demo_handle_create_catalog_invoke(self) -> None:
        from extensions.sop_converter.resource_catalog import (
            ResourceCatalog,
            ResourceRecord,
            get_resource_record,
            resolve_resource_catalog_path,
        )
        from extensions.sop_converter.resource_handlers import (
            ResourceHandler,
            register_resource_handler,
            require_resource_handler,
        )

        def materialize(record: ResourceRecord) -> dict:
            return {"demo": {"id": record.resource_id, "payload": record.payload}}

        def invoke(
            record: ResourceRecord,
            query: str = "",
            inputs: object = None,
        ) -> dict:
            del inputs
            resource = materialize(record)
            return {"text": query or "ok", "resource_id": resource["demo"]["id"]}

        register_resource_handler(
            ResourceHandler(
                resource_type="DemoHandle",
                materialize=materialize,
                invoke=invoke,
                public_output_schema={"type": "object", "required": ["text"]},
                error_codes=frozenset({"resource_materialize_failed"}),
            ),
            replace=True,
        )
        record = ResourceRecord(
            resource_type="DemoHandle",
            resource_id="demo-1",
            bundle_id="test",
            source_tool="create-demo",
            materializer={"kind": "demo"},
            invoker={"kind": "demo"},
            payload={"handle_field": "id", "value": "demo-1"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            path = resolve_resource_catalog_path(bundle).path
            catalog = ResourceCatalog()
            catalog.upsert(record)
            catalog.save(path)
            loaded = get_resource_record(
                "demo-1",
                resource_type="DemoHandle",
                bundle_path=bundle,
            )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        output = require_resource_handler("DemoHandle").invoke(loaded, query="ping")
        self.assertEqual(output["text"], "ping")
        self.assertEqual(output["resource_id"], "demo-1")


class TestE5UnregisteredTypeDoesNotUseAgentPath(unittest.TestCase):
    def test_unknown_type_rejected(self) -> None:
        from extensions.sop_converter.resource_handlers import require_resource_handler

        with self.assertRaises(Exception) as raised:
            require_resource_handler("totally-unknown-type")
        error = raised.exception
        code = getattr(error, "error_code", None) or (error.args[0] if error.args else "")
        self.assertIn("resource_type_unregistered", f"{code}{error}")


class TestSidecarResourcesOverride(unittest.TestCase):
    def test_parse_resources_yaml(self) -> None:
        from extensions.sop_converter.core.bundle_resources import load_resource_bindings

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            sidecar = bundle / ".clawcodex" / "resources.yaml"
            sidecar.parent.mkdir()
            sidecar.write_text(
                "resources:\n"
                "  - type: TeamSession\n"
                "    create: create-team\n"
                "    invoke: run-team\n"
                "    handle_field: session_id\n",
                encoding="utf-8",
            )
            bindings = load_resource_bindings(bundle)

        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0].resource_type, "TeamSession")
        self.assertEqual(bindings[0].create, "create-team")
        self.assertEqual(bindings[0].invoke, "run-team")
        self.assertEqual(bindings[0].handle_field, "session_id")

    def test_sidecar_applies_to_convert_pairing_and_schema(self) -> None:
        from extensions.sop_converter.source_parser import SourceCodeParser
        from extensions.sop_converter.tool_registry_bridge import register_component_tools

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sdk"
            source.mkdir()
            (source / "session_api.py").write_text(
                "def provision(label: str):\n"
                '    """Create a reusable team session."""\n'
                "    return {'session_id': label}\n\n"
                "def execute(session_id: str, query: str):\n"
                '    """Invoke a previously created team session."""\n'
                "    return {'text': query}\n",
                encoding="utf-8",
            )
            bundle = root / "bundle"
            sidecar = bundle / ".clawcodex" / "resources.yaml"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text(
                "resources:\n"
                "  - type: TeamSession\n"
                "    create: provision\n"
                "    invoke: execute\n"
                "    handle_field: session_id\n",
                encoding="utf-8",
            )
            name_map = register_component_tools(
                SourceCodeParser(str(source)).parse(),
                str(source),
                persist=True,
                bundle_dir=bundle,
                bundle_id="sidecar-test",
            )
            create_name = name_map["session_api.provision"]
            invoke_name = name_map["session_api.execute"]
            create_spec = json.loads((bundle / "agent-tools" / f"{create_name}.json").read_text(encoding="utf-8"))
            invoke_spec = json.loads((bundle / "agent-tools" / f"{invoke_name}.json").read_text(encoding="utf-8"))

        create_tokens = shlex.split(create_spec["call_impl"])
        invoke_tokens = shlex.split(invoke_spec["call_impl"])
        create_meta = json.loads(create_tokens[create_tokens.index("--catalog-metadata") + 1])
        invoke_meta = json.loads(invoke_tokens[invoke_tokens.index("--catalog-fallback") + 1])
        self.assertEqual(create_meta["resource_type"], "teamsession")
        self.assertEqual(create_meta["handle_field"], "session_id")
        self.assertEqual(invoke_meta["resource_type"], "teamsession")
        self.assertEqual(
            invoke_spec["input_schema"]["properties"]["resource_ref"]["type"],
            "string",
        )
        self.assertIn("resource_ref", invoke_spec["input_schema"]["required"])


if __name__ == "__main__":
    unittest.main()
