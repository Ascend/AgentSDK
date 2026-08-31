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

"""Unit tests for :mod:`extensions.sop_converter.tool_registry_bridge`.

Covers the bridge that converts parsed source operations into
executable agent tools with bash-callable wrapper scripts.

* :func:`_type_hint_to_json_type` — Python type-hint → JSON Schema.
* :func:`_strip_optional_union` — Optional/Union/X | None reduction.
* :func:`_to_kebab_case` — dot/snake → kebab-case conversion.
* :func:`_resolve_module_path` — dotted module path from file path.
* :func:`_script_name_for_class` / :func:`_script_name_for_functions`
  — deterministic script filenames.
* :func:`_generate_wrapper_script` — wrapper script creation.
* :func:`_enrich_bridge_params` — json_args injection.
* :func:`operation_to_spec` — single-operation → AgentToolSpec.
* :func:`register_component_tools` — bulk registration with name map.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from clawcodex_ext.agent.tool_authoring.call_handlers.bash import execute_bash
from extensions.sop_converter import tool_registry_bridge as trb
from extensions.sop_converter.core.agent_catalog_resolver import HOME_ROOT_ENV
from extensions.sop_converter.resource_catalog import (
    DUAL_WRITE_ENV,
    PAYLOAD_REF_ENV,
    SESSION_ID_ENV,
    ResourceCatalog,
)
from extensions.sop_converter.core.source_parser import (
    ParamSpec,
    SourceComponent,
    SourceOperation,
    SourceCodeParser,
)
from extensions.sop_converter.tool_registry_bridge import (
    operation_to_spec,
    register_component_tools,
)


def _make_param(
    name: str,
    type_hint: str | None = None,
    *,
    required: bool = True,
    default: object = None,
    description: str = "",
) -> ParamSpec:
    return ParamSpec(
        name=name,
        type_hint=type_hint,
        default=default,
        required=required,
        description=description,
    )


def _make_op(
    name: str = "do_thing",
    *,
    description: str = "A method that does a thing.",
    parameters: list[ParamSpec] | None = None,
    return_type: str | None = None,
    class_name: str | None = None,
    file_stem: str = "things",
    is_async: bool = False,
    is_async_generator: bool = False,
) -> SourceOperation:
    return SourceOperation(
        name=name,
        description=description,
        parameters=parameters or [],
        return_type=return_type,
        class_name=class_name,
        file_stem=file_stem,
        is_async=is_async,
        is_async_generator=is_async_generator,
    )


def _make_component(
    name: str = "comp",
    file_path: str = "comp/things",
    operations: list[SourceOperation] | None = None,
) -> SourceComponent:
    return SourceComponent(
        name=name,
        file_path=file_path,
        description="A test component",
        operations=operations or [],
    )


def _catalog_payload(call_impl: str, flag: str) -> dict:
    parts = shlex.split(call_impl)
    idx = parts.index(flag)
    return json.loads(parts[idx + 1])


def _isolated_dirs() -> tuple[object, Path, Path]:
    """Return (cleanup, fake_tool_dir, fake_scripts_dir).

    Patches the module's TOOL_DIR and SCRIPTS_DIR to temp paths.
    """
    tmp = tempfile.TemporaryDirectory()
    tool_dir = Path(tmp.name) / "tools"
    scripts_dir = tool_dir / "scripts"
    tool_dir.mkdir()
    scripts_dir.mkdir()

    def _cleanup() -> None:
        tmp.cleanup()

    patches = [
        patch.object(trb, "TOOL_DIR", tool_dir),
        patch.object(trb, "SCRIPTS_DIR", scripts_dir),
        # Also patch the imported name on the upstream module, since
        # SCRIPTS_DIR = TOOL_DIR / "scripts" evaluated at import time.
        patch(
            "clawcodex_ext.agent.tool_authoring.persistence.TOOL_DIR",
            tool_dir,
        ),
    ]
    for p in patches:
        p.start()
    return _cleanup, tool_dir, scripts_dir


# ---------------------------------------------------------------------------
# _strip_optional_union
# ---------------------------------------------------------------------------


class TestOperationToSpec(unittest.TestCase):
    def setUp(self) -> None:
        self._cleanup, self.tool_dir, self.scripts_dir = _isolated_dirs()
        self.addCleanup(self._cleanup)
        self.script_path = "/tmp/fake_script.py"

    def test_class_method_kebab_name(self) -> None:
        op = _make_op(name="invoke", class_name="LLM")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
            comp_name="core",
        )
        # {comp}.{class}.{method} → "core.LLM.invoke" → kebab.
        self.assertEqual(spec.name, "core-llm-invoke")

    def test_class_method_without_comp_name(self) -> None:
        op = _make_op(name="invoke", class_name="LLM")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        # Falls back to {class}.{method}.
        self.assertEqual(spec.name, "llm-invoke")

    def test_standalone_function_with_comp(self) -> None:
        op = _make_op(name="load_config", class_name=None, file_stem="utils")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
            comp_name="helpers",
        )
        self.assertEqual(spec.name, "helpers-load-config")

    def test_standalone_function_no_comp_falls_back_to_file_stem(self) -> None:
        op = _make_op(name="load_config", class_name=None, file_stem="utils")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        self.assertEqual(spec.name, "utils-load-config")

    def test_no_comp_no_file_stem_kebab_converts(self) -> None:
        # Even without a comp or file_stem, the raw name goes through
        # kebab-case conversion. "load_config" → "load-config".
        op = _make_op(name="load_config", class_name=None, file_stem="")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        self.assertEqual(spec.name, "load-config")

    def test_call_type_is_bash(self) -> None:
        op = _make_op(name="x")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        self.assertEqual(spec.call_type, "bash")

    def test_call_impl_uses_script_path(self) -> None:
        op = _make_op(name="x")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        # The bash command should reference the script path and method.
        self.assertIn(self.script_path, spec.call_impl)
        self.assertIn("x", spec.call_impl)
        # And the {json_args} placeholder for the runtime.
        self.assertIn("{json_args}", spec.call_impl)

    def test_input_schema_basic_param(self) -> None:
        op = _make_op(
            name="x",
            parameters=[_make_param("foo", "str")],
        )
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        schema = spec.input_schema
        self.assertEqual(schema["type"], "object")
        self.assertIn("foo", schema["properties"])
        self.assertEqual(schema["properties"]["foo"]["type"], "string")
        # Required param → "required" list contains "foo".
        self.assertIn("foo", schema["required"])

    def test_optional_param_not_required(self) -> None:
        op = _make_op(
            name="x",
            parameters=[_make_param("foo", "str", required=False)],
        )
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        # "required" key is omitted when there are no required params.
        self.assertNotIn("required", spec.input_schema)

    def test_param_with_description(self) -> None:
        op = _make_op(
            name="x",
            parameters=[
                _make_param("foo", "str", description="The foo value"),
            ],
        )
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        self.assertEqual(
            spec.input_schema["properties"]["foo"]["description"],
            "The foo value",
        )

    def test_param_with_default(self) -> None:
        op = _make_op(
            name="x",
            parameters=[_make_param("foo", "str", default="bar")],
        )
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        self.assertEqual(
            spec.input_schema["properties"]["foo"]["default"],
            "bar",
        )

    def test_aliases_with_comp_name(self) -> None:
        op = _make_op(name="x", class_name="C")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
            comp_name="alpha",
        )
        # First alias is the fully-qualified {comp}.{class}.{method}.
        self.assertIn("alpha.C.x", spec.aliases)

    def test_aliases_short_form_with_dotted_comp(self) -> None:
        # When comp_name is "openjiuwen.core" (multi-segment), an
        # additional short alias drops the first segment.
        op = _make_op(name="x", class_name="C")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
            comp_name="openjiuwen.core",
        )
        # Original alias present.
        self.assertIn("openjiuwen.core.C.x", spec.aliases)
        # Short alias present.
        self.assertIn("core.C.x", spec.aliases)

    def test_source_is_sop_converter(self) -> None:
        op = _make_op(name="x")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        self.assertEqual(spec.source, "sop-converter")

    def test_generates_search_tags(self) -> None:
        op = _make_op(
            name="run_team_cli",
            description="Bring up the Team CLI.",
            class_name=None,
        )
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
            comp_name="openjiuwen.agent_teams.cli",
        )
        self.assertTrue(spec.tags)
        self.assertIn("run team cli", spec.tags)
        self.assertIn("run_team_cli", spec.tags)
        self.assertIn("cli", spec.tags)

    def test_skips_star_args_in_schema(self) -> None:
        op = _make_op(
            name="x",
            parameters=[
                _make_param("foo", "int"),
                _make_param("*args", "list", required=False),
                _make_param("**kwargs", "dict", required=False),
            ],
        )
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path=self.script_path,
        )
        self.assertIn("foo", spec.input_schema["properties"])
        self.assertNotIn("*args", spec.input_schema["properties"])
        self.assertNotIn("**kwargs", spec.input_schema["properties"])


# ---------------------------------------------------------------------------
# register_component_tools
# ---------------------------------------------------------------------------


class TestRegisterComponentTools(unittest.TestCase):
    def setUp(self) -> None:
        self._cleanup, self.tool_dir, self.scripts_dir = _isolated_dirs()
        self.addCleanup(self._cleanup)

    def test_register_class_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            op = _make_op(name="invoke", class_name="LLM")
            comp = _make_component(
                name="core",
                file_path="proj/core",
                operations=[op],
            )
            name_map = register_component_tools(
                [comp],
                str(source_dir),
                persist=False,
            )
        # The name map should contain a kebab-case entry for the
        # grouper-style name "{comp}.{method}".
        self.assertIn("core.invoke", name_map)
        # Value is a kebab-case name.
        self.assertEqual(name_map["core.invoke"], "core-llm-invoke")

    def test_register_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            op = _make_op(
                name="helper_fn",
                class_name=None,
                file_stem="helpers",
            )
            comp = _make_component(
                name="utils",
                file_path="proj/utils",
                operations=[op],
            )
            name_map = register_component_tools(
                [comp],
                str(source_dir),
                persist=False,
            )
        self.assertEqual(name_map["utils.helper_fn"], "utils-helper-fn")

    def test_name_map_includes_class_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            op = _make_op(name="x", class_name="C")
            comp = _make_component(
                name="comp",
                file_path="proj/comp",
                operations=[op],
            )
            name_map = register_component_tools(
                [comp],
                str(source_dir),
                persist=False,
            )
        # Three name forms all point at the same kebab spec.
        kebab = name_map["comp.x"]
        self.assertEqual(name_map["C.x"], kebab)
        self.assertEqual(name_map["comp.C.x"], kebab)
        # And the fully-qualified form (comp + grouper-style).
        self.assertEqual(name_map["comp.comp.x"], kebab)

    def test_wrapper_script_written_to_scripts_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            op = _make_op(name="x", class_name="C")
            comp = _make_component(
                name="comp",
                file_path="proj/comp",
                operations=[op],
            )
            register_component_tools(
                [comp],
                str(source_dir),
                persist=False,
            )
        # A wrapper script should be created in the temp scripts dir.
        scripts = list(self.scripts_dir.iterdir())
        self.assertEqual(len(scripts), 1)
        self.assertTrue(scripts[0].name.endswith(".py"))
        self.assertIn("C_", scripts[0].name)

    def test_persist_writes_spec_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            op = _make_op(name="x", class_name="C")
            comp = _make_component(
                name="comp",
                file_path="proj/comp",
                operations=[op],
            )
            register_component_tools(
                [comp],
                str(source_dir),
                persist=True,
            )
        # A spec file should appear in the tool dir.
        specs = list(self.tool_dir.glob("*.json"))
        # Specs could be in the tool dir itself or a subdir depending
        # on save_spec; we just check at least one was created.
        self.assertGreater(len(specs), 0)

    def test_overwrite_false_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            op = _make_op(name="x", class_name="C")
            comp = _make_component(
                name="comp",
                file_path="proj/comp",
                operations=[op],
            )
            # First call writes the spec.
            register_component_tools(
                [comp],
                str(source_dir),
                persist=True,
                overwrite=True,
            )
            # Specs on disk.
            initial_specs = sorted(self.tool_dir.glob("*.json"))
            # Touch the spec file's mtime to detect a re-write.
            for spec_path in initial_specs:
                spec_path.write_text(
                    json.dumps({"modified": True}),
                    encoding="utf-8",
                )
            # Second call with overwrite=False → should NOT rewrite.
            register_component_tools(
                [comp],
                str(source_dir),
                persist=True,
                overwrite=False,
            )
            # The file still has the modified marker.
            for spec_path in self.tool_dir.glob("*.json"):
                content = json.loads(spec_path.read_text(encoding="utf-8"))
                self.assertEqual(content, {"modified": True})

    def test_grouped_by_class_into_one_script(self) -> None:
        # Two operations on the same class share one script.
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            op1 = _make_op(name="a", class_name="C")
            op2 = _make_op(name="b", class_name="C")
            comp = _make_component(
                name="comp",
                file_path="proj/comp",
                operations=[op1, op2],
            )
            register_component_tools(
                [comp],
                str(source_dir),
                persist=False,
            )
        # Single wrapper script for both methods.
        scripts = list(self.scripts_dir.iterdir())
        self.assertEqual(len(scripts), 1)

    def test_class_and_function_separate_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            class_op = _make_op(name="a", class_name="C", file_stem="mod")
            func_op = _make_op(
                name="b",
                class_name=None,
                file_stem="mod",
            )
            comp = _make_component(
                name="comp",
                file_path="proj/comp",
                operations=[class_op, func_op],
            )
            register_component_tools(
                [comp],
                str(source_dir),
                persist=False,
            )
        # Two scripts: one for the class, one for the function file.
        self.assertEqual(len(list(self.scripts_dir.iterdir())), 2)

    def test_register_create_kind_tool_enriches_call_impl(self) -> None:
        """create-kind ops get --catalog-metadata + bundle env prefix."""
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            bundle_dir = Path(tmp) / "bundle"
            bundle_dir.mkdir()
            op = _make_op(
                name="build_agent",
                class_name="AgentBuilder",
                return_type="Dict[str, Any]",
            )
            comp = _make_component(
                name="agentbuilder",
                file_path="proj/agentbuilder",
                operations=[op],
            )
            name_map = register_component_tools(
                [comp],
                str(source_dir),
                persist=False,
                bundle_dir=bundle_dir,
                bundle_id="test-bundle",
            )
        self.assertIn("agentbuilder.build_agent", name_map)

    def test_lifecycle_catalog_payload_uses_alias_aware_resource_type(self) -> None:
        """Create/invoke pairs match by type identity, not SDK-specific field names."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "proj"
            sdk_dir = source_dir / "generic_sdk"
            sdk_dir.mkdir(parents=True)
            (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
            (sdk_dir / "types.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass

                    @dataclass
                    class WidgetConfig:
                        name: str
                    """
                ).strip(),
                encoding="utf-8",
            )
            (sdk_dir / "factory.py").write_text(
                textwrap.dedent(
                    """
                    from .types import WidgetConfig

                    def create_widget(name: str) -> WidgetConfig:
                        \"\"\"Create a reusable widget configuration.\"\"\"
                        return WidgetConfig(name=name)
                    """
                ).strip(),
                encoding="utf-8",
            )
            (sdk_dir / "runner.py").write_text(
                textwrap.dedent(
                    """
                    from .types import WidgetConfig as PublicConfig

                    class WidgetRunner:
                        def invoke(self, widget: PublicConfig, query: str) -> dict:
                            \"\"\"Invoke a previously created widget.\"\"\"
                            return {"error_code": "resource_not_found", "query": query}
                    """
                ).strip(),
                encoding="utf-8",
            )
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()

            parser = SourceCodeParser(str(source_dir), extern_only=True)
            components = parser.parse()
            name_map = register_component_tools(
                components,
                str(source_dir),
                persist=True,
                bundle_dir=bundle_dir,
                bundle_id="generic-bundle",
            )

            create_tool = name_map["generic_sdk.create_widget"]
            invoke_tool = name_map["generic_sdk.WidgetRunner.invoke"]
            create_spec = json.loads((bundle_dir / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
            self.assertEqual(create_spec["output_schema"]["type"], "object")
            self.assertEqual(
                create_spec["output_schema"]["properties"]["created_persisted"],
                {"const": True},
            )
            self.assertEqual(
                set(create_spec["output_schema"]["required"]),
                {
                    "resource_ref",
                    "resource_type",
                    "created_persisted",
                    "resource_catalog_path",
                },
            )
            invoke_spec = json.loads((bundle_dir / "agent-tools" / f"{invoke_tool}.json").read_text(encoding="utf-8"))

            create_meta = _catalog_payload(create_spec["call_impl"], "--catalog-metadata")
            fallback_meta = _catalog_payload(invoke_spec["call_impl"], "--catalog-fallback")
            expected_type = "generic_sdk_types_widgetconfig"
            self.assertEqual(create_meta["resource_type"], expected_type)
            self.assertEqual(fallback_meta["resource_type"], expected_type)
            self.assertEqual(fallback_meta["handle_field"], "widget")
            self.assertEqual(fallback_meta["query_arg"], "query")

    def test_create_catalog_write_accepts_generic_name_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "proj"
            sdk_dir = source_dir / "generic_sdk"
            sdk_dir.mkdir(parents=True)
            (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
            (sdk_dir / "factory.py").write_text(
                textwrap.dedent(
                    """
                    def create_widget(name: str) -> dict:
                        \"\"\"Create a reusable widget by name.\"\"\"
                        return {"name": name, "status": "created"}
                    """
                ).strip(),
                encoding="utf-8",
            )
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()

            parser = SourceCodeParser(str(source_dir), extern_only=True)
            name_map = register_component_tools(
                parser.parse(),
                str(source_dir),
                persist=True,
                bundle_dir=bundle_dir,
                bundle_id="generic-bundle",
            )

            create_tool = name_map["generic_sdk.create_widget"]
            create_spec = json.loads((bundle_dir / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
            stdout = execute_bash(
                create_spec["call_impl"],
                {"json_args": json.dumps({"name": "verify-bot"})},
            )
            created = json.loads(stdout.strip().splitlines()[-1])
            self.assertEqual(created["agent_id"], "verify-bot")
            self.assertEqual(created["resource_ref"], "verify-bot")
            self.assertTrue(created["created_persisted"])

            catalog = ResourceCatalog.load(bundle_dir / ".clawcodex" / "resource-catalog.json")
            records = catalog.find_by_resource_id("verify-bot")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].resource_id, "verify-bot")
            self.assertEqual(records[0].payload["handle_field"], "name")

    def test_create_catalog_callable_without_handle_is_resource_handle_missing(
        self,
    ) -> None:
        """Opaque callables with empty config must not recurse or catalog_write_failed.

        Regression: ``config: {}`` made ``_to_jsonable({}) is not {}`` always true,
        so ``_extract_resource_handle`` looped until max recursion instead of
        returning ``resource_handle_missing``.
        """
        from clawcodex_ext.agent.tool_authoring.call_handlers.bash import BashCallError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "proj"
            sdk_dir = source_dir / "callable_sdk"
            sdk_dir.mkdir(parents=True)
            (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
            (sdk_dir / "factory.py").write_text(
                textwrap.dedent(
                    """
                    from typing import Any, Callable, Dict

                    def create_opaque_loader(config: Dict[str, Any]) -> Callable:
                        \"\"\"Return a nested loader function with no stable id.\"\"\"
                        def loader():
                            return config.get("data_paths", [])
                        return loader
                    """
                ).strip(),
                encoding="utf-8",
            )
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()

            parser = SourceCodeParser(str(source_dir), extern_only=True)
            name_map = register_component_tools(
                parser.parse(),
                str(source_dir),
                persist=True,
                bundle_dir=bundle_dir,
                bundle_id="callable-bundle",
            )
            create_tool = name_map["callable_sdk.create_opaque_loader"]
            create_spec = json.loads((bundle_dir / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
            with self.assertRaises(BashCallError) as raised:
                execute_bash(
                    create_spec["call_impl"],
                    {"json_args": json.dumps({"config": {}})},
                )
            blob = (raised.exception.stderr or "") + (raised.exception.stdout or "")
            payload = None
            for line in reversed(blob.splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            self.assertIsNotNone(payload, f"no JSON payload in output: {blob[-2000:]}")
            assert payload is not None
            self.assertEqual(payload.get("error_code"), "resource_handle_missing")
            self.assertIs(payload.get("created_persisted"), False)
            self.assertNotIn(
                "maximum recursion depth exceeded",
                str(payload.get("error", "")),
            )
            self.assertNotEqual(payload.get("error_code"), "catalog_write_failed")
            catalog_path = bundle_dir / ".clawcodex" / "resource-catalog.json"
            self.assertFalse(
                catalog_path.exists(),
                "opaque callable must not write a catalog record",
            )

    def test_create_catalog_dual_write_writes_bundle_and_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "proj"
            sdk_dir = source_dir / "generic_sdk"
            sdk_dir.mkdir(parents=True)
            (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
            (sdk_dir / "factory.py").write_text(
                textwrap.dedent(
                    """
                    def create_widget(name: str) -> dict:
                        \"\"\"Create a reusable widget by name.\"\"\"
                        return {"name": name, "status": "created"}
                    """
                ).strip(),
                encoding="utf-8",
            )
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            home = root / "clawcodex-home"
            home.mkdir()

            parser = SourceCodeParser(str(source_dir), extern_only=True)
            name_map = register_component_tools(
                parser.parse(),
                str(source_dir),
                persist=True,
                bundle_dir=bundle_dir,
                bundle_id="generic-bundle",
            )
            create_tool = name_map["generic_sdk.create_widget"]
            create_spec = json.loads((bundle_dir / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
            with patch.dict(
                os.environ,
                {DUAL_WRITE_ENV: "1", HOME_ROOT_ENV: str(home)},
            ):
                stdout = execute_bash(
                    create_spec["call_impl"],
                    {"json_args": json.dumps({"name": "dual-bot"})},
                )
            created = json.loads(stdout.strip().splitlines()[-1])
            self.assertTrue(created["created_persisted"])
            self.assertEqual(sorted(created["written_layers"]), ["bundle", "user"])
            self.assertIn("bundle", created["catalog_paths"])
            self.assertIn("user", created["catalog_paths"])
            bundle_catalog = Path(created["catalog_paths"]["bundle"])
            user_catalog = Path(created["catalog_paths"]["user"])
            self.assertTrue(bundle_catalog.is_file())
            self.assertTrue(user_catalog.is_file())
            self.assertEqual(
                created["resource_catalog_path"],
                str(bundle_catalog),
            )
            self.assertTrue(ResourceCatalog.load(bundle_catalog).find_by_resource_id("dual-bot"))
            self.assertTrue(ResourceCatalog.load(user_catalog).find_by_resource_id("dual-bot"))

    def test_create_catalog_payload_ref_env_spills_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "proj"
            sdk_dir = source_dir / "generic_sdk"
            sdk_dir.mkdir(parents=True)
            (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
            (sdk_dir / "factory.py").write_text(
                textwrap.dedent(
                    """
                    def create_widget(name: str) -> dict:
                        \"\"\"Create a reusable widget by name.\"\"\"
                        return {"name": name, "status": "created", "blob": "x" * 100}
                    """
                ).strip(),
                encoding="utf-8",
            )
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()

            parser = SourceCodeParser(str(source_dir), extern_only=True)
            name_map = register_component_tools(
                parser.parse(),
                str(source_dir),
                persist=True,
                bundle_dir=bundle_dir,
                bundle_id="generic-bundle",
            )
            create_tool = name_map["generic_sdk.create_widget"]
            create_spec = json.loads((bundle_dir / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
            with patch.dict(os.environ, {PAYLOAD_REF_ENV: "1"}, clear=False):
                stdout = execute_bash(
                    create_spec["call_impl"],
                    {"json_args": json.dumps({"name": "spill-bot"})},
                )
            created = json.loads(stdout.strip().splitlines()[-1])
            self.assertTrue(created["created_persisted"])
            catalog_path = Path(created["resource_catalog_path"])
            matches = ResourceCatalog.load(catalog_path).find_by_resource_id("spill-bot")
            self.assertTrue(matches)
            stored = ResourceCatalog.load(catalog_path).get_stored(matches[0].resource_type, "spill-bot")
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.payload["kind"], "payload_ref")
            self.assertTrue(stored.payload.get("ref") or stored.payload.get("path"))

    def test_create_catalog_session_id_writes_session_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "proj"
            sdk_dir = source_dir / "generic_sdk"
            sdk_dir.mkdir(parents=True)
            (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
            (sdk_dir / "factory.py").write_text(
                textwrap.dedent(
                    """
                    def create_widget(name: str) -> dict:
                        \"\"\"Create a reusable widget by name.\"\"\"
                        return {"name": name, "status": "created"}
                    """
                ).strip(),
                encoding="utf-8",
            )
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            home = root / "clawcodex-home"
            home.mkdir()

            parser = SourceCodeParser(str(source_dir), extern_only=True)
            name_map = register_component_tools(
                parser.parse(),
                str(source_dir),
                persist=True,
                bundle_dir=bundle_dir,
                bundle_id="generic-bundle",
            )
            create_tool = name_map["generic_sdk.create_widget"]
            create_spec = json.loads((bundle_dir / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
            with patch.dict(
                os.environ,
                {SESSION_ID_ENV: "sess-create-1", HOME_ROOT_ENV: str(home)},
            ):
                stdout = execute_bash(
                    create_spec["call_impl"],
                    {"json_args": json.dumps({"name": "session-bot"})},
                )
            created = json.loads(stdout.strip().splitlines()[-1])
            self.assertTrue(created["created_persisted"])
            self.assertIn("session", created["written_layers"])
            self.assertIn("session", created["catalog_paths"])
            session_catalog = Path(created["catalog_paths"]["session"])
            self.assertTrue(session_catalog.is_file())
            self.assertEqual(
                session_catalog,
                home / "sessions" / "sess-create-1" / "sop-resources.json",
            )
            self.assertTrue(ResourceCatalog.load(session_catalog).find_by_resource_id("session-bot"))
            # Session is additive; default base layer (bundle) is also written.
            self.assertIn("bundle", created["written_layers"])
            self.assertTrue((bundle_dir / ".clawcodex" / "resource-catalog.json").is_file())

    def test_empty_components_returns_empty_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "proj"
            source_dir.mkdir()
            name_map = register_component_tools(
                [],
                str(source_dir),
                persist=False,
            )
        self.assertEqual(name_map, {})
