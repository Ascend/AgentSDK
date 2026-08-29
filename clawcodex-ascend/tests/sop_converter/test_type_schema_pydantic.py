#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------
"""Tests for Pydantic- and dataclass-aware JSON Schema generation in pos convert."""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from extensions.sop_converter.core.source_parser import ParamSpec, SourceOperation
from extensions.sop_converter.tool_registry_bridge import operation_to_spec
from extensions.sop_converter.core.type_schema import (
    param_to_json_schema_property,
    pydantic_schema_for_type,
)

_JIUWEN_AGENT_ROOT = Path(__file__).resolve().parents[2].parent / "JiuwenAgent"


class TestTypeSchemaPydantic(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        pkg = self.root / "demo_models"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "team.py").write_text(
            textwrap.dedent(
                """
                from pydantic import BaseModel, Field


                class TeamCard(BaseModel):
                    id: str = Field(description="Team id")
                    name: str
                    description: str = ""
                """
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_pydantic_schema_includes_properties_and_example(self) -> None:
        schema = pydantic_schema_for_type(str(self.root), "TeamCard")
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertIn("properties", schema)
        self.assertIn("id", schema["properties"])
        self.assertIn("examples", schema)
        self.assertEqual(schema["examples"][0]["id"], "")

    def test_operation_to_spec_uses_pydantic_schema(self) -> None:
        op = SourceOperation(
            name="make_team",
            description="Make a team session.",
            parameters=[
                ParamSpec(
                    name="card",
                    type_hint="TeamCard",
                    required=True,
                    description="Team identity card",
                )
            ],
            file_stem="teams",
        )
        spec = operation_to_spec(
            op,
            source_dir=str(self.root),
            script_path="/tmp/fake_script.py",
            comp_name="demo",
        )
        card = spec.input_schema["properties"]["card"]
        self.assertIn("properties", card)
        self.assertIn("examples", card)
        self.assertEqual(card["properties"]["name"]["type"], "string")


class TestTypeSchemaDataclass(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        pkg = self.root / "demo_models"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "config.py").write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass, field

                from pydantic import BaseModel, Field


                class EndpointInfo(BaseModel):
                    api_key: str = Field(default="")
                    api_base: str = Field(min_length=1)
                    model_name: str = Field(default="", alias="model")


                @dataclass
                class ModelConfig:
                    model_provider: str
                    model_info: EndpointInfo = field(default_factory=EndpointInfo)
                """
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_dataclass_schema_includes_nested_pydantic_fields(self) -> None:
        schema = pydantic_schema_for_type(str(self.root), "ModelConfig")
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertEqual(schema["type"], "object")
        self.assertIn("model_provider", schema["properties"])
        self.assertIn("model_info", schema["properties"])
        self.assertEqual(schema["required"], ["model_provider"])
        model_info = schema["properties"]["model_info"]
        self.assertIn("properties", model_info)
        self.assertIn("api_base", model_info["properties"])
        self.assertIn("examples", schema)
        self.assertEqual(schema["examples"][0]["model_provider"], "")

    def test_param_model_config_is_object_not_string(self) -> None:
        prop = param_to_json_schema_property(
            type_hint="ModelConfig",
            source_dir=str(self.root),
            fallback_json_type="string",
        )
        self.assertEqual(prop["type"], "object")
        self.assertIn("properties", prop)
        self.assertIn("model_provider", prop["properties"])
        self.assertNotEqual(prop.get("type"), "string")

    def test_operation_to_spec_model_param_is_structured(self) -> None:
        op = SourceOperation(
            name="create_agent_config",
            description="Create agent config.",
            parameters=[
                ParamSpec(
                    name="model",
                    type_hint="ModelConfig",
                    required=True,
                    description="LLM model configuration",
                )
            ],
            file_stem="config",
        )
        spec = operation_to_spec(
            op,
            source_dir=str(self.root),
            script_path="/tmp/fake_script.py",
            comp_name="demo",
        )
        model = spec.input_schema["properties"]["model"]
        self.assertEqual(model["type"], "object")
        self.assertIn("model_info", model["properties"])


@unittest.skipUnless(
    (_JIUWEN_AGENT_ROOT / "openjiuwen" / "core" / "foundation" / "llm" / "schema" / "mode_info.py").is_file(),
    "JiuwenAgent checkout not available",
)
class TestJiuwenModelConfigSchema(unittest.TestCase):
    def test_jiuwen_model_config_schema(self) -> None:
        prop = param_to_json_schema_property(
            type_hint="ModelConfig",
            source_dir=str(_JIUWEN_AGENT_ROOT),
            fallback_json_type="string",
        )
        self.assertEqual(prop["type"], "object")
        self.assertIn("model_provider", prop["properties"])
        self.assertIn("model_info", prop["properties"])
        model_info = prop["properties"]["model_info"]
        self.assertIn("properties", model_info)
        self.assertIn("api_base", model_info["properties"])


class TestImportSystemExitIsolation(unittest.TestCase):
    """SDK demos that call sys.exit() on ImportError must not abort convert."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        pkg = self.root / "exit_demo"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "bad_mod.py").write_text(
            textwrap.dedent(
                """
                import sys
                from dataclasses import dataclass

                try:
                    from missing_sibling import something  # noqa: F401
                except ImportError:
                    print("Error importing required modules: No module named 'missing_sibling'")
                    sys.exit(1)


                @dataclass
                class PipelineConfig:
                    name: str = "default"
                """
            ),
            encoding="utf-8",
        )
        (pkg / "good_mod.py").write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass


                @dataclass
                class GoodConfig:
                    value: int = 1
                """
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_import_resolved_type_survives_sys_exit(self) -> None:
        from extensions.sop_converter.core.type_schema import _import_resolved_type

        result = _import_resolved_type(str(self.root), "PipelineConfig", module_path="exit_demo.bad_mod")
        self.assertIsNone(result)

    def test_operation_to_spec_survives_and_continues(self) -> None:
        bad_op = SourceOperation(
            name="run_pipeline",
            description="Runs pipeline.",
            parameters=[
                ParamSpec(
                    name="config",
                    type_hint="PipelineConfig",
                    required=True,
                    description="Pipeline config",
                )
            ],
            file_stem="bad_mod",
        )
        good_op = SourceOperation(
            name="use_good",
            description="Uses good config.",
            parameters=[
                ParamSpec(
                    name="config",
                    type_hint="GoodConfig",
                    required=True,
                    description="Good config",
                )
            ],
            file_stem="good_mod",
        )

        bad_spec = operation_to_spec(
            bad_op,
            source_dir=str(self.root),
            script_path="/tmp/fake_bad.py",
            comp_name="exit_demo",
            module_path="exit_demo.bad_mod",
        )
        good_spec = operation_to_spec(
            good_op,
            source_dir=str(self.root),
            script_path="/tmp/fake_good.py",
            comp_name="exit_demo",
            module_path="exit_demo.good_mod",
        )

        # Bad module degrades; convert must continue for the good op.
        self.assertEqual(bad_spec.input_schema["properties"]["config"]["type"], "object")
        good_cfg = good_spec.input_schema["properties"]["config"]
        self.assertEqual(good_cfg["type"], "object")
        self.assertIn("value", good_cfg.get("properties", {}))


class TestSiblingSrcLayoutImport(unittest.TestCase):
    """Root script + modules under ./src/ must import during schema extraction."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        demo = self.root / "demo_app"
        src = demo / "src"
        src.mkdir(parents=True)
        (src / "run_data_pipeline.py").write_text(
            textwrap.dedent(
                """
                def run_data_pipeline():
                    return {"ok": True}
                """
            ),
            encoding="utf-8",
        )
        (src / "finetune_qwen3.py").write_text(
            textwrap.dedent(
                """
                def run_fine_tuning():
                    return True
                """
            ),
            encoding="utf-8",
        )
        (demo / "run_full_pipeline.py").write_text(
            textwrap.dedent(
                """
                import os
                import sys
                from dataclasses import dataclass

                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

                try:
                    from run_data_pipeline import run_data_pipeline
                    from finetune_qwen3 import run_fine_tuning
                except ImportError as e:
                    print(f"Error importing required modules: {e}")
                    sys.exit(1)


                @dataclass
                class PipelineConfig:
                    name: str = "default"
                """
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if (
                name == "demo_app"
                or name.startswith("demo_app.")
                or name
                in {
                    "run_data_pipeline",
                    "finetune_qwen3",
                }
            ):
                sys.modules.pop(name, None)
        self._tmpdir.cleanup()

    def test_import_resolved_type_finds_pipeline_config(self) -> None:
        from extensions.sop_converter.core.type_schema import _import_resolved_type

        cls = _import_resolved_type(
            str(self.root),
            "PipelineConfig",
            module_path="demo_app.run_full_pipeline",
        )
        self.assertIsNotNone(cls)
        assert cls is not None
        self.assertEqual(cls.__name__, "PipelineConfig")

    def test_operation_to_spec_gets_dataclass_fields(self) -> None:
        op = SourceOperation(
            name="run_full_pipeline",
            description="Run full pipeline.",
            parameters=[
                ParamSpec(
                    name="config",
                    type_hint="PipelineConfig",
                    required=True,
                    description="Pipeline config",
                )
            ],
            file_stem="run_full_pipeline",
        )
        spec = operation_to_spec(
            op,
            source_dir=str(self.root),
            script_path="/tmp/fake.py",
            comp_name="demo_app",
            module_path="demo_app.run_full_pipeline",
        )
        cfg = spec.input_schema["properties"]["config"]
        self.assertEqual(cfg["type"], "object")
        self.assertIn("name", cfg.get("properties", {}))


if __name__ == "__main__":
    unittest.main()
