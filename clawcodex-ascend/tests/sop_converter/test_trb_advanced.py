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
import shlex
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.sop_converter import tool_registry_bridge as trb
from extensions.sop_converter.core.source_parser import (
    ParamSpec,
    SourceComponent,
    SourceOperation,
    SourceCodeParser,
)
from extensions.sop_converter.tool_registry_bridge import (
    _generate_method_stub,
    _generate_wrapper_script,
    _merge_init_and_method_params,
    _param_signature_parts,
    operation_to_spec,
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

    def _cleanup() -> None:
        for p in patches:
            p.stop()
        tmp.cleanup()

    return _cleanup, tool_dir, scripts_dir


# ---------------------------------------------------------------------------
# _strip_optional_union
# ---------------------------------------------------------------------------


class TestClassInitParamsWrapper(unittest.TestCase):
    def test_wrapper_passes_init_kwargs_to_constructor(self) -> None:
        source = '''
def team_memory_dir(team_name: str = "team") -> str:
    """Resolve team memory directory path."""
    return f"/tmp/{team_name}/team-memory"

class SharedMemoryManager:
    """Team shared memory."""

    def __init__(self, team_memory_dir: str) -> None:
        self.team_memory_dir = team_memory_dir

    def ensure_dir(self) -> str:
        """Ensure team-memory directory exists."""
        import os
        os.makedirs(self.team_memory_dir, exist_ok=True)
        return self.team_memory_dir
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            py_file = pkg / "memory.py"
            py_file.write_text(source)

            init_params = [ParamSpec(name="team_memory_dir", type_hint="str", required=True)]
            op = SourceOperation(
                name="ensure_dir",
                description="Ensure team-memory directory exists.",
                class_name="SharedMemoryManager",
                file_stem="memory",
            )
            script_path = _generate_wrapper_script(
                [op],
                class_name="SharedMemoryManager",
                module_name="demo_pkg.memory",
                file_stem="memory",
                source_dir=str(tmp),
                init_params=init_params,
                scripts_dir=tmp / "scripts",
            )

            args = json.dumps({"team_memory_dir": "/tmp/my-team/team-memory"})
            result = __import__("subprocess").run(
                [__import__("sys").executable, str(script_path), "ensure_dir", args],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout.strip()),
                "/tmp/my-team/team-memory",
            )
            self.assertTrue(Path("/tmp/my-team/team-memory").is_dir())

    def test_operation_to_spec_merges_init_params(self) -> None:
        op = SourceOperation(
            name="ensure_dir",
            description="Ensure dir.",
            class_name="SharedMemoryManager",
        )
        init_params = [ParamSpec(name="team_memory_dir", type_hint="str", required=True)]
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path="/tmp/wrapper.py",
            comp_name="memory",
            init_params=init_params,
        )
        self.assertIn("team_memory_dir", spec.input_schema["properties"])
        self.assertIn("team_memory_dir", spec.input_schema.get("required", []))

    def test_merge_puts_required_init_params_before_optional(self) -> None:
        init_params = [
            ParamSpec(name="team_memory_dir", type_hint="str", required=True),
            ParamSpec(name="sys_operation", type_hint="str", required=False, default="None"),
        ]
        merged = _merge_init_and_method_params(init_params, [])
        self.assertEqual([p.name for p in merged], ["team_memory_dir", "sys_operation"])
        signature = ", ".join(_param_signature_parts(merged))
        self.assertEqual(signature, "team_memory_dir, sys_operation=None")

    def test_generated_stub_is_valid_python(self) -> None:
        import ast

        init_params = [
            ParamSpec(name="team_memory_dir", type_hint="str", required=True),
            ParamSpec(name="sys_operation", type_hint="str", required=False, default="None"),
        ]
        op = SourceOperation(
            name="ensure_dir",
            description="Ensure team-memory directory exists.",
            class_name="SharedMemoryManager",
        )
        stub, _imports = _generate_method_stub(
            op,
            is_class_method=True,
            module_name="openjiuwen.agent_teams.memory.shared_memory",
            init_params=init_params,
        )
        ast.parse(stub)
        self.assertIn("def ensure_dir(team_memory_dir, sys_operation=None)", stub)

    def test_wrapper_with_optional_init_param_is_runnable(self) -> None:
        source = """
class SharedMemoryManager:
    def __init__(self, team_memory_dir: str, sys_operation=None) -> None:
        self.team_memory_dir = team_memory_dir

    def ensure_dir(self) -> str:
        import os
        os.makedirs(self.team_memory_dir, exist_ok=True)
        return self.team_memory_dir
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "memory.py").write_text(source)

            init_params = [
                ParamSpec(name="team_memory_dir", type_hint="str", required=True),
                ParamSpec(name="sys_operation", type_hint="str", required=False, default="None"),
            ]
            op = SourceOperation(
                name="ensure_dir",
                description="Ensure.",
                class_name="SharedMemoryManager",
                file_stem="memory",
            )
            script_path = _generate_wrapper_script(
                [op],
                class_name="SharedMemoryManager",
                module_name="demo_pkg.memory",
                file_stem="memory",
                source_dir=str(tmp),
                init_params=init_params,
                scripts_dir=tmp / "scripts",
            )
            import ast

            ast.parse(script_path.read_text(encoding="utf-8"))

            args = json.dumps({"team_memory_dir": "/tmp/p0-fix-team-memory"})
            result = __import__("subprocess").run(
                [__import__("sys").executable, str(script_path), "ensure_dir", args],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(Path("/tmp/p0-fix-team-memory").is_dir())


# ---------------------------------------------------------------------------
# Pydantic / dataclass parameter coercion (wrapper runtime)
# ---------------------------------------------------------------------------


class TestPydanticParamCoercion(unittest.TestCase):
    """JSON dict tool args must be coerced back to SDK model instances."""

    def setUp(self) -> None:
        self._cleanup, self.tool_dir, self.scripts_dir = _isolated_dirs()
        self.addCleanup(self._cleanup)

    def test_stub_includes_model_coercion(self) -> None:
        op = SourceOperation(
            name="create_llm_agent",
            description="Create an LLM agent.",
            parameters=[
                ParamSpec(
                    name="agent_config",
                    type_hint="AgentConfig",
                    required=True,
                ),
            ],
            file_stem="agent",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "agent.py").write_text(
                textwrap.dedent(
                    """
                    from pydantic import BaseModel


                    class AgentConfig(BaseModel):
                        controller_type: str = "react"
                        model: str = "gpt-4"
                    """
                ),
                encoding="utf-8",
            )
            stub, imports = _generate_method_stub(
                op,
                is_class_method=False,
                module_name="demo_pkg.agent",
                source_dir=str(tmp),
            )
            self.assertIn("_coerce_sdk_type(AgentConfig, agent_config)", stub)
            self.assertIn(("demo_pkg.agent", "AgentConfig"), imports)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("pydantic") is not None,
        "pydantic not installed",
    )
    def test_wrapper_coerces_pydantic_dict_at_runtime(self) -> None:
        source = textwrap.dedent(
            '''
            from pydantic import BaseModel


            class AgentConfig(BaseModel):
                controller_type: str = "react"
                model: str = "gpt-4"


            def create_llm_agent(agent_config: AgentConfig) -> str:
                """Create agent and return controller type."""
                return agent_config.controller_type
            '''
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "agent.py").write_text(source, encoding="utf-8")

            op = SourceOperation(
                name="create_llm_agent",
                description="Create agent.",
                parameters=[
                    ParamSpec(
                        name="agent_config",
                        type_hint="AgentConfig",
                        required=True,
                    ),
                ],
                file_stem="agent",
            )
            script_path = _generate_wrapper_script(
                [op],
                class_name=None,
                module_name="demo_pkg.agent",
                file_stem="agent",
                source_dir=str(tmp),
                scripts_dir=self.scripts_dir,
            )
            content = script_path.read_text(encoding="utf-8")
            path_idx = content.index("sys.path.insert(0, _SOURCE_DIR)")
            import_idx = content.index("from demo_pkg.agent import AgentConfig")
            self.assertLess(path_idx, import_idx)

            args = json.dumps({"agent_config": {"controller_type": "react", "model": "gpt-4"}})
            result = __import__("subprocess").run(
                [__import__("sys").executable, str(script_path), "create_llm_agent", args],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.strip()), "react")

    def test_wrapper_coerces_dataclass_init_param(self) -> None:
        source = textwrap.dedent(
            """
            from dataclasses import dataclass


            @dataclass
            class Settings:
                mode: str


            class Service:
                def __init__(self, settings: Settings) -> None:
                    self.settings = settings

                def mode(self) -> str:
                    return self.settings.mode
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "service.py").write_text(source, encoding="utf-8")

            init_params = [
                ParamSpec(name="settings", type_hint="Settings", required=True),
            ]
            op = SourceOperation(
                name="mode",
                description="Return settings mode.",
                class_name="Service",
                file_stem="service",
            )
            script_path = _generate_wrapper_script(
                [op],
                class_name="Service",
                module_name="demo_pkg.service",
                file_stem="service",
                source_dir=str(tmp),
                init_params=init_params,
                scripts_dir=self.scripts_dir,
            )
            args = json.dumps({"settings": {"mode": "debug"}})
            result = __import__("subprocess").run(
                [__import__("sys").executable, str(script_path), "mode", args],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.strip()), "debug")

    def test_wrapper_resolves_model_alias_from_import_context(self) -> None:
        """When two modules define the same name, coercion uses the imported alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "new_config.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass


                    @dataclass
                    class ReActAgentConfig:
                        model: str = "gpt-4"
                    """
                ),
                encoding="utf-8",
            )
            (pkg / "legacy_config.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass


                    @dataclass
                    class LegacyReActAgentConfig:
                        controller_type: str = "react"
                        model: str = "gpt-4"


                    ReActAgentConfig = LegacyReActAgentConfig
                    """
                ),
                encoding="utf-8",
            )
            (pkg / "agent.py").write_text(
                textwrap.dedent(
                    '''
                    from demo_pkg.legacy_config import ReActAgentConfig


                    def create_llm_agent(agent_config: ReActAgentConfig) -> str:
                        """Create agent and return controller type."""
                        return agent_config.controller_type
                    '''
                ),
                encoding="utf-8",
            )

            op = SourceOperation(
                name="create_llm_agent",
                description="Create agent.",
                parameters=[
                    ParamSpec(
                        name="agent_config",
                        type_hint="ReActAgentConfig",
                        required=True,
                    ),
                ],
                file_stem="agent",
            )
            script_path = _generate_wrapper_script(
                [op],
                class_name=None,
                module_name="demo_pkg.agent",
                file_stem="agent",
                source_dir=str(tmp),
                scripts_dir=self.scripts_dir,
            )
            content = script_path.read_text(encoding="utf-8")
            # The wrapper must import the legacy class, not the new one.
            self.assertIn("from demo_pkg.legacy_config import LegacyReActAgentConfig", content)
            self.assertNotIn("from demo_pkg.new_config import ReActAgentConfig", content)

            args = json.dumps({"agent_config": {"controller_type": "react", "model": "gpt-4"}})
            result = __import__("subprocess").run(
                [__import__("sys").executable, str(script_path), "create_llm_agent", args],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.strip()), "react")

        jiouwen_root = Path("D:/projects/JiuwenAgent")
        llm_agent_py = jiouwen_root / "openjiuwen" / "core" / "application" / "llm_agent" / "llm_agent.py"
        if not llm_agent_py.is_file():
            self.skipTest("JiuwenAgent source tree not available")

        parser = SourceCodeParser(str(jiouwen_root / "openjiuwen"))
        components = parser.parse()
        llm_ops = [
            op
            for comp in components
            for op in comp.operations
            if op.file_stem == "llm_agent" and op.name == "create_llm_agent"
        ]
        self.assertTrue(llm_ops, "create_llm_agent operation not found in SDK parse")
        op = llm_ops[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = _generate_wrapper_script(
                [op],
                class_name=None,
                module_name="openjiuwen.core.application.llm_agent.llm_agent",
                file_stem="llm_agent",
                source_dir=str(jiouwen_root),
                scripts_dir=Path(tmpdir),
            )
            content = script_path.read_text(encoding="utf-8")
            self.assertIn(
                "from openjiuwen.core.single_agent.legacy.config import LegacyReActAgentConfig",
                content,
            )
            self.assertNotIn(
                "from openjiuwen.core.single_agent.agents.react_agent import ReActAgentConfig",
                content,
            )
            self.assertIn(
                "_coerce_sdk_type(LegacyReActAgentConfig, agent_config)",
                content,
            )
            compile(content, str(script_path), "exec")


# ---------------------------------------------------------------------------
# CLI handler subprocess bridge
# ---------------------------------------------------------------------------


_SAMPLE_CLI = textwrap.dedent(
    '''\
    """Sample CLI module."""
    import argparse


    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="sample")
        sub = parser.add_subparsers(dest="command")
        proj = sub.add_parser("project", help="Multi-project management")
        proj.add_argument(
            "project_action",
            choices=["list", "create"],
            help="Project action",
        )
        return parser


    def cmd_project(args: argparse.Namespace) -> int:
        """C1: Multi-project management commands."""
        if args.project_action == "list":
            print("no projects")
            return 0
        print(f"created:{getattr(args, 'name', '')}")
        return 0


    def main(argv=None) -> int:
        parser = build_parser()
        args = parser.parse_args(argv)
        command = args.command
        if command == "project":
            return cmd_project(args)
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    '''
)
