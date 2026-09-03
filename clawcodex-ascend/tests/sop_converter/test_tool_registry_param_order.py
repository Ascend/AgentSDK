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

"""Lightweight tests for wrapper param ordering (no clawcodex_ext import)."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_bridge():
    pos = ROOT / "extensions" / "sop_converter"
    sp_name = "extensions.sop_converter.source_parser"
    if sp_name not in sys.modules:
        pkg = type(sys)("extensions.sop_converter")
        sys.modules["extensions"] = type(sys)("extensions")
        sys.modules["extensions.sop_converter"] = pkg
        _load(sp_name, pos / "source_parser.py")

    # Stub heavy imports used at module level in tool_registry_bridge.
    for mod_name in (
        "clawcodex_ext.agent.tool_authoring.persistence",
        "clawcodex_ext.agent.tool_authoring.spec",
        "clawcodex_ext.agent.tool_authoring.validators",
        "extensions.sop_converter.search_tags",
    ):
        if mod_name not in sys.modules:
            stub = type(sys)(mod_name)
            if mod_name.endswith("persistence"):
                stub.TOOL_DIR = Path("/tmp/agent-tools")
                stub.SCRIPTS_DIR = Path("/tmp/agent-tools/scripts")
                stub.bundle_tool_dir = lambda _p: Path("/tmp/agent-tools")
                stub.save_spec = lambda *a, **k: None
                stub.scripts_dir_for = lambda _p: Path("/tmp/agent-tools/scripts")
            if mod_name.endswith("spec"):
                stub.AgentToolSpec = type("AgentToolSpec", (), {})
            if mod_name.endswith("validators"):
                stub.validate_spec = lambda _s: None
            if mod_name.endswith("search_tags"):
                stub.generate_search_tags = lambda *a, **k: ()
            sys.modules[mod_name] = stub

    trb_name = "extensions.sop_converter.tool_registry_bridge"
    _load(trb_name, pos / "tool_registry_bridge.py")
    # Stub swaps ``sys.modules[trb_name]`` to the runtime impl; return that.
    return sys.modules[trb_name]


class TestWrapperParamOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trb = _load_bridge()
        cls.ParamSpec = cls.trb.ParamSpec
        cls.SourceOperation = cls.trb.SourceOperation

    def test_required_before_optional_after_merge(self) -> None:
        ParamSpec = self.ParamSpec
        init = [
            ParamSpec(name="team_memory_dir", type_hint="str", required=True),
            ParamSpec(name="sys_operation", type_hint="str", required=False, default="None"),
        ]
        merged = self.trb._merge_init_and_method_params(init, [])
        self.assertEqual([p.name for p in merged], ["team_memory_dir", "sys_operation"])

    def test_shared_memory_manager_stub_parses(self) -> None:
        ParamSpec = self.ParamSpec
        SourceOperation = self.SourceOperation
        init = [
            ParamSpec(name="team_memory_dir", type_hint="str", required=True),
            ParamSpec(name="sys_operation", type_hint="str", required=False, default="None"),
        ]
        op = SourceOperation(
            name="ensure_dir",
            description="Ensure.",
            class_name="SharedMemoryManager",
        )
        stub, _imports = self.trb._generate_method_stub(
            op,
            is_class_method=True,
            module_name="demo.memory",
            init_params=init,
        )
        ast.parse(stub)
        self.assertIn("def ensure_dir(team_memory_dir, sys_operation=None)", stub)

    def test_full_wrapper_script_runs(self) -> None:
        ParamSpec = self.ParamSpec
        SourceOperation = self.SourceOperation
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
            init = [
                ParamSpec(name="team_memory_dir", type_hint="str", required=True),
                ParamSpec(name="sys_operation", type_hint="str", required=False, default="None"),
            ]
            op = SourceOperation(
                name="ensure_dir",
                description="Ensure.",
                class_name="SharedMemoryManager",
                file_stem="memory",
            )
            script = self.trb._generate_wrapper_script(
                [op],
                class_name="SharedMemoryManager",
                module_name="demo_pkg.memory",
                file_stem="memory",
                source_dir=str(tmp),
                init_params=init,
                scripts_dir=tmp / "scripts",
            )
            ast.parse(script.read_text(encoding="utf-8"))
            args = json.dumps({"team_memory_dir": str(tmp / "team-memory")})
            result = subprocess.run(
                [sys.executable, str(script), "ensure_dir", args],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((tmp / "team-memory").is_dir())

    def test_module_fn_required_kwonly_after_optional_parses(self) -> None:
        """Regression: a module-level function whose source uses ``*`` to mark
        a required keyword-only arg (e.g. ``timeout_seconds``) after optional
        args must generate a syntactically valid wrapper stub.

        Previously the non-class-method branch passed ``op.parameters`` raw
        (no required-before-optional reordering), producing
        ``def f(a=None, b, c=None)`` which raises
        ``SyntaxError: non-default argument follows default argument``.
        """
        from extensions.sop_converter.source_parser import SourceCodeParser

        src = '''\nasync def request(session, method, url, *, headers=None,\n    json_body=None, timeout_seconds, max_bytes=None):\n    """Perform an HTTP request."""\n    return None\n'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "_http.py").write_text(src)
            parser = SourceCodeParser(tmp, extern_only=False)
            components = parser.parse()
            op = next(o for o in components[0].operations if o.name == "request")
            stub, _imports = self.trb._generate_method_stub(
                op,
                is_class_method=False,
                module_name="demo_pkg._http",
            )
            ast.parse(stub)  # must not raise
            # required kwonly arg stays required in the signature (no ``=...``)
            def_line = stub.splitlines()[0]
            self.assertIn("timeout_seconds", def_line)
            self.assertNotIn("timeout_seconds=", def_line)

    def test_source_parser_skips_clawcodex_output_dir(self) -> None:
        """Regression: ``.clawcodex/`` inside a source tree (prior convert
        output) must not be re-parsed as source — otherwise VALID generated
        wrappers pollute the component list (and broken ones add noise).

        Uses a *valid* wrapper so the assertion fails when ``.clawcodex`` is
        not excluded (a valid file produces a real SourceComponent, unlike a
        SyntaxError file which is skipped regardless).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "real.py").write_text('def api(x: int) -> str:\n    """Doc."""\n    return str(x)\n')
            bundle = tmp / ".clawcodex" / "proj" / "agent-tools" / "scripts"
            bundle.mkdir(parents=True)
            # A valid generated wrapper (mimics real _http_fn_*.py output).
            (bundle / "generated_wrapper.py").write_text(
                'def call_tool(name: str) -> str:\n    """Generated wrapper."""\n    return name\n'
            )
            from extensions.sop_converter.source_parser import SourceCodeParser

            parser = SourceCodeParser(tmp, extern_only=False)
            components = parser.parse()
            names = {c.name for c in components}
            self.assertIn(tmp.name, names)
            # No component derived from the generated wrapper under .clawcodex
            self.assertFalse(
                any("clawcodex" in n or "generated_wrapper" in n for n in names),
                f"generated wrapper leaked into components: {names}",
            )


if __name__ == "__main__":
    unittest.main()
