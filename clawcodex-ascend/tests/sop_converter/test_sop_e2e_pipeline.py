"""End-to-end SOP pipeline scenarios (design + executable smoke)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from clawcodex_ext.agent.tool_authoring.call_handlers.bash import execute_bash
from extensions.sop_converter.core.agent_catalog_resolver import HOME_ROOT_ENV
from extensions.sop_converter.resource_catalog import ResourceCatalog
from extensions.sop_converter.core.source_parser import SourceCodeParser
from extensions.sop_converter.tool_registry_bridge import register_component_tools

REPO_ROOT = Path(__file__).resolve().parents[2]
INVOKE_WRAPPER = (
    REPO_ROOT
    / "extensions"
    / "sop_converter"
    / "runtime"
    / "composite_tools"
    / "scripts"
    / "invoke_existing_agent_wrapper.py"
)


def _last_json_line(raw: str) -> dict:
    text = raw.strip()
    if not text:
        raise AssertionError("no output")
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON line found in output: {raw!r}")


def _write_sdk(parent: Path) -> Path:
    sdk = parent / "e2e_sdk"
    sdk.mkdir(parents=True)
    (sdk / "__init__.py").write_text("", encoding="utf-8")
    (sdk / "agent.py").write_text(
        textwrap.dedent(
            """
            from typing import Any

            class DemoAgent:
                def __init__(self, temperature: float = 0.0, model: str = "gpt-4o"):
                    self.temperature = temperature
                    self.model = model

                def build_agent(self, query: str) -> dict[str, Any]:
                    \"\"\"Create a new agent and return its stable id.\"\"\"
                    return {"agent_id": "e2e-agent-1", "query": query}

                def invoke(self, query: str = "") -> dict[str, Any]:
                    \"\"\"Invoke this agent instance.\"\"\"
                    return {"echo": query, "model": self.model, "temperature": self.temperature}

                def run_agent(self, agent_id: str, query: str) -> dict[str, Any]:
                    \"\"\"Native invoke-by-id tool used with catalog fallback.\"\"\"
                    return {"agent_id": agent_id, "echo": query}
            """
        ).strip(),
        encoding="utf-8",
    )
    return sdk


class TestSopE2ECreateThenInvoke(unittest.TestCase):
    """End-to-end: create persists the resource catalog, invoke recovers in a subprocess."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.home = self.root / "clawcodex-home"
        self.home.mkdir()
        self._saved_home = os.environ.get(HOME_ROOT_ENV)
        os.environ[HOME_ROOT_ENV] = str(self.home)
        _write_sdk(self.root)

    def tearDown(self) -> None:
        if self._saved_home is None:
            os.environ.pop(HOME_ROOT_ENV, None)
        else:
            os.environ[HOME_ROOT_ENV] = self._saved_home
        self._tmp.cleanup()

    def test_create_persists_resource_catalog_then_invoke_recovers(self) -> None:
        if not INVOKE_WRAPPER.is_file():
            self.skipTest(f"missing wrapper at {INVOKE_WRAPPER}")

        components = SourceCodeParser(str(self.root), extern_only=True).parse()
        name_map = register_component_tools(
            components,
            str(self.root),
            persist=True,
            bundle_dir=self.bundle,
            bundle_id=self.bundle.name,
        )
        build_tool = next(v for k, v in name_map.items() if k.endswith(".build_agent"))
        spec = json.loads((self.bundle / "agent-tools" / f"{build_tool}.json").read_text(encoding="utf-8"))
        self.assertIn("--catalog-metadata", spec["call_impl"])

        created = _last_json_line(execute_bash(spec["call_impl"], {"json_args": json.dumps({"query": "hello"})}))
        self.assertEqual(created.get("agent_id"), "e2e-agent-1")
        self.assertTrue(created.get("created_persisted"))
        self.assertEqual(created.get("catalog_reason"), "f56_resource_catalog")
        self.assertFalse((self.bundle / ".clawcodex" / "agent-catalog.json").exists())
        resource_path = self.bundle / ".clawcodex" / "resource-catalog.json"
        self.assertTrue(resource_path.is_file())
        self.assertTrue(ResourceCatalog.load(resource_path).find_by_resource_id("e2e-agent-1"))

        proc = subprocess.run(
            [
                sys.executable,
                str(INVOKE_WRAPPER),
                "invoke_existing_agent",
                json.dumps({"agent_id": "e2e-agent-1", "query": "ping"}),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "CLAWCODEX_BUNDLE_PATH": str(self.bundle)},
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = _last_json_line(proc.stdout)
        self.assertEqual(out.get("agent_id"), "e2e-agent-1")
        payload = out.get("output") or out
        self.assertEqual(payload.get("echo") or (payload.get("raw") or {}).get("echo"), "ping")


if __name__ == "__main__":
    unittest.main()
