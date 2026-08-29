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

"""E2E integration test for the create → catalog → invoke lifecycle.

This test exercises the full create → catalog → invoke chain:

1. A temporary SDK exposes ``build_agent()`` (create) and ``run_agent(agent_id, query)``
   (invoke).
2. ``register_component_tools`` emits a wrapper for ``build_agent`` with the
   ``--catalog-metadata`` flag and writes the spec to the bundle-local tool dir.
3. Running the create wrapper writes ``agent_id`` into
   ``<bundle>/.clawcodex/resource-catalog.json``.
4. The ``invoke-existing-agent`` composite macro wrapper reads the resource
   catalog via ResourceHandler, materializes the SDK class, and invokes.

The test drives the wrapper scripts via ``subprocess.run`` exactly as the
Agent tool's bash call handler does.
"""

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
from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
from clawcodex_ext.agent.tool_authoring.persistence import bundle_tool_dir, load_spec
from clawcodex_ext.tool_system.context import ToolContext
from extensions.sop_converter.core.agent_catalog_resolver import (
    HOME_ONLY_ENV,
    HOME_ROOT_ENV,
)
from extensions.sop_converter.core.source_parser import SourceCodeParser
from extensions.sop_converter.resource_catalog import ResourceCatalog
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
    """Return the last non-empty line parsed as JSON."""
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


def _write_fake_sdk(parent: Path) -> Path:
    """Create a tiny importable SDK and return the directory containing it."""
    sdk_dir = parent / "fake_sdk"
    sdk_dir.mkdir(parents=True, exist_ok=True)
    (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
    (sdk_dir / "agent.py").write_text(
        textwrap.dedent(
            """
            from typing import Any


            class DemoAgent:
                def __init__(self, temperature: float = 0.0, model: str = "gpt-4o"):
                    self.temperature = temperature
                    self.model = model

                def build_agent(self, query: str) -> dict[str, Any]:
                    \"\"\"Create a new agent and return its stable id.\"\"\"
                    return {"agent_id": "agent-1", "query": query}

                def invoke(self, query: str) -> dict[str, Any]:
                    \"\"\"Run this agent on the provided query.\"\"\"
                    return {
                        "echo": query,
                        "model": self.model,
                        "temperature": self.temperature,
                    }

                def run_agent(self, agent_id: str, query: str) -> dict[str, Any]:
                    \"\"\"Run an existing agent by id on the provided query.\"\"\"
                    return {"error_code": "agent_not_found", "agent_id": agent_id}
            """
        ).strip(),
        encoding="utf-8",
    )
    return parent


def _write_factory_sdk(parent: Path) -> Path:
    """Create a factory-style SDK matching Jiuwen's LLMAgent shape."""
    sdk_dir = parent / "factory_sdk"
    sdk_dir.mkdir(parents=True, exist_ok=True)
    (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
    (sdk_dir / "factory_agent.py").write_text(
        textwrap.dedent(
            """
            class FactoryAgent:
                def __init__(self, agent_config: dict):
                    self.agent_config = agent_config

                def invoke(self, inputs: dict) -> dict:
                    '''Reply to a message using the configured agent.'''
                    return {
                        "echo": inputs["query"],
                        "agent_id": self.agent_config["id"],
                        "model": self.agent_config["model"]["model_info"]["model"],
                    }


            def create_llm_agent(agent_config: dict) -> FactoryAgent:
                '''Create an agent from its serializable configuration.'''
                return FactoryAgent(agent_config)
            """
        ).strip(),
        encoding="utf-8",
    )
    return parent


def _write_opaque_factory_sdk(parent: Path) -> Path:
    """Factory fixture whose returned object exposes no serializable identity."""
    sdk_dir = parent / "opaque_factory_sdk"
    sdk_dir.mkdir(parents=True, exist_ok=True)
    (sdk_dir / "__init__.py").write_text("", encoding="utf-8")
    (sdk_dir / "factory_agent.py").write_text(
        textwrap.dedent(
            """
            class OpaqueAgent:
                def __init__(self, agent_config: dict):
                    self._agent_config = agent_config

                def invoke(self, query: str) -> dict:
                    return {"echo": query, "agent_id": self._agent_config["id"]}


            def create_llm_agent(agent_config: dict) -> OpaqueAgent:
                '''Create an agent whose identity is only present in the input config.'''
                return OpaqueAgent(agent_config)
            """
        ).strip(),
        encoding="utf-8",
    )
    return parent


class TestCreateToInvokeLifecycleE2E(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()
        self.sdk_parent = _write_fake_sdk(self.tmp)
        self._saved_env = {
            HOME_ROOT_ENV: os.environ.pop(HOME_ROOT_ENV, None),
            HOME_ONLY_ENV: os.environ.pop(HOME_ONLY_ENV, None),
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_create_wrapper_writes_catalog_then_invoke_recovers(self) -> None:
        # Parse the fake SDK into SourceComponents.
        parser = SourceCodeParser(str(self.sdk_parent), extern_only=True)
        components = parser.parse()
        self.assertTrue(components, "SourceCodeParser found no operations")

        # Register tools into the bundle-local registry.
        name_map = register_component_tools(
            components,
            str(self.sdk_parent),
            persist=True,
            bundle_dir=self.bundle,
            bundle_id="test-bundle",
        )

        # Locate the registered build-agent tool.
        build_tool_name: str | None = None
        for key, value in name_map.items():
            if key.endswith(".build_agent"):
                build_tool_name = value
                break
        self.assertIsNotNone(build_tool_name, f"build_agent not found in {name_map}")
        assert build_tool_name is not None

        spec_path = self.bundle / "agent-tools" / f"{build_tool_name}.json"
        self.assertTrue(spec_path.exists(), f"spec not found at {spec_path}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        call_impl = spec["call_impl"]
        self.assertIn("--catalog-metadata", call_impl)

        # Execute the create wrapper.
        args = {"query": "hello"}
        create_stdout = execute_bash(call_impl, {"json_args": json.dumps(args)})
        create_result = _last_json_line(create_stdout)
        self.assertEqual(create_result.get("agent_id"), "agent-1")
        self.assertIs(create_result.get("created_persisted"), True)
        self.assertIs(create_result.get("callable_by_agent_id"), True)
        self.assertEqual(create_result.get("agent_id_call_contract"), "catalog_persisted")
        self.assertIn("catalog_path", create_result)
        self.assertIn("resource_catalog_path", create_result)
        self.assertEqual(
            create_result.get("catalog_reason"),
            "f56_resource_catalog",
        )

        quoted_stdout = execute_bash(
            call_impl,
            {"json_args": json.dumps({"query": "it's fine"})},
        )
        quoted_result = _last_json_line(quoted_stdout)
        self.assertEqual(quoted_result.get("agent_id"), "agent-1")
        self.assertEqual(quoted_result.get("query"), "it's fine")

        # Canonical catalog is resource-catalog.json only.
        catalog_path = self.bundle / ".clawcodex" / "agent-catalog.json"
        self.assertFalse(
            catalog_path.exists(),
            "legacy agent-catalog.json must not be written",
        )
        resource_catalog_path = self.bundle / ".clawcodex" / "resource-catalog.json"
        self.assertTrue(resource_catalog_path.exists(), "resource catalog was not written")
        resource_catalog = ResourceCatalog.load(resource_catalog_path)
        resource_record = resource_catalog.find_by_resource_id("agent-1")[0]
        self.assertEqual(resource_record.resource_id, "agent-1")
        self.assertEqual(resource_record.payload.get("handle_field"), "agent_id")
        self.assertEqual(resource_record.materializer["class_name"], "DemoAgent")
        self.assertNotIn("agent_catalog_entry", resource_record.payload)

        # The native run_agent tool itself should carry catalog fallback metadata.
        run_tool_name: str | None = None
        for key, value in name_map.items():
            if key.endswith(".run_agent"):
                run_tool_name = value
                break
        self.assertIsNotNone(run_tool_name, f"run_agent not found in {name_map}")
        assert run_tool_name is not None
        run_spec = json.loads((self.bundle / "agent-tools" / f"{run_tool_name}.json").read_text(encoding="utf-8"))
        run_call_impl = run_spec["call_impl"]
        self.assertIn("--catalog-fallback", run_call_impl)
        fallback_args = {"agent_id": "agent-1", "query": "via native run"}
        # Simulate SDK in-memory state loss: the generated wrapper sees
        # agent_not_found and must recover via the persisted catalog.
        persisted_run_spec = load_spec(
            run_tool_name,
            tool_dir=bundle_tool_dir(self.bundle),
        )
        self.assertIsNotNone(persisted_run_spec)
        assert persisted_run_spec is not None
        fallback_tool = build_tool_from_spec(persisted_run_spec)
        fallback_call_result = fallback_tool.call(
            fallback_args,
            ToolContext(workspace_root=self.tmp, session_id="fallback-session"),
        )
        self.assertFalse(fallback_call_result.is_error, fallback_call_result.output)
        fallback_result = fallback_call_result.output
        self.assertEqual(fallback_result.get("agent_id"), "agent-1")
        self.assertEqual(
            fallback_result.get("echo") or fallback_result.get("output", {}).get("echo"),
            "via native run",
        )
        self.assertTrue(fallback_result.get("catalog_fallback_attempted"))

        # Execute the invoke-existing-agent macro wrapper.
        invoke_cmd = [
            sys.executable,
            str(INVOKE_WRAPPER),
            "invoke_existing_agent",
            json.dumps({"agent_id": "agent-1", "query": "hello"}),
        ]
        env = os.environ.copy()
        env["CLAWCODEX_BUNDLE_PATH"] = str(self.bundle)
        invoke_proc = subprocess.run(
            invoke_cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        self.assertEqual(
            invoke_proc.returncode,
            0,
            f"invoke wrapper failed: stderr={invoke_proc.stderr!r}",
        )
        invoke_result = _last_json_line(invoke_proc.stdout)
        self.assertEqual(invoke_result.get("agent_id"), "agent-1")
        self.assertEqual(invoke_result.get("output", {}).get("echo"), "hello")
        self.assertEqual(invoke_result.get("output", {}).get("model"), "gpt-4o")


class TestFactoryResultCatalogE2E(unittest.TestCase):
    """Regression: factories return agent objects, not an agent_id dict."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()
        self.sdk_parent = _write_factory_sdk(self.tmp)
        self._saved_env = {
            HOME_ROOT_ENV: os.environ.pop(HOME_ROOT_ENV, None),
            HOME_ONLY_ENV: os.environ.pop(HOME_ONLY_ENV, None),
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for key, value in self._saved_env.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    def test_factory_agent_config_id_is_persisted_and_recreated(self) -> None:
        components = SourceCodeParser(str(self.sdk_parent), extern_only=True).parse()
        name_map = register_component_tools(
            components,
            str(self.sdk_parent),
            persist=True,
            bundle_dir=self.bundle,
            bundle_id="factory-bundle",
        )
        create_tool = next(value for key, value in name_map.items() if key.endswith(".create_llm_agent"))
        spec = json.loads((self.bundle / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
        self.assertIn("--catalog-metadata", spec["call_impl"])
        self.assertIn('"factory"', spec["call_impl"])

        agent_config = {
            "id": "verify-bot",
            "model": {
                "model_provider": "deepseek",
                "model_info": {"model": "deepseek-v4-flash"},
            },
        }
        create_result = _last_json_line(
            execute_bash(
                spec["call_impl"],
                {"json_args": json.dumps({"agent_config": agent_config})},
            )
        )
        self.assertEqual(create_result["agent_id"], "verify-bot")
        self.assertTrue(create_result["created_persisted"])

        self.assertFalse(
            (self.bundle / ".clawcodex" / "agent-catalog.json").exists(),
            "legacy agent-catalog.json must not be written",
        )
        resource_catalog = ResourceCatalog.load(self.bundle / ".clawcodex" / "resource-catalog.json")
        record = resource_catalog.find_by_resource_id("verify-bot")[0]
        self.assertEqual(record.payload.get("model"), "deepseek-v4-flash")
        self.assertEqual(record.payload.get("provider"), "deepseek")
        self.assertEqual(record.materializer["kind"], "python_function")
        self.assertEqual(record.materializer["name"], "create_llm_agent")
        self.assertEqual(record.invoker.get("input_param"), "inputs")
        self.assertEqual(record.metadata.get("factory", {}).get("name"), "create_llm_agent")

        invoke_cmd = [
            sys.executable,
            str(INVOKE_WRAPPER),
            "invoke_existing_agent",
            json.dumps({"agent_id": "verify-bot", "query": "ping"}),
        ]
        env = os.environ.copy()
        env["CLAWCODEX_BUNDLE_PATH"] = str(self.bundle)
        invoke_proc = subprocess.run(
            invoke_cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        self.assertEqual(invoke_proc.returncode, 0, invoke_proc.stderr)
        invoke_result = _last_json_line(invoke_proc.stdout)
        self.assertEqual(invoke_result["output"]["echo"], "ping")
        self.assertEqual(invoke_result["output"]["agent_id"], "verify-bot")

    def test_runtime_tool_call_executes_catalog_cli_hook(self) -> None:
        components = SourceCodeParser(str(self.sdk_parent), extern_only=True).parse()
        name_map = register_component_tools(
            components,
            str(self.sdk_parent),
            persist=True,
            bundle_dir=self.bundle,
            bundle_id="factory-bundle",
        )
        create_tool_name = next(value for key, value in name_map.items() if key.endswith(".create_llm_agent"))
        spec = load_spec(create_tool_name, tool_dir=bundle_tool_dir(self.bundle))
        self.assertIsNotNone(spec)
        assert spec is not None

        tool = build_tool_from_spec(spec)
        result = tool.call(
            {
                "agent_config": {
                    "id": "verify-bot",
                    "model": {
                        "model_provider": "deepseek",
                        "model_info": {"model": "deepseek-v4-flash"},
                    },
                }
            },
            ToolContext(workspace_root=self.tmp, session_id="verify-session"),
        )

        self.assertFalse(result.is_error, result.output)
        self.assertIsInstance(result.output, dict)
        self.assertEqual(result.output["agent_id"], "verify-bot")
        self.assertTrue(result.output["created_persisted"])
        self.assertIn("resource_catalog_path", result.output)
        self.assertTrue((self.bundle / ".clawcodex" / "resource-catalog.json").exists())


class TestOpaqueFactoryCatalogFallbackE2E(unittest.TestCase):
    """A config ID is sufficient when a factory result exposes no identity."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()
        self.sdk_parent = _write_opaque_factory_sdk(self.tmp)
        self._saved_env = {
            HOME_ROOT_ENV: os.environ.pop(HOME_ROOT_ENV, None),
            HOME_ONLY_ENV: os.environ.pop(HOME_ONLY_ENV, None),
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for key, value in self._saved_env.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    def test_agent_config_id_is_used_when_factory_result_is_opaque(self) -> None:
        components = SourceCodeParser(str(self.sdk_parent), extern_only=True).parse()
        name_map = register_component_tools(
            components,
            str(self.sdk_parent),
            persist=True,
            bundle_dir=self.bundle,
            bundle_id="opaque-factory-bundle",
        )
        create_tool = next(value for key, value in name_map.items() if key.endswith(".create_llm_agent"))
        spec = json.loads((self.bundle / "agent-tools" / f"{create_tool}.json").read_text(encoding="utf-8"))
        create_result = _last_json_line(
            execute_bash(
                spec["call_impl"],
                {"json_args": json.dumps({"agent_config": {"id": "verify-bot"}})},
            )
        )
        self.assertEqual(create_result["agent_id"], "verify-bot")
        self.assertTrue(create_result["created_persisted"])
        self.assertTrue((self.bundle / ".clawcodex" / "resource-catalog.json").exists())

        invoke_proc = subprocess.run(
            [
                sys.executable,
                str(INVOKE_WRAPPER),
                "invoke_existing_agent",
                json.dumps({"agent_id": "verify-bot", "query": "ping"}),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "CLAWCODEX_BUNDLE_PATH": str(self.bundle)},
            timeout=30,
        )
        self.assertEqual(invoke_proc.returncode, 0, invoke_proc.stderr)
        invoke_result = _last_json_line(invoke_proc.stdout)
        self.assertEqual(invoke_result["output"]["echo"], "ping")
        self.assertEqual(invoke_result["output"]["agent_id"], "verify-bot")


if __name__ == "__main__":
    unittest.main()
