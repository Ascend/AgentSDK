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

"""Regression tests for SDK reconstruction and secret references."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from extensions.sop_converter.core.agent_runtime import AgentRuntimeError, materialize_agent
from extensions.sop_converter.resource_catalog import (
    RESOURCE_PAYLOAD_REF_MISSING,
    ResourceCatalogError,
    ResourceRecord,
    spill_payload_if_needed,
)
from extensions.sop_converter.resource_runtime import materialize_resource
from extensions.sop_converter.core.sdk_serialization import resolve_env_references


class TestEnvironmentReferences(unittest.TestCase):
    def test_only_explicit_env_references_are_resolved(self) -> None:
        value = {
            "api_key": "env:DEEPSEEK_API_KEY",
            "shell_style": "$DEEPSEEK_API_KEY",
        }

        resolved = resolve_env_references(
            value,
            environ={"DEEPSEEK_API_KEY": "secret-value"},
        )

        self.assertEqual(resolved["api_key"], "secret-value")
        self.assertEqual(resolved["shell_style"], "$DEEPSEEK_API_KEY")

    def test_missing_environment_reference_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment variable is not set"):
            resolve_env_references("env:MISSING_DEEPSEEK_API_KEY", environ={})

    def test_shell_style_api_key_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "use env:NAME"):
            resolve_env_references({"api_key": "$DEEPSEEK_API_KEY"}, environ={})


class TestAgentMaterialization(unittest.TestCase):
    def test_missing_record_secret_has_stable_error_code(self) -> None:
        record = ResourceRecord(
            resource_type="agent",
            resource_id="verify-bot",
            source_tool="create-agent",
            materializer={"kind": "python_function", "module": "x", "name": "create"},
            invoker={},
            payload={"init_kwargs": {"api_key": "env:TEST_MISSING_SECRET_KEY"}},
            secrets={"env_refs": ["TEST_MISSING_SECRET_KEY"]},
        )
        with self.assertRaises(AgentRuntimeError) as raised:
            materialize_agent(record)
        self.assertEqual(raised.exception.error_code, "resource_secret_missing")
        self.assertIn("TEST_MISSING_SECRET_KEY", str(raised.exception))

    def test_factory_reconstructs_typed_config_and_resolves_env_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            package = tmp / "typed_sdk"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "factory.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass


                    @dataclass
                    class AgentConfig:
                        id: str
                        api_key: str


                    class Agent:
                        def __init__(self, config: AgentConfig):
                            self.config = config


                    def create_agent(agent_config: AgentConfig) -> Agent:
                        return Agent(agent_config)
                    """
                ).strip(),
                encoding="utf-8",
            )
            record = ResourceRecord(
                resource_type="agent",
                resource_id="verify-bot",
                source_tool="create-agent",
                materializer={
                    "kind": "python_function",
                    "module": "typed_sdk.factory",
                    "name": "create_agent",
                },
                invoker={"kind": "python_method", "method": "invoke"},
                payload={
                    "init_kwargs": {
                        "agent_config": {
                            "id": "verify-bot",
                            "api_key": "env:TEST_DEEPSEEK_API_KEY",
                        }
                    }
                },
                sdk={"source_dir": str(tmp)},
            )
            previous = os.environ.get("TEST_DEEPSEEK_API_KEY")
            os.environ["TEST_DEEPSEEK_API_KEY"] = "secret-value"
            try:
                materialized = materialize_agent(record)
            finally:
                if previous is None:
                    os.environ.pop("TEST_DEEPSEEK_API_KEY", None)
                else:
                    os.environ["TEST_DEEPSEEK_API_KEY"] = previous
                sys.modules.pop("typed_sdk.factory", None)
                sys.modules.pop("typed_sdk", None)

        agent = materialized["agent"]
        self.assertEqual(agent.config.id, "verify-bot")
        self.assertEqual(agent.config.api_key, "secret-value")

    def test_omitted_list_factory_defaults_match_create_wrapper(self) -> None:
        """Rematerialize must apply create-wrapper ``(param or [])``.

        Create wrappers coerce omitted ``List[...] = None`` factory args to
        ``[]`` before calling the SDK. Catalog ``init_kwargs`` only stores
        args the model passed (often just ``agent_config``). Materialize must
        still supply empty lists so SDK helpers like ``len(workflows)`` do not
        crash with ``object of type 'NoneType' has no len()``.
        """
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            package = tmp / "list_default_sdk"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "factory.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass
                    from typing import List, Optional


                    @dataclass
                    class AgentConfig:
                        id: str


                    class Agent:
                        def __init__(self, config: AgentConfig, workflows, tools, note):
                            self.config = config
                            self.workflows = workflows
                            self.tools = tools
                            self.note = note


                    def create_agent(
                        agent_config: AgentConfig,
                        workflows: List[str] = None,
                        tools: List[str] = None,
                        note: Optional[str] = None,
                    ) -> Agent:
                        # Mirrors Jiuwen BaseAgent.add_workflows(len(workflows)).
                        if len(workflows) != 0:
                            raise AssertionError("expected empty workflows list")
                        if len(tools) != 0:
                            raise AssertionError("expected empty tools list")
                        return Agent(agent_config, workflows, tools, note)
                    """
                ).strip(),
                encoding="utf-8",
            )
            record = ResourceRecord(
                resource_type="agent",
                resource_id="verify-bot",
                source_tool="create-agent",
                materializer={
                    "kind": "python_function",
                    "module": "list_default_sdk.factory",
                    "name": "create_agent",
                },
                invoker={"kind": "python_method", "method": "invoke"},
                payload={"init_kwargs": {"agent_config": {"id": "verify-bot"}}},
                sdk={"source_dir": str(tmp)},
            )
            try:
                materialized = materialize_agent(record)
            finally:
                sys.modules.pop("list_default_sdk.factory", None)
                sys.modules.pop("list_default_sdk", None)

        agent = materialized["agent"]
        self.assertEqual(agent.config.id, "verify-bot")
        self.assertEqual(agent.workflows, [])
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.note)

    def test_materialize_agent_requires_catalog_dir_for_payload_ref(self) -> None:
        record = ResourceRecord(
            resource_type="agent",
            resource_id="spilled-bot",
            source_tool="create-agent",
            materializer={
                "kind": "python_function",
                "module": "typed_sdk.factory",
                "name": "create_agent",
            },
            invoker={"kind": "python_method", "method": "invoke"},
            payload={
                "kind": "payload_ref",
                "ref": "payloads/spilled-bot.json",
                "init_kwargs": {"agent_config": {"id": "spilled-bot"}},
            },
        )
        with self.assertRaises(ResourceCatalogError) as raised:
            materialize_agent(record)
        self.assertEqual(raised.exception.error_code, RESOURCE_PAYLOAD_REF_MISSING)

    def test_materialize_agent_resolves_spilled_payload_with_catalog_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            package = tmp / "payload_ref_sdk"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "factory.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass


                    @dataclass
                    class AgentConfig:
                        id: str
                        note: str


                    class Agent:
                        def __init__(self, config: AgentConfig):
                            self.config = config


                    def create_agent(agent_config: AgentConfig) -> Agent:
                        return Agent(agent_config)
                    """
                ).strip(),
                encoding="utf-8",
            )
            record = ResourceRecord(
                resource_type="agent",
                resource_id="spilled-bot",
                source_tool="create-agent",
                materializer={
                    "kind": "python_function",
                    "module": "payload_ref_sdk.factory",
                    "name": "create_agent",
                },
                invoker={"kind": "python_method", "method": "invoke"},
                payload={
                    "init_kwargs": {
                        "agent_config": {
                            "id": "spilled-bot",
                            "note": "from-spilled-payload",
                        }
                    },
                    "dsl": {"name": "spilled-bot", "blob": "x" * 70000},
                },
                sdk={"source_dir": str(tmp)},
            )
            catalog_dir = tmp / "catalog"
            catalog_dir.mkdir()
            spilled = spill_payload_if_needed(record, catalog_dir, force_ref=True)
            self.assertEqual(spilled.payload["kind"], "payload_ref")
            try:
                materialized = materialize_agent(spilled, catalog_dir=catalog_dir)
            finally:
                sys.modules.pop("payload_ref_sdk.factory", None)
                sys.modules.pop("payload_ref_sdk", None)

        agent = materialized["agent"]
        self.assertEqual(agent.config.id, "spilled-bot")
        self.assertEqual(agent.config.note, "from-spilled-payload")

    def test_materialize_resource_resolves_spilled_payload_with_catalog_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            package = tmp / "payload_ref_sdk"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "factory.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass


                    @dataclass
                    class AgentConfig:
                        id: str


                    class Agent:
                        def __init__(self, config: AgentConfig):
                            self.config = config


                    def create_agent(agent_config: AgentConfig) -> Agent:
                        return Agent(agent_config)
                    """
                ).strip(),
                encoding="utf-8",
            )
            record = ResourceRecord(
                resource_type="agent",
                resource_id="spilled-bot",
                source_tool="create-agent",
                materializer={
                    "kind": "python_function",
                    "module": "payload_ref_sdk.factory",
                    "name": "create_agent",
                },
                invoker={"kind": "python_method", "method": "invoke"},
                payload={
                    "init_kwargs": {"agent_config": {"id": "spilled-bot"}},
                    "dsl": {"name": "spilled-bot", "blob": "x" * 70000},
                },
                sdk={"source_dir": str(tmp)},
            )
            catalog_dir = tmp / "catalog"
            catalog_dir.mkdir()
            spilled = spill_payload_if_needed(record, catalog_dir, force_ref=True)
            self.assertEqual(spilled.payload["kind"], "payload_ref")
            # Verify the spill actually happened before exercising the
            # spilled-payload resolution path (mirrors the agent variant).
            self.assertEqual(spilled.payload["kind"], "payload_ref")
            try:
                materialized = materialize_resource(
                    spilled,
                    resource_type="agent",
                    catalog_dir=catalog_dir,
                )
            finally:
                sys.modules.pop("payload_ref_sdk.factory", None)
                sys.modules.pop("payload_ref_sdk", None)

        self.assertEqual(materialized["agent"].config.id, "spilled-bot")


if __name__ == "__main__":
    unittest.main()
