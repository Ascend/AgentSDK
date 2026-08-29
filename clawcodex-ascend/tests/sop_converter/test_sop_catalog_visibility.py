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

"""Tests for SOP catalog visibility (A prompt block, B tool, D read allow)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from extensions.capabilities.agent_definition_protocol import AgentToolConstants
from extensions.sop_converter.core.agent_catalog_resolver import HOME_ROOT_ENV
from extensions.sop_converter.catalog_cli import _cmd_list
from extensions.sop_converter.resource_catalog import (
    CatalogExecutionContext,
    ResourceRecord,
    format_resource_catalog_locations_block,
    write_record,
)
from extensions.sop_converter.runtime.catalog_tools import (
    ResourceCatalogTool,
    register_resource_catalog_tool,
)
from extensions.sop_converter.core.sop_prompts import append_sop_overview_routing
from clawcodex_ext.permissions.filesystem import (
    check_readable_catalog_path,
    check_read_permission_for_tool,
)
from clawcodex_ext.permissions.types import PermissionAllowDecision
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.registry import ToolRegistry


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="sop_catalog_vis_"))


def _make_record(*, resource_id: str = "verify-bot") -> ResourceRecord:
    return ResourceRecord(
        resource_type="AgentConfig",
        resource_id=resource_id,
        bundle_id="core",
        source_tool="agentbuilder-build-agent",
        materializer={
            "kind": "python_class",
            "module": "fake_sdk.agent",
            "class_name": "DemoAgent",
        },
        invoker={"kind": "python_method", "method": "invoke", "input_param": "query"},
        payload={
            "kind": "inline",
            "dsl": {"name": resource_id, "api_key": "sk-secret"},
            "init_kwargs": {},
        },
        secrets={"policy": "env_refs_only", "env_refs": []},
    )


class FormatCatalogBlockTests(unittest.TestCase):
    def test_block_has_template_and_b_read_guidance(self) -> None:
        home = _tmp()
        bundle = _tmp() / "JiuwenAgent_v7.24"
        bundle.mkdir()
        (bundle / ".clawcodex").mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            text = format_resource_catalog_locations_block(bundle)
        self.assertIn("sessions/<session_id>/sop-resources.json", text)
        self.assertIn("resource-catalog", text)
        self.assertIn("resource-catalog.json", text)
        self.assertIn("no Grep", text)
        self.assertNotIn("If you must Read/Grep", text)

    def test_overview_routing_includes_catalog_block(self) -> None:
        home = _tmp()
        bundle = _tmp() / "bundle"
        bundle.mkdir()
        (bundle / ".clawcodex").mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            out = append_sop_overview_routing("base", bundle_path=bundle)
        self.assertIn("Resource catalogs", out)
        self.assertIn("resource-catalog", out)


class ResourceCatalogToolTests(unittest.TestCase):
    def test_list_redacts_and_finds_verify_bot(self) -> None:
        home = _tmp()
        bundle = _tmp() / "core"
        bundle.mkdir()
        cat_dir = bundle / ".clawcodex"
        cat_dir.mkdir()
        ctx = CatalogExecutionContext(bundle_path=bundle, bundle_id="core", session_id="", dual_write=False)
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            write_record(_make_record(), ctx)
            rows = _cmd_list(ctx, scope="bundle", resource_type="")
        self.assertTrue(any(r.get("resource_id") == "verify-bot" for r in rows))
        blob = json.dumps(rows)
        self.assertNotIn("sk-secret", blob)

    def test_tool_requires_bundle(self) -> None:
        context = ToolContext(workspace_root=_tmp())
        context.bundle_context = None
        result = ResourceCatalogTool.call({"action": "list"}, context)
        self.assertTrue(result.is_error)
        self.assertIn("sop_bundle_required", str(result.output))

    def test_tool_get_by_id(self) -> None:
        home = _tmp()
        bundle = _tmp() / "core"
        bundle.mkdir()
        (bundle / ".clawcodex").mkdir()
        ctx_env = CatalogExecutionContext(bundle_path=bundle, bundle_id="core", session_id="", dual_write=False)
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            write_record(_make_record(), ctx_env)

        bundle_ctx = MagicMock()
        bundle_ctx.bundle_path = bundle
        bundle_ctx.bundle_name = "core"
        context = ToolContext(workspace_root=bundle)
        context.bundle_context = bundle_ctx
        context.session_id = ""

        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            result = ResourceCatalogTool.call(
                {"action": "get", "resource_id": "verify-bot"},
                context,
            )
        self.assertFalse(result.is_error)
        self.assertIn("verify-bot", str(result.output))
        self.assertNotIn("sk-secret", str(result.output))

    def test_register_idempotent(self) -> None:
        reg = ToolRegistry()
        register_resource_catalog_tool(reg)
        register_resource_catalog_tool(reg)
        self.assertIsNotNone(reg.get("resource-catalog"))

    def test_allowlists_include_tool(self) -> None:
        self.assertIn("resource-catalog", AgentToolConstants.POS_PROXY_BASE_TOOLS)
        self.assertIn("resource-catalog", AgentToolConstants.POS_SOP_DOMAIN_AGENT_TOOLS)
        from clawcodex_ext.permissions.check import NO_PERMISSION_TOOLS

        self.assertIn("resource-catalog", NO_PERMISSION_TOOLS)


class CatalogReadAllowTests(unittest.TestCase):
    def test_allows_catalog_basenames_under_home(self) -> None:
        home = _tmp()
        path = home / "sop-resources" / "core" / "catalog.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        sess = home / "sessions" / "abc" / "sop-resources.json"
        sess.parent.mkdir(parents=True)
        sess.write_text("{}", encoding="utf-8")
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            self.assertTrue(check_readable_catalog_path(str(path)))
            self.assertTrue(check_readable_catalog_path(str(sess)))

    def test_denies_transcript_and_config(self) -> None:
        home = _tmp()
        transcript = home / "sessions" / "abc" / "transcript.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("x", encoding="utf-8")
        config = home / "config.json"
        config.write_text("{}", encoding="utf-8")
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            self.assertFalse(check_readable_catalog_path(str(transcript)))
            self.assertFalse(check_readable_catalog_path(str(config)))
            result = check_read_permission_for_tool(str(config), None)
            self.assertFalse(isinstance(result, PermissionAllowDecision))

    def test_allows_via_check_read_permission_for_tool(self) -> None:
        home = _tmp()
        path = home / "sessions" / "s1" / "sop-resources.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        context = MagicMock()
        context.allowed_roots.return_value = []
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(home)}):
            with patch(
                "clawcodex_ext.permissions.filesystem.check_readable_internal_path",
                return_value=False,
            ):
                result = check_read_permission_for_tool(str(path), context)
        self.assertIsInstance(result, PermissionAllowDecision)


if __name__ == "__main__":
    unittest.main()
