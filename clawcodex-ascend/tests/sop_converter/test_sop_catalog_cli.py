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

"""Tests for the diagnostic ``sop catalog`` CLI."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from extensions.sop_converter.core.agent_catalog_resolver import HOME_ROOT_ENV
from extensions.sop_converter.resource_catalog import (
    CatalogExecutionContext,
    ResourceCatalog,
    ResourceRecord,
    write_record,
)


def _tmp_path() -> Path:
    return Path(tempfile.mkdtemp(prefix="catalog_cli_test_"))


def _make_record(*, resource_id: str = "agent-1", name: str = "verify-bot") -> ResourceRecord:
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
            "dsl": {"name": name, "api_key": "sk-secret"},
            "model": "gpt-4o",
        },
        sdk={"source_dir": "/tmp/sdk"},
    )


class TestSopCatalogCli(unittest.TestCase):
    def setUp(self) -> None:
        self.home = _tmp_path()
        self.bundle = _tmp_path() / "mybundle"
        self.bundle.mkdir()
        self.env = {HOME_ROOT_ENV: str(self.home)}

    def _write_bundle_record(self, record: ResourceRecord | None = None) -> ResourceRecord:
        rec = record or _make_record()
        ctx = CatalogExecutionContext(bundle_path=self.bundle, bundle_id="mybundle")
        with patch.dict(os.environ, self.env, clear=False):
            write_record(rec, ctx)
        return rec

    def _write_dual(self, record: ResourceRecord | None = None) -> ResourceRecord:
        rec = record or _make_record(resource_id="dual-1")
        ctx = CatalogExecutionContext(
            bundle_path=self.bundle,
            bundle_id="mybundle",
            dual_write=True,
        )
        with patch.dict(os.environ, self.env, clear=False):
            write_record(rec, ctx)
        return rec

    def _run(self, argv: list[str], *, extra_env: dict[str, str] | None = None) -> tuple[int, str, str]:
        from extensions.sop_converter.catalog_cli import run_catalog_command

        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        stdout = StringIO()
        stderr = StringIO()
        with patch.dict(os.environ, env, clear=False):
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = run_catalog_command(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_list_includes_layer(self) -> None:
        self._write_bundle_record()
        code, out, _err = self._run(
            ["list", "--bundle", str(self.bundle), "--json"],
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["layer"], "bundle")
        self.assertEqual(payload[0]["resource_id"], "agent-1")

    def test_get_includes_layer_and_does_not_restore_secrets(self) -> None:
        self._write_bundle_record()
        code, out, _err = self._run(
            [
                "get",
                "--bundle",
                str(self.bundle),
                "--type",
                "agentconfig",
                "--id",
                "agent-1",
                "--json",
            ],
            extra_env={"CLAWCODEX_CORE_API_KEY": "sk-live-restored"},
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["layer"], "bundle")
        self.assertNotIn("sk-live-restored", out)
        self.assertNotIn("sk-secret", out)
        api_key = payload["payload"]["dsl"]["api_key"]
        self.assertNotEqual(api_key, "sk-live-restored")
        self.assertNotEqual(api_key, "sk-secret")

    def test_latest_includes_layer(self) -> None:
        self._write_bundle_record(_make_record(resource_id="older", name="a"))
        self._write_bundle_record(_make_record(resource_id="newer", name="b"))
        code, out, _err = self._run(
            ["latest", "--bundle", str(self.bundle), "--type", "agentconfig", "--json"],
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["layer"], "bundle")
        self.assertEqual(payload["resource_id"], "newer")

    def test_delete_removes_record(self) -> None:
        self._write_bundle_record()
        code, _out, _err = self._run(
            [
                "delete",
                "--bundle",
                str(self.bundle),
                "--type",
                "agentconfig",
                "--id",
                "agent-1",
                "--scope",
                "bundle",
                "--json",
            ],
        )
        self.assertEqual(code, 0)
        from extensions.sop_converter.resource_catalog import resolve_resource_catalog_path

        loc = resolve_resource_catalog_path(self.bundle, bundle_id="mybundle", scope="bundle")
        loaded = ResourceCatalog.load(loc.path)
        self.assertIsNone(loaded.get_stored("agentconfig", "agent-1"))

    def test_mark_failed_sets_status(self) -> None:
        self._write_bundle_record()
        code, out, _err = self._run(
            [
                "mark-failed",
                "--bundle",
                str(self.bundle),
                "--type",
                "agentconfig",
                "--id",
                "agent-1",
                "--scope",
                "bundle",
                "--reason",
                "broken",
                "--json",
            ],
        )
        self.assertEqual(code, 0)
        from extensions.sop_converter.resource_catalog import resolve_resource_catalog_path

        loc = resolve_resource_catalog_path(self.bundle, bundle_id="mybundle", scope="bundle")
        loaded = ResourceCatalog.load(loc.path)
        rec = loaded.get_stored("agentconfig", "agent-1")
        assert rec is not None
        self.assertEqual(rec.status, "failed")
        self.assertEqual(rec.metadata.get("failure_reason"), "broken")
        payload = json.loads(out)
        self.assertEqual(payload["status"], "failed")

    def test_scope_all_list_and_delete_dual_write(self) -> None:
        self._write_dual()
        code, out, _err = self._run(
            ["list", "--bundle", str(self.bundle), "--scope", "all", "--json"],
        )
        self.assertEqual(code, 0)
        rows = json.loads(out)
        layers = sorted({row["layer"] for row in rows})
        self.assertEqual(layers, ["bundle", "user"])

        code, _out, _err = self._run(
            [
                "delete",
                "--bundle",
                str(self.bundle),
                "--type",
                "agentconfig",
                "--id",
                "dual-1",
                "--scope",
                "all",
                "--json",
            ],
        )
        self.assertEqual(code, 0)

        code, out, _err = self._run(
            ["list", "--bundle", str(self.bundle), "--scope", "all", "--json"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [])

    def test_type_filter_uses_normalize_resource_type(self) -> None:
        self._write_bundle_record()
        code, out, _err = self._run(
            [
                "list",
                "--bundle",
                str(self.bundle),
                "--type",
                "Agent-Config",
                "--json",
            ],
        )
        self.assertEqual(code, 0)
        rows = json.loads(out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["resource_id"], "agent-1")

    def test_get_by_id_without_type(self) -> None:
        self._write_bundle_record(_make_record(resource_id="verify-bot"))
        code, out, err = self._run(
            [
                "get",
                "--bundle",
                str(self.bundle),
                "--scope",
                "bundle",
                "--id",
                "verify-bot",
                "--json",
            ],
        )
        self.assertEqual(code, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["resource_id"], "verify-bot")
        self.assertEqual(payload["layer"], "bundle")

    def test_get_type_agent_matches_fq_agent_family(self) -> None:
        self._write_bundle_record(
            ResourceRecord(
                resource_type="openjiuwencoresingleagentlegacyconfiglegacyreactagentconfig",
                resource_id="verify-bot",
                bundle_id="core",
                source_tool="openjiuwen-core-application-llm-agent-create-llm-agent",
                materializer={
                    "kind": "python_function",
                    "module": "fake",
                },
                invoker={"kind": "python_method", "method": "invoke", "input_param": "query"},
                payload={"kind": "inline", "dsl": {"id": "verify-bot", "api_key": "sk-secret"}},
            )
        )
        code, out, err = self._run(
            [
                "get",
                "--bundle",
                str(self.bundle),
                "--scope",
                "bundle",
                "--type",
                "agent",
                "--id",
                "verify-bot",
                "--json",
            ],
        )
        self.assertEqual(code, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["resource_id"], "verify-bot")
        self.assertIn("agent", payload["resource_type"])

    def test_get_by_id_ambiguous_requires_type(self) -> None:
        self._write_bundle_record(_make_record(resource_id="shared"))
        self._write_bundle_record(
            ResourceRecord(
                resource_type="DemoHandle",
                resource_id="shared",
                bundle_id="core",
                source_tool="demo-create",
                materializer={"kind": "python_class", "module": "fake", "class_name": "Demo"},
                invoker={"kind": "python_method", "method": "invoke", "input_param": "query"},
                payload={"kind": "inline", "note": "other"},
            )
        )
        code, _out, err = self._run(
            [
                "get",
                "--bundle",
                str(self.bundle),
                "--scope",
                "bundle",
                "--id",
                "shared",
                "--json",
            ],
        )
        self.assertEqual(code, 2)
        self.assertIn("ambiguous", err.lower())
        self.assertIn("--type", err.lower())

    def test_dispatch_from_sop_commands(self) -> None:
        from clawcodex_ext.cli.sop_cmd.commands import run_sop_command

        self._write_bundle_record()
        stdout = StringIO()
        with patch.dict(os.environ, self.env, clear=False):
            with patch("sys.stdout", stdout):
                code = run_sop_command(
                    ["catalog", "list", "--bundle", str(self.bundle), "--json"],
                )
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout.getvalue()))


if __name__ == "__main__":
    unittest.main()
