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

"""Tests for POS bundle manifest read/write."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pos = ROOT / "extensions" / "sop_converter"
_manifest = _load_module(
    "extensions.sop_converter.bundle_manifest",
    _pos / "bundle_manifest.py",
)
_sop = _load_module(
    "extensions.sop_converter.sop_prompts",
    _pos / "sop_prompts.py",
)

write_bundle_manifest = _manifest.write_bundle_manifest
read_bundle_manifest = _manifest.read_bundle_manifest
resolve_sdk_source_dir = _manifest.resolve_sdk_source_dir
BUNDLE_MANIFEST_NAME = _manifest.BUNDLE_MANIFEST_NAME
format_sdk_source_dir_block = _sop.format_sdk_source_dir_block
domain_agent_sop_body = _sop.domain_agent_sop_body
append_sop_overview_routing = _sop.append_sop_overview_routing


class TestBundleManifest(unittest.TestCase):
    def test_write_and_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            bundle_dir = ws / ".clawcodex" / "JiuwenAgent_tool_test"
            sdk_root = ws / "JiuwenAgent"
            sdk_root.mkdir()
            bundle_dir.mkdir(parents=True)

            path = write_bundle_manifest(
                bundle_dir,
                sdk_source_dir=sdk_root,
                bundle_id="JiuwenAgent_tool_test",
                sdk_requirements=("openai>=1", "pydantic>=2"),
                bundle_venv_dir=str(bundle_dir / ".venv"),
            )
            self.assertEqual(path.name, BUNDLE_MANIFEST_NAME)
            manifest = read_bundle_manifest(bundle_dir)
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual(manifest.bundle_id, "JiuwenAgent_tool_test")
            self.assertEqual(manifest.sdk_source_dir, sdk_root.resolve())
            self.assertEqual(manifest.sdk_requirements, ("openai>=1", "pydantic>=2"))
            self.assertEqual(manifest.bundle_venv_dir, str(bundle_dir / ".venv"))

    def test_resolve_from_workspace_clawcodex_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            sdk_root = ws / "JiuwenAgent"
            sdk_root.mkdir()
            bundle_dir = ws / ".clawcodex" / "JiuwenAgent_tool_test"
            bundle_dir.mkdir(parents=True)
            write_bundle_manifest(bundle_dir, sdk_source_dir=sdk_root)

            skills_only = ws / "skills" / "JiuwenAgent_tool_test"
            skills_only.mkdir(parents=True)
            resolved = resolve_sdk_source_dir(skills_only, workspace_root=ws)
            self.assertEqual(resolved, sdk_root.resolve())

    def test_manifest_json_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            sdk_root = ws / "JiuwenAgent"
            sdk_root.mkdir()
            bundle_dir = ws / ".clawcodex" / "JiuwenAgent_tool_test"
            bundle_dir.mkdir(parents=True)
            write_bundle_manifest(bundle_dir, sdk_source_dir=sdk_root)
            raw = (bundle_dir / BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8")
            self.assertIn("sdk_source_dir", raw)
            self.assertIn("JiuwenAgent_tool_test", raw)


class TestSopSdkSourcePrompt(unittest.TestCase):
    def test_format_sdk_source_dir_block(self) -> None:
        block = format_sdk_source_dir_block("/mnt/d/projects/JiuwenAgent")
        self.assertIn("JiuwenAgent", block)
        self.assertIn("勿", block)
        self.assertIn("_SOURCE_DIR", block)

    def test_domain_agent_includes_sdk_block(self) -> None:
        body = domain_agent_sop_body(
            agent_type="memory-agent",
            description="Memory APIs",
            skill_name="memory-skill",
            sdk_source_dir="/mnt/d/projects/JiuwenAgent",
        )
        self.assertIn("SDK 源码根", body)
        self.assertIn("JiuwenAgent/openjiuwen", body)

    def test_overview_routing_appends_sdk_block(self) -> None:
        body = append_sop_overview_routing("", sdk_source_dir="/mnt/d/projects/JiuwenAgent")
        self.assertIn("SOP 路由", body)
        self.assertIn("SDK 源码根", body)


if __name__ == "__main__":
    unittest.main()
