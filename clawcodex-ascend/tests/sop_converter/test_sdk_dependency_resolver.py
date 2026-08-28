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

"""Tests for SDK dependency discovery used by ``sop convert``."""

from __future__ import annotations

from pathlib import Path

from extensions.sop_converter.core.sdk_dependency_resolver import resolve_sdk_dependencies


def test_resolves_pyproject_dependencies_and_filters_self(tmp_path: Path) -> None:
    sdk = tmp_path / "demo-sdk"
    sdk.mkdir()
    (sdk / "pyproject.toml").write_text(
        """
[project]
name = "demo-sdk"
dependencies = [
  "demo-sdk",
  "openai>=1.0",
  "pydantic>=2; python_version >= '3.11'",
]
""",
        encoding="utf-8",
    )

    spec = resolve_sdk_dependencies(sdk)

    assert spec.source == "pyproject.toml"
    assert spec.requirements == (
        "openai>=1.0",
        "pydantic>=2; python_version >= '3.11'",
    )


def test_requirements_txt_fallback_skips_comments_and_options(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "requirements.txt").write_text(
        """
# runtime deps
requests>=2  # HTTP client
-r dev-requirements.txt
--find-links ./wheels
openai==1.2.3
""",
        encoding="utf-8",
    )

    spec = resolve_sdk_dependencies(sdk)

    assert spec.source == "requirements.txt"
    assert spec.requirements == ("requests>=2", "openai==1.2.3")


def test_empty_when_no_dependency_files(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()

    spec = resolve_sdk_dependencies(sdk)

    assert spec.source == "empty"
    assert spec.requirements == ()
