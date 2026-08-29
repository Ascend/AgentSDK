#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for stage7 no root shadow."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_ALLOWED_ROOT_DIRS = frozenset(
    {
        "src",
        "clawcodex_ext",
        "extensions",
        "upstream_sync",
        "tests",
        "scripts",
        "docs",
        "patches",
        "demos",
        "eval",
        "assets",
        "claude-code-wiki",
        "build",
        "dist",
        "clawcodex_dev.egg-info",
        "clawcodex_dev_mind.egg-info",
        "__pycache__",
        ".git",
        ".github",
        ".claude",
        ".idea",
        ".pytest_cache",
        ".atomcode",
        ".audit_temp",
        ".port_sessions",
    }
)

_KNOWN_LEGIT_ROOT_PY: frozenset[str] = frozenset()

_EAGER_IMPORT_RE = re.compile(r"^from\s+clawcodex_ext\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\s+import\b")

_FACADE_HEADER = "Facade \u2014"  # "Facade —" with em-dash


def _iter_root_python_files():
    """Test helper for iter root python files."""
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.suffix == ".py":
            yield p


def _iter_root_dirs():
    for p in sorted(ROOT.iterdir()):
        if p.is_dir():
            yield p


class TestNoRootLevelShadow:
    """Tests for TestNoRootLevelShadow."""

    def test_no_root_package_shadows_src(self) -> None:
        """Verify no root package shadows src."""
        violations: list[str] = []
        for d in _iter_root_dirs():
            if d.name in _ALLOWED_ROOT_DIRS:
                continue
            if not (d / "__init__.py").exists():
                continue
            if (ROOT / "src" / d.name).exists():
                violations.append(f"[package] /{d.name} (contains __init__.py, also exists as src/{d.name})")
        assert not violations, (
            f"Root-level package shadowing detected ({len(violations)} violation(s)):\n  - " + "\n  - ".join(violations)
        )

    def test_no_root_facade_module(self) -> None:
        """Verify no root facade module."""
        violations: list[str] = []
        for p in _iter_root_python_files():
            if p.name in _KNOWN_LEGIT_ROOT_PY:
                continue
            try:
                first_line = p.read_text(encoding="utf-8").splitlines()[0]
            except (IndexError, OSError):
                continue
            if first_line.startswith(_FACADE_HEADER):
                violations.append(f"[facade]   /{p.name} (starts with {first_line!r})")
        assert not violations, (
            f"Root-level facade module(s) detected ({len(violations)} violation(s)):\n  - " + "\n  - ".join(violations)
        )

    def test_no_root_eager_import(self) -> None:
        """Verify no root eager import."""
        violations: list[str] = []
        for p in _iter_root_python_files():
            if p.name in _KNOWN_LEGIT_ROOT_PY:
                continue
            try:
                source = p.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                tree = ast.parse(source, filename=str(p))
            except SyntaxError:
                continue
            for stmt in tree.body:
                if (
                    isinstance(stmt, ast.ImportFrom)
                    and stmt.module
                    and _EAGER_IMPORT_RE.match(f"from {stmt.module} import {ast.unparse(stmt)}")
                ):
                    violations.append(f"[eager]    /{p.name} (top-level `from {stmt.module} import ...`)")
                    break
        assert not violations, (
            f"Root-level eager re-import detected ({len(violations)} violation(s)):\n  - " + "\n  - ".join(violations)
        )

    def test_no_untracked_stale_dirs(self) -> None:
        """Verify no untracked stale dirs."""
        soft_violations: list[str] = []
        for name in ("agents", "build"):
            p = ROOT / name
            if p.exists() and p.is_dir():
                if name == "agents":
                    soft_violations.append(f"[stale]    /{name} (empty dir from refactor)")
                elif name == "build":
                    soft_violations.append(f"[stale]    /{name} (26M stale build output, in .gitignore)")
        assert not soft_violations, "Stale root-level directory/ies detected:\n  - " + "\n  - ".join(soft_violations)

    def test_known_special_case_litellm_adapter_documented(self) -> None:
        """Verify known special case litellm adapter documented."""
        shim = ROOT / "src" / "providers" / "_litellm_adapter.py"
        assert shim.exists(), (
            f"Expected Pattern D facade still present: {shim} (consumed by tests/provider/test_litellm_adapter.py)"
        )
        content = shim.read_text(encoding="utf-8")
        assert "from clawcodex_ext.providers._litellm_adapter import" in content, (
            f"{shim} should re-export from clawcodex_ext.providers._litellm_adapter "
            "(Phase K migration target — canonical location)"
        )
        assert "from extensions.providers_ext import" not in content, (
            f"{shim} unexpectedly re-exports from extensions.providers_ext "
            "— extensions/ is now a deprecated shim; src/ facade should "
            "point at clawcodex_ext.providers._litellm_adapter"
        )
        # The canonical implementation must exist in clawcodex_ext.
        canonical = ROOT / "clawcodex_ext" / "providers" / "_litellm_adapter.py"
        assert canonical.exists(), f"Expected canonical implementation at {canonical} after Phase K migration"
        # The deprecated extensions/ shim must still exist for backward compat.
        deprecated_shim = ROOT / "extensions" / "providers_ext" / "__init__.py"
        assert deprecated_shim.exists(), (
            f"Expected deprecated extensions shim at {deprecated_shim} for backward compatibility"
        )

    def test_known_legit_root_py_unchanged(self) -> None:
        """Verify known legit root py unchanged."""
        for name in _KNOWN_LEGIT_ROOT_PY:
            p = ROOT / name
            assert p.exists(), f"Expected root-level script missing: /{name}"
