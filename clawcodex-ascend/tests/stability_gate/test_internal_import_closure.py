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

"""Ensure production modules do not statically import absent internal modules."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INTERNAL_ROOTS = ("clawcodex_ext", "extensions", "src", "telemetry")
SKIP_PARTS = frozenset({".venv", "__pycache__", "examples", "patches", "tests"})


def _module_exists(module: str) -> bool:
    path = ROOT.joinpath(*module.split("."))
    return path.with_suffix(".py").is_file() or path.is_dir()


def _module_declares_name(module: str, name: str) -> bool:
    """Return whether a package initializer statically defines ``name``."""
    path = ROOT.joinpath(*module.split("."))
    source_path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    if not source_path.is_file():
        return False

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound_name = alias.asname or alias.name.rsplit(".", 1)[-1]
                if bound_name == name:
                    return True
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return True
    return False


def _resolve_relative(current: str, level: int, imported: str | None) -> str:
    package = current.split(".")[:-1]
    prefix = package[: max(len(package) - level + 1, 0)]
    if imported:
        prefix.extend(imported.split("."))
    return ".".join(prefix)


def _production_files() -> list[Path]:
    files: list[Path] = []
    for package in INTERNAL_ROOTS:
        package_root = ROOT / package
        for path in package_root.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
                continue
            files.append(path)
    return sorted(files)


def test_all_static_internal_imports_resolve() -> None:
    missing: list[str] = []
    for path in _production_files():
        relative = path.relative_to(ROOT)
        current = ".".join(relative.with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            imported = ""
            if isinstance(node, ast.ImportFrom):
                imported = _resolve_relative(current, node.level, node.module) if node.level else node.module or ""
                if node.level and node.module is None:
                    for alias in node.names:
                        candidate = f"{imported}.{alias.name}"
                        if not _module_exists(candidate) and not _module_declares_name(imported, alias.name):
                            missing.append(f"{relative}:{node.lineno}: {candidate}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(INTERNAL_ROOTS) and not _module_exists(alias.name):
                        missing.append(f"{relative}:{node.lineno}: {alias.name}")
                continue
            else:
                continue
            if imported.startswith(INTERNAL_ROOTS) and not _module_exists(imported):
                missing.append(f"{relative}:{node.lineno}: {imported}")

    assert not missing, "Missing internal import targets:\n" + "\n".join(missing)
