# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
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

from __future__ import annotations

import ast
from pathlib import Path

CLAWCODEX_ROOT = Path(__file__).resolve().parents[2]


def _parse(relative_path: str) -> ast.Module:
    source = (CLAWCODEX_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def test_package_initialization_uses_public_provider_boundary() -> None:
    imported_names: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(_parse("clawcodex_ext/__init__.py")):
        if isinstance(node, ast.ImportFrom) and node.module == "clawcodex_ext.providers":
            imported_names.update(alias.name for alias in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)

    assert "initialize_provider_extensions" in imported_names
    assert "_init_provider_extensions" not in imported_names
    assert "initialize_provider_extensions" in called_names


def test_public_provider_boundary_remains_declared() -> None:
    declared_names = {
        node.name
        for node in _parse("clawcodex_ext/providers/__init__.py").body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "initialize_provider_extensions" in declared_names
