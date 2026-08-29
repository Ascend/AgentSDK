# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

from __future__ import annotations

import ast
from pathlib import Path

CLAWCODEX_ROOT = Path(__file__).resolve().parents[2]


def _tree(relative_path: str) -> ast.Module:
    source = (CLAWCODEX_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def _imported_names(relative_path: str, module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_tree(relative_path)):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _aliases(relative_path: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in _tree(relative_path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
            aliases[target.id] = node.value.id
    return aliases


def test_query_uses_public_renderer_boundary() -> None:
    query_imports = _imported_names(
        "clawcodex_ext/query/query.py",
        "clawcodex_ext.tool_system.renderers",
    )
    assert "emit_text_chunks" in query_imports
    assert "_emit_text_chunks" not in query_imports


def test_public_renderer_alias_remains_declared() -> None:
    expected = {
        "clawcodex_ext/tool_system/renderers.py": {
            "emit_text_chunks": "_emit_text_chunks",
        },
    }
    for relative_path, aliases in expected.items():
        declared = _aliases(relative_path)
        for public_name, private_name in aliases.items():
            assert declared.get(public_name) == private_name
