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

CONSUMERS = {
    "clawcodex_ext/agent/session.py": (
        "clawcodex_ext.services.session_storage",
        "locked_append",
        "_locked_append",
    ),
    "clawcodex_ext/command_system/bg_commands.py": (
        "clawcodex_ext.tasks.bg_session_manager",
        "read_tail",
        "_read_tail",
    ),
    "clawcodex_ext/community_radar/fetcher.py": (
        "clawcodex_ext.utils.git",
        "run_git",
        "_run_git",
    ),
}

ALIASES = {
    "clawcodex_ext/services/session_storage.py": ("locked_append", "_locked_append"),
    "clawcodex_ext/tasks/bg_session_manager.py": ("read_tail", "_read_tail"),
    "clawcodex_ext/utils/git.py": ("run_git", "_run_git"),
}


def _parse(relative_path: str) -> ast.Module:
    source = (CLAWCODEX_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def test_consumers_use_public_session_helper_boundaries() -> None:
    for relative_path, (module, public_name, private_name) in CONSUMERS.items():
        imported_names: set[str] = set()
        for node in ast.walk(_parse(relative_path)):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                imported_names.update(alias.name for alias in node.names)
        assert public_name in imported_names
        assert private_name not in imported_names


def test_session_helper_public_aliases_remain_declared() -> None:
    for relative_path, (public_name, private_name) in ALIASES.items():
        aliases: dict[str, str] = {}
        for node in _parse(relative_path).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                aliases[target.id] = node.value.id
        assert aliases.get(public_name) == private_name
