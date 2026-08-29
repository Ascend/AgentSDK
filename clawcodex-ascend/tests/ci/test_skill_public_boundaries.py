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
    "clawcodex_ext/skills/invocation.py": (
        "clawcodex_ext.hooks.config_manager",
        "parse_hook_config",
        "_parse_hook_config",
    ),
    "clawcodex_ext/skills/visibility.py": (
        "clawcodex_ext.context_system.prompt_assembly",
        "build_skill_section",
        "_build_skill_section",
    ),
    "clawcodex_ext/tool_system/tools/skill.py": (
        "clawcodex_ext.skills.invocation",
        "effective_skill_root",
        "_effective_skill_root",
    ),
}

ALIASES = {
    "clawcodex_ext/hooks/config_manager.py": ("parse_hook_config", "_parse_hook_config"),
    "clawcodex_ext/context_system/prompt_assembly.py": (
        "build_skill_section",
        "_build_skill_section",
    ),
    "clawcodex_ext/skills/invocation.py": ("effective_skill_root", "_effective_skill_root"),
}


def _parse(relative_path: str) -> ast.Module:
    source = (CLAWCODEX_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def test_skill_consumers_use_public_cross_module_boundaries() -> None:
    for relative_path, (module, public_name, private_name) in CONSUMERS.items():
        imported_names: set[str] = set()
        for node in ast.walk(_parse(relative_path)):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                imported_names.update(alias.name for alias in node.names)
        assert public_name in imported_names
        assert private_name not in imported_names


def test_skill_public_aliases_remain_declared() -> None:
    for relative_path, (public_name, private_name) in ALIASES.items():
        aliases: dict[str, str] = {}
        for node in _parse(relative_path).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                aliases[target.id] = node.value.id
        assert aliases.get(public_name) == private_name
