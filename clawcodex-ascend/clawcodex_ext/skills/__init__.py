#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Canonical public surface for :mod:`clawcodex_ext.skills`."""

from __future__ import annotations

import importlib
from typing import Any

_SYMBOLS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "clawcodex_ext.skills.model": ("Skill", "PromptSkill"),
    "clawcodex_ext.skills.bundled_skills": (
        "BundledSkillDefinition",
        "SkillValidationError",
        "register_bundled_skill",
        "get_bundled_skills",
        "get_bundled_skill_by_name",
        "clear_bundled_skills",
        "validate_skill",
        "validate_skill_definition",
        "skill_from_mcp_tool",
    ),
    "clawcodex_ext.skills.create": ("create_skill",),
    "clawcodex_ext.skills.loader": (
        "create_skill_command",
        "parse_skill_frontmatter_fields",
        "get_skill_dir_commands",
        "get_skills_path",
        "load_skills_from_skills_dir",
        "load_skills_from_dir",
        "discover_skill_dirs_for_paths",
        "add_skill_directories",
        "activate_conditional_skills_for_paths",
        "get_dynamic_skills",
        "get_conditional_skill_count",
        "clear_skill_caches",
        "clear_dynamic_skills",
        "clear_skill_registry",
        "get_all_skills",
        "get_registered_skill",
    ),
    "clawcodex_ext.skills.frontmatter": ("parse_frontmatter",),
    "clawcodex_ext.skills.argument_substitution": (
        "parse_arguments",
        "substitute_arguments",
    ),
    "clawcodex_ext.skills.mcp_skill_builders": (
        "register_mcp_skill_builders",
        "get_mcp_skill_builders",
    ),
    "clawcodex_ext.skills.runtime_substitution": (
        "render_skill_prompt",
        "prepend_base_dir_header",
        "substitute_skill_dir",
        "substitute_session_id",
        "find_shell_blocks",
        "has_shell_blocks",
        "format_shell_output",
        "format_shell_error",
    ),
    "clawcodex_ext.skills.bundled": ("init_bundled_skills",),
}
_SYMBOL_MODULES = {symbol: module_name for module_name, symbols in _SYMBOLS_BY_MODULE.items() for symbol in symbols}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _SYMBOL_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
