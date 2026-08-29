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

from __future__ import annotations

from clawcodex_ext.permissions.types import PermissionRuleValue

LEGACY_TOOL_NAME_ALIASES: dict[str, str] = {
    "Task": "Agent",
    "KillShell": "TaskStop",
    "AgentOutputTool": "TaskOutput",
    "BashOutputTool": "TaskOutput",
}


def normalize_legacy_tool_name(name: str) -> str:
    return LEGACY_TOOL_NAME_ALIASES.get(name, name)


def get_legacy_tool_names(canonical_name: str) -> list[str]:
    return [legacy for legacy, canonical in LEGACY_TOOL_NAME_ALIASES.items() if canonical == canonical_name]


def escape_rule_content(content: str) -> str:
    return content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def unescape_rule_content(content: str) -> str:
    return content.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")


def _find_first_unescaped_char(s: str, char: str) -> int:
    for i, c in enumerate(s):
        if c == char:
            backslash_count = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                return i
    return -1


def _find_last_unescaped_char(s: str, char: str) -> int:
    for i in range(len(s) - 1, -1, -1):
        if s[i] == char:
            backslash_count = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                return i
    return -1


def permission_rule_value_from_string(rule_string: str) -> PermissionRuleValue:
    open_paren = _find_first_unescaped_char(rule_string, "(")
    if open_paren == -1:
        return PermissionRuleValue(tool_name=normalize_legacy_tool_name(rule_string))

    close_paren = _find_last_unescaped_char(rule_string, ")")
    if close_paren == -1 or close_paren <= open_paren:
        return PermissionRuleValue(tool_name=normalize_legacy_tool_name(rule_string))

    if close_paren != len(rule_string) - 1:
        return PermissionRuleValue(tool_name=normalize_legacy_tool_name(rule_string))

    tool_name = rule_string[:open_paren]
    raw_content = rule_string[open_paren + 1 : close_paren]

    if not tool_name:
        return PermissionRuleValue(tool_name=normalize_legacy_tool_name(rule_string))

    if raw_content == "" or raw_content == "*":  # pylint: disable=consider-using-in
        return PermissionRuleValue(tool_name=normalize_legacy_tool_name(tool_name))

    rule_content = unescape_rule_content(raw_content)
    return PermissionRuleValue(
        tool_name=normalize_legacy_tool_name(tool_name),
        rule_content=rule_content,
    )


def permission_rule_value_to_string(rule_value: PermissionRuleValue) -> str:
    if not rule_value.rule_content:
        return rule_value.tool_name
    escaped_content = escape_rule_content(rule_value.rule_content)
    return f"{rule_value.tool_name}({escaped_content})"
