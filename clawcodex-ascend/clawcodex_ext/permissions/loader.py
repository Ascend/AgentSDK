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

from clawcodex_ext.permissions.types import (
    PermissionRule,
    PermissionRuleSource,
    ToolPermissionContext,
)

from .rule_parser import permission_rule_value_from_string, permission_rule_value_to_string


def settings_to_rules(
    permissions_data: dict | None,
    source: PermissionRuleSource,
) -> list[PermissionRule]:
    if not permissions_data:
        return []

    rules: list[PermissionRule] = []
    for behavior in ("allow", "deny", "ask"):
        behavior_list = permissions_data.get(behavior, [])
        for rule_string in behavior_list:
            rules.append(
                PermissionRule(
                    source=source,
                    rule_behavior=behavior,  # type: ignore[arg-type]
                    rule_value=permission_rule_value_from_string(rule_string),
                )
            )
    return rules


def apply_rules_to_context(
    context: ToolPermissionContext,
    rules: list[PermissionRule],
) -> ToolPermissionContext:
    allow_rules = dict(context.always_allow_rules)
    deny_rules = dict(context.always_deny_rules)
    ask_rules = dict(context.always_ask_rules)

    for rule in rules:
        rule_string = permission_rule_value_to_string(rule.rule_value)
        if rule.rule_behavior == "allow":
            allow_rules.setdefault(rule.source, []).append(rule_string)
        elif rule.rule_behavior == "deny":
            deny_rules.setdefault(rule.source, []).append(rule_string)
        elif rule.rule_behavior == "ask":
            ask_rules.setdefault(rule.source, []).append(rule_string)

    return ToolPermissionContext(
        mode=context.mode,
        additional_working_directories=dict(context.additional_working_directories),
        always_allow_rules=allow_rules,
        always_deny_rules=deny_rules,
        always_ask_rules=ask_rules,
        is_bypass_permissions_mode_available=context.is_bypass_permissions_mode_available,
        should_avoid_permission_prompts=context.should_avoid_permission_prompts,
        await_automated_checks_before_dialog=context.await_automated_checks_before_dialog,
    )
