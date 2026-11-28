#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

from abc import ABC, abstractmethod
from collections.abc import Iterable

from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import ValidationIssue


class ValidationRule(ABC):
    """Base class for a configuration validation rule."""

    rule_id: str = ""
    scopes: frozenset[str] = frozenset({"common"})
    io_bound: bool = False
    default_value_notice: bool = False

    def is_enabled(self, context: ValidationContext) -> bool:
        """Return whether this rule applies to the supplied context."""
        return True

    @abstractmethod
    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Return every issue found by this rule without mutating context."""


class RuleRegistry:
    """Ordered registry of uniquely identified validation rules."""

    def __init__(self) -> None:
        self._rules: dict[str, ValidationRule] = {}

    def register(self, rule: ValidationRule) -> None:
        """Register one rule and reject empty or duplicate identifiers."""
        if not isinstance(rule, ValidationRule):
            raise TypeError("rule must be an instance of ValidationRule")
        if not isinstance(rule.rule_id, str) or not rule.rule_id.strip():
            raise ValueError("rule_id must be a non-empty string")
        if not isinstance(rule.scopes, frozenset) or not rule.scopes:
            raise ValueError(f"Rule {rule.rule_id} must declare at least one scope")
        if any(not isinstance(scope, str) or not scope.strip() for scope in rule.scopes):
            raise ValueError(f"Rule {rule.rule_id} contains an invalid scope")
        if rule.rule_id in self._rules:
            raise ValueError(f"Duplicate validation rule: {rule.rule_id}")
        self._rules[rule.rule_id] = rule

    def register_many(self, rules: Iterable[ValidationRule]) -> None:
        """Register multiple rules in iteration order."""
        for rule in rules:
            self.register(rule)

    def get_rules(self, scopes: set[str] | None = None) -> tuple[ValidationRule, ...]:
        """Return registered rules filtered by at least one matching scope."""
        if scopes is None:
            return tuple(self._rules.values())
        return tuple(rule for rule in self._rules.values() if rule.scopes & scopes)

    def __len__(self) -> int:
        return len(self._rules)
