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

import math
from typing import Any

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


SUPPORTED_ADVANTAGE_ESTIMATORS = frozenset(
    {
        "gae",
        "gdpo",
        "grpo",
        "reinforce_with_advantage",
        "remax",
    }
)


class AuraValueRule(ValidationRule):
    """Base class for required Aura value constraints."""

    scopes = frozenset({"aura"})

    def __init__(self, rule_id: str, path: str, expected: str):
        self.rule_id = rule_id
        self.path = path
        self.expected = expected

    def _read(self, context: ValidationContext) -> Any:
        return get_config_value(context.train_config, self.path)

    def _issue(self, context: ValidationContext, value: Any) -> ValidationIssue:
        current_value = "<missing>" if value is MISSING else value
        return ValidationIssue(
            rule_id=self.rule_id,
            severity=Severity.ERROR,
            path=self.path,
            value=current_value,
            expected=self.expected,
            source=context.source_for(self.path),
            message=f"Aura parameter {self.path} does not satisfy its required constraint.",
            suggestion=f"Set {self.path} to {self.expected}.",
        )


class IntegerRangeRule(AuraValueRule):
    """Require a non-boolean integer above a configured lower bound."""

    def __init__(self, rule_id: str, path: str, minimum: int, inclusive: bool):
        operator = ">=" if inclusive else ">"
        super().__init__(rule_id, path, f"integer {operator} {minimum}")
        self.minimum = minimum
        self.inclusive = inclusive

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        valid_type = isinstance(value, int) and not isinstance(value, bool)
        valid_range = valid_type and (value >= self.minimum if self.inclusive else value > self.minimum)
        return [] if valid_range else [self._issue(context, value)]


class NumberIntervalRule(AuraValueRule):
    """Require a finite, non-boolean number inside a closed interval."""

    def __init__(self, rule_id: str, path: str, lower: float, upper: float):
        super().__init__(rule_id, path, f"finite number in [{lower}, {upper}]")
        self.lower = lower
        self.upper = upper

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        valid_range = valid_type and self.lower <= value <= self.upper and math.isfinite(value)
        return [] if valid_range else [self._issue(context, value)]


class BooleanRule(AuraValueRule):
    """Require an Aura parameter to be an explicit boolean."""

    def __init__(self, rule_id: str, path: str):
        super().__init__(rule_id, path, "boolean")

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        return [] if isinstance(value, bool) else [self._issue(context, value)]


class ChoiceRule(AuraValueRule):
    """Require a string value from an explicit supported set."""

    def __init__(self, rule_id: str, path: str, choices: frozenset[str]):
        expected = "one of {" + ", ".join(sorted(choices)) + "}"
        super().__init__(rule_id, path, expected)
        self.choices = choices

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        return [] if isinstance(value, str) and value in self.choices else [self._issue(context, value)]


def build_aura_rules() -> tuple[ValidationRule, ...]:
    """Build the complete first-party Aura parameter rule set."""
    return (
        IntegerRangeRule(
            "aura.async_training.trigger_parameter_sync_step",
            "async_training.trigger_parameter_sync_step",
            minimum=0,
            inclusive=False,
        ),
        NumberIntervalRule(
            "aura.async_training.staleness_threshold",
            "async_training.staleness_threshold",
            lower=0,
            upper=1,
        ),
        IntegerRangeRule(
            "aura.async_training.require_batches",
            "async_training.require_batches",
            minimum=0,
            inclusive=False,
        ),
        BooleanRule(
            "aura.async_training.partial_rollout",
            "async_training.partial_rollout",
        ),
        BooleanRule(
            "aura.async_training.use_trainer_do_validate",
            "async_training.use_trainer_do_validate",
        ),
        IntegerRangeRule(
            "aura.rollout.nnodes",
            "rollout.nnodes",
            minimum=0,
            inclusive=False,
        ),
        IntegerRangeRule(
            "aura.rollout.n_gpus_per_node",
            "rollout.n_gpus_per_node",
            minimum=0,
            inclusive=False,
        ),
        IntegerRangeRule(
            "aura.rollout.n",
            "rollout.n",
            minimum=0,
            inclusive=False,
        ),
        IntegerRangeRule(
            "aura.rollout.total_rollout_steps",
            "rollout.total_rollout_steps",
            minimum=0,
            inclusive=False,
        ),
        IntegerRangeRule(
            "aura.extras.delta",
            "extras.delta",
            minimum=0,
            inclusive=False,
        ),
        IntegerRangeRule(
            "aura.extras.update_weights_interval",
            "extras.update_weights_interval",
            minimum=0,
            inclusive=False,
        ),
        IntegerRangeRule(
            "aura.extras.validate_num_samples",
            "extras.validate_num_samples",
            minimum=0,
            inclusive=True,
        ),
        IntegerRangeRule(
            "aura.extras.max_queue_size",
            "extras.max_queue_size",
            minimum=0,
            inclusive=False,
        ),
        ChoiceRule(
            "aura.algorithm.adv_estimator",
            "algorithm.adv_estimator",
            SUPPORTED_ADVANTAGE_ESTIMATORS,
        ),
    )
