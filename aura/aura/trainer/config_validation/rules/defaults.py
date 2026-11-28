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

from collections.abc import Sequence
from typing import Any, Callable

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule
from aura.trainer.config_validation.sources import VERL_DEFAULT


NoticePredicate = Callable[[Any, ValidationContext], bool]


def _always_notice(value: Any, context: ValidationContext) -> bool:
    return True


def _zero_integer(value: Any, context: ValidationContext) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _math_or_code_task(value: Any, context: ValidationContext) -> bool:
    task = context.metadata.get("task_type", context.metadata.get("task", ""))
    if isinstance(task, Sequence) and not isinstance(task, (str, bytes, bytearray)):
        task = " ".join(str(item) for item in task)
    task_name = str(task).lower()
    return "math" in task_name or "code" in task_name


class VerlDefaultNoticeRule(ValidationRule):
    """Emit a non-blocking notice for a risky or tunable verl default."""

    scopes = frozenset({"defaults"})
    default_value_notice = True

    def __init__(
        self,
        rule_id: str,
        path: str,
        severity: Severity,
        recommendation: str,
        predicate: NoticePredicate = _always_notice,
    ):
        self.rule_id = rule_id
        self.path = path
        self.severity = severity
        self.recommendation = recommendation
        self.predicate = predicate

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        source = context.source_for(self.path)
        value = get_config_value(context.train_config, self.path)
        if source != VERL_DEFAULT or value is MISSING or not self.predicate(value, context):
            return []

        return [
            ValidationIssue(
                rule_id=self.rule_id,
                severity=self.severity,
                path=self.path,
                value=value,
                expected="an explicit value reviewed for the current Aura workload",
                source=source,
                message=f"Parameter {self.path} is inherited from verl defaults.",
                suggestion=self.recommendation,
            )
        ]


def build_default_notice_rules() -> tuple[ValidationRule, ...]:
    """Build warnings and tuning information for important verl defaults."""
    return (
        VerlDefaultNoticeRule(
            "defaults.trainer.save_freq",
            "trainer.save_freq",
            Severity.WARNING,
            "Set trainer.save_freq to a positive checkpoint interval if checkpoints are required.",
            _zero_integer,
        ),
        VerlDefaultNoticeRule(
            "defaults.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            Severity.WARNING,
            "Explicitly tune gpu_memory_utilization for the Ascend NPU model and topology.",
        ),
        VerlDefaultNoticeRule(
            "defaults.rollout.temperature",
            "actor_rollout_ref.rollout.temperature",
            Severity.WARNING,
            "Explicitly set rollout temperature for the math or code task.",
            _math_or_code_task,
        ),
        VerlDefaultNoticeRule(
            "defaults.actor.kl_loss_coef",
            "actor_rollout_ref.actor.kl_loss_coef",
            Severity.INFO,
            "Review kl_loss_coef for the selected advantage estimator and workload.",
        ),
        VerlDefaultNoticeRule(
            "defaults.actor.entropy_coeff",
            "actor_rollout_ref.actor.entropy_coeff",
            Severity.INFO,
            "Review entropy_coeff for the desired exploration behavior.",
        ),
        VerlDefaultNoticeRule(
            "defaults.actor.optim.lr",
            "actor_rollout_ref.actor.optim.lr",
            Severity.INFO,
            "Explicitly tune the actor learning rate for the model and batch size.",
        ),
        VerlDefaultNoticeRule(
            "defaults.data.max_prompt_length",
            "data.max_prompt_length",
            Severity.INFO,
            "Review max_prompt_length against the dataset and model context window.",
        ),
    )
