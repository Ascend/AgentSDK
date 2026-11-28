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

"""Startup constraints for the step-driven VERL one-step-off pipeline."""
from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


class OneStepOffStepBudgetRule(ValidationRule):
    rule_id = "linkage.one_step_off_step_budget"
    scopes = frozenset({"linkage"})

    def is_enabled(self, context):
        return context.work_mode == "one_step_off"

    def validate(self, context):
        issues = []

        def read(path):
            return get_config_value(context.train_config, path)

        def positive(value):
            return isinstance(value, int) and not isinstance(value, bool) and value > 0

        def add(path, value, expected, message, severity=Severity.ERROR, namespace="train_config"):
            issues.append(ValidationIssue(
                rule_id=self.rule_id, severity=severity, path=namespace + "." + path,
                value="<missing>" if value is MISSING else value, expected=expected,
                source=context.source_for(path, namespace), message=message,
                suggestion=expected,
            ))

        total = read("total_train_steps")
        if not positive(total):
            add("total_train_steps", total, "a positive non-boolean integer",
                "The fully-async trainer requires an explicit stopping step budget.")

        epochs = read("trainer.total_epochs")
        if not positive(epochs):
            add("trainer.total_epochs", epochs, "a positive non-boolean integer",
                "The configured epoch count is invalid.")
        else:
            add("trainer.total_epochs", epochs,
                "Use total_train_steps to control one-step-off termination; epochs need not equal steps.",
                "This one-step-off pipeline stops by steps, not trainer.total_epochs.", Severity.INFO)

        fields = [("extras.data_loader.train_iters", read("extras.data_loader.train_iters"), "train_config")]
        if context.rollout_config is not None:
            fields.append(("train_iters", get_config_value(context.rollout_config, "train_iters"), "rollout_config"))
        # Upstream permits None here; Aura overwrites optimizer budgets with total_train_steps.
        trainer_steps = read("trainer.total_training_steps")
        if trainer_steps is not MISSING and trainer_steps is not None:
            fields.append(("trainer.total_training_steps", trainer_steps, "train_config"))
        for path, value, namespace in fields:
            if not positive(value):
                add(path, value, "a positive non-boolean integer", "The step budget is invalid.", namespace=namespace)
            elif positive(total) and value != total:
                add(path, value, "equal total_train_steps (%s)" % total,
                    "Producer and trainer step budgets disagree; training may end early or wait for missing samples.",
                    namespace=namespace)

        rollout_steps = read("rollout.total_rollout_steps")
        if not positive(rollout_steps):
            add("rollout.total_rollout_steps", rollout_steps, "a positive non-boolean integer",
                "The declared rollout step count is invalid.")
        elif positive(total) and rollout_steps != total:
            add("rollout.total_rollout_steps", rollout_steps,
                "Prefer total_train_steps (%s); the actual rollout loop uses rollout_config.train_iters." % total,
                "The declared rollout steps differ from the effective budget; this field is not a loop limit here.",
                Severity.WARNING)
        return issues
