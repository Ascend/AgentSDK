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
"""Mode compatibility and known unused settings in the built-in hybrid path."""
from collections.abc import Mapping

from omegaconf import OmegaConf

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


class ModeCompatibilityRule(ValidationRule):
    rule_id = "mode.compatibility"
    scopes = frozenset({"mode"})

    # These are consumed by the separated controller/rollout, not HybridTaskRunner.
    # Shared fields such as extras.traj_output_path and service names remain valid.
    UNUSED_TRAIN_PATHS = {
        "extras.data_loader.train_data_path": "data.train_files",
        "extras.data_loader.test_data_path": "data.val_files",
        "extras.data_loader.train_iters": "trainer.total_training_steps / trainer.total_epochs",
        "extras.validate_num_samples": "data.val_max_samples",
        "rollout.total_rollout_steps": "trainer.total_training_steps / trainer.total_epochs",
        "rollout.test_freq": "trainer.test_freq",
    }
    UNUSED_ROLLOUT_PATHS = (
        "train_tensor_parallel_size", "infer_tensor_parallel_size", "infer_expert_parallel_size",
        "train_iters", "use_on_policy", "enable_version_control", "hybrid_batch_num",
        "wait_available_weight_timeout",
    )

    def is_enabled(self, context):
        return context.work_mode in {"hybrid", "one_step_off", "fully_async"}

    @staticmethod
    def engine_issues(context):
        value = get_config_value(context.train_config, "actor_rollout_ref.hybrid_engine")
        if value is MISSING:
            actor = get_config_value(context.train_config, "actor_rollout_ref", default={})
            if not isinstance(actor, Mapping) or "hybrid_engine" not in actor.keys():
                return []  # Allow a native default only when no override exists.
            value = "<unresolved>"
        elif OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=False)
        expected = context.work_mode == "hybrid"
        if type(value) is bool and value == expected:
            return []
        path = "actor_rollout_ref.hybrid_engine"
        return [ValidationIssue(
            rule_id="mode.hybrid_engine", severity=Severity.ERROR, path="train_config." + path,
            value=value, expected=repr(expected), source=context.source_for(path),
            message=f"work_mode={context.work_mode} requires {path}={expected}.",
            suggestion=f"Use a matching training preset or set {path} to {expected}.",
        )]

    def validate(self, context):
        issues = self.engine_issues(context)
        if context.work_mode != "hybrid":
            return issues
        fields = [("train_config", path, replacement) for path, replacement in self.UNUSED_TRAIN_PATHS.items()]
        fields += [("rollout_config", path, "actor_rollout_ref.rollout or trainer settings")
                   for path in self.UNUSED_ROLLOUT_PATHS]
        for namespace, path, replacement in fields:
            config = context.train_config if namespace == "train_config" else context.rollout_config
            value = get_config_value(config, path)
            if value is MISSING or value is None or (isinstance(value, str) and not value.strip()):
                continue
            if context.source_for(path, namespace) == "verl_default":
                continue
            issues.append(ValidationIssue(
                rule_id="mode.hybrid_unused_parameter", severity=Severity.WARNING,
                path=namespace + "." + path,
                value=OmegaConf.to_container(value, resolve=False) if OmegaConf.is_config(value) else value,
                expected="not consumed by the built-in hybrid training path",
                source=context.source_for(path, namespace),
                message=f"{namespace}.{path} is a separated-mode setting and does not control hybrid training.",
                suggestion=f"Remove the unused override; configure {replacement} for hybrid instead.",
            ))
        return issues
