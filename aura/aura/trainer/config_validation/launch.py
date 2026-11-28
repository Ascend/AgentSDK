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
"""Validate launch routing before starting Ray or external inference services."""
import os
from collections.abc import Sequence
from dataclasses import replace

from omegaconf import OmegaConf

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import ConfigValidationError, Severity, ValidationIssue, ValidationReport
from aura.trainer.config_validation.rules.modes import ModeCompatibilityRule


WORK_MODES = frozenset({"hybrid", "one_step_off", "fully_async", "dummy_train"})
BACKEND_MODES = {
    "verl": WORK_MODES,
    "mindspeed_rl": WORK_MODES - {"fully_async"},
}


def validate_launch_config(config, launcher_work_mode=None, train_engine=None):
    """Reject contradictory routing; a direct Python launch may omit WORK_MODE."""
    if launcher_work_mode is None:
        launcher_work_mode = os.environ.get("WORK_MODE")
    if train_engine is None:
        train_engine = os.environ.get("RL_TRAIN_BACKEND")
    issues = []

    def error(path, value, expected, message):
        issues.append(ValidationIssue(
            rule_id="mode.launch_consistency", severity=Severity.ERROR, path=path,
            value="<missing>" if value is MISSING else value, expected=expected,
            message=message, suggestion="Make base.conf and all selected train_instances use compatible work modes.",
        ))

    if launcher_work_mode is not None and (
        not isinstance(launcher_work_mode, str) or launcher_work_mode not in WORK_MODES - {"dummy_train"}
    ):
        error("WORK_MODE", launcher_work_mode, "hybrid, one_step_off or fully_async", "Invalid launcher work mode.")
    instances = get_config_value(config, "train_instances", default=())
    if OmegaConf.is_list(instances):
        instances = list(instances)
    if not isinstance(instances, Sequence) or isinstance(instances, (str, bytes)):
        error("train_instances", instances, "a list of training instances", "Invalid training instance list.")
        instances = ()
    for index, instance in enumerate(instances):
        prefix = f"train_instances.{index}.executor_kwargs."
        kwargs = get_config_value(instance, "executor_kwargs", default={})
        actual_engine = get_config_value(kwargs, "train_engine")
        if train_engine is not None and actual_engine != train_engine:
            error(prefix + "train_engine", actual_engine, f"match launcher backend={train_engine}",
                  "Launcher backend and YAML train_engine disagree.")
        mode = get_config_value(kwargs, "work_mode")
        if not isinstance(mode, str) or mode not in WORK_MODES:
            error(prefix + "work_mode", mode, ", ".join(sorted(WORK_MODES)), "Invalid YAML training work mode.")
            continue
        if isinstance(actual_engine, str) and actual_engine in BACKEND_MODES and mode not in BACKEND_MODES[actual_engine]:
            error(prefix + "work_mode", mode, ", ".join(sorted(BACKEND_MODES[actual_engine])),
                  f"The {actual_engine} backend does not register this work mode.")
        # dummy_train is the registered controller-only path for separated rollout tests.
        compatible = mode == launcher_work_mode or (launcher_work_mode == "one_step_off" and mode == "dummy_train")
        if launcher_work_mode is not None and not compatible:
            error(prefix + "work_mode", {"WORK_MODE": launcher_work_mode, "yaml": mode},
                  f"match launcher WORK_MODE={launcher_work_mode}",
                  "Launcher and training YAML disagree; they would select different inference/training paths.")
        if actual_engine == "verl" and mode != "dummy_train":
            context = ValidationContext(train_config=get_config_value(kwargs, "train_config", default={}), work_mode=mode)
            for issue in ModeCompatibilityRule.engine_issues(context):
                issues.append(replace(issue, path=prefix + issue.path))
    report = ValidationReport(issues=tuple(issues), scopes=("mode",))
    if report.has_errors:
        raise ConfigValidationError(report)
    return report
