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

from copy import deepcopy

import pytest

from aura.trainer.config_validation import (
    Severity,
    ValidationContext,
    ValidationOptions,
)
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.defaults import build_default_notice_rules
from aura.trainer.config_validation.sources import build_source_map


class ConfigValidator:
    def __init__(self, registry):
        self.registry = registry

    def validate(self, context, scopes=None):
        return ValidationReport(
            issues=tuple(issue for rule in self.registry.get_rules(scopes)
                         if (context.options.show_default_warnings or not rule.default_value_notice)
                         and rule.is_enabled(context) for issue in rule.validate(context)),
            sources=context.source_map,
        )

    def validate_or_raise(self, context, scopes=None):
        report = self.validate(context, scopes)
        if report.has_errors:
            from aura.trainer.config_validation.models import ConfigValidationError
            raise ConfigValidationError(report)
        return report


def build_default_registry():
    registry = RuleRegistry()
    registry.register_many(build_default_notice_rules())
    return registry

def verl_default_config():
    return {
        "trainer": {"save_freq": 0},
        "actor_rollout_ref": {
            "rollout": {
                "gpu_memory_utilization": 0.6,
                "temperature": 1.0,
            },
            "actor": {
                "kl_loss_coef": 0.001,
                "entropy_coeff": 0.0,
                "optim": {"lr": 1e-6},
            },
        },
        "data": {"max_prompt_length": 4096},
    }


def validate(config=None, source_map=None, metadata=None, show_default_warnings=True):
    config = config if config is not None else verl_default_config()
    context = ValidationContext(
        train_config=config,
        source_map=source_map if source_map is not None else build_source_map(verl_default=config),
        metadata=metadata or {},
        options=ValidationOptions(show_default_warnings=show_default_warnings),
    )
    return ConfigValidator(build_default_registry()).validate(context, scopes={"defaults"})


def test_default_registry_contains_every_default_notice_rule():
    rules = build_default_registry().get_rules({"defaults"})

    assert len(rules) == 7
    assert all(rule.default_value_notice for rule in rules)


def test_math_workload_reports_risky_warnings_and_tuning_information():
    report = validate(metadata={"task_type": "math"})

    assert report.error_count == 0
    assert report.warning_count == 3
    assert report.info_count == 4
    assert {issue.rule_id for issue in report.issues if issue.severity is Severity.WARNING} == {
        "defaults.rollout.gpu_memory_utilization",
        "defaults.rollout.temperature",
        "defaults.trainer.save_freq",
    }


def test_temperature_notice_only_applies_to_math_or_code_tasks():
    report = validate(metadata={"task_type": "general_chat"})

    assert "defaults.rollout.temperature" not in {issue.rule_id for issue in report.issues}
    assert report.warning_count == 2


def test_positive_default_save_frequency_does_not_warn():
    config = verl_default_config()
    config["trainer"]["save_freq"] = 100

    report = validate(config=config)

    assert "defaults.trainer.save_freq" not in {issue.rule_id for issue in report.issues}


def test_user_explicit_value_is_not_reported_even_when_equal_to_default():
    config = verl_default_config()
    sources = build_source_map(
        verl_default=config,
        user_explicit={"trainer": {"save_freq": 0}},
    )

    report = validate(config=config, source_map=sources)

    assert "defaults.trainer.save_freq" not in {issue.rule_id for issue in report.issues}


def test_aura_override_is_not_reported_as_a_verl_default():
    config = verl_default_config()
    sources = build_source_map(
        verl_default=config,
        aura_default={"actor_rollout_ref": {"rollout": {"gpu_memory_utilization": 0.6}}},
    )

    report = validate(config=config, source_map=sources)

    assert "defaults.rollout.gpu_memory_utilization" not in {
        issue.rule_id for issue in report.issues
    }


def test_default_notices_can_be_disabled_together():
    report = validate(
        metadata={"task_type": "math"},
        show_default_warnings=False,
    )

    assert report.issues == ()


def test_missing_default_value_does_not_create_a_notice():
    config = deepcopy(verl_default_config())
    del config["actor_rollout_ref"]["actor"]["kl_loss_coef"]
    sources = build_source_map(verl_default=verl_default_config())

    report = validate(config=config, source_map=sources)

    assert "defaults.actor.kl_loss_coef" not in {issue.rule_id for issue in report.issues}


def test_report_exports_parameter_sources_and_notice_source():
    report = validate()
    payload = report.to_dict()

    assert payload["sources"]["train_config.trainer.save_freq"] == "verl_default"
    save_notice = next(
        issue for issue in payload["issues"] if issue["rule_id"] == "defaults.trainer.save_freq"
    )
    assert save_notice["source"] == "verl_default"


def test_default_rules_are_excluded_from_unrelated_scopes():
    rules = build_default_registry().get_rules({"linkage"})

    assert all("defaults" not in rule.scopes for rule in rules)
