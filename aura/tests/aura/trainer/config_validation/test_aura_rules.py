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

from aura.trainer.config_validation import ValidationContext
from aura.trainer.config_validation.rules.aura import SUPPORTED_ADVANTAGE_ESTIMATORS
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.aura import build_aura_rules


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
    registry.register_many(build_aura_rules())
    return registry

INTEGER_RULE_CASES = (
    ("async_training.trigger_parameter_sync_step", "aura.async_training.trigger_parameter_sync_step", 0),
    ("async_training.require_batches", "aura.async_training.require_batches", 0),
    ("rollout.nnodes", "aura.rollout.nnodes", 0),
    ("rollout.n_gpus_per_node", "aura.rollout.n_gpus_per_node", 0),
    ("rollout.n", "aura.rollout.n", 0),
    ("rollout.total_rollout_steps", "aura.rollout.total_rollout_steps", 0),
    ("extras.delta", "aura.extras.delta", 0),
    ("extras.update_weights_interval", "aura.extras.update_weights_interval", 0),
    ("extras.validate_num_samples", "aura.extras.validate_num_samples", -1),
    ("extras.max_queue_size", "aura.extras.max_queue_size", 0),
)

OTHER_REQUIRED_RULE_CASES = (
    ("async_training.staleness_threshold", "aura.async_training.staleness_threshold"),
    ("async_training.partial_rollout", "aura.async_training.partial_rollout"),
    ("async_training.use_trainer_do_validate", "aura.async_training.use_trainer_do_validate"),
    ("algorithm.adv_estimator", "aura.algorithm.adv_estimator"),
)


def valid_config():
    return {
        "async_training": {
            "trigger_parameter_sync_step": 1,
            "staleness_threshold": 0.5,
            "require_batches": 1,
            "partial_rollout": False,
            "use_trainer_do_validate": False,
        },
        "rollout": {
            "nnodes": 2,
            "n_gpus_per_node": 8,
            "n": 8,
            "total_rollout_steps": 100,
        },
        "extras": {
            "delta": 2,
            "update_weights_interval": 1,
            "validate_num_samples": 0,
            "max_queue_size": 256,
        },
        "algorithm": {"adv_estimator": "grpo"},
    }


def set_path(config, path, value):
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def remove_path(config, path):
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        current = current[part]
    del current[parts[-1]]


def validate(config, source_map=None):
    context = ValidationContext(train_config=config, source_map=source_map or {})
    return ConfigValidator(build_default_registry()).validate(context, scopes={"aura"})


def test_default_registry_contains_every_aura_rule():
    registry = build_default_registry()

    assert len(registry.get_rules({"aura"})) == 14


def test_valid_aura_configuration_passes_every_rule():
    report = validate(valid_config())

    assert report.valid is True
    assert report.issues == ()


@pytest.mark.parametrize("path,rule_id,invalid_value", INTEGER_RULE_CASES)
def test_integer_rules_reject_out_of_range_values(path, rule_id, invalid_value):
    config = valid_config()
    set_path(config, path, invalid_value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert report.issues[0].path == path
    assert report.issues[0].value == invalid_value


@pytest.mark.parametrize("path,rule_id,_", INTEGER_RULE_CASES)
@pytest.mark.parametrize("invalid_value", [True, 1.0, "1", None])
def test_integer_rules_reject_invalid_types(path, rule_id, _, invalid_value):
    config = valid_config()
    set_path(config, path, invalid_value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert "integer" in report.issues[0].expected


@pytest.mark.parametrize("path,rule_id,_", INTEGER_RULE_CASES)
def test_integer_rules_reject_missing_values(path, rule_id, _):
    config = valid_config()
    remove_path(config, path)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert report.issues[0].value == "<missing>"


@pytest.mark.parametrize("path,rule_id", OTHER_REQUIRED_RULE_CASES)
def test_other_required_rules_reject_missing_values(path, rule_id):
    config = valid_config()
    remove_path(config, path)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert report.issues[0].value == "<missing>"


@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0])
def test_staleness_threshold_accepts_closed_interval_boundaries(value):
    config = valid_config()
    set_path(config, "async_training.staleness_threshold", value)

    assert validate(config).valid is True


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan"), True, "0.5", None])
def test_staleness_threshold_rejects_invalid_values(value):
    config = valid_config()
    set_path(config, "async_training.staleness_threshold", value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == ["aura.async_training.staleness_threshold"]


@pytest.mark.parametrize(
    "path,rule_id",
    [
        ("async_training.partial_rollout", "aura.async_training.partial_rollout"),
        ("async_training.use_trainer_do_validate", "aura.async_training.use_trainer_do_validate"),
    ],
)
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_boolean_rules_reject_non_boolean_values(path, rule_id, value):
    config = valid_config()
    set_path(config, path, value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]


@pytest.mark.parametrize("value", sorted(SUPPORTED_ADVANTAGE_ESTIMATORS))
def test_advantage_estimator_accepts_supported_values(value):
    config = valid_config()
    set_path(config, "algorithm.adv_estimator", value)

    assert validate(config).valid is True


@pytest.mark.parametrize("value", ["unsupported", "GRPO", 1, None])
def test_advantage_estimator_rejects_unsupported_values(value):
    config = valid_config()
    set_path(config, "algorithm.adv_estimator", value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == ["aura.algorithm.adv_estimator"]


def test_issue_contains_parameter_source_and_repair_information():
    config = valid_config()
    set_path(config, "extras.delta", 0)
    report = validate(
        config,
        source_map={"train_config.extras.delta": "user_explicit"},
    )

    issue = report.issues[0]
    assert issue.source == "user_explicit"
    assert issue.path == "extras.delta"
    assert issue.expected == "integer > 0"
    assert issue.suggestion
