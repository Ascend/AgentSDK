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

import pytest
from omegaconf import OmegaConf

from aura.trainer.config_validation import ValidationContext
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.linkage import build_linkage_rules


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
    registry.register_many(build_linkage_rules())
    return registry

def valid_config():
    return {
        "async_training": {"use_trainer_do_validate": False},
        "extras": {"max_queue_size": 256},
        "data": {
            "train_batch_size": 16,
            "val_files": None,
        },
        "trainer": {"test_freq": 0},
    }


def validate(config, source_map=None):
    context = ValidationContext(
        train_config=config,
        source_map=source_map or {},
    )
    return ConfigValidator(build_default_registry()).validate(context, scopes={"linkage"})


def test_valid_linkage_configuration_passes_every_rule():
    report = validate(valid_config())

    assert report.valid is True
    assert report.issues == ()


def test_queue_capacity_must_hold_at_least_one_training_batch():
    config = valid_config()
    config["extras"]["max_queue_size"] = 8

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == ["linkage.queue_capacity"]
    issue = report.issues[0]
    assert issue.value == {
        "extras.max_queue_size": 8,
        "data.train_batch_size": 16,
    }
    assert "at least 16" in issue.suggestion


def test_queue_capacity_equal_to_training_batch_is_valid():
    config = valid_config()
    config["extras"]["max_queue_size"] = 16

    assert validate(config).valid is True


@pytest.mark.parametrize(
    "queue_size,train_batch_size",
    [
        (0, 16),
        (256, 0),
        (True, 16),
        (256, "16"),
        (None, 16),
    ],
)
def test_queue_rule_skips_invalid_prerequisites_to_avoid_cascade(queue_size, train_batch_size):
    config = valid_config()
    config["extras"]["max_queue_size"] = queue_size
    config["data"]["train_batch_size"] = train_batch_size

    assert validate(config).issues == ()


@pytest.mark.parametrize("val_files", ["/data/val.parquet", ["/data/val.parquet"]])
def test_trainer_validation_accepts_non_empty_validation_data(val_files):
    config = valid_config()
    config["async_training"]["use_trainer_do_validate"] = True
    config["data"]["val_files"] = val_files

    assert validate(config).valid is True


@pytest.mark.parametrize("val_files", [None, "", "   ", []])
def test_trainer_validation_requires_validation_data(val_files):
    config = valid_config()
    config["async_training"]["use_trainer_do_validate"] = True
    config["data"]["val_files"] = val_files

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [
        "linkage.trainer_validate_requires_val_data"
    ]


@pytest.mark.parametrize("val_files", [None, "", "   ", []])
def test_positive_trainer_test_frequency_requires_validation_data(val_files):
    config = valid_config()
    config["trainer"]["test_freq"] = 10
    config["data"]["val_files"] = val_files

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [
        "linkage.test_freq_requires_val_data"
    ]


@pytest.mark.parametrize("test_freq", [0, -1, True, 1.5, "10", None])
def test_test_frequency_rule_only_activates_for_positive_integers(test_freq):
    config = valid_config()
    config["trainer"]["test_freq"] = test_freq

    assert validate(config).issues == ()


def test_both_validation_triggers_report_both_causes():
    config = valid_config()
    config["async_training"]["use_trainer_do_validate"] = True
    config["trainer"]["test_freq"] = 10

    report = validate(config)

    assert {issue.rule_id for issue in report.issues} == {
        "linkage.test_freq_requires_val_data",
        "linkage.trainer_validate_requires_val_data",
    }


def test_omegaconf_empty_and_non_empty_validation_lists_are_supported():
    config = OmegaConf.create(valid_config())
    config.async_training.use_trainer_do_validate = True
    config.data.val_files = []

    report = validate(config)
    assert [issue.rule_id for issue in report.issues] == [
        "linkage.trainer_validate_requires_val_data"
    ]

    config.data.val_files = ["/data/val.parquet"]
    assert validate(config).valid is True


def test_linkage_issue_contains_parameter_source_and_repair_information():
    config = valid_config()
    config["extras"]["max_queue_size"] = 8
    report = validate(
        config,
        source_map={
            "train_config.extras.max_queue_size": "aura_default",
            "train_config.data.train_batch_size": "user_explicit",
        },
    )

    issue = report.issues[0]
    assert issue.source == "mixed"
    assert issue.expected == "max_queue_size >= train_batch_size (16)"
    assert issue.suggestion


def test_linkage_rules_are_excluded_from_unrelated_scopes():
    rules = build_default_registry().get_rules({"paths"})

    assert all("linkage" not in rule.scopes for rule in rules)
