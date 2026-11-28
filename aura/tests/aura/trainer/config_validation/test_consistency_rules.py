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
from omegaconf import OmegaConf

from aura.trainer.config_validation import ValidationContext
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.consistency import build_consistency_rules


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
    registry.register_many(build_consistency_rules())
    return registry

MISMATCH_CASES = (
    ("rollout.n", "consistency.train.rollout_n", 4),
    ("rollout.nnodes", "consistency.train.rollout_nnodes", 1),
    ("rollout.n_gpus_per_node", "consistency.train.rollout_n_gpus_per_node", 4),
)

ROLLOUT_MISMATCH_CASES = (
    ("n_samples_per_prompt", "consistency.rollout.n_samples_per_prompt", 4),
    ("tokenizer_name_or_path", "consistency.rollout.tokenizer_name_or_path", "/models/other"),
    ("dataset_additional_keys", "consistency.rollout.dataset_additional_keys", ["scores"]),
    ("global_batch_size", "consistency.rollout.global_batch_size", 8),
    ("train_tensor_parallel_size", "consistency.rollout.train_tensor_parallel_size", 4),
)


def valid_train_config():
    return {
        "actor_rollout_ref": {
            "model": {"path": "/models/Qwen3-4B"},
            "rollout": {
                "n": 8,
                "tensor_model_parallel_size": 8,
            },
        },
        "trainer": {
            "n_gpus_per_node": 8,
            "nnodes": 2,
        },
        "rollout": {
            "n": 8,
            "n_gpus_per_node": 8,
            "nnodes": 2,
        },
        "extras": {
            "data_loader": {
                "dataset_additional_keys": ["labels"],
                "global_batch_size": 16,
            },
        },
        "data": {"train_batch_size": 16},
    }


def valid_rollout_config():
    return {
        "n_samples_per_prompt": 8,
        "tokenizer_name_or_path": "/models/Qwen3-4B",
        "dataset_additional_keys": ["labels"],
        "global_batch_size": 16,
        "train_tensor_parallel_size": 8,
    }


def validate(train_config=None, rollout_config=None, source_map=None):
    context = ValidationContext(
        train_config=train_config if train_config is not None else valid_train_config(),
        rollout_config=rollout_config if rollout_config is not None else valid_rollout_config(),
        source_map=source_map or {},
    )
    return ConfigValidator(build_default_registry()).validate(context, scopes={"consistency"})


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


def test_matching_train_and_rollout_configs_pass_every_rule():
    report = validate()

    assert report.valid is True
    assert report.issues == ()


@pytest.mark.parametrize("path,rule_id,value", MISMATCH_CASES)
def test_train_internal_mismatches_are_rejected(path, rule_id, value):
    train_config = valid_train_config()
    set_path(train_config, path, value)

    report = validate(train_config=train_config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]


@pytest.mark.parametrize("path,rule_id,value", ROLLOUT_MISMATCH_CASES)
def test_train_to_rollout_mismatches_are_rejected(path, rule_id, value):
    rollout_config = valid_rollout_config()
    set_path(rollout_config, path, value)

    report = validate(rollout_config=rollout_config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert report.issues[0].path == f"rollout_config.{path}"


def test_equal_but_different_types_are_rejected():
    rollout_config = valid_rollout_config()
    rollout_config["n_samples_per_prompt"] = 8.0

    report = validate(rollout_config=rollout_config)

    assert [issue.rule_id for issue in report.issues] == [
        "consistency.rollout.n_samples_per_prompt"
    ]


@pytest.mark.parametrize("path,rule_id,_", ROLLOUT_MISMATCH_CASES)
def test_missing_rollout_values_are_rejected(path, rule_id, _):
    rollout_config = valid_rollout_config()
    remove_path(rollout_config, path)

    report = validate(rollout_config=rollout_config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    issue = report.issues[0]
    assert issue.value[f"rollout_config.{path}"]["value"] == "<missing>"


def test_missing_train_source_value_is_rejected():
    train_config = valid_train_config()
    remove_path(train_config, "actor_rollout_ref.model.path")

    report = validate(train_config=train_config)

    assert [issue.rule_id for issue in report.issues] == [
        "consistency.rollout.tokenizer_name_or_path"
    ]
    issue = report.issues[0]
    assert issue.value["train_config.actor_rollout_ref.model.path"]["value"] == "<missing>"


def test_rollout_rules_are_skipped_when_no_rollout_config_exists():
    context = ValidationContext(train_config=valid_train_config(), rollout_config=None)

    report = ConfigValidator(build_default_registry()).validate(context, scopes={"consistency"})

    assert report.valid is True
    assert report.issues == ()


def test_issue_contains_both_values_and_both_parameter_sources():
    rollout_config = valid_rollout_config()
    rollout_config["global_batch_size"] = 8
    report = validate(
        rollout_config=rollout_config,
        source_map={
            "train_config.extras.data_loader.global_batch_size": "aura_default",
            "rollout_config.global_batch_size": "user_explicit",
        },
    )

    issue = report.issues[0]
    assert issue.source == "mixed"
    assert issue.value == {
        "train_config.extras.data_loader.global_batch_size": {
            "value": 16,
            "source": "aura_default",
        },
        "rollout_config.global_batch_size": {
            "value": 8,
            "source": "user_explicit",
        },
    }
    assert issue.suggestion


def test_omegaconf_lists_compare_as_plain_ordered_values():
    plain_train_config = valid_train_config()
    plain_rollout_config = valid_rollout_config()
    plain_train_config["extras"]["data_loader"]["dataset_additional_keys"] = ["labels", "scores"]
    plain_rollout_config["dataset_additional_keys"] = ["labels", "scores"]
    train_config = OmegaConf.create(plain_train_config)
    rollout_config = OmegaConf.create(plain_rollout_config)

    assert validate(train_config=train_config, rollout_config=rollout_config).valid is True

    rollout_config.dataset_additional_keys = ["scores", "labels"]
    report = validate(train_config=train_config, rollout_config=rollout_config)

    assert [issue.rule_id for issue in report.issues] == [
        "consistency.rollout.dataset_additional_keys"
    ]


def test_multiple_consistency_errors_are_aggregated():
    train_config = valid_train_config()
    rollout_config = valid_rollout_config()
    train_config["rollout"]["nnodes"] = 1
    rollout_config["global_batch_size"] = 8
    rollout_config["tokenizer_name_or_path"] = "/models/other"

    report = validate(train_config=train_config, rollout_config=rollout_config)

    assert {issue.rule_id for issue in report.issues} == {
        "consistency.train.rollout_nnodes",
        "consistency.rollout.global_batch_size",
        "consistency.rollout.tokenizer_name_or_path",
    }


def test_consistency_rules_do_not_run_for_unrelated_scope():
    train_config = deepcopy(valid_train_config())
    train_config["rollout"]["n"] = 4
    context = ValidationContext(train_config=train_config, rollout_config=valid_rollout_config())

    rules = build_default_registry().get_rules({"aura"})

    assert all("consistency" not in rule.scopes for rule in rules)
