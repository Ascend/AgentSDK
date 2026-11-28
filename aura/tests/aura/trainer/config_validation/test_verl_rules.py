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

from aura.trainer.config_validation import ValidationContext
from aura.trainer.config_validation.rules.verl import SUPPORTED_ACTOR_STRATEGIES
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.verl import build_verl_rules


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
    registry.register_many(build_verl_rules())
    return registry

EXACT_RULE_CASES = (
    ("actor_rollout_ref.actor.strategy", "verl.actor.strategy", "ddp"),
    ("actor_rollout_ref.rollout.name", "verl.rollout.name", "sglang"),
    ("actor_rollout_ref.rollout.mode", "verl.rollout.mode", "sync"),
    ("actor_rollout_ref.rollout.calculate_log_probs", "verl.rollout.calculate_log_probs", False),
    ("actor_rollout_ref.actor.use_rollout_log_probs", "verl.actor.use_rollout_log_probs", False),
    ("actor_rollout_ref.model.use_remove_padding", "verl.model.use_remove_padding", False),
    ("trainer.device", "verl.trainer.device", "cuda"),
)

POSITIVE_INTEGER_CASES = (
    ("actor_rollout_ref.rollout.data_parallel_size", "verl.rollout.data_parallel_size"),
    ("actor_rollout_ref.rollout.n", "verl.rollout.n"),
    ("trainer.n_gpus_per_node", "verl.trainer.n_gpus_per_node"),
    ("trainer.nnodes", "verl.trainer.nnodes"),
    ("data.train_batch_size", "verl.data.train_batch_size"),
)


def valid_config():
    return {
        "actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp2",
                "use_rollout_log_probs": True,
            },
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "calculate_log_probs": True,
                "tensor_model_parallel_size": 8,
                "data_parallel_size": 2,
                "n": 8,
            },
            "model": {"use_remove_padding": True},
        },
        "trainer": {
            "device": "npu",
            "n_gpus_per_node": 8,
            "nnodes": 2,
        },
        "data": {"train_batch_size": 16},
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


def validate(config, work_mode="one_step_off", source_map=None):
    context = ValidationContext(
        train_config=config,
        work_mode=work_mode,
        source_map=source_map or {},
    )
    return ConfigValidator(build_default_registry()).validate(context, scopes={"verl"})


def test_default_registry_contains_every_verl_rule():
    registry = build_default_registry()

    assert len(registry.get_rules({"verl"})) == 15


def test_valid_verl_npu_configuration_passes_every_rule():
    report = validate(valid_config())

    assert report.valid is True
    assert report.issues == ()


@pytest.mark.parametrize("calculate,use_rollout,remove_padding", [
    (calculate, use_rollout, remove_padding)
    for calculate in (False, True)
    for use_rollout in (False, True)
    for remove_padding in (False, True)
])
def test_hybrid_does_not_force_one_step_off_boolean_values(calculate, use_rollout, remove_padding):
    config = valid_config()
    set_path(config, "actor_rollout_ref.rollout.calculate_log_probs", calculate)
    set_path(config, "actor_rollout_ref.actor.use_rollout_log_probs", use_rollout)
    set_path(config, "actor_rollout_ref.model.use_remove_padding", remove_padding)

    assert validate(config, work_mode="hybrid").valid is True


@pytest.mark.parametrize("strategy", sorted(SUPPORTED_ACTOR_STRATEGIES))
def test_actor_strategy_accepts_every_supported_value(strategy):
    config = valid_config()
    set_path(config, "actor_rollout_ref.actor.strategy", strategy)

    assert validate(config).valid is True


@pytest.mark.parametrize("path,rule_id,invalid_value", EXACT_RULE_CASES)
def test_exact_requirements_reject_invalid_values(path, rule_id, invalid_value):
    config = valid_config()
    set_path(config, path, invalid_value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert report.issues[0].path == path
    assert report.issues[0].value == invalid_value


@pytest.mark.parametrize("path,rule_id,_", EXACT_RULE_CASES)
def test_exact_requirements_reject_missing_values(path, rule_id, _):
    config = valid_config()
    remove_path(config, path)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert report.issues[0].value == "<missing>"


def test_rollout_mode_rule_only_applies_to_one_step_off_policy():
    config = valid_config()
    set_path(config, "actor_rollout_ref.rollout.mode", "sync")

    report = validate(config, work_mode="hybrid")

    assert report.valid is True


@pytest.mark.parametrize(
    "path,rule_id",
    [
        ("actor_rollout_ref.rollout.calculate_log_probs", "verl.rollout.calculate_log_probs"),
        ("actor_rollout_ref.actor.use_rollout_log_probs", "verl.actor.use_rollout_log_probs"),
        ("actor_rollout_ref.model.use_remove_padding", "verl.model.use_remove_padding"),
    ],
)
@pytest.mark.parametrize("invalid_value", [1, "true", None])
def test_boolean_requirements_reject_truthy_non_boolean_values(path, rule_id, invalid_value):
    config = valid_config()
    set_path(config, path, invalid_value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]


@pytest.mark.parametrize("path,rule_id", POSITIVE_INTEGER_CASES)
@pytest.mark.parametrize("invalid_value", [0, -1, True, 1.5, "1", None])
def test_positive_integer_requirements_reject_invalid_values(path, rule_id, invalid_value):
    config = valid_config()
    set_path(config, path, invalid_value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]


@pytest.mark.parametrize("path,rule_id", POSITIVE_INTEGER_CASES)
def test_positive_integer_requirements_reject_missing_values(path, rule_id):
    config = valid_config()
    remove_path(config, path)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == [rule_id]
    assert report.issues[0].value == "<missing>"


@pytest.mark.parametrize("value", [1, 2, 4, 8, 16])
def test_tensor_parallel_size_accepts_powers_of_two(value):
    config = valid_config()
    set_path(config, "actor_rollout_ref.rollout.tensor_model_parallel_size", value)
    set_path(config, "actor_rollout_ref.rollout.data_parallel_size", 16 // value)

    assert validate(config).valid is True


@pytest.mark.parametrize("value", [0, -2, 3, 6, True, 1.5, "8", None])
def test_tensor_parallel_size_rejects_invalid_values_without_topology_cascade(value):
    config = valid_config()
    set_path(config, "actor_rollout_ref.rollout.tensor_model_parallel_size", value)

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == ["verl.rollout.tensor_model_parallel_size"]


def test_tensor_parallel_size_rejects_missing_value():
    config = valid_config()
    remove_path(config, "actor_rollout_ref.rollout.tensor_model_parallel_size")

    report = validate(config)

    assert [issue.rule_id for issue in report.issues] == ["verl.rollout.tensor_model_parallel_size"]
    assert report.issues[0].value == "<missing>"


def test_multiple_verl_errors_are_aggregated():
    config = valid_config()
    set_path(config, "actor_rollout_ref.actor.strategy", "ddp")
    set_path(config, "trainer.device", "cuda")
    set_path(config, "actor_rollout_ref.rollout.tensor_model_parallel_size", 4)

    report = validate(config)

    assert {issue.rule_id for issue in report.issues} == {
        "verl.actor.strategy",
        "verl.topology.tp_dp",
        "verl.trainer.device",
    }
