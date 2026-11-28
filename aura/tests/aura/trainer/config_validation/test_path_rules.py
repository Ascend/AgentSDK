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

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aura.trainer.config_validation import ValidationContext
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.paths import build_path_rules
from aura.trainer.config_validation.rules.parallelism import build_parallelism_rules


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
    registry.register_many(build_path_rules())
    registry.register_many(build_parallelism_rules())
    return registry

def make_valid_configs(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    train_data_path = tmp_path / "train-data"
    train_data_path.mkdir()
    (train_data_path / "data.jsonl").write_text('{"prompt": "1+1"}\n', encoding="utf-8")

    test_data_path = tmp_path / "test-data.jsonl"
    test_data_path.write_text('{"prompt": "2+2"}\n', encoding="utf-8")

    weight_save_dir = tmp_path / "weights"

    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    train_config = {
        "actor_rollout_ref": {"model": {"path": str(model_path)}},
        "extras": {
            "weight_save_dir": str(weight_save_dir),
            "data_loader": {
                "train_data_path": str(train_data_path),
                "test_data_path": str(test_data_path),
            },
        },
        "rollout": {"test_freq": 1},
    }
    rollout_config = {"llm_tokenizer_path": str(tokenizer_path)}
    return train_config, rollout_config


def validate(tmp_path, train_config=None, rollout_config=None, work_mode="one_step_off", source_map=None):
    if train_config is None and rollout_config is None:
        train_config, rollout_config = make_valid_configs(tmp_path)
    elif train_config is None or rollout_config is None:
        raise ValueError("test helper requires both explicit configs or neither")
    context = ValidationContext(
        train_config=train_config,
        rollout_config=rollout_config,
        work_mode=work_mode,
        source_map=source_map or {},
    )
    return ConfigValidator(build_default_registry()).validate(context, scopes={"paths"})


def test_valid_filesystem_configuration_passes(tmp_path):
    report = validate(tmp_path)

    assert report.valid is True
    assert report.issues == ()


def test_model_path_must_exist(tmp_path):
    train_config, rollout_config = make_valid_configs(tmp_path)
    train_config["actor_rollout_ref"]["model"]["path"] = str(tmp_path / "missing-model")

    report = validate(tmp_path, train_config, rollout_config)

    assert [issue.rule_id for issue in report.issues] == ["paths.model"]
    assert "does not exist" in report.issues[0].message


def test_model_path_must_be_a_directory_with_config_json(tmp_path):
    train_config, rollout_config = make_valid_configs(tmp_path)
    model_path = tmp_path / "model-without-config"
    model_path.mkdir()
    train_config["actor_rollout_ref"]["model"]["path"] = str(model_path)

    report = validate(tmp_path, train_config, rollout_config)

    assert [issue.rule_id for issue in report.issues] == ["paths.model"]
    assert "config.json" in report.issues[0].message


@pytest.mark.parametrize("config_key", ["train_data_path", "test_data_path"])
def test_consumed_dataset_path_must_be_non_empty(tmp_path, config_key):
    train_config, rollout_config = make_valid_configs(tmp_path)
    empty_path = tmp_path / f"empty-{config_key}"
    empty_path.mkdir()
    train_config["extras"]["data_loader"][config_key] = str(empty_path)

    report = validate(tmp_path, train_config, rollout_config)

    expected_rule = "paths.train_data" if config_key == "train_data_path" else "paths.test_data"
    assert [issue.rule_id for issue in report.issues] == [expected_rule]
    assert "empty" in report.issues[0].message


@pytest.mark.parametrize("work_mode", ["one_step_off", "fully_async"])
@pytest.mark.parametrize("frequency", [0, 100])
@pytest.mark.parametrize("path", [None, ""])
@pytest.mark.parametrize("omega", [False, True])
def test_unset_test_data_skips_checks_regardless_of_frequency(tmp_path, work_mode, frequency, path, omega):
    train_config, rollout_config = make_valid_configs(tmp_path)
    train_config["rollout"]["test_freq"] = frequency
    train_config["extras"]["data_loader"]["test_data_path"] = path
    if omega:
        train_config, rollout_config = OmegaConf.create(train_config), OmegaConf.create(rollout_config)
    report = validate(tmp_path, train_config, rollout_config, work_mode=work_mode)
    assert report.valid and report.issues == ()


@pytest.mark.parametrize("work_mode", ["one_step_off", "fully_async"])
@pytest.mark.parametrize("frequency", [-1, 0, 1])
def test_supplied_test_path_is_checked_regardless_of_frequency(tmp_path, work_mode, frequency):
    train_config, rollout_config = make_valid_configs(tmp_path)
    train_config["rollout"]["test_freq"] = frequency
    train_config["extras"]["data_loader"]["test_data_path"] = str(tmp_path / "missing-test-data")
    report = validate(tmp_path, train_config, rollout_config, work_mode=work_mode)
    assert [issue.rule_id for issue in report.issues] == ["paths.test_data"]


@pytest.mark.parametrize("path", [False, [], " "])
def test_malformed_supplied_test_path_is_not_treated_as_disabled(tmp_path, path):
    train_config, rollout_config = make_valid_configs(tmp_path)
    train_config["rollout"]["test_freq"] = 0
    train_config["extras"]["data_loader"]["test_data_path"] = path
    report = validate(tmp_path, train_config, rollout_config)
    assert [issue.rule_id for issue in report.issues] == ["paths.test_data"]


def test_separated_dataset_checks_are_skipped_in_hybrid_mode(tmp_path):
    train_config, rollout_config = make_valid_configs(tmp_path)
    train_config["extras"]["data_loader"]["train_data_path"] = None
    train_config["extras"]["data_loader"]["test_data_path"] = None

    report = validate(tmp_path, train_config, rollout_config, work_mode="hybrid")

    assert report.valid is True


@pytest.mark.parametrize("work_mode", ["one_step_off", "fully_async", "hybrid"])
@pytest.mark.parametrize("omega", [False, True])
def test_weight_save_directory_does_not_need_to_exist_before_startup(tmp_path, work_mode, omega):
    train_config, rollout_config = make_valid_configs(tmp_path)
    weight_path = tmp_path / "runtime-created" / "weights"
    train_config["extras"]["weight_save_dir"] = str(weight_path)
    if omega:
        train_config, rollout_config = OmegaConf.create(train_config), OmegaConf.create(rollout_config)
    context = ValidationContext(
        train_config=train_config, rollout_config=rollout_config, work_mode=work_mode,
    )
    report = ConfigValidator(build_default_registry()).validate_or_raise(context, scopes={"paths"})
    assert report.valid and report.issues == ()
    assert not weight_path.parent.exists()


def test_llm_tokenizer_path_must_contain_tokenizer_config(tmp_path):
    train_config, rollout_config = make_valid_configs(tmp_path)
    tokenizer_path = Path(rollout_config["llm_tokenizer_path"])
    (tokenizer_path / "tokenizer_config.json").unlink()

    report = validate(tmp_path, train_config, rollout_config)

    assert [issue.rule_id for issue in report.issues] == ["paths.llm_tokenizer"]
    assert "tokenizer_config.json" in report.issues[0].message


def test_rollout_path_rules_are_skipped_without_rollout_config(tmp_path):
    train_config, _ = make_valid_configs(tmp_path)
    context = ValidationContext(
        train_config=train_config,
        rollout_config=None,
        work_mode="one_step_off",
    )

    report = ConfigValidator(build_default_registry()).validate(context, scopes={"paths"})

    assert report.valid is True


def test_path_issue_contains_source_and_repair_information(tmp_path):
    train_config, rollout_config = make_valid_configs(tmp_path)
    train_config["actor_rollout_ref"]["model"]["path"] = str(tmp_path / "missing-model")
    report = validate(
        tmp_path,
        train_config,
        rollout_config,
        source_map={"train_config.actor_rollout_ref.model.path": "user_explicit"},
    )

    issue = report.issues[0]
    assert issue.source == "user_explicit"
    assert issue.path == "train_config.actor_rollout_ref.model.path"
    assert issue.suggestion


def test_multiple_path_errors_are_aggregated(tmp_path):
    train_config, rollout_config = make_valid_configs(tmp_path)
    train_config["actor_rollout_ref"]["model"]["path"] = str(tmp_path / "missing-model")
    train_config["extras"]["weight_save_dir"] = str(tmp_path / "missing-weights")
    rollout_config["llm_tokenizer_path"] = str(tmp_path / "missing-tokenizer")

    report = validate(tmp_path, train_config, rollout_config)

    assert {issue.rule_id for issue in report.issues} == {
        "paths.llm_tokenizer",
        "paths.model",
    }
