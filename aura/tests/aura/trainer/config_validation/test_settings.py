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
from threading import TIMEOUT_MAX

import pytest
from omegaconf import OmegaConf

from aura.trainer.config_validation import (
    STRICT_ENV_VAR,
    ValidationOptions,
    load_validation_options,
)


def test_missing_config_block_uses_framework_defaults():
    options = load_validation_options({}, environ={})

    assert options == ValidationOptions()


def test_yaml_block_controls_every_supported_option():
    config = {
        "config_validation": {
            "strict": False,
            "show_config_summary": False,
            "show_default_warnings": False,
            "report_path": "/tmp/config-validation.json",
            "max_io_workers": 4,
            "io_timeout": 2.5,
        }
    }

    options = load_validation_options(config, environ={})

    assert options == ValidationOptions(
        strict=False,
        show_config_summary=False,
        show_default_warnings=False,
        report_path="/tmp/config-validation.json",
        max_io_workers=4,
        io_timeout=2.5,
    )


def test_omegaconf_block_and_interpolation_are_resolved():
    config = OmegaConf.create(
        {
            "report_dir": "/tmp",
            "storage_timeout": 12,
            "config_validation": {
                "strict": True,
                "report_path": "${report_dir}/validation.json",
                "io_timeout": "${storage_timeout}",
            },
        }
    )

    options = load_validation_options(config, environ={})

    assert options.report_path == "/tmp/validation.json"
    assert options.io_timeout == 12


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_environment_can_enable_strict_mode(value):
    config = {"config_validation": {"strict": False}}

    options = load_validation_options(config, environ={STRICT_ENV_VAR: value})

    assert options.strict is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_environment_can_disable_strict_mode(value):
    config = {"config_validation": {"strict": True}}

    options = load_validation_options(config, environ={STRICT_ENV_VAR: value})

    assert options.strict is False


@pytest.mark.parametrize("value", ["", "enabled", "2", "none"])
def test_invalid_environment_boolean_is_rejected(value):
    with pytest.raises(ValueError, match=STRICT_ENV_VAR):
        load_validation_options({}, environ={STRICT_ENV_VAR: value})


def test_non_string_environment_value_is_rejected():
    with pytest.raises(TypeError, match=STRICT_ENV_VAR):
        load_validation_options({}, environ={STRICT_ENV_VAR: False})


@pytest.mark.parametrize("field", ["strict", "show_config_summary", "show_default_warnings"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_yaml_booleans_require_exact_boolean_type(field, value):
    config = {"config_validation": {field: value}}

    with pytest.raises(TypeError, match=field):
        load_validation_options(config, environ={})


@pytest.mark.parametrize("block", [None, True, [], "strict=false"])
def test_config_validation_block_must_be_a_mapping(block):
    with pytest.raises(TypeError, match="must be a mapping"):
        load_validation_options({"config_validation": block}, environ={})


def test_unknown_yaml_option_is_rejected():
    with pytest.raises(ValueError, match="unknown_option"):
        load_validation_options(
            {"config_validation": {"unknown_option": True}},
            environ={},
        )


@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("report_path", 1, TypeError),
        ("report_path", "", ValueError),
        ("max_io_workers", True, TypeError),
        ("max_io_workers", 0, ValueError),
        ("io_timeout", True, TypeError),
        ("io_timeout", "30", TypeError),
        ("io_timeout", None, TypeError),
        ("io_timeout", [], TypeError),
        ("io_timeout", 0, ValueError),
        ("io_timeout", -1, ValueError),
        ("io_timeout", float("nan"), ValueError),
        ("io_timeout", float("inf"), ValueError),
        ("io_timeout", float("-inf"), ValueError),
        ("io_timeout", TIMEOUT_MAX * 2, ValueError),
        ("io_timeout", 10 ** 500, ValueError),
    ],
)
def test_validation_options_reject_invalid_non_boolean_values(field, value, error_type):
    config = {"config_validation": {field: value}}

    with pytest.raises(error_type, match=field):
        load_validation_options(config, environ={})


def test_loading_options_does_not_mutate_input_config():
    config = {
        "config_validation": {
            "strict": False,
            "show_config_summary": False,
        }
    }
    original = deepcopy(config)

    load_validation_options(config, environ={})

    assert config == original


@pytest.mark.parametrize("timeout", [0.001, 1, 2.5, TIMEOUT_MAX])
def test_io_timeout_accepts_supported_positive_numbers(timeout):
    assert load_validation_options({"config_validation": {"io_timeout": timeout}}, environ={}).io_timeout == timeout
