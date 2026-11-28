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

import os
from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationOptions


STRICT_ENV_VAR = "AURA_CONFIG_VALIDATION_STRICT"
_BOOLEAN_FIELDS = ("strict", "show_config_summary", "show_default_warnings")
_OPTION_FIELDS = frozenset((*_BOOLEAN_FIELDS, "report_path", "max_io_workers", "io_timeout"))
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def load_validation_options(
    train_config: Any,
    environ: Mapping[str, str] | None = None,
) -> ValidationOptions:
    """Load validation controls from YAML and then apply environment overrides."""
    raw_options = get_config_value(train_config, "config_validation")
    if raw_options is MISSING:
        option_values: dict[str, Any] = {}
    else:
        option_values = _to_option_mapping(raw_options)

    unknown_fields = set(option_values) - _OPTION_FIELDS
    if unknown_fields:
        unknown = ", ".join(sorted(str(field) for field in unknown_fields))
        raise ValueError(f"Unknown config_validation option(s): {unknown}")

    defaults = ValidationOptions()
    resolved = {
        "strict": defaults.strict,
        "show_config_summary": defaults.show_config_summary,
        "show_default_warnings": defaults.show_default_warnings,
        "report_path": defaults.report_path,
        "max_io_workers": defaults.max_io_workers,
        "io_timeout": defaults.io_timeout,
    }
    for field in _BOOLEAN_FIELDS:
        if field in option_values:
            resolved[field] = _require_boolean(option_values[field], f"config_validation.{field}")
    if "report_path" in option_values:
        resolved["report_path"] = option_values["report_path"]
    if "max_io_workers" in option_values:
        resolved["max_io_workers"] = option_values["max_io_workers"]
    if "io_timeout" in option_values:
        resolved["io_timeout"] = option_values["io_timeout"]

    environment = os.environ if environ is None else environ
    if STRICT_ENV_VAR in environment:
        resolved["strict"] = _parse_environment_boolean(environment[STRICT_ENV_VAR], STRICT_ENV_VAR)

    return ValidationOptions(**resolved)


def _to_option_mapping(raw_options: Any) -> dict[str, Any]:
    if OmegaConf.is_config(raw_options):
        try:
            raw_options = OmegaConf.to_container(raw_options, resolve=True)
        except Exception as exc:
            raise ValueError(f"Unable to resolve config_validation: {exc}") from exc
    if not isinstance(raw_options, Mapping):
        raise TypeError("config_validation must be a mapping")
    return dict(raw_options)


def _require_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _parse_environment_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(f"{name} must be one of {{{accepted}}}, got {value!r}")
