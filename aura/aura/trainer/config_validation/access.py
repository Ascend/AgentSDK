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

from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf


class _MissingValue:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _MissingValue()


def get_config_value(config: Any, path: str, default: Any = MISSING) -> Any:
    """Safely read a dotted path from OmegaConf or plain containers."""
    if not path:
        return config
    if config is None:
        return default

    if OmegaConf.is_config(config):
        try:
            return OmegaConf.select(config, path, default=default)
        except Exception:
            # Invalid or unresolved interpolations are handled as unavailable
            # values so the calling rule can create a structured issue.
            return default

    current = config
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
            continue

        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                current = current[int(part)]
            except (IndexError, TypeError, ValueError):
                return default
            continue

        return default

    return current


def has_config_value(config: Any, path: str) -> bool:
    """Return whether a dotted configuration path is present."""
    return get_config_value(config, path) is not MISSING
