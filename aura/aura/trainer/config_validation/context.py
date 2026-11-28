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
from dataclasses import dataclass, field
import math
from threading import TIMEOUT_MAX
from typing import Any, Mapping

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ValidationOptions:
    """Runtime controls shared by the validator and its reporters."""

    strict: bool = True
    show_config_summary: bool = True
    show_default_warnings: bool = True
    report_path: str | None = None
    max_io_workers: int = 8
    io_timeout: float = 30.0

    def __post_init__(self) -> None:
        for name in ("strict", "show_config_summary", "show_default_warnings"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.report_path is not None:
            if not isinstance(self.report_path, str):
                raise TypeError("report_path must be a string or None")
            if not self.report_path.strip():
                raise ValueError("report_path must not be empty")
        if isinstance(self.max_io_workers, bool) or not isinstance(self.max_io_workers, int):
            raise TypeError("max_io_workers must be an integer")
        if self.max_io_workers <= 0:
            raise ValueError("max_io_workers must be greater than zero")
        if isinstance(self.io_timeout, bool) or not isinstance(self.io_timeout, (int, float)):
            raise TypeError("io_timeout must be a number")
        if not 0 < self.io_timeout <= TIMEOUT_MAX or not math.isfinite(self.io_timeout):
            raise ValueError("io_timeout must be finite, greater than zero and at most threading.TIMEOUT_MAX")


@dataclass(frozen=True)
class ValidationContext:
    """Immutable inputs made available to every configuration rule."""

    train_config: Any
    rollout_config: Any = None
    work_mode: str = ""
    source_map: Mapping[str, str] = field(default_factory=dict)
    options: ValidationOptions = field(default_factory=ValidationOptions)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def source_for(self, path: str, namespace: str = "train_config") -> str:
        """Return the recorded origin for a configuration path."""
        for candidate in (f"{namespace}.{path}", path):
            if candidate in self.source_map:
                return self.source_map[candidate]
        return "unknown"

    def snapshot(self) -> dict[str, Any]:
        """Create a detached, resolved configuration snapshot for reports."""
        return {
            "train_config": _to_plain_container(self.train_config),
            "rollout_config": _to_plain_container(self.rollout_config),
            "work_mode": self.work_mode,
        }


def _to_plain_container(value: Any) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        try:
            return OmegaConf.to_container(value, resolve=True)
        except Exception:
            # Preserve unresolved values in the report instead of losing all
            # previously aggregated issues because one interpolation is invalid.
            return OmegaConf.to_container(value, resolve=False)
    return deepcopy(value)
