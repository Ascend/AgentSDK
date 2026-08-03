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

"""Feature Gate runtime toggle types.

Implements the ``FeatureFlag`` dataclass used by the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=False)
class FeatureFlag:
    """Declaration of a single runtime feature flag.

    Attributes:
        name: Unique identifier for the feature flag.
        default: Whether the feature is enabled by default when not
            overridden by CLI, env-var, or config-file.
        deps: Names of other features that must be enabled for this
            feature to function correctly.
        mutex_with: Names of features that are mutually exclusive
            with this one (cannot be enabled simultaneously).
        description: Human-readable explanation of what the feature does.
    """

    name: str
    default: bool = False
    deps: list[str] = field(default_factory=list)
    mutex_with: list[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        # Ensure name is a stable key
        if not self.name:
            raise ValueError("FeatureFlag name must be a non-empty string")
        # Deduplicate lists
        object.__setattr__(self, "deps", list(dict.fromkeys(self.deps)))
        object.__setattr__(self, "mutex_with", list(dict.fromkeys(self.mutex_with)))
        # Sanity: deps and mutex lists must reference valid names (checked at
        # registry level; here we just ensure they are strings).
        for dep in self.deps:
            if not isinstance(dep, str):
                raise TypeError(f"FeatureFlag.deps items must be str, got {type(dep)}")
        for mx in self.mutex_with:
            if not isinstance(mx, str):
                raise TypeError(f"FeatureFlag.mutex_with items must be str, got {type(mx)}")
