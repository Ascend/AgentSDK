#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Public Python API for ClawCodex."""
# pylint: disable=undefined-all-variable

__all__ = [
    "OrchestrationSubsystem",
    "QueryConfig",
    "QueryRunner",
    "QueryEvent",
]


def __getattr__(name: str):
    if name == "OrchestrationSubsystem":
        from .orchestration import OrchestrationSubsystem

        return OrchestrationSubsystem
    if name in {"QueryConfig", "QueryRunner", "QueryEvent"}:
        from .query import QueryConfig, QueryEvent, QueryRunner

        values = {
            "QueryConfig": QueryConfig,
            "QueryRunner": QueryRunner,
            "QueryEvent": QueryEvent,
        }
        return values[name]
    raise AttributeError(name)
