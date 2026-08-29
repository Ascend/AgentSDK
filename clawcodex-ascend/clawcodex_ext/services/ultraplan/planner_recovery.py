#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Failure recovery helpers for LLM-driven ultraplan generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerRecoveryHint:
    message: str
    can_retry: bool = True
    manual_mode_suggested: bool = True


def recovery_hint(error: Exception) -> PlannerRecoveryHint:
    return PlannerRecoveryHint(
        message=(f"LLM planning failed. Try simplifying the goal or provide a manual plan JSON. Last error: {error}")
    )
