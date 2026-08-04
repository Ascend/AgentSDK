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

"""Skill search exceptions."""

from __future__ import annotations


class SkillSearchError(Exception):
    """Base exception for skill search module."""


class SkillSourceError(SkillSearchError):
    """Raised when a skill source cannot be parsed or extracted."""


class IndexCorruptError(SkillSearchError):
    """Raised when the persisted index is corrupt and cannot be loaded."""


class SearchDisabledError(SkillSearchError):
    """Raised when search is attempted while the feature flag is off."""


class EmptyQueryError(SkillSearchError):
    """Raised when a search query is empty."""
