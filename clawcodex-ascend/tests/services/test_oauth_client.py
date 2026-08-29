#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for ``src.services.oauth.client``."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.auth.claude_ai import ENV_ORG_UUID
from clawcodex_ext.services.oauth.client import get_organization_uuid


def _no_claude_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_AI_")}


@pytest.mark.asyncio
async def test_get_organization_uuid_none_when_unset() -> None:
    with patch.dict(os.environ, _no_claude_env(), clear=True):
        assert await get_organization_uuid() is None


@pytest.mark.asyncio
async def test_get_organization_uuid_returns_env_value() -> None:
    env = _no_claude_env() | {ENV_ORG_UUID: "org-abc"}
    with patch.dict(os.environ, env, clear=True):
        assert await get_organization_uuid() == "org-abc"
