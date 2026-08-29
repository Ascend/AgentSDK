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

from __future__ import annotations

from clawcodex_ext.services.mcp.normalization import normalize_name_for_mcp


class TestNormalizeNameForMcp:
    def test_simple_name(self) -> None:
        assert normalize_name_for_mcp("myserver") == "myserver"

    def test_with_hyphens(self) -> None:
        assert normalize_name_for_mcp("my-server") == "my-server"

    def test_with_underscores(self) -> None:
        assert normalize_name_for_mcp("my_server") == "my_server"

    def test_with_spaces(self) -> None:
        assert normalize_name_for_mcp("my server") == "my_server"

    def test_with_special_chars(self) -> None:
        assert normalize_name_for_mcp("my@server!") == "my_server_"

    def test_with_dots(self) -> None:
        assert normalize_name_for_mcp("my.server.name") == "my_server_name"

    def test_empty_string(self) -> None:
        assert normalize_name_for_mcp("") == ""

    def test_already_valid(self) -> None:
        assert normalize_name_for_mcp("valid_Name-123") == "valid_Name-123"

    def test_claude_ai_prefix(self) -> None:
        result = normalize_name_for_mcp("claude.ai some server")
        assert "__" not in result
        assert result.strip("_") == result
