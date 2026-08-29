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

# pylint: disable=E0611

from clawcodex_ext.agent.tool_authoring.call_handlers.bash import parse_sop_wrapper_stdout


def test_parse_sop_wrapper_stdout_last_json_line():
    raw = "2026-06-23 | common | INFO | Registered connector pool type: default\n\"/root/.openjiuwen\"\n"
    assert parse_sop_wrapper_stdout(raw) == "/root/.openjiuwen"


def test_parse_sop_wrapper_stdout_object():
    raw = 'log line\n{"ok": true}\n'
    assert parse_sop_wrapper_stdout(raw) == {"ok": True}
