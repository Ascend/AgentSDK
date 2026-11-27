#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from typing import Any


def _format_observation_as_messages_patch(self, obs: Any) -> list[dict]:
    """Helper to format observation into messages."""
    messages = []

    if isinstance(obs, dict):
        if "problem" in obs:
            messages.append({"role": "user", "content": obs["problem"]})
        elif "tool_outputs" in obs:
            for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_output_str,
                        "tool_call_id": tool_call_id,
                    }
                )
    elif isinstance(obs, str):
        messages.append({"role": "user", "content": obs})
    elif obs:
        messages.append({"role": "user", "content": str(obs)})

    return messages
