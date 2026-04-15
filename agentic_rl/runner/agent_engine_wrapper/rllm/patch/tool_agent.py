#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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