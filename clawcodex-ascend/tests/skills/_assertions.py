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

"""Assertions shared by migrated skill invocation tests."""

from __future__ import annotations

from typing import Any


def rendered_prompt(result: Any) -> str:
    """Return the prompt injected by either model or user skill invocation."""

    output = result.output
    if isinstance(output, dict) and isinstance(output.get("prompt"), str):
        return output["prompt"]

    messages = result.new_messages
    assert messages, f"skill invocation did not inject a prompt: {output!r}"
    content = messages[-1].content
    assert isinstance(content, str), f"skill prompt is not text: {content!r}"
    metadata, separator, prompt = content.partition("\n\n")
    assert separator and metadata.startswith("<command-message>"), f"skill prompt metadata is malformed: {content!r}"
    return prompt
