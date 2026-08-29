#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

import threading
import time
from typing import Any

from clawcodex_ext.query.outbox_types import ProactivePromptEvent
from clawcodex_ext.services.proactive import TICK_TAG, get_default_controller

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult


def _sleep_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    seconds = tool_input["seconds"]
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        raise ToolInputError("seconds must be a number")
    if seconds < 0 or seconds > 30:
        raise ToolInputError("seconds must be between 0 and 30")
    ctrl = get_default_controller()
    if ctrl.is_active():
        until_ms = time.time() * 1000 + float(seconds) * 1000
        ctrl.enter_sleep(until_ms)

        def _wake() -> None:
            ctrl.wake_from_sleep()
            outbox = getattr(context, "outbox", None)
            if hasattr(outbox, "append"):
                outbox.append(
                    ProactivePromptEvent(
                        prompt=f"<{TICK_TAG}>{time.strftime('%H:%M:%S')}</{TICK_TAG}>",
                        source="sleep",
                    )
                )

        timer = threading.Timer(float(seconds), _wake)
        timer.daemon = True
        timer.start()
        return ToolResult(
            name="Sleep",
            output={
                "sleeping_until_ms": until_ms,
                "proactive": True,
            },
        )
    time.sleep(float(seconds))
    return ToolResult(name="Sleep", output={"slept_seconds": float(seconds)})


SleepTool: Tool = build_tool(
    name="Sleep",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {"seconds": {"type": "number"}},
        "required": ["seconds"],
    },
    call=_sleep_call,
    prompt="Sleep for a short duration.",
    description="Sleep for a short duration.",
    max_result_size_chars=1000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    # Sleep has no TS counterpart toAutoClassifierInput; surface the
    # duration since that is the only meaningful axis.
    to_auto_classifier_input=lambda input_data: (f"{(input_data or {}).get('seconds', 0)}s"),
)
