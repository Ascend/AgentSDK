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

from agents.math_agent.reward.math_reward import RewardMathFn
from agents.math_agent.reward.reward_types import RewardConfig, RewardInput, RewardOutput
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

_REPETITIVE_GEN_WHITELIST = {",", ".", "*", "#"}
_REPETITIVE_GEN_MIN_LEN = 2
_REPETITIVE_GEN_MAX_OCCURRENCE = 3


def _detect_repetitive_generation(text,
                                  whitelist=_REPETITIVE_GEN_WHITELIST,
                                  min_len=_REPETITIVE_GEN_MIN_LEN,
                                  max_occurrence=_REPETITIVE_GEN_MAX_OCCURRENCE):
    if not isinstance(text, str) or not text:
        return False

    runs = []
    run = []
    for ch in text:
        if ch in whitelist:
            run.append(ch)
        else:
            if len(run) >= min_len:
                runs.append("".join(run))
            run = []
    if len(run) >= min_len:
        runs.append("".join(run))

    substrings = set()
    for r in runs:
        for k in range(min_len, len(r) + 1):
            for i in range(len(r) - k + 1):
                substrings.add(r[i:i + k])

    for s in substrings:
        if text.count(s) > max_occurrence:
            return True
    return False


def math_res_reward_fn(action: dict, task_info: dict) -> RewardOutput:
    reward_config = RewardConfig()
    res_reward_fn = RewardMathFn(reward_config)

    step_idx = action.get("step_idx", None)
    is_last = action.get("is_last_step", False)
    max_steps = action.get("max_steps", None)
    tool_outputs = action.get("tool_outputs", None)
    assistant_message = action.get("assistant_message", None)
    assert step_idx is not None
    assert max_steps is not None
    done = step_idx >= (max_steps - 1) or is_last

    if not done:
        tool_actions = {"assistant_message": assistant_message, "tool_outputs": tool_outputs}
        logger.info(f"[ozy_math_res_reward] NOT DONE, returning reward=0.0 (intermediate step), step_idx={step_idx}")
        return RewardOutput(reward=0.0, metadata={"tool_actions": tool_actions}, is_correct=False)
    else:
        if isinstance(assistant_message, str):
            llm_response = assistant_message
        elif isinstance(assistant_message, dict):
            # Find the finish tool call
            finish_action = None
            if "tool_calls" in assistant_message:
                for tool_call in assistant_message["tool_calls"]:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call
                        break
            if finish_action:
                arguments = finish_action.get("function", {}).get("arguments", {})
                llm_response = arguments.get("response", "")
            elif "content" in assistant_message and assistant_message["content"]:
                llm_response = assistant_message["content"]
            else:
                # No finish tool call found, use the action itself
                llm_response = assistant_message
        if _detect_repetitive_generation(llm_response):
            logger.warning(f"[ozy_math_res_reward] repetitive token/character generation detected, reward forced to 0, step_idx={step_idx}")
            return RewardOutput(reward=0.0, metadata={"repetition_detected": True}, is_correct=False)

        result = res_reward_fn(action=llm_response, task_info=task_info)
        logger.info(f"[ozy_math_res_reward] RESULT: reward={result.reward}, is_correct={result.is_correct}")
        return result
