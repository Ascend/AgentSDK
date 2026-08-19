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

import os
import copy
from typing import List
from aura.runner.agent_engine_wrapper.vaee.vaee_types import Trajectory, Episode, RequestRecord, Step
from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import extract_token_ids_and_logprobs, \
    get_tool_outputs, get_action_from_assistant, concat_history_message
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

def get_no_compress_steps_in_step_trajectories(records: List[RequestRecord], task: dict, tokenizer=None):
    """
    Trajectory step filtering for step-mode training

    Description: Collect all step trajectories of the current task into one trajectory based on whether compression is applied.
    Once compression occurs, place into a new trajectory.
    Notes:
        1. Ensure input records are sorted by start_time.
        2. Input step trajectories do not contain subagent messages.
    """

    records_num = len(records)
    trajectories: List[Trajectory] = []
    for i in range(records_num):
        record = records[i]
        next_record = records[i + 1] if (i + 1) < records_num else None
        matched_traj_idx = None

        # Try to match current messages with the last Step of existing Trajectory
        for traj_idx, traj in enumerate(trajectories):
            if not traj.steps:
                continue

            last_step = traj.steps[-1]
            last_messages = last_step.chat_completions

            # Check if messages are prefixed by the previous Step's chat_completions
            # and messages are 2 longer (tool message + assistant response)
            if (len(record.messages) >= len(last_messages) + 2 and
                    record.messages[:len(last_messages)] == last_messages):
                matched_traj_idx = traj_idx
                break

        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(record, tokenizer)
        tool_outputs = get_tool_outputs(record, next_record)
        action = get_action_from_assistant(record)

        step = Step(
            id=record.request_id,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=logprobs,
            reward=0.0,
            chat_completions=record.messages,
            model_response=record.response_text or "",
            action=action,
            tool_outputs=tool_outputs,
        )

        if matched_traj_idx is not None:
            # Append to matched Trajectory
            trajectories[matched_traj_idx].steps.append(step)
        else:
            # Create a new Trajectory
            new_traj = Trajectory(steps=[step], task=task)
            trajectories.append(new_traj)

    return trajectories

def is_messages_match_in(last_record: RequestRecord, record: RequestRecord):
    """
    Compare whether the messages in record contain the messages + response from last_record.
    messages field: record.messages
    response message: record.raw_response['choices'][0]['message']
    """
    # note: Force matching prompt_messages + assistant_message here to avoid the inference engine's thinking removal issue
    last_messages = [*last_record.messages, last_record.raw_response['choices'][0]['message']]
    cur_messages = record.messages
    if len(cur_messages) >= len(last_messages) and cur_messages[:len(last_messages)] == last_messages:
        return True
    return False

def get_no_compress_steps_in_token_trajectories(records: List[RequestRecord], task: dict, tokenizer=None):
    """
    Trajectory step filtering for step-mode training and non-compressed step trajectory concatenation into token trajectory

    Description: Collect all step trajectories of the current task into one trajectory based on whether compression is applied.
    Once compression occurs, place into a new trajectory.
    Notes:
        1. Ensure input records are sorted by start_time.
        2. Input step trajectories do not contain subagent messages.
        3. Concatenate the entire trajectory into a token trajectory.
    """

    records_num = len(records)
    records_list = []
    for i in range(records_num):
        record = records[i]
        matched_traj_idx = None

        # Try to match current messages with the last Step of existing Trajectory
        for traj_idx, i_records in enumerate(records_list):
            if not i_records:
                continue

            last_record = i_records[-1]
            # Check if messages are prefixed by the previous Step's chat_completions + assistant
            # and messages are 1 longer (tool message)
            if is_messages_match_in(last_record, record):
                matched_traj_idx = traj_idx
                break

        if matched_traj_idx is not None:
            records_list[matched_traj_idx].append(record)
        else:
            new_records = [record]
            records_list.append(new_records)

    trajectories: List[Trajectory] = []
    for k in range(len(records_list)):
        sub_records = records_list[k]
        records_num = len(sub_records)
        steps = []
        for i in range(records_num):
            record = sub_records[i]
            next_record = sub_records[i + 1] if (i + 1) < records_num else None
            logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(record, tokenizer)
            tool_outputs = get_tool_outputs(record, next_record)
            action = get_action_from_assistant(record)
            step = Step(
                id=record.request_id,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                logprobs=logprobs,
                reward=0.0,
                chat_completions=record.messages,
                model_response=record.response_text or "",
                action=action,
                tool_outputs=tool_outputs,
            )
            steps.append(step)
        # Create a new Trajectory
        new_traj = Trajectory(steps=steps, task=task)
        trajectories.append(new_traj)
    return trajectories

def prefix_message_match_check(prev_prompt: List[int], prev_response: List[int], next_prompt: List[int], task_id="",
    step=0, print_str="") -> bool:
    """
    Verify that the previous round's prompt_ids + response_ids are a complete prefix of the next round's prompt_ids.
    If the next round is shorter, it indicates a mismatch (likely due to thinking removal).
    """
    full_prev = prev_prompt + prev_response
    len_prev = len(full_prev)
    # If the next round prompt is shorter than the previous round total, thinking info was definitely removed
    if len(next_prompt) < len_prev:
        logger.warning(
            f"[prefix_message_match_check Mismatch]{print_str} task_id:{task_id}, step_idx:{step}. "
            f"Next prompt is shorter than expected. "
            f"Expected min len: {len_prev}, Actual len: {len(next_prompt)}. "
            f"Likely thinking info was removed."
        )
        return False
    # Check if the next prompt's prefix exactly equals the previous total
    if next_prompt[:len_prev] == full_prev:
        return True

    if os.environ.get("ENABLE_MESSAGE_DIFF_CHECK", 'false').lower() == "false":
        return False

    # 3. Tolerance check: Use two-pointer to find the first mismatch point and try to skip the error
    p_prev, p_next = 0, 0
    mismatch_idx_prev = None
    mismatch_idx_next = None
    has_diff = False

    while p_prev < len_prev and p_next < len(next_prompt):
        if full_prev[p_prev] == next_prompt[p_next]:
            p_prev += 1
            p_next += 1
        else:
            mismatch_idx_prev = p_prev
            mismatch_idx_next = p_next
            has_diff = True

            # Try to resolve 3 possible misalignment cases caused by single-token tokenization differences:
            # Case A: Replacement-type difference (1-to-1, e.g., 198 -> 271), same length, skip 1 each
            if p_prev + 1 < len_prev and p_next + 1 < len(next_prompt) and full_prev[p_prev + 1] == next_prompt[p_next + 1]:
                p_prev += 1
                p_next += 1
            # Case B: Forced merge-type difference (2-to-1), next_prompt merged here, next skip 1, prev skip 2
            elif p_prev + 2 < len_prev and full_prev[p_prev + 2] == next_prompt[p_next + 1]:
                p_prev += 2
                p_next += 1
            # Case C: Forced split-type difference (1-to-2), next_prompt split here, next skip 2, prev skip 1
            elif p_next + 2 < len(next_prompt) and full_prev[p_prev + 1] == next_prompt[p_next + 2]:
                p_prev += 1
                p_next += 2
            else:
                # Difference exceeds 1 token range, or subsequent tokens cannot re-align, fail directly
                break

            # After successfully skipping 1 difference, remaining tokens must be exactly equal (no second difference allowed)
            rem_prev = full_prev[p_prev:]
            rem_next = next_prompt[p_next : p_next + len(rem_prev)]
            if rem_prev == rem_next:
                p_prev = len_prev
                break
            else:
                break

    # 4. Determine final alignment result
    if p_prev == len_prev:
        if has_diff:
            # Although there is a difference, it aligned within tolerance, log an INFO or DEBUG message
            ctx_start = max(0, mismatch_idx_prev - 3)
            logger.debug(
                f"[prefix_message_match_check Mismatch]{print_str} task_id:{task_id}, step_idx:{step}."
                f" Single token difference tolerated at index {mismatch_idx_prev}.\n"
                f"Expected clip: {full_prev[ctx_start : mismatch_idx_prev + 5]}\n"
                f"Actual clip:   {next_prompt[ctx_start : mismatch_idx_next + 5]}"
            )
        return True

    # 5. Complete mismatch, print warning log
    mismatch_idx = mismatch_idx_prev if mismatch_idx_prev is not None else 0
    ctx_start = max(0, mismatch_idx - 5)
    logger.warning(
        f"[prefix_message_match_check Mismatch]{print_str} task_id:{task_id}, step_idx:{step}."
        f" Mismatch detected (exceeded tolerance) near token index {mismatch_idx}.\n"
        f"Expected tokens (ctx): {full_prev[ctx_start:mismatch_idx+20]}\n"
        f"Actual tokens (ctx):   {next_prompt[ctx_start:mismatch_idx+20]}"
    )
    return False

def convert_step_traj_to_token_traj(steps: List[Step], task_id="") -> Step:
    cancel_logprobs = False
    res_step = copy.deepcopy(steps[-1])
    res_step.prompt_ids = steps[0].prompt_ids
    # Calculate response_masks: response portion filled with 1, tool portion filled with 0.
    # The tool portion is formed by the current round's prompt minus the previous round's (prompt + response).
    # Calculate logprobs: response portion keeps original logprobs, tool portion filled with 0.
    # If any logprobs value is None, skip calculation and return [].
    response_tokens = []
    response_masks = []
    logprobs = []
    for j in range(len(steps)):
        response_tokens.extend(steps[j].response_ids)
        response_masks.extend([1] * len(steps[j].response_ids))
        if steps[j].logprobs is not None and len(steps[j].logprobs) != 0 and not cancel_logprobs:
            logprobs.extend(steps[j].logprobs)
        else:
            cancel_logprobs = True
        if j < len(steps) - 1:
            prefix_len = len(steps[j].prompt_ids) + len(steps[j].response_ids)
            prefix_message_match_check(steps[j].prompt_ids, steps[j].response_ids, steps[j + 1].prompt_ids, task_id, j,
                "convert_step_traj_to_token_traj")
            tool_tokens = steps[j + 1].prompt_ids[prefix_len:]
            response_tokens.extend(tool_tokens)
            response_masks.extend([0] * len(tool_tokens))
            if not cancel_logprobs:
                logprobs.extend([0] * len(tool_tokens))
    res_step.response_ids = response_tokens
    res_step.response_masks = response_masks
    res_step.logprobs = logprobs
    return res_step

def compress_trajectories_steps(trajectories: List[Trajectory], task_id: str=""):
    """
    Step compression for step-mode training

    Description: Each step's messages within a trajectory are uncompressed (no thinking removal).
    Merge the entire trajectory into a single step to reduce performance issues caused by too many steps in step mode.
    Notes:
        1. Ensure steps within each trajectory are consecutive multi-round interactions, with no system/tool compression or thinking removal in historical messages.
        2. Since each step in a trajectory is uncompressed, similar to token mode, the last step can be used directly.
        3. Each trajectory needs to concatenate multiple steps into a single token trajectory.
    """
    assert len(trajectories) > 0
    steps = []
    origin_steps = []
    for traj in trajectories:
        last_step = convert_step_traj_to_token_traj(traj.steps, task_id)
        steps.append(last_step)
        origin_steps.extend(traj.steps)

    logger.info(f"compress_trajectories_steps task_id:{task_id} origin_steps:{len(origin_steps)} -> steps:{len(steps)}")
    trajectory = copy.deepcopy(trajectories[0])
    trajectory.steps = steps
    return trajectory
