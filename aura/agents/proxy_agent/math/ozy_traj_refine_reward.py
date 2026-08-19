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

from typing import List

from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import extract_token_ids_and_logprobs, \
    get_episode_summary, default_traj_filter_func, get_tool_outputs, get_action_from_assistant, concat_history_message
from aura.runner.agent_engine_wrapper.vaee.vaee_types import Trajectory, Episode, RequestRecord, Step
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()




def _filter_agent_records(task_id: str, task: dict, records: List[RequestRecord]) -> List[RequestRecord]:
    """Filter records to keep only Agent requests (first user message matches task.problem)."""
    agent_records = []
    for record in records:
        if not record.messages:
            logger.warning(f"{task_id=} record.messages is empty, record.unique_id={record.unique_id}")
            continue
        first_user_msg = None
        for msg in record.messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                first_user_msg = msg
                break
        if first_user_msg is None:
            logger.warning(f"{task_id=} no user message in record.messages, msgs_roles={[m.get('role') if isinstance(m, dict) else type(m).__name__ for m in record.messages]}")
            continue
        user_content = first_user_msg.get("content", "") or ""
        if user_content == task["problem"]:
            agent_records.append(record)
        else:
            logger.warning(f"{task_id=} user_content mismatch! "
                          f"user_content[:200]={repr(user_content[:200])}, "
                          f"task_problem[:200]={repr(task['problem'][:200])}")
    return agent_records


def ozy_token_traj_refine_func(
        task_id: str,
        task: dict,
        records: List[RequestRecord],
        tokenizer=None,
        *args, **kwargs
) -> Episode:
    """
    Refine raw request records into a structured Episode with trajectory steps.

    Filters and sorts records by time, selects agent records matching the task problem,
    builds steps with token IDs, logprobs, tool outputs, and handles truncation
    when the prompt + response exceeds max_model_len.

    Args:
        task_id: Unique task identifier.
        task: Task dictionary containing the problem and metadata.
        records: Raw request records to be refined.
        tokenizer: Tokenizer for encoding/decoding text.
        *args: Additional positional arguments.
        **kwargs: Keyword arguments, may include max_model_len.

    Returns:
        Episode: Structured episode with a single trajectory containing refined steps.
    """
    sorted_records = sorted(
        records,
        key=lambda r: r.start_time
    )
    # Filter invalid records (SAFE response, sub agent messages, messages with empty raw_desponse, content and toolcall are empty but reasoning is not empty)
    valid_records = default_traj_filter_func(task_id, task, sorted_records)

    # Select Agent requests: The first role == "user" message with content == task.problem
    agent_records = _filter_agent_records(task_id, task, valid_records)

    logger.info(f"{task_id=} records={len(records)}, valid_records={len(valid_records)}, agent_records={len(agent_records)}")

    # Build Trajectory
    steps = []
    max_model_len = kwargs.get("max_model_len", None)

    # Concatenate messages in each round of output to prevent alignment issues caused by think and tool compression
    history_records = concat_history_message(agent_records)
    termination_reason = "ENV_DONE"
    records_num = len(history_records)
    for i in range(records_num):
        record = history_records[i]
        next_record = history_records[i + 1] if (i + 1) < records_num else None
        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(record, tokenizer)
        tool_outputs = get_tool_outputs(record, next_record)
        action = get_action_from_assistant(record)
        # Truncate response_ids if max_model_len is set
        if max_model_len is not None:
            curr_prompt_length = len(prompt_ids)
            curr_response_length = len(response_ids)
            if curr_prompt_length + curr_response_length >= max_model_len:
                # Truncate responsive_ids, keep max_madel_1en - curr_prompt_1ength tokens
                dropped_steps = records_num - i - 1
                logger.warning(f"trajectory truncated, {task_id=}, step_idx:{i}, "
                               f"curr_prompt_length:{curr_prompt_length}, curr_response_length:{curr_response_length}, "
                               f"dropped_steps:{dropped_steps}, "
                               f"max_model_len:{max_model_len}")
                termination_reason = "TRUNCATION"
                break

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

        if not steps:
            steps.append(step)
        else:
            last_messages = steps[-1].chat_completions
            if record.messages == last_messages:
                steps[-1] = step
            else:
                steps.append(step)
    if termination_reason == "TRUNCATION":
        steps = [steps[0]]
    trajectory = Trajectory(steps=steps, task=task)
    # Build Episode
    episode = Episode(
        id=task_id,
        trajectories=[trajectory],
    )
    episode.task = task
    episode.id = task_id

    logger.debug(f"Refine completed: {get_episode_summary(episode)}")
    return episode
