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
import json
import copy
from typing import List, Any

from aura.runner.agent_engine_wrapper.vaee.vaee_types import Trajectory, Episode, RequestRecord, Step
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def get_episode_summary(episode: Episode):
    summary = {
        "task_id": episode.id,
        "trajectories": [
            {
                "traj_id": i,
                "step_count": len(traj.steps),
                "steps": [s.id for s in traj.steps]
            }
            for i, traj in enumerate(episode.trajectories)
        ]
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


def preprocess_messages_arguments_str2json(messages):
    new_messages = copy.deepcopy(messages)
    for msg in new_messages:
        if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        logger.warning(f"Warning: Failed to decode arguments: {args}")
    return new_messages

def extract_logprobs(record):
    """
    Extract the logprob list from the raw_desponse of chat.completion
    """
    logprobs_data = record.raw_response['choices'][0].get("logprobs")
    if not logprobs_data:
        return []

    content_list = logprobs_data.get("content")
    if content_list and isinstance(content_list, list):
        return [item["logprob"] for item in content_list if "logprob" in item]

    if "token_logprobs" in logprobs_data:
        return logprobs_data.get("token_logprobs", [])

    return []

def _log_token_difference(stage_name, infer_ids, concat_ids, tokenizer, window_size=20):
    """Identify the first difference location and print the context"""
    diff_idx = next(
        (i for i, (a, b) in enumerate(zip(infer_ids, concat_ids)) if a != b),
        min(len(infer_ids), len(concat_ids))
    )

    start_idx = max(0, diff_idx - window_size)
    end_idx = diff_idx + window_size

    infer_window = infer_ids[start_idx:end_idx]
    concat_window = concat_ids[start_idx:end_idx]

    logger.warning(
        f"[{stage_name} Token Mismatch] infer_len={len(infer_ids)}, concat_len={len(concat_ids)}. first-diff-idx: {diff_idx}\n"
        f"  - infer Token idx ({start_idx}): {infer_window}\n"
        f"  - concat Token idx ({start_idx}): {concat_window}\n"
        f"  - infer: {repr(tokenizer.decode(infer_window))}\n"
        f"  - concat: {repr(tokenizer.decode(concat_window))}"
    )
def check_token_ids_and_messages(record: RequestRecord, tokenizer):
    if os.environ.get("ENABLE_MESSAGE_DIFF_CHECK", 'false').lower() == "false":
        return

    if "prompt_token_ids" in record.raw_response:
        infer_prompt_ids = record.raw_response['prompt_token_ids']
        prompt_messages = preprocess_messages_arguments_str2json(record.messages)
        concat_prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tools=record.raw_request["tools"] if "tools" in record.raw_request else None,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False
        )

        if infer_prompt_ids != concat_prompt_ids:
            _log_token_difference("Prompt", infer_prompt_ids, concat_prompt_ids, tokenizer)

        if "token_ids" in record.raw_response['choices'][0]:
            infer_response_ids = record.raw_response['choices'][0]['token_ids']
            response_msgs = preprocess_messages_arguments_str2json([record.raw_response['choices'][0]['message']])
            messages = prompt_messages + response_msgs
            try:
                concat_response_ids = tokenizer.apply_chat_template(
                    messages,
                    tools=record.raw_request["tools"] if "tools" in record.raw_request else None,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False
                )[len(concat_prompt_ids):]
                if concat_response_ids and tokenizer.decode([concat_response_ids[-1]]) == '\n' \
                    and len(concat_response_ids) != len(infer_response_ids):
                    concat_response_ids = concat_response_ids[:-1]
            except Exception as e:
                logger.error(f"====tokenizer.apply_chat_template exception: e:{e} messages={messages}")
                return

            if infer_response_ids != concat_response_ids:
                _log_token_difference("Response", infer_response_ids, concat_response_ids, tokenizer)
                if len(concat_response_ids) < 256:
                    logger.info(f"check_token_ids_and_messages infer_response_ids:{infer_response_ids}"
                        f" concat_response_ids:{concat_response_ids}")

def extract_token_ids_and_logprobs(record: RequestRecord, tokenizer, get_from_infer=False) -> tuple[list[int], list[int], Any]:
    if record.token_ids and record.response_ids:
        # TITO mode
        prompt_ids = record.token_ids
        response_ids = record.response_ids
        logprobs = record.token_response['choices'][0]['logprobs']['token_logprobs']
    else:
        # Non TITO mode
        check_token_ids_and_messages(record, tokenizer)
        if "prompt_token_ids" in record.raw_response and get_from_infer:
            prompt_ids = record.raw_response['prompt_token_ids']
        else:
            prompt_messages = preprocess_messages_arguments_str2json(record.messages)
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tools=record.raw_request["tools"] if "tools" in record.raw_request else None,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False
            )

        logprobs = extract_logprobs(record)
        if "token_ids" in record.raw_response['choices'][0] and logprobs:
            response_ids = record.raw_response['choices'][0]['token_ids']
        else:
            response_msgs = preprocess_messages_arguments_str2json([record.raw_response['choices'][0]['message']])
            prompt_messages = preprocess_messages_arguments_str2json(record.messages)
            messages = prompt_messages + response_msgs
            try:
                response_ids = tokenizer.apply_chat_template(
                    messages,
                    tools=record.raw_request["tools"] if "tools" in record.raw_request else None,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False
                )[len(prompt_ids):]
            except Exception as e:
                logger.error(f"====tokenizer.apply_chat_template exception: e:{e} messages={messages}")
                response_ids = []
        if len(response_ids) != len(logprobs) and len(logprobs) > 0:
            logger.warning(f"=====extract_token_ids_and_logprobs len(response_ids)[{len(response_ids)}] "
                f"!= len(logprobs)[{len(logprobs)}] len(prompt_ids)={len(prompt_ids)}")
    return logprobs, prompt_ids, response_ids

def get_action_from_assistant(record: RequestRecord):
    assistant_message = record.raw_response['choices'][0]["message"]
    if assistant_message is None:
        # or "tool_calls" not in assistant_message:
        return record.response_text or ""
    return assistant_message


def get_tool_outputs(record: RequestRecord, next_record: RequestRecord):
    if next_record is None:
        return []
    tool_outputs = []
    messages = next_record.raw_request['messages']
    for msg in reversed(messages):
        if msg["role"] != "tool":
            break
        tool_outputs.append(msg)
    return tool_outputs

def get_first_message(role: str, in_messages: list, is_reversed: bool = False):
    messages = reversed(in_messages) if is_reversed else in_messages
    for msg in messages:
        if msg["role"] == role:
            return msg["content"], msg
    return None, None

def default_traj_filter_func(task_id, task: dict, records: List[RequestRecord]) ->List[RequestRecord]:
    # 1. Filter out records with response text as '\n\nSAFE'
    filtered_safe_records = [record for record in records if record.response_text != "\n\nSAFE"]

    filtered_records = []
    for record in filtered_safe_records:
        # 2. Filter out empty raw_desponse
        if record.raw_response is None or "choices" not in record.raw_response:
            logger.info(f"{task_id=} filter record.raw_response is None")
            continue
        try:
            # 3. Filter out sub-agent records
            problem, _ = get_first_message("user", record.raw_request['messages'])
            task_problem = task["problem"]
            if problem != task_problem:
                logger.info(f"{task_id=} filter subagent record, problem={problem} task.problem={task_problem}")
                continue
            filtered_records.append(record)
        except Exception as e:
            logger.info(f"{task_id=} KeyError e={e}")
            continue

    # 4. Deduplicate records with same messages, keep the one with latest start_time
    seen_msg_idx = {}
    deduped_records = []
    for record in filtered_records:
        msg_key = str(record.messages)
        if msg_key not in seen_msg_idx:
            seen_msg_idx[msg_key] = len(deduped_records)
            deduped_records.append(record)
        else:
            deduped_records[seen_msg_idx[msg_key]] = record

    logger.info(f"{task_id=} filtered special/no raw_response records count:{len(filtered_records)}, remove rows:"
        f"{len(filtered_safe_records) - len(filtered_records)} remove dup:{len(filtered_records) - len(deduped_records)}")
    return deduped_records


def default_step_traj_refine_func(
        task_id: str,
        task: dict,
        records: List[RequestRecord],
        tokenizer=None,
        *args, **kwargs
) -> Episode:
    sorted_records = sorted(
        records,
        key=lambda r: r.start_time
    )

    trajectories: List[Trajectory] = []

    records_num = len(sorted_records)
    for i in range(records_num):
        record = sorted_records[i]
        next_record = sorted_records[i + 1] if (i + 1) < records_num else None
        matched_traj_idx = None

        for traj_idx, traj in enumerate(trajectories):
            if not traj.steps:
                continue

            last_step = traj.steps[-1]
            last_messages = last_step.chat_completions

            if (len(record.messages) >= len(last_messages) + 2 and
                    record.messages[:len(last_messages)] == last_messages):
                matched_traj_idx = traj_idx
                break

        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(record, tokenizer)
        action = get_action_from_assistant(record)
        tool_outputs = get_tool_outputs(record, next_record)

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
            trajectories[matched_traj_idx].steps.append(step)
        else:
            new_traj = Trajectory(steps=[step], task=task)
            trajectories.append(new_traj)

    episode = Episode(
        id=task_id,
        trajectories=trajectories,
    )
    episode.task = task

    episode.id = task_id
    logger.info(f"Refine completed: {get_episode_summary(episode)}")
    return episode

def concat_history_message(records: List[RequestRecord]) -> List[RequestRecord]:
    if not records:
        return []

    new_records = copy.deepcopy(records)

    cumulative_messages = list(new_records[0].messages)

    for i, record in enumerate(new_records):
        if i == 0:
            record.messages = copy.deepcopy(cumulative_messages)
        else:
            current_tool_messages = []
            for msg in reversed(record.messages):
                if msg["role"] == "tool":
                    current_tool_messages.append(msg)
                else:
                    break
            if current_tool_messages:
                cumulative_messages.extend(reversed(current_tool_messages))
                record.messages = copy.deepcopy(cumulative_messages)

        cumulative_messages.append(record.raw_response["choices"][0]["message"])
    return new_records


def default_token_traj_refine_func(
        task_id: str,
        task: dict,
        records: List[RequestRecord],
        tokenizer=None,
        *args, **kwargs
) -> Episode:
    sorted_records = sorted(records, key=lambda r: len(r.messages), reverse=False)

    max_model_len = kwargs.get("max_model_len", None)
    steps = []
    records_num = len(sorted_records)
    for i in range(records_num):
        record = sorted_records[i]
        next_record = sorted_records[i + 1] if (i + 1) < records_num else None
        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(record, tokenizer)
        tool_outputs = get_tool_outputs(record, next_record)
        action = get_action_from_assistant(record)

        if max_model_len is not None:
            curr_prompt_length = len(prompt_ids)
            curr_response_length = len(response_ids)
            if curr_prompt_length + curr_response_length >= max_model_len:
                dropped_steps = records_num - i - 1
                logger.warning(f"trajectory truncated, {task_id=}, step_idx:{i}, "
                               f"curr_prompt_length:{curr_prompt_length}, curr_response_length:{curr_response_length}, "
                               f"dropped_steps:{dropped_steps}, "
                               f"max_model_len:{max_model_len}")
                break

        step = Step(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=logprobs,
            reward=0.0,
            action=action,
            tool_outputs=tool_outputs,
        )
        steps.append(step)

    trajectory = Trajectory(
        steps=steps,
        reward=0.0,
        task=task,
    )

    episode = Episode(
        trajectories=[trajectory]
    )
    episode.id = task_id
    episode.task = task

    logger.info(f"Refine completed: {get_episode_summary(episode)}")
    return episode
