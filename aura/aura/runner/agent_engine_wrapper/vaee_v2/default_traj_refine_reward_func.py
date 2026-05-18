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

import json
import logging
from typing import List, Optional, Any

from aura.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
from aura.runner.agent_engine_wrapper.vaee_v2.vaee_types import Trajectory, Episode, RequestRecord, Step

logger = logging.getLogger(__name__)


def get_episode_summary(episode: Episode):
    summary = {
        "task_id": episode.id,
        "trajectories": [
            {"traj_id": i, "step_count": len(traj.steps), "steps": [s.id for s in traj.steps]}
            for i, traj in enumerate(episode.trajectories)
        ],
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


def get_episode_data(episode_dict):
    def truncate_data(data, max_str_len=100, max_list_len=10):
        """
        Recursively traverse dictionary/list, truncate long strings
        """
        if isinstance(data, dict):
            return {k: truncate_data(v, max_str_len) for k, v in data.items()}
        elif isinstance(data, list):
            return [truncate_data(i, max_str_len) for i in data[:max_list_len]]
        elif isinstance(data, str):
            if len(data) > max_str_len:
                return data[:max_str_len] + "..."
            return data
        else:
            return data

    return json.dumps(truncate_data(episode_dict), indent=4, ensure_ascii=False)


def extract_token_ids_and_logprobs(record: RequestRecord, tokenizer) -> tuple[list[int], list[int], Any]:
    if record.token_ids and record.response_ids:
        # TITO mode
        prompt_ids = record.token_ids
        response_ids = record.response_ids
        logprobs = record.token_response['choices'][0]['logprobs']['token_logprobs']
    else:
        # not TITO mode
        prompt_ids = tokenizer.apply_chat_template(record.messages, tokenize=True, add_generation_prompt=True)
        response_ids = tokenizer.apply_chat_template(
            record.messages + [record.raw_response['choices'][0]['message']], tokenize=True, add_generation_prompt=False
        )[len(prompt_ids) :]
        logprobs = []
    return logprobs, prompt_ids, response_ids


def default_step_traj_refine_func(
    task_id: str,
    task: AgentTask,
    records: List[RequestRecord],
    tokenizer=None,
) -> Episode:
    """
    Trajectory Filtering and Stitching in Step Mode

    Description: Use all inference requests to construct an Episode.
    Applicable:
        1. Multiple Steps within an Agent/SubAgent are executed linearly, i.e., a complete multi-turn conversation.
        2. A complete multi-turn conversation is considered a single Trajectory.
        3. In the case of conversation compression, it is regarded as a new Trajectory.
    Processing:
        1. Sort all inference requests by their starting timestamp.
        2. Starting from the first inference request, match the internal messages field one by one.
            2.1 If the messages fully contain the messages of the last Step in any existing Trajectory and
                the length of the messages list is 2 more than that Step's messages, treat it as an inference
                request belonging to that Trajectory and append it as a new Step.
            2.2 If the messages cannot match the messages of the last Step in any existing Trajectory, treat it as
                the first Step of a new Trajectory.
        3. Multiple Trajectories form an Episode.

        x. Additional processing details
        x.1 The request_id in the RequestRecord is used as the Step id.
        x.2 The session_id in the RequestRecord is used as the Episode id.
        x.3 The id of a Trajectory naturally increments from 0 -> N based on the above splitting.
    """
    sorted_records = sorted(records, key=lambda r: r.start_time)

    trajectories: List[Trajectory] = []

    for record in sorted_records:
        matched_traj_idx = None

        # Try to match the current messages with the last step of the existing Trajectory
        for traj_idx, traj in enumerate(trajectories):
            if not traj.steps:
                continue

            last_step = traj.steps[-1]
            last_messages = last_step.chat_completions

            # 2.1 Check whether messages have the previous Step's chat_completions as a prefix
            # and whether messages are longer than it by 2 (user message and assistant reply)
            if (
                len(record.messages) >= len(last_messages) + 2
                and record.messages[: len(last_messages)] == last_messages
            ):
                matched_traj_idx = traj_idx
                break

        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(record, tokenizer)

        step = Step(
            id=record.request_id,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=logprobs,
            reward=0.0,
            chat_completions=record.messages,
            model_response=record.response_text or "",
        )

        if matched_traj_idx is not None:
            trajectories[matched_traj_idx].steps.append(step)
        else:
            new_traj = Trajectory(steps=[step])
            trajectories.append(new_traj)

    episode = Episode(
        id=task_id,
        trajectories=trajectories,
    )
    episode.task = task

    episode.id = task_id
    logger.info(f"Refine completed: {get_episode_summary(episode)}")
    print(f"Refine completed: {get_episode_summary(episode)}")
    return episode


def default_token_traj_refine_func(
    task_id: str,
    task: AgentTask,
    records: List[RequestRecord],
    tokenizer=None,
) -> Episode:
    sorted_records = sorted(records, key=lambda r: len(r.messages), reverse=False)

    steps = []
    for record in sorted_records:
        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(record, tokenizer)
        step = Step(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=logprobs,
            reward=0.0,
        )
        steps.append(step)

    trajectory = Trajectory(
        steps=steps,
        reward=0.0,
    )

    episode = Episode(trajectories=[trajectory])
    episode.id = task_id
    episode.task = task

    logger.info(f"Refine completed: {get_episode_summary(episode)}")
    print(f"Refine completed: {get_episode_summary(episode)}")
    return episode


def default_traj_reward_func(episode: Episode, answer: Optional[str] = None) -> Episode:
    logger.info(f"Reward completed: {get_episode_data(episode.to_dict())}, answer: {answer}")
    print(f"Reward completed: {get_episode_data(episode.to_dict())}, answer: {answer}")
    return episode
