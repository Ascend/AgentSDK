#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
import os
import string
import random
from rllm.rewards.reward_types import RewardOutput, RewardConfig
from aura.runner.agent_engine_wrapper.base.agent.base_agent import Trajectory

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    print(f"normalized_prediction: {normalized_prediction}")
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    print(f"subem_check score:{score}")
    return score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        logger.info("--------------------------------")
        logger.info(f"Golden answers: {ground_truth}")
        logger.info(f"Extracted answer: {answer}")
        logger.info(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth):
            return score
        else:
            return format_score


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0.0, score=1.0):
    answer = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1
    if do_print:
        logger.info("--------------------------------")
        logger.info(f"Golden answers: {ground_truth}")
        logger.info(f"Extracted answer: {answer}")
        logger.info(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth):
            return score
        else:
            return format_score


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    print(f"normalized_prediction: {normalized_prediction}")
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


class SearchR1ResRewardFn:
    def __init__(self, config: RewardConfig):
        self.config = config
        self.compute_func = os.getenv("COMPUTE_REWARD_FUNC", "subem")
        self.SCORE_METHODS = {
            'em': compute_score_em,
            'subem': compute_score_subem,
        }

    def __call__(self, action, task_info: dict = None) -> RewardOutput:
        # Extract information from task_info and action
        model_response = action
        ground_truth = task_info.get("task", {}).get("ground_truth")
        print(f"SearchR1ResRewardFn model_response:{model_response}")
        print(f"SearchR1ResRewardFn ground_truth:{ground_truth}")

        if ground_truth is None:
            return RewardOutput(
                reward=self.config.unk_error_reward, is_correct=False, metadata={"error": "No ground truth provided"}
            )
        logger.info(f"model response: {model_response}, ground truth: {ground_truth}")
        reward = self.SCORE_METHODS[self.compute_func](model_response, ground_truth)
        is_correct = True if reward == 1.0 else False
        return RewardOutput(reward=reward, is_correct=is_correct)


def compute_search_r1_trajectory_reward(trajectory: Trajectory) -> Trajectory:
    """
    Add trajectory reward to the dict of each interaction.

    Args:
        trajectory: List of dictionaries representing each step in the trajectory.

    Returns:
        The updated trajectory with trajectory_reward added to each step.
    """
    if not trajectory:
        return trajectory
    res_rewards = [d.reward for d in trajectory.steps if d.done]
    if res_rewards:
        res_reward = res_rewards[-1]
    else:
        res_reward = 0
    trajectory.res_reward = res_reward
    trajectory.reward = res_reward
    return trajectory
