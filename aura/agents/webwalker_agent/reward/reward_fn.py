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
import re
from collections import Counter
from enum import Enum
import logging
import unicodedata
from agents.math_agent.reward.reward_types import RewardOutput
from agents.webwalker_agent.constants import WEBWALKER_PARSE_TOOL_ERROR
from agents.webwalker_agent.golden_path_utils import (
    get_target_golden_node,
    match_golden_click,
    normalize_button_label,
)
from .reward_config import WebWalkerConfig


logger = logging.getLogger(__name__)

DEFAULT_PREVIEW_LIMIT = 100
ACTION_PREVIEW_LIMIT = 200
STRUCTURED_SHORT_MAX_CHARS = 24
STRUCTURED_SHORT_MAX_TOKENS = 12
STRUCTURED_DATE_MIN_DIGITS = 2
SHORT_CODE_MAX_CHARS = 16
STRUCTURED_DIGIT_RATIO_MIN = 0.2


def _safe_preview(value, limit: int = DEFAULT_PREVIEW_LIMIT) -> str:
    """Return a type-safe, truncated preview for debug logging."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _match_current_golden_node(task_info: dict | None) -> str:
    if not isinstance(task_info, dict):
        return ""

    golden_path = task_info.get("golden_path") or []
    click_index = task_info.get("golden_step_index")
    if not isinstance(click_index, int):
        step_count = task_info.get("step_count")
        click_index = step_count - 1 if isinstance(step_count, int) else -1

    clicked_button = normalize_button_label(task_info.get("clicked_button", ""))
    is_match, target_node = match_golden_click(golden_path, clicked_button, click_index)
    if is_match:
        return target_node
    return get_target_golden_node(golden_path, click_index)


class WebWalkerRewardStage(Enum):
    TOOLS_FORMAT = "TOOLS_FORMAT"
    TOOLS_RETURN = "TOOLS_RETURN"
    DONE = "DONE"

class WebWalkerRewardFn:
    def __init__(self, config: WebWalkerConfig):
        self.config = config

    def __call__(self, action, stage, task_info: dict = None) -> RewardOutput:
        logger.debug("[WebWalkerRewardFn] action=%s", _safe_preview(action, ACTION_PREVIEW_LIMIT))
        logger.debug("[WebWalkerRewardFn] stage=%s", stage)
        logger.debug("[WebWalkerRewardFn] task_info=%s", _safe_preview(task_info, 200))

        reward = 0
        obs = ""
        metadata = {}
        logger.debug("[WebWalkerRewardFn] initial reward=%s obs=%s", reward, obs)

        if stage == WebWalkerRewardStage.TOOLS_FORMAT:
            logger.debug("[WebWalkerRewardFn] checking tool-call format")
            for idx, tool_call in enumerate(action):
                logger.debug("[WebWalkerRewardFn] tool_call[%s]=%s", idx, tool_call)
                if "function" in tool_call:
                    tool_name = tool_call["function"].get("name", "").strip()
                    tool_args = tool_call["function"].get("arguments", "")
                else:
                    tool_name = tool_call.get("name", "").strip()
                    tool_args = tool_call.get("arguments", "")
                logger.debug("[WebWalkerRewardFn] tool_name=%s tool_args=%s", tool_name, tool_args)

                is_format_ok = True
                if tool_name != "visit_page":
                    logger.warning("[WebWalkerRewardFn] invalid tool: %s", tool_name)
                    is_format_ok, obs = False, f"Invalid tool: {tool_name}"
                elif not isinstance(tool_args, dict) or "button" not in tool_args:
                    logger.warning("[WebWalkerRewardFn] invalid tool arguments")
                    is_format_ok, obs = False, "Missing 'button' in arguments"

                if not is_format_ok:
                    obs = obs if obs else "Format error for WebWalker function call."
                    metadata["tool_format_error"] = True
                    logger.debug("[WebWalkerRewardFn] reward=%s obs=%s", reward, obs)

        elif stage == WebWalkerRewardStage.TOOLS_RETURN:
            logger.debug("[WebWalkerRewardFn] evaluating tool return")
            tool_returns = action.get("tool_outputs", {}).values()
            reward_mode = str((task_info or {}).get("reward_mode", "step")).strip().lower()

            click_success = False
            for res in tool_returns:
                res_text = res if isinstance(res, str) else str(res)
                logger.debug("[WebWalkerRewardFn] tool return: %s", _safe_preview(res_text, 100))
                obs = res_text.strip()

                if WEBWALKER_PARSE_TOOL_ERROR in res_text:
                    logger.debug("[WebWalkerRewardFn] tool parsing error; no return-stage reward")
                elif "can not be clicked" in res_text:
                    logger.debug("[WebWalkerRewardFn] unclickable button; no return-stage reward")
                elif "information of the current page is not accessible" in res_text:
                    logger.debug("[WebWalkerRewardFn] inaccessible page information; no return-stage reward")
                elif "Error accessing page" in res_text or "Error: " in res_text:
                    logger.debug("[WebWalkerRewardFn] page access error; no return-stage reward")
                else:
                    logger.debug("[WebWalkerRewardFn] navigation succeeded; checking golden/source reward")
                    click_success = True

            progress_match = (task_info or {}).get("golden_progress_match", {})
            golden_node = ""
            if isinstance(progress_match, dict) and progress_match.get("matched"):
                golden_node = str(progress_match.get("node") or "")
            source_url_hit = bool((task_info or {}).get("source_url_hit", False))
            if reward_mode == "step" and click_success and golden_node:
                reward += self.config.explore_reward_pos
                logger.info(
                    "[WebWalkerRewardFn] step=%s matched golden node %r: +%s",
                    task_info.get("step_count"),
                    golden_node,
                    self.config.explore_reward_pos,
                )
            elif reward_mode == "step" and click_success and source_url_hit:
                reward += self.config.explore_reward_pos
                logger.info(
                    "[WebWalkerRewardFn] step=%s matched source website: +%s",
                    task_info.get("step_count"),
                    self.config.explore_reward_pos,
                )

            logger.debug("[WebWalkerRewardFn] reward=%s obs=%s", reward, _safe_preview(obs, 100))

        else:
            logger.debug("[WebWalkerRewardFn] done stage has no final-answer reward")
            obs = "DONE_NO_REWARD"
            logger.debug("[WebWalkerRewardFn] reward=%s obs=%s", reward, obs)

        metadata["reward_obs"] = obs
        reward_output = RewardOutput(reward=reward, metadata=metadata)
        logger.debug("[WebWalkerRewardFn] reward output: %s", reward_output)
        return reward_output

def webwalker_reward_fn(action, stage, task_info=None):
    from .reward_config import get_webwalker_reward_config

    reward_fn = WebWalkerRewardFn(get_webwalker_reward_config())
    return reward_fn(action, stage, task_info)

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize mixed Chinese/English text for token-level matching."""
    normalized_chars = []
    for char in str(text).lower():
        category = unicodedata.category(char)
        if category.startswith(("P", "S")):
            normalized_chars.append(" ")
        else:
            normalized_chars.append(char)
    normalized_text = "".join(normalized_chars)
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    return normalized_text.strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize Chinese as characters and Latin text as word pieces."""
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []
    return _TOKEN_PATTERN.findall(normalized_text)


def is_structured_short_answer(text: str) -> bool:
    """Prefer exact match for dates, IDs, abbreviations, and other short structured answers."""
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False

    tokens = tokenize_text(normalized_text)
    if not tokens:
        return False

    compact_text = normalized_text.replace(" ", "")
    if len(compact_text) > STRUCTURED_SHORT_MAX_CHARS or len(tokens) > STRUCTURED_SHORT_MAX_TOKENS:
        return False

    # Obvious date-like / number-heavy answers such as "2003年7月", "2023-07-06", "第504研究所".
    has_many_digits = sum(char.isdigit() for char in compact_text) >= STRUCTURED_DATE_MIN_DIGITS
    has_date_markers = any(marker in compact_text for marker in ("年", "月", "日", "号"))
    if has_many_digits and has_date_markers:
        return True

    # Short abbreviations / codes such as C2PA, GPT-4, ISBN-like fragments.
    if re.fullmatch(r"[a-z0-9]+", compact_text, re.IGNORECASE):
        return len(compact_text) <= SHORT_CODE_MAX_CHARS

    # Short mixed structured spans with strong numeric/code patterns should be matched exactly.
    if re.search(r"[a-z]", compact_text, re.IGNORECASE):
        digit_ratio = sum(char.isdigit() for char in compact_text) / max(len(compact_text), 1)
        if digit_ratio >= STRUCTURED_DIGIT_RATIO_MIN:
            return True

    if compact_text.startswith("第") and has_many_digits:
        return True

    return False


def f1_score(model_response: str, ground_truth: str):
    pred_tokens = tokenize_text(model_response)
    gt_tokens = tokenize_text(ground_truth)

    if not gt_tokens or not pred_tokens:
        return 0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    common = pred_counter & gt_counter
    common_count = sum(common.values())

    if common_count == 0:
        return 0

    precision = common_count / sum(pred_counter.values())
    recall = common_count / sum(gt_counter.values())

    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def answer_match_score(model_response: str, ground_truth: str) -> float:
    """Use exact match for structured short answers and F1 for free-form sentences."""
    normalized_pred = normalize_text(model_response)
    normalized_gt = normalize_text(ground_truth)

    if not normalized_pred or not normalized_gt:
        return 0

    if is_structured_short_answer(normalized_pred) or is_structured_short_answer(normalized_gt):
        return 1.0 if normalized_pred == normalized_gt else 0.0

    return f1_score(normalized_pred, normalized_gt)
