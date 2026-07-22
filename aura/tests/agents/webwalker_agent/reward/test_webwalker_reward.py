#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

        http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import sys
from pathlib import Path


def _ensure_aura_src_on_path():
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "agents").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)


def test_golden_path_parsing_and_progress_matching():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.golden_path_utils import (
        extract_golden_click_paths_from_task,
        is_exact_golden_click_path,
        is_golden_click_path_prefix,
        match_golden_progress,
    )

    task = {"info": {"golden_path": ["root -> <button>Products</button> -> Pricing"]}}
    paths = extract_golden_click_paths_from_task(task)

    assert paths == [["Products", "Pricing"]]
    assert is_golden_click_path_prefix(paths, ["Products"])
    assert is_exact_golden_click_path(paths, ["Products", "Pricing"])

    match = match_golden_progress(paths, "Pricing", [1])

    assert match["matched"]
    assert match["is_current"]
    assert match["is_final"]
    assert match["advance_to"] == 2


def test_golden_path_parsing_keeps_arrow_inside_list_labels():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.golden_path_utils import parse_golden_click_path

    assert parse_golden_click_path(["root", "A->B", "Pricing"]) == ["A->B", "Pricing"]
    assert parse_golden_click_path('["root", "A->B", "Pricing"]') == ["A->B", "Pricing"]


def test_webwalker_reward_step_success_and_format_error():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.reward.reward_config import WebWalkerConfig
    from agents.webwalker_agent.reward.reward_fn import WebWalkerRewardFn, WebWalkerRewardStage

    config = WebWalkerConfig()
    config.explore_reward_pos = 2.0
    reward_fn = WebWalkerRewardFn(config)

    success = reward_fn(
        {"tool_outputs": {"0": "opened page"}},
        WebWalkerRewardStage.TOOLS_RETURN,
        {"reward_mode": "step", "golden_progress_match": {"matched": True, "node": "Products"}},
    )
    bad_format = reward_fn(
        [{"name": "finish", "arguments": {"response": "done"}}],
        WebWalkerRewardStage.TOOLS_FORMAT,
        {},
    )

    assert success.reward == 2.0
    assert bad_format.reward == 0
    assert bad_format.metadata["tool_format_error"]


def test_webwalker_reward_parse_error_gets_no_step_reward():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.constants import WEBWALKER_PARSE_TOOL_ERROR
    from agents.webwalker_agent.reward.reward_config import WebWalkerConfig
    from agents.webwalker_agent.reward.reward_fn import WebWalkerRewardFn, WebWalkerRewardStage

    config = WebWalkerConfig()
    config.explore_reward_pos = 2.0
    reward_fn = WebWalkerRewardFn(config)

    reward = reward_fn(
        {"tool_outputs": {"0": WEBWALKER_PARSE_TOOL_ERROR}},
        WebWalkerRewardStage.TOOLS_RETURN,
        {"reward_mode": "step", "golden_progress_match": {"matched": True, "node": "Products"}},
    )

    assert reward.reward == 0


def test_answer_match_prefers_exact_for_structured_short_answers():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.reward.reward_fn import answer_match_score

    assert answer_match_score("C2PA", "c2pa") == 1.0
    assert answer_match_score("C2PA", "C2PA draft") == 0.0
    assert 0 < answer_match_score(
        "alpha beta product overview for release",
        "alpha gamma product overview for release",
    ) < 1
