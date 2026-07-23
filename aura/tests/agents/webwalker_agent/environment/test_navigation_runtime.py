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


def test_safe_asyncio_run_returns_coroutine_result():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.environment.runtime import safe_asyncio_run

    async def _value():
        return "ok"

    assert safe_asyncio_run(_value()) == "ok"


def test_navigation_helpers_extract_actions_and_source_hits():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.environment.navigation import WebWalkerNavigationMixin

    class DummyEnv(WebWalkerNavigationMixin):
        pass

    env = DummyEnv()
    env.task = {
        "source_website": ["https://example.com/final/"],
        "reward_mode": "trajectory",
    }
    env.step_count = 1
    env.golden_path = [["Products", "Pricing"]]
    env.golden_progress_indices = [0]
    env.stop_mode = "golden_path_horizon_or_finish"
    env.clicked_buttons = ["Products"]
    env.reached_answer_page = False

    actions = [{"name": "visit_page", "arguments": {"button": "<button>Products</button>"}}]

    assert env._extract_clicked_button(actions) == "Products"
    assert env._is_source_url_hit("https://example.com/final#section")
    assert env._is_chain_click_prefix_valid()
    assert env._get_chain_click_horizon() == 2
