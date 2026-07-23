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
import types
from pathlib import Path
from unittest.mock import MagicMock


def _ensure_aura_src_on_path():
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "agents").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)
    sys.modules.setdefault(
        "torch",
        types.SimpleNamespace(distributed=types.SimpleNamespace(is_initialized=lambda: False)),
    )
    sys.modules.setdefault(
        "torch.distributed",
        types.SimpleNamespace(
            is_initialized=lambda: False,
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
        ),
    )


def _mock_openai(monkeypatch):
    openai = types.ModuleType("openai")
    openai.OpenAI = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", openai)
    return openai.OpenAI


def test_webwalker_environment_init_and_reset(monkeypatch):
    _ensure_aura_src_on_path()
    openai_cls = _mock_openai(monkeypatch)
    from agents.webwalker_agent.environment.webwalker_env import WebWalkerEnvironment

    env = WebWalkerEnvironment(
        task={
            "root_url": "https://example.com",
            "question": "Where is pricing?",
            "golden_path": [["root", "Products"]],
            "reward_mode": "trajectory",
        },
        cache_mode="off",
        enable_critic_early_stop=False,
        chat_model_max_tokens=16,
    )
    env._fetch_page_success_only = MagicMock(return_value=("<a href='/products'>Products</a>", "home"))
    env.extract_links_with_text = MagicMock(return_value="<button>Products</button>")
    env._observation_information_extraction = MagicMock(return_value=(None, {"stage": "info"}))

    obs, info = env.reset()

    openai_cls.assert_called_once()
    assert env.root_url == "https://example.com"
    assert env.golden_path == [["Products"]]
    assert "home" in obs["initial_observation"]
    assert info["metadata"]["critic_traces"] == [{"stage": "info"}]


def test_execute_tool_visit_page_uses_button_url(monkeypatch):
    _ensure_aura_src_on_path()
    _mock_openai(monkeypatch)
    from agents.webwalker_agent.environment.webwalker_env import WebWalkerEnvironment

    env = WebWalkerEnvironment(task={}, cache_mode="off", enable_critic_early_stop=False)
    env.button_url_dict = {"Products": "https://example.com/products"}
    env._fetch_page_success_only = MagicMock(return_value=("<html></html>", "product page"))
    env.extract_links_with_text = MagicMock(return_value="<button>Pricing</button>")

    tool_id, result = env.execute_tool(
        {"id": "call-1", "name": "visit_page", "arguments": {"button": "<button>Products</button>"}}
    )

    assert tool_id == "call-1"
    assert "product page" in result
    assert env.current_page_url == "https://example.com/products"


def test_step_reward_task_info_does_not_mutate_original_task(monkeypatch):
    _ensure_aura_src_on_path()
    _mock_openai(monkeypatch)
    from agents.webwalker_agent.environment.webwalker_env import WebWalkerEnvironment

    captured_task_infos = []

    def reward_fn(action, stage, task_info):
        captured_task_infos.append(task_info)
        return types.SimpleNamespace(reward=0.0, metadata={})

    original_task = {
        "root_url": "https://example.com",
        "question": "Where is pricing?",
        "golden_path": [["root", "Products"]],
    }
    env = WebWalkerEnvironment(
        task=original_task,
        reward_fn=reward_fn,
        cache_mode="off",
        enable_critic_early_stop=False,
        max_steps=3,
    )
    env.golden_path = [["Products"]]
    env.current_page_url = "https://example.com"
    env.button_url_dict = {"Products": "https://example.com/products"}
    env._execute_tool_calls = MagicMock(return_value={"call-1": "opened page"})
    env._observation_information_extraction = MagicMock(return_value=(None, {"stage": "info"}))

    env.step({"name": "visit_page", "arguments": {"button": "Products"}})

    assert captured_task_infos
    assert captured_task_infos[0]["step_count"] == 1
    assert "step_count" not in env.task
    assert "clicked_button" not in env.task
    assert "source_url_hit" not in env.task
    assert original_task == {
        "root_url": "https://example.com",
        "question": "Where is pricing?",
        "golden_path": [["root", "Products"]],
    }
