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
from types import SimpleNamespace

import pytest


def _ensure_aura_src_on_path():
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "agents").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)


def _dummy_critic():
    _ensure_aura_src_on_path()
    sys.modules.setdefault(
        "torch",
        SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
    )
    sys.modules.setdefault(
        "torch.distributed",
        SimpleNamespace(
            is_initialized=lambda: False,
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
        ),
    )
    from agents.webwalker_agent.environment.critic import WebWalkerCriticMixin

    class DummyCritic(WebWalkerCriticMixin):
        pass

    env = DummyCritic()
    env.enable_critic_early_stop = True
    env.model = "critic-model"
    env.chat_model_temperature = 0.1
    env.chat_model_top_p = 0.9
    env.chat_model_max_tokens = 128
    env.tokenizer = None
    env.critic_failure_dump_dir = ""
    env.api_url = "http://critic"
    env.root_url = "http://example.com"
    env.current_page_url = "http://example.com/page"
    env.fail_on_critic_error = False
    env.webwalker_memory = []
    return env


def test_parse_critic_json_accepts_plain_fenced_and_embedded_json():
    env = _dummy_critic()

    assert env._parse_critic_json('{"judge": true, "answer": "ok"}') == {"judge": True, "answer": "ok"}
    assert env._parse_critic_json('```json\n{"usefulness": false}\n```') == {"usefulness": False}
    assert env._parse_critic_json('prefix {"usefulness": true, "information": "fact"} suffix') == {
        "usefulness": True,
        "information": "fact",
    }

    with pytest.raises(ValueError):
        env._parse_critic_json("not json")


def test_critic_metadata_trace_and_request_kwargs_are_stable():
    env = _dummy_critic()
    messages = [{"role": "user", "content": "hello"}]

    metadata = env._empty_critic_metadata()
    trace = env._build_critic_trace(stage="answer", messages=messages, result="done")
    request_kwargs = env._critic_request_kwargs(messages)

    assert metadata["critic_early_stop_enabled"] is True
    assert metadata["critic_traces"] == []
    assert trace["model"] == "critic-model"
    assert trace["result"] == "done"
    assert request_kwargs["messages"] == messages
    assert request_kwargs["max_tokens"] == 128


def test_critic_information_extraction_imports_prompt_and_parses_response():
    env = _dummy_critic()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"usefulness": true, "information": "found fact"}'
                )
            )
        ]
    )
    env.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )

    information, trace = env._observation_information_extraction("query", "observation")

    assert information == "found fact"
    assert trace["result"] == "found fact"


def test_critic_answer_generation_imports_prompt_and_parses_response():
    env = _dummy_critic()
    env.webwalker_memory = ["found fact"]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"judge": true, "answer": "final answer"}'
                )
            )
        ]
    )
    env.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )

    answer, trace = env._critic_information("query")

    assert answer == "final answer"
    assert trace["result"] == "final answer"
