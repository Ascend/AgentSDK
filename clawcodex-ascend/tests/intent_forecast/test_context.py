#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import json

from src.agent.conversation import Conversation
from src.types.messages import Message

from clawcodex_ext.intent_forecast.config import IntentForecastConfig
from clawcodex_ext.intent_forecast.context import (
    build_user_intent,
    IntentForecastContextBuilder,
    infer_response_language,
)


def test_context_reads_memory_workspace_and_current_messages(tmp_path) -> None:
    (tmp_path / "CLAUDE.md").write_text("memory note", encoding="utf-8")
    conv = Conversation()
    conv.messages = [Message(role="user", content="continue forecast")]

    context = IntentForecastContextBuilder(
        conversation=conv,
        workspace_root=tmp_path,
        config=IntentForecastConfig(),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    assert context.current_messages[-1]["content"] == "continue forecast"
    assert context.user_intent["initial_user_input"] == "continue forecast"
    assert context.user_intent["latest_user_input"] == "continue forecast"
    assert context.intent_strategy == "user"
    assert context.response_language == "English"
    assert context.intent_stage in {"explore", "implement"}
    assert "active_goal" in context.task_state
    assert context.memory_files[0]["path"] == "CLAUDE.md"
    assert "permission_mode" in context.workspace
    assert "changed_files" in context.workspace
    assert "git_branch" in context.workspace
    assert "git_diff_names" in context.workspace
    assert "README.md" not in context.workspace["project_files"]


def test_context_falls_back_to_transcript_tail_and_enqueues_summary(tmp_path, monkeypatch) -> None:
    enqueued: list[str] = []
    monkeypatch.setattr(
        "clawcodex_ext.intent_forecast.context.enqueue_summary_job",
        lambda session_id, **kwargs: enqueued.append(session_id),
    )
    sessions = tmp_path / "sessions"
    session_dir = sessions / "s1"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(
        json.dumps({"session_id": "s1", "last_updated": 10, "title": "T"}),
        encoding="utf-8",
    )
    (session_dir / "transcript.jsonl").write_text(
        '{"role":"user","content":"tail task"}\n',
        encoding="utf-8",
    )

    context = IntentForecastContextBuilder(
        conversation=None,
        workspace_root=tmp_path,
        config=IntentForecastConfig(summary_lazy_generate=True),
        sessions_dir=sessions,
        feedback_base_dir=tmp_path,
    ).build()

    assert context.sessions[0]["transcript_tail"][0]["content"] == "tail task"
    assert enqueued == ["s1"]


def test_context_infers_chinese_from_recent_user_messages(tmp_path) -> None:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="please inspect this repo"),
        Message(role="assistant", content="ok"),
        Message(
            role="user",
            content="\u7ee7\u7eed\u8865\u9f50\u4e0a\u8ff0\u4ea4\u4e92\u548c\u540e\u53f0\u4fa7\u8f66\u80fd\u529b",
        ),
    ]

    context = IntentForecastContextBuilder(
        conversation=conv,
        workspace_root=tmp_path,
        config=IntentForecastConfig(),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    assert context.response_language == "Chinese"


def test_infer_language_falls_back_to_session_tail() -> None:
    assert (
        infer_response_language(
            [],
            [
                {
                    "transcript_tail": [
                        {
                            "role": "user",
                            "content": "\u7ee7\u7eed\u5b8c\u6210\u6d4b\u8bd5\u8986\u76d6",
                        }
                    ]
                }
            ],
        )
        == "Chinese"
    )


def test_infer_language_reads_session_summary_candidates() -> None:
    assert (
        infer_response_language(
            [],
            [
                {
                    "title": "Mostly English title",
                    "summary": {
                        "next_action_candidates": [
                            "\u7ee7\u7eed\u5b8c\u6210\u610f\u56fe\u9884\u6d4b\u7684\u4ea4\u4e92\u4fee\u590d"
                        ]
                    },
                }
            ],
        )
        == "Chinese"
    )


def test_context_respects_configured_response_language(tmp_path) -> None:
    conv = Conversation()
    conv.messages = [Message(role="user", content="finish tests")]

    context = IntentForecastContextBuilder(
        conversation=conv,
        workspace_root=tmp_path,
        config=IntentForecastConfig(response_language="Chinese"),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    assert context.response_language == "Chinese"


def test_context_truncates_assistant_more_than_user_and_preserves_user_intent(tmp_path) -> None:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="请先写文档，不要实现代码"),
        Message(role="assistant", content="A" * 2000),
    ]

    context = IntentForecastContextBuilder(
        conversation=conv,
        workspace_root=tmp_path,
        config=IntentForecastConfig(),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    assistant = [msg for msg in context.current_messages if msg["role"] == "assistant"][0]
    assert len(assistant["content"]) == 360
    assert context.user_intent["latest_user_input"] == "请先写文档，不要实现代码"
    assert context.intent_stage == "document"


def test_build_user_intent_tracks_initial_latest_and_preferences() -> None:
    intent = build_user_intent(
        [
            {"role": "user", "content": "先分析架构"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "不要实现，改成写文档"},
        ]
    )

    assert intent["initial_user_input"] == "先分析架构"
    assert intent["latest_user_input"] == "不要实现，改成写文档"
    assert intent["previous_user_inputs"] == ["先分析架构"]
    assert intent["explicit_preferences"] == ["不要实现，改成写文档"]


def test_context_carries_configured_intent_strategy(tmp_path) -> None:
    conv = Conversation()
    conv.messages = [Message(role="user", content="finish tests")]

    context = IntentForecastContextBuilder(
        conversation=conv,
        workspace_root=tmp_path,
        config=IntentForecastConfig(intent_strategy="workspace"),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    assert context.intent_strategy == "workspace"
    assert context.to_prompt_dict()["intent_strategy"] == "workspace"
