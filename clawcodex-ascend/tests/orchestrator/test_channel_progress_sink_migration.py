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

# ruff: noqa: UP009

"""Focused migration tests for the channel progress sink."""

from __future__ import annotations

from extensions.orchestrator.channel_sink import ChannelProgressSink
from extensions.orchestrator.events import EventLevel, OrchestratorEvent


def _event(issue_id: str = "I1") -> OrchestratorEvent:
    return OrchestratorEvent("issue.started", issue_id, EventLevel.INFO, "started")


def test_channel_sink_delivers_formatted_text() -> None:
    delivered: list[tuple[str, str]] = []
    sink = ChannelProgressSink(lambda event, text: delivered.append((event.event_type, text)))

    sink(_event())

    assert delivered[0][0] == "issue.started"
    assert delivered[0][1]


def test_delivery_failure_is_isolated_and_redacted(caplog) -> None:
    def _fail(_event, _text) -> None:
        raise RuntimeError("private-chat-payload")

    sink = ChannelProgressSink(_fail)

    with caplog.at_level("WARNING", logger="extensions.orchestrator.channel_sink"):
        sink(_event())

    assert "RuntimeError" in caplog.text
    assert "private-chat-payload" not in caplog.text
