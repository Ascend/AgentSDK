#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# ruff: noqa: UP009

"""Focused migration tests for session-scoped Feishu activity cards."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from clawcodex_ext.services.channels.capabilities import (
    ChannelCapability,
    ChannelCapabilitySet,
    InboundActivityContext,
)
from extensions.api.query import PhaseComplete, SessionComplete, TurnComplete
from extensions.orchestrator.feishu_activity_sink import FeishuActivitySink
from extensions.orchestrator.status_dashboard import SessionStatus


class _Adapter:
    channel_id = "feishu"
    capabilities = ChannelCapabilitySet.of(ChannelCapability.CARD_UPDATE)

    def __init__(self) -> None:
        self.context = InboundActivityContext(message_id="om_1", chat_id="oc_original")
        self.calls: list[tuple[str, str, dict]] = []

    def last_inbound_context(self):
        return self.context

    async def send_placeholder_card(self, chat_id: str, card: dict) -> str:
        self.calls.append(("send", chat_id, card))
        return "om_placeholder"

    async def update_progress_card(self, message_id: str, card: dict) -> bool:
        self.calls.append(("update", message_id, card))
        return True


class _Dashboard:
    def __init__(self) -> None:
        self.listeners = []
        self.running: dict[str, SessionStatus] = {}

    def add_session_start_listener(self, listener):
        self.listeners.append(listener)

        def _remove() -> None:
            if listener in self.listeners:
                self.listeners.remove(listener)

        return _remove

    def state(self):
        return SimpleNamespace(running=self.running)

    def on_session_start(self, status: SessionStatus) -> None:
        self.running[status.issue_id] = status
        for listener in list(self.listeners):
            listener(status)


def _session():
    return SimpleNamespace(issue=SimpleNamespace(identifier="ISSUE-1"))


async def _drain(sink: FeishuActivitySink) -> None:
    while sink._pending_tasks:
        await asyncio.gather(*tuple(sink._pending_tasks), return_exceptions=True)
        await asyncio.sleep(0)


async def _started_sink(
    adapter: _Adapter,
    *,
    task_id: str = "ISSUE-1",
    **kwargs,
) -> tuple[FeishuActivitySink, _Dashboard]:
    dashboard = _Dashboard()
    dashboard.on_session_start(SessionStatus(issue_id=task_id, issue_identifier=f"{task_id} title"))
    sink = FeishuActivitySink(
        task_id=task_id,
        feishu_adapter=adapter,
        status_dashboard=dashboard,
        **kwargs,
    )
    await _drain(sink)
    return sink, dashboard


@pytest.mark.asyncio
async def test_late_registration_replays_running_session() -> None:
    adapter = _Adapter()

    sink, dashboard = await _started_sink(adapter, phases_total=4)

    assert adapter.calls[0][0:2] == ("send", "oc_original")
    assert "ISSUE-1 title" in adapter.calls[0][2]["elements"][0]["text"]["content"]
    assert dashboard.listeners == []
    assert sink._session_started is True


@pytest.mark.asyncio
async def test_listener_filters_other_tasks_then_detaches_on_match() -> None:
    adapter = _Adapter()
    dashboard = _Dashboard()
    sink = FeishuActivitySink(
        task_id="ISSUE-1",
        feishu_adapter=adapter,
        status_dashboard=dashboard,
    )

    dashboard.on_session_start(SessionStatus(issue_id="other", issue_identifier="other"))
    assert adapter.calls == []
    assert len(dashboard.listeners) == 1

    dashboard.on_session_start(SessionStatus(issue_id="ISSUE-1", issue_identifier="match"))
    await _drain(sink)

    assert adapter.calls[0][0] == "send"
    assert dashboard.listeners == []


@pytest.mark.asyncio
async def test_destination_is_frozen_before_matching_start() -> None:
    adapter = _Adapter()
    dashboard = _Dashboard()
    sink = FeishuActivitySink(
        task_id="ISSUE-1",
        feishu_adapter=adapter,
        status_dashboard=dashboard,
    )
    adapter.context = InboundActivityContext(message_id="om_2", chat_id="oc_newer")

    dashboard.on_session_start(SessionStatus(issue_id="ISSUE-1", issue_identifier="match"))
    await _drain(sink)

    assert adapter.calls[0][0:2] == ("send", "oc_original")


@pytest.mark.asyncio
async def test_progress_and_terminal_cards_update_placeholder() -> None:
    adapter = _Adapter()
    sink, dashboard = await _started_sink(adapter, phases_total=4)

    sink.on_phase_complete(PhaseComplete(phase=2, turn_count=2), _session())
    await _drain(sink)
    sink.on_session_complete(SessionComplete(reason="success"), _session())
    await _drain(sink)

    updates = [call for call in adapter.calls if call[0] == "update"]
    progress = next(element for element in updates[0][2]["elements"] if element.get("tag") == "progress")
    assert progress["percent"] == 50
    assert "Completed" in updates[1][2]["header"]["title"]["content"]
    assert sink._placeholder_message_id is None
    assert sink._inbound_context is None
    assert dashboard.listeners == []


def test_no_running_loop_drops_replayed_operation() -> None:
    adapter = _Adapter()
    dashboard = _Dashboard()
    dashboard.on_session_start(SessionStatus(issue_id="ISSUE-1", issue_identifier="ISSUE-1 title"))

    sink = FeishuActivitySink(
        task_id="ISSUE-1",
        feishu_adapter=adapter,
        status_dashboard=dashboard,
    )

    assert adapter.calls == []
    assert sink._pending_tasks == set()
    assert dashboard.listeners == []


@pytest.mark.asyncio
async def test_pending_operation_limit_is_enforced() -> None:
    adapter = _Adapter()
    sink, _dashboard = await _started_sink(
        adapter,
        max_pending_tasks=1,
        operation_timeout_seconds=1.0,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocked(message_id: str, card: dict) -> bool:
        adapter.calls.append(("update", message_id, card))
        started.set()
        await release.wait()
        return True

    adapter.update_progress_card = _blocked
    sink.on_phase_complete(PhaseComplete(phase=1, turn_count=1), _session())
    await started.wait()
    sink.on_phase_complete(PhaseComplete(phase=2, turn_count=2), _session())

    assert len(sink._pending_tasks) == 1
    release.set()
    await _drain(sink)
    assert len([call for call in adapter.calls if call[0] == "update"]) == 1


@pytest.mark.asyncio
async def test_operation_timeout_and_exception_logging_are_safe(caplog) -> None:
    adapter = _Adapter()
    sink, _dashboard = await _started_sink(adapter, operation_timeout_seconds=0.01)
    blocker = asyncio.Event()

    async def _blocked(_message_id: str, _card: dict) -> bool:
        await blocker.wait()
        return True

    adapter.update_progress_card = _blocked
    with caplog.at_level("WARNING", logger="extensions.orchestrator.feishu_activity_sink"):
        sink.on_phase_complete(PhaseComplete(phase=1, turn_count=1), _session())
        await _drain(sink)

    assert "timed out" in caplog.text
    assert sink._pending_tasks == set()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pending_tasks": 0},
        {"max_pending_tasks": True},
        {"operation_timeout_seconds": 0},
        {"operation_timeout_seconds": float("inf")},
    ],
)
def test_invalid_async_limits_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        FeishuActivitySink(
            task_id="ISSUE-1",
            feishu_adapter=_Adapter(),
            **kwargs,
        )


def test_terminal_without_start_detaches_waiting_listener() -> None:
    dashboard = _Dashboard()
    sink = FeishuActivitySink(
        task_id="ISSUE-1",
        feishu_adapter=_Adapter(),
        status_dashboard=dashboard,
    )

    sink.on_session_complete(SessionComplete(reason="failed"), _session())

    assert dashboard.listeners == []


def test_turn_complete_remains_a_progress_sink_noop() -> None:
    adapter = _Adapter()
    sink = FeishuActivitySink(task_id="ISSUE-1", feishu_adapter=adapter)

    sink.on_turn_complete(TurnComplete(turn=1), _session())

    assert adapter.calls == []
