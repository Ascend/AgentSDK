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

"""Fire-and-forget task lifecycle tests for ``WebSocketTransport``.

The transport schedules ``ws.send`` / ``ws.close`` coroutines without
awaiting them. Without a strong reference the event loop only weakly
holds those tasks (they can be GC'd mid-flight → "Task was destroyed but
it is pending"), and on disconnect the pending sends leak. These tests
exercise the tracking set + cancel-on-disconnect path directly, without a
real socket, by populating ``_send_tasks`` and driving ``_do_disconnect``.
"""

from __future__ import annotations

import asyncio

import pytest

from extensions.ports.transports.websocket_v1 import WebSocketTransport


def _make_transport() -> WebSocketTransport:
    return WebSocketTransport("wss://example.invalid/ws", session_id="s-1")


@pytest.mark.asyncio
async def test_send_task_tracking_set_starts_empty() -> None:
    transport = _make_transport()
    assert transport._send_tasks == set()


@pytest.mark.asyncio
async def test_do_disconnect_cancels_pending_send_tasks() -> None:
    """An in-flight ``ws.send`` task is cancelled when the socket tears down."""
    transport = _make_transport()

    async def _never() -> None:
        await asyncio.Event().wait()

    task = asyncio.get_running_loop().create_task(_never())
    transport._send_tasks.add(task)
    task.add_done_callback(transport._send_tasks.discard)

    # No live ``_ws`` — disconnect just cancels reader/sends and no-ops the
    # socket close.
    transport._do_disconnect()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    assert task not in transport._send_tasks


@pytest.mark.asyncio
async def test_close_cancels_pending_send_tasks() -> None:
    """The public ``close()`` path also drains tracked send tasks."""
    transport = _make_transport()

    async def _never() -> None:
        await asyncio.Event().wait()

    task = asyncio.get_running_loop().create_task(_never())
    transport._send_tasks.add(task)
    task.add_done_callback(transport._send_tasks.discard)

    transport.close()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    assert transport.is_closed_status()


@pytest.mark.asyncio
async def test_already_done_send_task_is_left_untouched() -> None:
    """A completed send task is not re-cancelled and is discarded cleanly."""
    transport = _make_transport()

    async def _quick() -> None:
        return None

    task = asyncio.get_running_loop().create_task(_quick())
    transport._send_tasks.add(task)
    task.add_done_callback(transport._send_tasks.discard)
    await task  # let it finish; done-callback discards it

    assert task not in transport._send_tasks
    # Disconnect must not raise on an empty/settled set.
    transport._do_disconnect()
    assert not task.cancelled()
