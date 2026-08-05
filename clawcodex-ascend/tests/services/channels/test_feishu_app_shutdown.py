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

# pylint: disable=no-name-in-module
from __future__ import annotations

import asyncio

import pytest

from clawcodex_ext.services.channels.feishu_app import _cancel_feishu_sdk_ws_tasks


class _SdkLikeExpiringCache:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._cron = loop.create_task(self._start_clear_cron())

    async def _start_clear_cron(self) -> None:
        await asyncio.sleep(3600)


class _SdkLikeWsClient:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._close_seen = loop.create_future()
        self._cache = _SdkLikeExpiringCache(loop)
        self.ping_task = loop.create_task(self._ping_loop())
        self.receive_task = loop.create_task(self._receive_message_loop())

    @property
    def tasks(self) -> list[asyncio.Task]:
        return [self._cache._cron, self.ping_task, self.receive_task]

    async def _ping_loop(self) -> None:
        await asyncio.sleep(3600)

    async def _receive_message_loop(self) -> None:
        await self._close_seen
        raise RuntimeError("sent 1000 (OK); no close frame received")

    def close_from_disconnect(self) -> None:
        if not self._close_seen.done():
            self._close_seen.set_result(None)
        if not self.receive_task.done():
            self._close_seen.get_loop().run_until_complete(asyncio.sleep(0))


@pytest.mark.asyncio
async def test_cancel_feishu_sdk_ws_tasks_prevents_receive_close_exception() -> None:
    sdk_loop = asyncio.new_event_loop()
    ws_client = _SdkLikeWsClient(sdk_loop)

    try:
        await _cancel_feishu_sdk_ws_tasks(sdk_loop)
        ws_client.close_from_disconnect()

        assert ws_client.receive_task.cancelled() is True
        assert all(task.done() for task in ws_client.tasks)
    finally:
        await asyncio.to_thread(_drain_test_loop, sdk_loop, ws_client.tasks)


def _drain_test_loop(loop: asyncio.AbstractEventLoop, tasks: list[asyncio.Task]) -> None:
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.close()
