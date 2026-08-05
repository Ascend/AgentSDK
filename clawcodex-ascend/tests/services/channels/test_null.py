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

"""NullChannel tests: it must never touch the network."""

from __future__ import annotations

# Sibling clawcodex_ext packages (channels source) are migrated in separate
# branches; suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import pytest

from clawcodex_ext.services.channels import (
    ChannelConfig,
    ChannelMessage,
    ChannelType,
    MessageLevel,
    NullChannel,
)


def _config() -> ChannelConfig:
    # Loopback hostname is not validated by NullChannel by design.
    return ChannelConfig(
        type=ChannelType.SLACK,
        webhook_url="https://localhost/hook/abcdef0123456789",
        name="null-1",
    )


def test_null_channel_constructs_with_loopback_url() -> None:
    # If NullChannel ran the URL safety check, this would raise — it must not.
    channel = NullChannel(_config())
    assert channel.name == "null-1"
    assert channel.enabled is True


@pytest.mark.asyncio
async def test_null_channel_send_records_payload() -> None:
    channel = NullChannel(_config())
    msg = ChannelMessage(text="hi", level=MessageLevel.WARN)
    ok = await channel.send(msg)
    assert ok is True

    log = channel.log
    assert len(log) == 1
    entry = log[0]
    assert entry.message is msg
    # Body is JSON; verify it round-trips and includes the level.
    import json

    payload = json.loads(entry.body.decode("utf-8"))
    assert payload["text"] == "hi"
    assert payload["level"] == "warn"
    assert entry.headers.get("Content-Type") == "application/json"


@pytest.mark.asyncio
async def test_null_channel_clear_empties_log() -> None:
    channel = NullChannel(_config())
    await channel.send(ChannelMessage(text="x"))
    assert len(channel.log) == 1
    channel.clear()
    assert channel.log == []


@pytest.mark.asyncio
async def test_null_channel_send_is_thread_safe() -> None:
    import asyncio

    channel = NullChannel(_config())
    n = 50

    async def fire(i: int) -> None:
        await channel.send(ChannelMessage(text=f"m{i}"))

    await asyncio.gather(*(fire(i) for i in range(n)))
    assert len(channel.log) == n


def test_null_channel_does_not_touch_transport_calls() -> None:
    # The internal _NullTransport records network calls; default channel
    # should be using it, not the urllib transport.
    from clawcodex_ext.services.channels.null_channel import _NullTransport

    channel = NullChannel(_config())
    assert isinstance(channel.transport, _NullTransport)
    assert channel.transport.calls == []
