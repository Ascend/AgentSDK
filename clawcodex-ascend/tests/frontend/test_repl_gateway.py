# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Tests for the REPL opt-in wrapper."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from clawcodex_ext.frontend.repl_gateway import QueueFull, ReplGatewayClient
from clawcodex_ext.services.im_gateway.models import (
    IM_DIRECT_ALL_ORIGIN,
)
# -- ReplGatewayClient (capacity + dedup, no live UDS) -----------------


def _make_repl_client(capacity=3):
    enqueued: list[str] = []
    size = [0]

    def enqueue(text: str) -> None:
        enqueued.append(text)
        size[0] += 1

    client = ReplGatewayClient(
        "/tmp/gw.sock",
        session_id="repl_main",
        origin="wechat:direct:default:u",
        enqueue=enqueue,
        queue_size=lambda: size[0],
        queue_capacity=capacity,
    )
    return client, enqueued


class _FakeCommandRegistry:
    def __init__(self) -> None:
        self.commands: dict[str, object] = {}

    def register(self, command) -> None:
        self.commands[command.name] = command


class _FakeConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str = "", *args, **kwargs) -> None:
        self.messages.append(str(message))


class _FakeReplForGatewayCommand:
    def __init__(self) -> None:
        self.command_registry = _FakeCommandRegistry()
        self.console = _FakeConsole()
        self._queued_prompts: list[str] = []
        self._gateway_client = None
        self._im_reply_controller = None
        self._gateway_init = None
        self._gateway_session_counter = 0

    def _update_built_in_commands_with_command_system(self) -> None:
        return None

    def _enqueue_prompt(self, text: str) -> None:
        self._queued_prompts.append(text)

    def _wake_prompt_for_im(self) -> None:
        return None


def _gateway_ctx(*, gateway=False, origin=None, sock=None):
    return SimpleNamespace(
        options=SimpleNamespace(
            gateway=gateway,
            gateway_origin=origin,
            gateway_sock=sock,
        )
    )


def test_repl_can_enqueue_under_capacity() -> None:
    client, _ = _make_repl_client(capacity=5)
    assert client.can_enqueue() is True


def test_repl_gateway_command_status_tracks_connect_disconnect(monkeypatch) -> None:
    """The runtime /gateway command reports disconnected, connected, then disconnected."""
    from clawcodex_ext.frontend.repl_extensions import _install_gateway_client

    created: list[object] = []

    class _FakeRuntimeClient:
        def __init__(self, sock, *, session_id, origin, **_kwargs):
            self.socket_path = sock
            self.session_id = session_id
            self.origin = origin
            self.connected = False
            self.closed = False
            created.append(self)

        @property
        def is_connected(self):
            return self.connected

        async def connect(self):
            self.connected = True
            return SimpleNamespace(ack_layer="accepted")

        async def start_heartbeat(self, interval=30.0):
            return None

        async def close(self):
            self.connected = False
            self.closed = True

    monkeypatch.setattr(
        "clawcodex_ext.frontend.repl_gateway.ReplGatewayClient",
        _FakeRuntimeClient,
    )

    repl = _FakeReplForGatewayCommand()
    _install_gateway_client(repl, _gateway_ctx())

    command = repl.command_registry.commands["gateway"]
    assert "connect" in command.description.lower()
    assert "status" in command.description.lower()
    assert "disconnect" in command.description.lower()
    assert "disconnected" in repl._handle_gateway_command("status")
    assert "connected" in repl._handle_gateway_command(
        "connect --origin wechat:direct:default:user --sock /tmp/gateway.sock"
    )
    assert "connected" in repl._handle_gateway_command("status")
    assert "wechat:direct:default:user" in repl._handle_gateway_command("status")
    assert "disconnected" in repl._handle_gateway_command("disconnect")
    assert "disconnected" in repl._handle_gateway_command("status")
    assert created[-1].closed is True


def test_repl_gateway_is_advertised_in_slash_suggestions() -> None:
    """Typing /g should surface /gateway with the runtime actions in the popup."""
    from clawcodex_ext.repl.core import ClawcodexREPL

    description = ClawcodexREPL._REPL_EXTRA_BUILTIN_DESCRIPTIONS.get("gateway")

    assert description is not None
    lowered = description.lower()
    for action in ("connect", "status", "disconnect"):
        assert action in lowered


def test_repl_gateway_connect_missing_daemon_fails_without_client(monkeypatch) -> None:
    """Missing gateway socket should fail visibly and leave the REPL disconnected."""
    from clawcodex_ext.frontend.repl_extensions import _install_gateway_client

    class _MissingGatewayClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def connect(self):
            raise FileNotFoundError("/tmp/missing.sock")

        async def close(self):
            return None

    monkeypatch.setattr(
        "clawcodex_ext.frontend.repl_gateway.ReplGatewayClient",
        _MissingGatewayClient,
    )

    repl = _FakeReplForGatewayCommand()
    _install_gateway_client(repl, _gateway_ctx())

    result = repl._handle_gateway_command("connect --sock /tmp/missing.sock")

    assert "IM gateway daemon is not running" in result
    assert repl._gateway_client is None


@pytest.mark.asyncio
async def test_repl_close_unregisters_before_closing_socket(monkeypatch) -> None:
    client, _ = _make_repl_client(capacity=5)
    calls: list[tuple[str, str | None]] = []

    async def _fake_unregister(session_id: str):
        calls.append(("unregister", session_id))
        return None

    async def _fake_close():
        calls.append(("close", None))
        return None

    monkeypatch.setattr(client._client, "unregister", _fake_unregister, raising=False)
    monkeypatch.setattr(client._client, "close", _fake_close)

    await client.close()

    assert calls == [("unregister", "repl_main"), ("close", None)]


@pytest.mark.asyncio
async def test_repl_on_pushed_deliver_enqueues_into_repl(monkeypatch) -> None:
    """A server-pushed DELIVER frame (gateway→REPL) lands in the prompt queue."""
    client, enqueued = _make_repl_client(capacity=5)

    async def _fake_ack(*, delivery_id, layer, message=None):
        return None

    monkeypatch.setattr(client._client, "ack", _fake_ack)

    # simulate the IPC read loop delivering a pushed frame
    from clawcodex_ext.services.im_gateway.ipc_protocol import GatewayFrame

    frame = GatewayFrame.deliver(
        delivery_id="d_push",
        session_id="repl_main",
        origin="wechat:direct:default:u",
        text="hello from wechat",
        context_token="ctx_abc",
    )
    await client._on_pushed_deliver(frame)
    assert enqueued == ["hello from wechat"]
    assert client.peek_reply_context_token() == "ctx_abc"


@pytest.mark.asyncio
async def test_repl_pushed_deliver_queue_full_reports_failure(monkeypatch) -> None:
    client, enqueued = _make_repl_client(capacity=1)
    client._queue_size = lambda: 1
    completed: list[tuple[str, str, str]] = []

    async def _fake_complete(*, message_id, outcome, reason):
        completed.append((message_id, outcome, reason))

    monkeypatch.setattr(client._client, "complete_processing", _fake_complete, raising=False)

    from clawcodex_ext.services.im_gateway.ipc_protocol import GatewayFrame

    await client._on_pushed_deliver(
        GatewayFrame.deliver(
            delivery_id="d_full",
            session_id="repl_main",
            origin="feishu:dm:cli_app:ou_user",
            text="hello",
        )
    )

    assert enqueued == []
    assert completed == [("d_full", "failure", "REPL prompt queue full")]


def test_im_reply_controller_sends_when_loop_is_not_running() -> None:
    """OUTBOUND must be scheduled even when ``_cron_loop`` is stopped.

    After ``chat()`` returns the REPL is between ``run_until_complete``
    calls so ``_cron_loop.is_running()`` is ``False``. If the controller
    guards on ``is_running()``, the OUTBOUND frame is never scheduled and
    WeChat never receives the assistant reply. ``run_coroutine_threadsafe``
    is designed for this case — it queues the coroutine and the loop
    executes it on the next ``run_until_complete``.
    """
    import asyncio
    import collections

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    sent: list[tuple[str, str]] = []

    class _FakeIpc:
        async def send_outbound(self, *, origin, text):
            sent.append((origin, text))

    class _FakeClient:
        def __init__(self):
            self._client = _FakeIpc()
            self._reply_origins = collections.deque(["wechat:direct:a:u1"])

        def pop_reply_origin(self) -> str | None:
            return self._reply_origins.popleft() if self._reply_origins else None

    class _FakeRepl:
        def __init__(self, text):
            self._text = text
            self._cron_loop = asyncio.new_event_loop()

        def _get_last_assistant_text(self):
            return self._text

    repl = _FakeRepl("agent reply")
    controller = _ImReplyController(repl, _FakeClient(), "wechat:direct:*:*")

    # Simulate the post-chat() state: loop is NOT running
    assert not repl._cron_loop.is_running()

    # Controller must schedule send_outbound even though loop is stopped
    controller.on_assistant_turn_complete()

    # Run the loop briefly so the scheduled send_outbound actually executes
    try:
        import time

        deadline = time.time() + 1.0
        while not sent and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()
    assert sent == [("wechat:direct:a:u1", "agent reply")], (
        "OUTBOUND must be sent when loop is NOT running (the post-chat() state)"
    )


def test_im_reply_controller_skips_keyboard_turns() -> None:
    """Keyboard-initiated turns must NOT send replies to WeChat.

    When the user types at the REPL prompt (not driven by an IM message),
    ``_reply_origins`` is empty. The controller must check this BEFORE
    calling ``send_outbound`` — otherwise every keyboard interaction would
    attempt to send a reply to an invalid WeChat target.
    """
    import collections

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    sent: list[tuple[str, str]] = []

    class _FakeIpc:
        async def send_outbound(self, *, origin, text):
            sent.append((origin, text))

    class _FakeClient:
        def __init__(self):
            self._client = _FakeIpc()
            self._reply_origins = collections.deque()  # empty = keyboard turn

        def pop_reply_origin(self) -> str | None:
            return self._reply_origins.popleft() if self._reply_origins else None

    class _FakeRepl:
        _cron_loop = None  # should never be accessed
        _last_sent = None

        def _get_last_assistant_text(self):
            return "keyboard reply"

    repl = _FakeRepl()
    controller = _ImReplyController(repl, _FakeClient(), "wechat:direct:*:*")

    controller.on_assistant_turn_complete()

    assert sent == [], "keyboard turn must NOT send OUTBOUND to WeChat"


def test_repl_all_private_binding_replies_to_actual_inbound_origin(monkeypatch) -> None:
    """Wildcard REPL bindings reply to the sender origin carried by DELIVER."""
    import asyncio

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController
    from clawcodex_ext.services.im_gateway.ipc_protocol import GatewayFrame

    sent: list[tuple[str, str]] = []

    async def _fake_ack(*, delivery_id, layer, message=None):
        return None

    class _FakeInner:
        async def send_outbound(self, *, origin, text):
            sent.append((origin, text))

    class _FakeRepl:
        def __init__(self):
            self._text = "reply to sender"
            self._cron_loop = asyncio.new_event_loop()

        def _get_last_assistant_text(self):
            return self._text

    client, _ = _make_repl_client(capacity=5)
    client._origin = "wechat:direct:*:*"
    client._client = _FakeInner()
    monkeypatch.setattr(client, "_ack", _fake_ack, raising=False)
    monkeypatch.setattr(client._client, "ack", _fake_ack, raising=False)

    frame = GatewayFrame.deliver(
        delivery_id="d_sender",
        session_id="repl_main",
        origin="wechat:direct:acct:user_sender",
        text="hello",
    )
    asyncio.run(client._on_pushed_deliver(frame))

    repl = _FakeRepl()
    controller = _ImReplyController(repl, client, "wechat:direct:*:*")
    try:
        repl._cron_loop.call_soon(controller.on_assistant_turn_complete)
        import time

        deadline = time.time() + 1.0
        while not sent and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()

    assert sent == [("wechat:direct:acct:user_sender", "reply to sender")]


def test_repl_feishu_reply_preserves_context_token_for_chat_id(monkeypatch) -> None:
    import asyncio

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController
    from clawcodex_ext.services.im_gateway.ipc_protocol import GatewayFrame

    sent: list[tuple[str, str, str | None, str | None]] = []

    async def _fake_ack(*, delivery_id, layer, message=None):
        return None

    class _FakeInner:
        async def send_outbound(self, *, origin, text, context_token=None, in_reply_to=None):
            sent.append((origin, text, context_token, in_reply_to))

    class _FakeRepl:
        def __init__(self):
            self._text = "reply to feishu"
            self._cron_loop = asyncio.new_event_loop()

        def _get_last_assistant_text(self):
            return self._text

    client, _ = _make_repl_client(capacity=5)
    client._origin = "feishu:dm:*:*"
    client._client = _FakeInner()
    monkeypatch.setattr(client._client, "ack", _fake_ack, raising=False)

    frame = GatewayFrame.deliver(
        delivery_id="d_feishu",
        session_id="repl_main",
        origin="feishu:dm:cli_app:ou_user",
        text="hello",
        context_token="oc_chat",
    )
    asyncio.run(client._on_pushed_deliver(frame))

    repl = _FakeRepl()
    controller = _ImReplyController(repl, client, "feishu:dm:*:*")
    try:
        repl._cron_loop.call_soon(controller.on_assistant_turn_complete)
        import time

        deadline = time.time() + 1.0
        while not sent and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()

    assert sent == [("feishu:dm:cli_app:ou_user", "reply to feishu", "oc_chat", "d_feishu")]


@pytest.mark.parametrize(
    ("run_outcome", "assistant_text", "expected_outcome"),
    [
        ("success", "", "success"),
        ("failure", "unused", "failure"),
        ("cancelled", "unused", "cancelled"),
    ],
)
def test_im_reply_controller_reports_terminal_outcome_without_reply(
    run_outcome: str,
    assistant_text: str,
    expected_outcome: str,
) -> None:
    import asyncio
    import time
    from types import SimpleNamespace

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    completed: list[tuple[str, str]] = []

    class _FakeInner:
        async def complete_processing(self, *, message_id, outcome):
            completed.append((message_id, outcome))

    class _FakeClient:
        def __init__(self):
            self._client = _FakeInner()
            self.pending = SimpleNamespace(
                delivery_id="d_terminal",
                origin="feishu:dm:cli_app:ou_user",
                context_token="oc_chat",
            )

        def pop_reply_delivery(self):
            pending, self.pending = self.pending, None
            return pending

    class _FakeRepl:
        def __init__(self):
            self._cron_loop = asyncio.new_event_loop()

        def _get_last_assistant_text(self):
            return assistant_text

    repl = _FakeRepl()
    controller = _ImReplyController(repl, _FakeClient(), "feishu:dm:*:*")
    controller.on_run_finish(run_outcome)
    controller.on_assistant_turn_complete()
    try:
        deadline = time.time() + 1.0
        while not completed and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()

    assert completed == [("d_terminal", expected_outcome)]


def test_im_reply_controller_does_not_resend_previous_assistant_message() -> None:
    import asyncio
    import time

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController
    from clawcodex_ext.frontend.repl_gateway import PendingImReply

    sent: list[dict] = []
    completed: list[tuple[str, str]] = []

    class _FakeInner:
        async def send_outbound(self, **kwargs):
            sent.append(kwargs)

        async def complete_processing(self, *, message_id, outcome):
            completed.append((message_id, outcome))

    class _FakeClient:
        def __init__(self):
            self._client = _FakeInner()
            self.pending = PendingImReply(
                "d_no_reply",
                "feishu:dm:cli_app:ou_user",
                "oc_chat",
            )

        def pop_reply_delivery(self):
            pending, self.pending = self.pending, None
            return pending

    class _FakeRepl:
        def __init__(self):
            self._cron_loop = asyncio.new_event_loop()
            self.session = SimpleNamespace(
                conversation=SimpleNamespace(messages=[SimpleNamespace(role="assistant", content="previous reply")])
            )

        def _get_last_assistant_text(self):
            return "previous reply"

    repl = _FakeRepl()
    controller = _ImReplyController(repl, _FakeClient(), "feishu:dm:*:*")
    controller.on_run_start()
    controller.on_run_finish("success")
    controller.on_assistant_turn_complete()
    try:
        deadline = time.time() + 1.0
        while not completed and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()

    assert sent == []
    assert completed == [("d_no_reply", "success")]


def test_im_reply_controller_sends_command_feedback_with_context() -> None:
    import asyncio
    import time

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    sent: list[dict] = []

    class _FakeInner:
        async def send_outbound(self, **kwargs):
            sent.append(kwargs)

    class _FakeClient:
        def __init__(self):
            self._client = _FakeInner()
            self._reply_context = [("feishu:dm:cli_app:ou_user", "oc_chat")]

        def pop_reply_context(self) -> tuple[str | None, str | None]:
            return self._reply_context.pop(0) if self._reply_context else (None, None)

    class _FakeRepl:
        def __init__(self):
            self._cron_loop = asyncio.new_event_loop()

    repl = _FakeRepl()
    client = _FakeClient()
    controller = _ImReplyController(repl, client, "feishu:dm:*:*")
    try:
        repl._cron_loop.call_soon(controller.send_command_feedback, "/clear")
        deadline = time.time() + 1.0
        while not sent and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()

    assert sent == [
        {
            "origin": "feishu:dm:cli_app:ou_user",
            "text": "命令已执行：/clear",
            "context_token": "oc_chat",
            "metadata": {
                "intent": "command_feedback",
                "command": "/clear",
                "success": True,
            },
            "semantic_tags": ["command_feedback"],
        }
    ]
    assert client._reply_context == []


def test_im_reply_controller_command_failure_reports_failure_separately() -> None:
    import asyncio
    import time

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController
    from clawcodex_ext.frontend.repl_gateway import PendingImReply

    sent: list[dict] = []
    completed: list[tuple[str, str]] = []

    class _FakeInner:
        async def send_outbound(self, **kwargs):
            sent.append(kwargs)

        async def complete_processing(self, *, message_id, outcome):
            completed.append((message_id, outcome))

    class _FakeClient:
        def __init__(self):
            self._client = _FakeInner()
            self._pending = [PendingImReply("d_failed", "feishu:dm:cli_app:ou_user", "oc_chat")]

        def pop_reply_delivery(self):
            return self._pending.pop(0) if self._pending else None

    class _FakeRepl:
        def __init__(self):
            self._cron_loop = asyncio.new_event_loop()

    repl = _FakeRepl()
    controller = _ImReplyController(repl, _FakeClient(), "feishu:dm:*:*")
    try:
        repl._cron_loop.call_soon(lambda: controller.send_command_feedback("/clear", success=False))
        deadline = time.time() + 1.0
        while (not sent or not completed) and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()

    assert len(sent) == 1
    assert "in_reply_to" not in sent[0]
    assert completed == [("d_failed", "failure")]


def test_im_reply_controller_sends_permission_prompt_without_consuming_origin(monkeypatch) -> None:
    """Permission menus shown during an IM turn must also be visible in WeChat.

    The permission prompt is not the final assistant reply, so sending it must
    peek at the current IM origin instead of popping it. Otherwise the final
    assistant reply would lose its WeChat target.

    The permission OUTBOUND leaves immediately via the one-shot thread path
    (the loop is NOT running during the permission wait — that is the whole
    point of the 1a fix). The final assistant reply is scheduled on
    ``_cron_loop`` and drained on the next ``run_until_complete``.
    """
    import asyncio
    import collections
    import time

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    sent: list[tuple[str, str]] = []

    class _FakeOneShot:
        """Replaces GatewayIpcClient inside _send_outbound_text_from_thread."""

        def __init__(self, socket_path, instance_id=None, **_kwargs) -> None:
            pass

        async def connect(self) -> None:
            return None

        async def send_outbound(self, *, origin, text) -> None:
            sent.append((origin, text))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "clawcodex_ext.services.im_gateway.ipc_client.GatewayIpcClient",
        _FakeOneShot,
    )

    class _FakeIpc:
        async def send_outbound(self, *, origin, text):
            sent.append((origin, text))

    class _FakeClient:
        _socket_path = "/tmp/gw.sock"

        def __init__(self):
            self._client = _FakeIpc()
            self._reply_origins = collections.deque(["wechat:direct:a:u1"])

        def peek_reply_origin(self) -> str | None:
            return self._reply_origins[0] if self._reply_origins else None

        def pop_reply_origin(self) -> str | None:
            return self._reply_origins.popleft() if self._reply_origins else None

    class _FakeRepl:
        def __init__(self):
            self._cron_loop = asyncio.new_event_loop()
            self._text = "done"

        def _get_last_assistant_text(self):
            return self._text

    repl = _FakeRepl()
    client = _FakeClient()
    controller = _ImReplyController(repl, client, "wechat:direct:*:*")

    try:
        assert controller.send_permission_prompt(
            message="Tool Bash wants to delete files",
            options=[("y", "Yes, allow this action"), ("n", "No, deny this action")],
        )

        # Thread path delivers the permission menu without running _cron_loop.
        deadline = time.time() + 1.0
        while len(sent) < 1 and time.time() < deadline:
            time.sleep(0.02)

        # The permission menu used the origin without consuming it.
        assert list(client._reply_origins) == ["wechat:direct:a:u1"]

        controller.on_assistant_turn_complete()
        deadline = time.time() + 1.0
        while len(sent) < 2 and time.time() < deadline:
            repl._cron_loop.run_until_complete(asyncio.sleep(0.02))
    finally:
        repl._cron_loop.close()

    assert sent[0][0] == "wechat:direct:a:u1"
    assert "需要权限" in sent[0][1]
    assert "1. [y] 是，允许此操作" in sent[0][1]
    assert sent[1] == ("wechat:direct:a:u1", "done")


def test_im_permission_prompt_sent_immediately_via_thread_path(monkeypatch) -> None:
    """1a: when ``_cron_loop`` is NOT running (the sync permission path, about
    to block on ``_safe_input``), the OUTBOUND permission menu must leave via
    the one-shot thread path immediately — NOT be queued on ``_cron_loop``.

    ``_cron_loop`` won't drain until ``chat()`` returns and the main loop
    re-enters ``run_until_complete``, i.e. AFTER the user already approved in
    the REPL. Queuing the OUTBOUND there is the stale-permission-notice bug.
    """
    import asyncio
    import collections
    import time

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    sent: list[tuple[str, str]] = []

    class _FakeOneShot:
        """Replaces GatewayIpcClient inside _send_outbound_text_from_thread."""

        def __init__(self, socket_path, instance_id=None, **_kwargs) -> None:
            pass

        async def connect(self) -> None:
            return None

        async def send_outbound(self, *, origin, text) -> None:
            sent.append((origin, text))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "clawcodex_ext.services.im_gateway.ipc_client.GatewayIpcClient",
        _FakeOneShot,
    )

    class _FakeInner:
        async def send_outbound(self, *, origin, text) -> None:
            sent.append((origin, text))  # only reached by the buggy loop-queue path

    class _FakeClient:
        _socket_path = "/tmp/gw.sock"

        def __init__(self) -> None:
            self._client = _FakeInner()
            self._reply_origins = collections.deque(["wechat:direct:a:u1"])

        def peek_reply_origin(self) -> str | None:
            return self._reply_origins[0] if self._reply_origins else None

    class _FakeRepl:
        def __init__(self) -> None:
            self._cron_loop = asyncio.new_event_loop()  # NOT running

    repl = _FakeRepl()
    controller = _ImReplyController(repl, _FakeClient(), "wechat:direct:*:*")

    assert not repl._cron_loop.is_running()
    assert controller.send_permission_prompt(
        message="Tool Bash wants to delete files",
        options=[("y", "Yes, allow this action"), ("n", "No, deny this action")],
    )

    # Deliberately do NOT run _cron_loop. The thread path must deliver without it.
    deadline = time.time() + 1.0
    while len(sent) < 1 and time.time() < deadline:
        time.sleep(0.02)
    repl._cron_loop.close()

    assert sent, "permission OUTBOUND must be sent immediately via the thread path"
    assert sent[0][0] == "wechat:direct:a:u1"
    assert "需要权限" in sent[0][1]


def test_im_permission_prompt_sends_structured_metadata() -> None:
    """Interactive permission prompts carry generic metadata for rich channels."""
    import collections

    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    sent: list[dict] = []

    class _FakeClient:
        def __init__(self):
            self._reply_origins = collections.deque(["wechat:direct:a:u1"])

        def peek_reply_origin(self) -> str | None:
            return self._reply_origins[0] if self._reply_origins else None

    class _FakeRepl:
        _cron_loop = None

    controller = _ImReplyController(_FakeRepl(), _FakeClient(), "wechat:direct:*:*")

    def _capture(im_origin, text, *, metadata=None, semantic_tags=None):
        sent.append(
            {
                "origin": im_origin,
                "text": text,
                "metadata": metadata,
                "semantic_tags": semantic_tags,
            }
        )
        return True

    controller._send_outbound_text = _capture

    assert controller.send_permission_prompt(
        message="Claude wants to use Bash. Allow?",
        options=[
            ("y", "Yes, allow this action"),
            ("s", "Yes, and don't ask again for Bash"),
            ("n", "No, deny this action"),
        ],
        suggestion="Review command",
        interactive=True,
        allow_choices={"y", "s"},
    )

    assert sent[0]["origin"] == "wechat:direct:a:u1"
    assert sent[0]["metadata"]["intent"] == "permission_approval"
    assert sent[0]["metadata"]["permission"]["message"].startswith("ClawCodex wants")
    assert sent[0]["metadata"]["permission"]["suggestion"] == "Review command"
    assert [option["value"] for option in sent[0]["metadata"]["permission"]["options"]] == [
        "y",
        "s",
        "n",
    ]
    assert [option["decision"] for option in sent[0]["metadata"]["permission"]["options"]] == [
        "allow",
        "allow",
        "deny",
    ]
    assert sent[0]["semantic_tags"] == ["approval"]


def test_permission_prompt_renders_chinese_and_rebrands_claude() -> None:
    """The WeChat-facing permission prompt must be Chinese and self-refer as
    ClawCodex (not Claude). The upstream ``message`` ("Claude wants to use X.
    Allow?") and English option descriptions are translated/rebranded in
    ``_format_permission_prompt`` only — the REPL console is untouched.
    """
    from clawcodex_ext.frontend.repl_extensions import _ImReplyController

    text = _ImReplyController._format_permission_prompt(
        message="Claude wants to use Bash. Allow?",
        options=[
            ("y", "Yes, allow this action"),
            ("n", "No, deny this action"),
        ],
        suggestion="可在设置中开启 allow_docs",
        interactive=True,
    )
    assert "需要权限" in text
    assert "ClawCodex 想使用 Bash，是否允许？" in text
    assert "Claude" not in text
    assert "1. [y] 是，允许此操作" in text
    assert "2. [n] 否，拒绝此操作" in text
    assert "选项：" in text
    assert "建议：" in text
    assert "请回复选项编号或字母" in text

    # Enable-and-allow option form is translated for any setting name.
    text2 = _ImReplyController._format_permission_prompt(
        message="Claude wants to use Read. Allow?",
        options=[
            ("e", "Enable allow_docs and allow"),
            ("y", "Yes, allow this action"),
            ("n", "No, deny this action"),
        ],
        interactive=False,
    )
    assert "1. [e] 启用 allow_docs 并允许" in text2
    assert "请在 REPL 中选择对应的选项以继续" in text2


def test_handle_im_permission_reply_maps_choices_and_stop() -> None:
    """1b: the permission-reply probe maps menu choices and /stop correctly."""
    import threading

    from clawcodex_ext.frontend.repl_extensions import _handle_im_permission_reply

    def _repl_with_wait(valid):
        class _FakeRepl:
            pass

        r = _FakeRepl()
        r._im_permission_lock = threading.Lock()
        r._im_permission_wait = {
            "event": threading.Event(),
            "choice": None,
            "valid": set(valid),
        }
        return r

    valid = {"1", "2", "y", "n", "yes", "no"}

    # numeric choice → consumed, choice recorded, event set
    r = _repl_with_wait(valid)
    assert _handle_im_permission_reply(r, "1") is True
    assert r._im_permission_wait["choice"] == "1"
    assert r._im_permission_wait["event"].is_set()

    # letter choice
    r = _repl_with_wait(valid)
    assert _handle_im_permission_reply(r, "Y") is True  # case-insensitive
    assert r._im_permission_wait["choice"] == "y"

    # /stop during permission wait → deny
    r = _repl_with_wait(valid)
    assert _handle_im_permission_reply(r, "/stop") is True
    assert r._im_permission_wait["choice"] == "n"

    # non-choice → not consumed, fall through to enqueue
    r = _repl_with_wait(valid)
    assert _handle_im_permission_reply(r, "hello there") is False
    assert r._im_permission_wait["choice"] is None
    assert not r._im_permission_wait["event"].is_set()

    # no active wait → not consumed
    r = _repl_with_wait(valid)
    r._im_permission_wait = None
    assert _handle_im_permission_reply(r, "1") is False


@pytest.mark.asyncio
async def test_repl_deliver_permission_reply_intercepted_before_enqueue(monkeypatch) -> None:
    """1b: when a permission wait is active, a WeChat reply matching a valid
    menu choice is consumed by ``permission_probe`` (ack ``processed``) and is
    NOT enqueued as a new prompt. A non-choice message falls through normally.
    """
    client, enqueued = _make_repl_client(capacity=5)
    acks: list[tuple] = []

    def _probe(text: str) -> bool:
        return (text or "").strip().lower() in {"1", "2", "y", "n"}

    client._permission_probe = _probe  # injected by _install_gateway_client

    async def _fake_ack(*, delivery_id, layer, message=None):
        acks.append((delivery_id, layer, message))
        return None

    monkeypatch.setattr(client._client, "ack", _fake_ack)

    # valid choice → consumed, acked processed, NOT enqueued
    await client.deliver(delivery_id="d_perm", text="1", origin="wechat:direct:a:u")
    assert enqueued == []
    assert acks == [("d_perm", "processed", "permission reply")]

    # non-choice → falls through to enqueue
    await client.deliver(delivery_id="d_msg", text="hello", origin="wechat:direct:a:u")
    assert enqueued == ["hello"]


@pytest.mark.asyncio
async def test_gateway_ipc_ack_is_fire_and_forget(monkeypatch) -> None:
    from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient
    from clawcodex_ext.services.im_gateway.ipc_protocol import FrameType

    client = GatewayIpcClient("/tmp/gw.sock")
    written = []

    async def _send(_frame):
        raise AssertionError("ACK replies to server-pushed DELIVER must not wait for another ACK")

    async def _write_frame_no_reply(frame):
        written.append(frame)

    monkeypatch.setattr(client, "_send", _send)
    monkeypatch.setattr(client, "_write_frame_no_reply", _write_frame_no_reply, raising=False)

    response = await client.ack(delivery_id="d_perm", layer="processed", message="permission reply")

    assert response is None
    assert len(written) == 1
    assert written[0].type is FrameType.ACK
    assert written[0].delivery_id == "d_perm"
    assert written[0].ack_layer == "processed"


def test_repl_gateway_uses_runtime_options_and_registers_once(monkeypatch) -> None:
    """Runtime options opt into IM; init uses ReplGatewayClient.connect only once."""
    import asyncio

    from clawcodex_ext.frontend import repl_extensions

    monkeypatch.delenv("CLAWCODEX_GATEWAY_ORIGIN", raising=False)
    monkeypatch.delenv("CLAWCODEX_IM_ORIGIN", raising=False)
    monkeypatch.delenv("CLAWCODEX_GATEWAY_SOCK", raising=False)
    monkeypatch.delenv("CLAWCODEX_IM_GATEWAY_SOCK", raising=False)

    class _FakeInner:
        def __init__(self):
            self.raw_registers = []

        async def register(self, **kwargs):
            self.raw_registers.append(kwargs)

    class _FakeClient:
        def __init__(self, socket_path, **kwargs):
            self.socket_path = socket_path
            self.kwargs = kwargs
            self._client = _FakeInner()
            self.connect_calls = 0
            self.heartbeat_intervals = []

        async def connect(self):
            self.connect_calls += 1

        async def _heartbeat_loop(self, interval):
            self.heartbeat_intervals.append(interval)

    monkeypatch.setattr("clawcodex_ext.frontend.repl_gateway.ReplGatewayClient", _FakeClient)

    class _FakeRepl:
        _queued_prompts = []

        def _enqueue_prompt(self, text):
            return None

        def _wake_prompt_for_im(self):
            return None

    repl = _FakeRepl()
    ctx = SimpleNamespace(
        options=SimpleNamespace(
            gateway_origin="wechat:direct:default:user1",
            gateway_sock="/tmp/clawcodex-gateway.sock",
        )
    )
    repl_extensions._install_gateway_client(repl, ctx)

    client = repl._gateway_client
    assert client.socket_path == "/tmp/clawcodex-gateway.sock"
    assert client.kwargs["origin"] == "wechat:direct:default:user1"
    asyncio.run(repl._gateway_init(None))
    assert client.connect_calls == 1
    assert client._client.raw_registers == []
    assert callable(client.kwargs["control_handler"])


def test_repl_gateway_switch_registers_all_private_messages(monkeypatch) -> None:
    """--gateway binds REPL to all supported private IM messages by default."""
    from clawcodex_ext.frontend import repl_extensions

    monkeypatch.delenv("CLAWCODEX_GATEWAY_ORIGIN", raising=False)
    monkeypatch.delenv("CLAWCODEX_IM_ORIGIN", raising=False)
    monkeypatch.delenv("CLAWCODEX_GATEWAY_SOCK", raising=False)
    monkeypatch.delenv("CLAWCODEX_IM_GATEWAY_SOCK", raising=False)

    class _FakeClient:
        def __init__(self, socket_path, **kwargs):
            self.socket_path = socket_path
            self.kwargs = kwargs

    monkeypatch.setattr("clawcodex_ext.frontend.repl_gateway.ReplGatewayClient", _FakeClient)

    class _FakeRepl:
        _queued_prompts = []

        def _enqueue_prompt(self, text):
            return None

        def _wake_prompt_for_im(self):
            return None

    repl = _FakeRepl()
    ctx = SimpleNamespace(
        options=SimpleNamespace(
            gateway=True,
            gateway_origin=None,
            gateway_sock="/tmp/clawcodex-gateway.sock",
        )
    )
    repl_extensions._install_gateway_client(repl, ctx)

    assert repl._gateway_client.kwargs["origin"] == IM_DIRECT_ALL_ORIGIN
    assert callable(repl._gateway_client.kwargs["control_handler"])


def test_repl_gateway_im_init_no_auto_start_on_missing_socket(monkeypatch) -> None:
    """First connect FileNotFoundError → logs error and returns (no auto-start)."""
    import asyncio

    from clawcodex_ext.frontend import repl_extensions

    class _FakeInner:
        async def register(self, **kwargs):
            return None

    class _FakeClient:
        def __init__(self, socket_path, **kwargs):
            self.socket_path = socket_path
            self.kwargs = kwargs
            self._client = _FakeInner()
            self.connect_calls = 0
            self.heartbeat_intervals = []

        async def connect(self):
            self.connect_calls += 1
            raise FileNotFoundError(2, "No such file or directory")

        async def _heartbeat_loop(self, interval):
            self.heartbeat_intervals.append(interval)

    monkeypatch.setattr("clawcodex_ext.frontend.repl_gateway.ReplGatewayClient", _FakeClient)

    class _FakeRepl:
        _queued_prompts = []

        def _enqueue_prompt(self, text):
            return None

        def _wake_prompt_for_im(self):
            return None

    repl = _FakeRepl()
    ctx = SimpleNamespace(
        options=SimpleNamespace(
            gateway_origin="wechat:direct:default:user1",
            gateway_sock="/tmp/clawcodex-gateway.sock",
        )
    )
    repl_extensions._install_gateway_client(repl, ctx)
    asyncio.run(repl._gateway_init(None))

    client = repl._gateway_client
    # No auto-start: connect fails once, then bails — no retry.
    assert client.connect_calls == 1
    assert client.heartbeat_intervals == []


def test_repl_gateway_im_init_no_retry_when_daemon_start_fails(monkeypatch) -> None:
    """Socket missing → log error and bail immediately without retry."""
    import asyncio

    from clawcodex_ext.frontend import repl_extensions

    class _FakeInner:
        async def register(self, **kwargs):
            return None

    class _FakeClient:
        def __init__(self, socket_path, **kwargs):
            self.socket_path = socket_path
            self.kwargs = kwargs
            self._client = _FakeInner()
            self.connect_calls = 0

        async def connect(self):
            self.connect_calls += 1
            raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr("clawcodex_ext.frontend.repl_gateway.ReplGatewayClient", _FakeClient)

    class _FakeRepl:
        _queued_prompts = []

        def _enqueue_prompt(self, text):
            return None

        def _wake_prompt_for_im(self):
            return None

    repl = _FakeRepl()
    ctx = SimpleNamespace(
        options=SimpleNamespace(
            gateway_origin="wechat:direct:default:user1",
            gateway_sock="/tmp/clawcodex-gateway.sock",
        )
    )
    repl_extensions._install_gateway_client(repl, ctx)
    asyncio.run(repl._gateway_init(None))

    # Only the initial connect attempt; no auto-start, no retry.
    assert repl._gateway_client.connect_calls == 1


def test_repl_gateway_control_handler_stops_active_run(monkeypatch) -> None:
    """The installed REPL control callback maps IM /stop to active-run cancel."""
    from clawcodex_ext.frontend import repl_extensions

    monkeypatch.delenv("CLAWCODEX_GATEWAY_ORIGIN", raising=False)
    monkeypatch.delenv("CLAWCODEX_IM_ORIGIN", raising=False)
    monkeypatch.delenv("CLAWCODEX_GATEWAY_SOCK", raising=False)
    monkeypatch.delenv("CLAWCODEX_IM_GATEWAY_SOCK", raising=False)

    class _FakeClient:
        def __init__(self, socket_path, **kwargs):
            self.socket_path = socket_path
            self.kwargs = kwargs

    monkeypatch.setattr("clawcodex_ext.frontend.repl_gateway.ReplGatewayClient", _FakeClient)

    class _FakeRepl:
        _queued_prompts = []

        def __init__(self):
            self.interrupts: list[str] = []

        def _enqueue_prompt(self, text):
            raise AssertionError("/stop should not be enqueued as a normal prompt")

        def _wake_prompt_for_im(self):
            return None

        def _interrupt_active_chat_from_im(self) -> bool:
            self.interrupts.append("stop")
            return True

    repl = _FakeRepl()
    ctx = SimpleNamespace(
        options=SimpleNamespace(
            gateway_origin="wechat:direct:default:user1",
            gateway_sock="/tmp/clawcodex-gateway.sock",
        )
    )
    repl_extensions._install_gateway_client(repl, ctx)

    handler = repl._gateway_client.kwargs["control_handler"]
    assert handler("/stop", "wechat:direct:default:user1") is True
    assert repl.interrupts == ["stop"]


@pytest.mark.asyncio
async def test_repl_deliver_wakes_blocked_prompt_loop(monkeypatch) -> None:
    """Enqueuing an IM prompt must wake the REPL's blocked prompt loop.

    Without the wake, the enqueued prompt sits in ``_queued_prompts`` while
    the main loop is stuck in ``prompt_async('❯ ')`` — the message is never
    displayed, processed, or replied to (the "IM message reaches the REPL
    but nothing happens" symptom). ``deliver()`` must invoke the injected
    wake callback right after enqueueing so the loop iterates and drains.
    """
    client, enqueued = _make_repl_client(capacity=5)
    wakes: list[int] = []

    def _wake() -> None:
        wakes.append(1)

    client._wake = _wake  # injected by _install_gateway_client

    async def _fake_ack(*, delivery_id, layer, message=None):
        return None

    monkeypatch.setattr(client._client, "ack", _fake_ack)
    await client.deliver(delivery_id="d_wake", text="hello from wechat")
    assert enqueued == ["hello from wechat"]
    assert wakes == [1], "deliver() must wake the REPL prompt loop after enqueue"


def test_repl_im_wake_erases_abandoned_prompt_row() -> None:
    """An IM wake must not leave an empty prompt line in REPL scrollback."""
    from unittest.mock import Mock

    from clawcodex_ext.repl.core import ClawcodexREPL

    app = SimpleNamespace(future=object(), erase_when_done=False, exit=Mock())
    repl = ClawcodexREPL.__new__(ClawcodexREPL)
    repl.prompt_session = SimpleNamespace(app=app)

    repl._exit_pending_prompt_for_im()

    assert app.erase_when_done is True
    app.exit.assert_called_once()


@pytest.mark.asyncio
async def test_repl_im_wake_restores_normal_prompt_rendering() -> None:
    """Erasing the IM-woken row must not erase later keyboard prompts."""
    from unittest.mock import Mock

    from clawcodex_ext.repl.core import ClawcodexREPL

    app = SimpleNamespace(
        future=object(),
        erase_when_done=False,
        exit=Mock(),
        default_buffer=None,
    )
    repl = ClawcodexREPL.__new__(ClawcodexREPL)

    class _PromptSession:
        async def prompt_async(self, _prompt):
            repl._exit_pending_prompt_for_im()
            assert app.erase_when_done is True
            return "ignored"

    repl.prompt_session = _PromptSession()
    repl.prompt_session.app = app
    repl.tool_context = SimpleNamespace(outbox=[])

    await repl._prompt_with_cron_watch()

    assert app.erase_when_done is False


@pytest.mark.asyncio
async def test_repl_stop_command_uses_priority_control_path(monkeypatch) -> None:
    """IM /stop should interrupt the active REPL run instead of waiting in queue."""
    client, enqueued = _make_repl_client(capacity=5)
    calls: list[tuple[str, str | None]] = []

    def _control(text: str, origin: str | None = None) -> bool:
        calls.append((text, origin))
        return True

    client._control_handler = _control

    async def _fake_ack(*, delivery_id, layer, message=None):
        return None

    monkeypatch.setattr(client._client, "ack", _fake_ack)
    await client.deliver(
        delivery_id="d_stop",
        text="/stop",
        origin="wechat:direct:default:u",
        semantic="command",
    )
    await client.deliver(
        delivery_id="d_interrupt",
        text="please stop",
        origin="wechat:direct:default:u",
        semantic="interrupt",
    )

    assert calls == [
        ("/stop", "wechat:direct:default:u"),
        ("please stop", "wechat:direct:default:u"),
    ]
    assert enqueued == []


@pytest.mark.asyncio
async def test_repl_rejects_at_capacity_without_silent_drop(monkeypatch) -> None:
    client, enqueued = _make_repl_client(capacity=2)

    async def _fake_deliver(*, delivery_id, session_id, origin, text, semantic=None):
        return None

    monkeypatch.setattr(client._client, "deliver", _fake_deliver)
    # simulate a full queue
    client._queue_size = lambda: 2
    with pytest.raises(QueueFull):
        await client.deliver(delivery_id="d1", text="a")
    assert enqueued == []  # not silently dropped


@pytest.mark.asyncio
async def test_repl_dedups_by_delivery_id(monkeypatch) -> None:
    client, enqueued = _make_repl_client(capacity=5)

    async def _fake_ack(*, delivery_id, layer, message=None):
        return None

    monkeypatch.setattr(client._client, "ack", _fake_ack)
    # first deliver enqueues "a"
    await client.deliver(delivery_id="d1", text="a")
    # second with same id → deduped, returns None, no extra enqueue
    res = await client.deliver(delivery_id="d1", text="a")
    assert res is None
    assert len(enqueued) == 1


@pytest.mark.asyncio
async def test_repl_ack_uses_ack_frame_not_deliver(monkeypatch) -> None:
    client, enqueued = _make_repl_client(capacity=5)
    calls: list[tuple] = []

    async def _fake_ack(*, delivery_id, layer, message=None):
        calls.append(("ack", delivery_id, layer, message))
        return None

    async def _bad_deliver(**_kwargs):
        raise AssertionError("REPL enqueue ack must not send a second DELIVER frame")

    monkeypatch.setattr(client._client, "ack", _fake_ack, raising=False)
    monkeypatch.setattr(client._client, "deliver", _bad_deliver)
    await client.deliver(delivery_id="d1", text="a")
    assert enqueued == ["a"]
    assert calls == [("ack", "d1", "enqueued", "enqueued")]
