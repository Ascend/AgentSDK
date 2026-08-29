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

"""P108-A Tests for agent bridge freeze."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from src.tui.agent_bridge import AgentBridge
from src.tui.messages import (
    AskUserQuestionRequested,
    PermissionRequested,
)
from src.tui.state import AppState


# ----------------------------------------------------------------------
# Helpers (mirror the fixture in test_ask_user_question.py)
# ----------------------------------------------------------------------


def _build_bridge() -> tuple[AgentBridge, list, AppState]:
    """Construct a minimal ``AgentBridge`` with a recording post_message.

    Returns ``(bridge, posted_messages, app_state)``. We don't spin up a
    real provider/session — the tests only exercise the two modal
    handlers, which never touch them.
    """

    post_calls: list = []
    app_state = AppState(model="test-model", provider="test-provider")
    tool_context = __import__("src.tool_system.context", fromlist=["ToolContext"]).ToolContext(workspace_root="/tmp")
    bridge = AgentBridge(
        post_message=lambda msg: post_calls.append(msg),
        session=Mock(),
        provider=Mock(),
        tool_registry=Mock(),
        tool_context=tool_context,
        app_state=app_state,
        run_worker=lambda *_a, **_kw: None,
        max_turns=1,
        stream=False,
    )
    return bridge, post_calls, app_state


@pytest.fixture
def short_timeout(monkeypatch: pytest.MonkeyPatch):
    """Shrink the modal timeout so tests finish in milliseconds."""

    monkeypatch.setattr("clawcodex_ext.tui.agent_bridge._PERMISSION_TIMEOUT_S", 0.05)
    yield


# ----------------------------------------------------------------------
# P108-A — Permission handler
# ----------------------------------------------------------------------


def test_permission_handler_auto_denies_when_ui_does_not_respond(
    short_timeout,
) -> None:
    """Risk #2: stuck permission modal must not block the worker thread.

    We deliberately never call ``pending.decide()``. After the timeout
    the handler must return ``(False, False)`` (deny, do not remember).
    """
    bridge, posted, state = _build_bridge()

    result: dict = {}

    def _worker() -> None:
        allowed, enable = bridge._permission_handler(
            tool_name="Bash",
            message="run rm -rf?",
            suggestion=None,
        )
        result["allowed"] = allowed
        result["enable"] = enable

    t = threading.Thread(target=_worker, name="perm-worker")
    t.start()

    # Wait until the bridge has posted the request + enqueued the pending.
    deadline = time.monotonic() + 1.0
    while not posted and time.monotonic() < deadline:
        time.sleep(0.005)
    assert posted, "PermissionRequested was never posted"
    assert state.pending_permissions, "PendingPermission was never enqueued"

    # Do NOT call decide() — simulate a stuck modal.
    t.join(timeout=2.0)
    assert not t.is_alive(), "worker thread is still blocked after timeout"
    assert result == {"allowed": False, "enable": False}, f"auto-deny should return (False, False); got {result!r}"


def test_permission_handler_returns_user_choice_when_ui_responds_in_time(
    short_timeout,
) -> None:
    """Regression: the timeout path must not eat legitimate responses.

    UI calls ``decide(allowed=True, enable=True)`` before the 50 ms
    timeout elapses; the handler must return those exact values.
    """
    bridge, posted, state = _build_bridge()

    result: dict = {}

    def _worker() -> None:
        allowed, enable = bridge._permission_handler(
            tool_name="Bash",
            message="ok?",
            suggestion=None,
        )
        result["allowed"] = allowed
        result["enable"] = enable

    t = threading.Thread(target=_worker, name="perm-worker-fast")
    t.start()

    deadline = time.monotonic() + 1.0
    while not state.pending_permissions and time.monotonic() < deadline:
        time.sleep(0.005)

    # Respond immediately — well inside the 50 ms window.
    state.pending_permissions[0].decide(True, True)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert result == {"allowed": True, "enable": True}


def test_permission_handler_resolves_state_after_timeout(
    short_timeout,
) -> None:
    """After the auto-deny, the pending entry must be cleared so the
    focus router drops back to PROMPT (mirrors the existing resolve
    test for AskUserQuestion).
    """

    bridge, posted, state = _build_bridge()

    def _worker() -> None:
        bridge._permission_handler(tool_name="Bash", message="x", suggestion=None)

    t = threading.Thread(target=_worker, name="perm-worker-cleanup")
    t.start()

    deadline = time.monotonic() + 1.0
    while not state.pending_permissions and time.monotonic() < deadline:
        time.sleep(0.005)
    pending_id = state.pending_permissions[0].request_id

    t.join(timeout=2.0)
    assert not t.is_alive()
    assert all(p.request_id != pending_id for p in state.pending_permissions), (
        "state queue not drained after timeout-driven auto-deny"
    )


# ----------------------------------------------------------------------
# P108-A — AskUser handler
# ----------------------------------------------------------------------


_QUESTIONS = [
    {
        "question": "pick one",
        "header": "p",
        "multiSelect": False,
        "options": [{"label": "A", "description": ""}, {"label": "B", "description": ""}],
    },
]


def test_ask_user_handler_returns_empty_dict_when_ui_does_not_respond(
    short_timeout,
) -> None:
    """Risk #3: stuck AskUserQuestion modal must not block the worker.

    UI never calls ``decide()``; after the timeout the handler returns
    ``{}`` (parity with the Esc-cancel path already covered by
    ``test_ask_user_question.py``).
    """
    bridge, posted, state = _build_bridge()

    result: dict = {}

    def _worker() -> None:
        answers = bridge._ask_user_handler(list(_QUESTIONS))
        result["answers"] = answers

    t = threading.Thread(target=_worker, name="ask-user-worker-stuck")
    t.start()

    deadline = time.monotonic() + 1.0
    while not posted and time.monotonic() < deadline:
        time.sleep(0.005)
    assert posted, "AskUserQuestionRequested was never posted"
    assert state.pending_ask_users

    # No decide() call — wait for the timeout.
    t.join(timeout=2.0)
    assert not t.is_alive(), "worker thread is still blocked after timeout"
    assert result.get("answers") == {}, f"timeout should return empty dict; got {result.get('answers')!r}"


def test_ask_user_handler_resolves_state_after_timeout(short_timeout) -> None:
    """State queue must be drained after a timeout-driven resolution,
    matching the explicit decide() path.
    """

    bridge, _, state = _build_bridge()

    def _worker() -> None:
        bridge._ask_user_handler(list(_QUESTIONS))

    t = threading.Thread(target=_worker, name="ask-user-cleanup")
    t.start()

    deadline = time.monotonic() + 1.0
    while not state.pending_ask_users and time.monotonic() < deadline:
        time.sleep(0.005)
    pending_id = state.pending_ask_users[0].request_id

    t.join(timeout=2.0)
    assert not t.is_alive()
    assert all(p.request_id != pending_id for p in state.pending_ask_users), (
        "state queue not drained after AskUser timeout"
    )


# ----------------------------------------------------------------------
# P108-A — timeout=0 disables the auto-resolve (legacy behavior)
# ----------------------------------------------------------------------


def test_permission_handler_no_timeout_when_disabled(monkeypatch) -> None:
    """Verify permission handler no timeout when disabled."""
    monkeypatch.setattr("clawcodex_ext.tui.agent_bridge._PERMISSION_TIMEOUT_S", 0)

    bridge, posted, state = _build_bridge()

    started = threading.Event()
    result: dict = {"done": False}

    def _worker() -> None:
        started.set()
        bridge._permission_handler(tool_name="Bash", message="x", suggestion=None)
        result["done"] = True

    t = threading.Thread(target=_worker, name="perm-worker-legacy")
    t.start()
    started.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while not posted and time.monotonic() < deadline:
        time.sleep(0.005)
    assert posted, "request must still be posted with timeout=0"

    # 200 ms is far longer than any sane real timeout — the worker
    # must still be blocked.
    time.sleep(0.2)
    assert t.is_alive(), "worker should still be blocked when timeout=0"
    assert result["done"] is False

    # Cleanup: unblock the worker so the test process can exit.
    state.pending_permissions[0].decide(False, False)
    t.join(timeout=2.0)
    assert result["done"] is True


# ----------------------------------------------------------------------
# Sanity: the message classes match what we expect to be posted.
# ----------------------------------------------------------------------


def test_posted_message_is_permission_requested(short_timeout) -> None:
    """Type check: the bridge posts a ``PermissionRequested`` dataclass
    so the modal screen can route on it.
    """
    bridge, posted, _ = _build_bridge()

    def _worker() -> None:
        bridge._permission_handler(tool_name="Bash", message="y", suggestion="n")

    t = threading.Thread(target=_worker, name="perm-msg-type")
    t.start()
    deadline = time.monotonic() + 1.0
    while not posted and time.monotonic() < deadline:
        time.sleep(0.005)
    # Clean up so the worker exits.
    bridge._state.pending_permissions[0].decide(True, False)
    t.join(timeout=2.0)

    assert posted, "no message was posted"
    assert isinstance(posted[0], PermissionRequested)
    assert posted[0].tool_name == "Bash"


def test_posted_message_is_ask_user_question_requested(short_timeout) -> None:
    """Type check: AskUserQuestion posts the matching dataclass."""
    bridge, posted, _ = _build_bridge()

    def _worker() -> None:
        bridge._ask_user_handler(list(_QUESTIONS))

    t = threading.Thread(target=_worker, name="ask-msg-type")
    t.start()
    deadline = time.monotonic() + 1.0
    while not posted and time.monotonic() < deadline:
        time.sleep(0.005)
    bridge._state.pending_ask_users[0].decide({_QUESTIONS[0]["question"]: "A"})
    t.join(timeout=2.0)

    assert posted
    assert isinstance(posted[0], AskUserQuestionRequested)
