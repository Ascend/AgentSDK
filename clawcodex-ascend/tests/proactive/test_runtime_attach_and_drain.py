#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Runtime-attach and wake-drain tests for the kairos closed loop.

Covers the mirror of cron's runtime wiring introduced for proactive ticks:

* :func:`attach_proactive_runtime` mounts exactly one tick emitter when
  KAIROS / PROACTIVE is enabled and is a strict no-op when it is not;
* the REPL's :meth:`~clawcodex_ext.repl.core.ClawcodexREPL._drain_proactive_outbox`
  picks tick / sleep-wake events up from the outbox and injects them through
  the cron prompt queue (same echo / permission semantics as cron fires);
* ``/proactive`` reuses the runtime-attached emitter so on/off controls one
  scheduler, never a second one.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from clawcodex_ext.command_system.builtins import execute_command_sync
from clawcodex_ext.command_system.engine import create_command_context
from clawcodex_ext.command_system.proactive_command import _EMITTERS_BY_CONTEXT_ID
from clawcodex_ext.feature_gate import FeatureFlag, get_registry
from clawcodex_ext.query.outbox_types import CronPromptEvent, ProactivePromptEvent
from clawcodex_ext.repl.core import ClawcodexREPL  # pylint: disable=no-name-in-module
from clawcodex_ext.services.proactive import TickEmitter, get_default_controller
from clawcodex_ext.services.proactive.runtime import (
    attach_proactive_runtime,
    drain_proactive_prompts,
    is_proactive_feature_enabled,
)
from clawcodex_ext.tool_system.context import ToolContext

_FEATURE_NAMES = ("KAIROS", "PROACTIVE")


@pytest.fixture(autouse=True)
def _feature_gate_baseline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin KAIROS / PROACTIVE to their registered defaults.

    The registry is a process-wide singleton and ``enable_feature`` is an
    in-memory override, so every test here must start from (and restore)
    the pristine no-override state.
    """
    reg = get_registry()
    for name in _FEATURE_NAMES:
        if reg.get_flag(name) is None:
            reg.register(FeatureFlag(name, default=False))
    for name in ("CLAWCODEX_FEATURE_KAIROS", "CLAWCODEX_FEATURE_PROACTIVE"):
        monkeypatch.delenv(name, raising=False)
    reg.clear_all_overrides()
    try:
        yield
    finally:
        reg.clear_all_overrides()


def _enable_proactive() -> None:
    get_registry().enable_feature("PROACTIVE")


def _enable_kairos_only() -> None:
    get_registry().enable_feature("KAIROS")


def _force_features_off() -> None:
    reg = get_registry()
    for name in _FEATURE_NAMES:
        reg.set_override(name, False)


def _make_repl(tmp_path: Path) -> ClawcodexREPL:
    """Build a minimally wired REPL instance (mirror of the cron tests)."""
    repl = ClawcodexREPL.__new__(ClawcodexREPL)
    repl.tool_context = SimpleNamespace(workspace_root=tmp_path, outbox=[])
    repl._queued_prompts = deque()
    repl._cron_queued_prompts = deque()
    repl._queued_prompts_lock = threading.Lock()
    return repl


# ---------------------------------------------------------------------------
# attach_proactive_runtime: feature-off no-op
# ---------------------------------------------------------------------------


def test_attach_is_noop_when_feature_off_on_real_context(tmp_path: Path) -> None:
    _force_features_off()

    tool_context = ToolContext(workspace_root=tmp_path)
    emitter = attach_proactive_runtime(tool_context, autostart=True)

    assert emitter is None
    # ToolContext declares the slot (default None) so the attach must leave
    # it untouched -- no emitter recorded, nothing started.
    assert getattr(tool_context, "proactive_emitter", None) is None


def test_attach_sets_no_attributes_when_feature_off() -> None:
    _force_features_off()

    bare = SimpleNamespace()

    assert attach_proactive_runtime(bare, autostart=True) is None
    assert not hasattr(bare, "proactive_emitter")
    assert not hasattr(bare, "outbox")


# ---------------------------------------------------------------------------
# attach_proactive_runtime: feature-on mounting
# ---------------------------------------------------------------------------


def test_attach_mounts_exactly_one_scheduler_when_feature_on(tmp_path: Path) -> None:
    _enable_proactive()
    tool_context = ToolContext(workspace_root=tmp_path)
    original_outbox = tool_context.outbox

    emitter = attach_proactive_runtime(tool_context, autostart=True)
    assert emitter is not None
    assert isinstance(emitter, TickEmitter)
    try:
        # Recorded on the context, wired to the shared outbox + default
        # controller, and running -- with exactly one scheduler.
        assert getattr(tool_context, "proactive_emitter", None) is emitter
        assert emitter._outbox is original_outbox
        assert emitter._ctrl is get_default_controller()
        assert emitter.scheduler.start() is False  # second start is a no-op
        assert emitter.scheduler.config.interval_seconds == 30
    finally:
        emitter.stop()


def test_attach_is_idempotent_and_returns_same_instance(tmp_path: Path) -> None:
    _enable_proactive()
    tool_context = ToolContext(workspace_root=tmp_path)

    first = attach_proactive_runtime(tool_context, autostart=True)
    second = attach_proactive_runtime(tool_context, autostart=True)
    assert first is not None
    assert second is first
    try:
        # autostart on the second call must not spawn a second scheduler.
        assert first.scheduler.start() is False
    finally:
        first.stop()


def test_attach_is_gated_by_kairos_flag_too(tmp_path: Path) -> None:
    _enable_kairos_only()
    assert is_proactive_feature_enabled()
    tool_context = ToolContext(workspace_root=tmp_path)

    emitter = attach_proactive_runtime(tool_context, autostart=True)
    assert emitter is not None
    try:
        assert getattr(tool_context, "proactive_emitter", None) is emitter
    finally:
        emitter.stop()


# ---------------------------------------------------------------------------
# /proactive reuse of the runtime-attached emitter
# ---------------------------------------------------------------------------


def test_proactive_on_off_reuses_attached_emitter(tmp_path: Path) -> None:
    _enable_proactive()
    tool_context = ToolContext(workspace_root=tmp_path)
    emitter = attach_proactive_runtime(tool_context, autostart=True)
    assert emitter is not None
    context = create_command_context(tmp_path, tool_context=tool_context)
    try:
        success, _, error = execute_command_sync("proactive", "on", context)
        assert success
        assert error is None
        # The slash command must control the mounted emitter, not start a
        # second scheduler of its own. Runtime-mounted emitters are reused
        # via the context attribute and are never recorded in the module
        # dict (that dict holds only command-created instances, so it can
        # never pin a live context for the process lifetime).
        assert _EMITTERS_BY_CONTEXT_ID.get(id(tool_context)) is None
        assert get_default_controller().state.phase == "active"
        assert emitter.scheduler.start() is False
    finally:
        off_success, _, off_error = execute_command_sync("proactive", "off", context)
        assert off_success
        assert off_error is None
        assert get_default_controller().state.phase == "inactive"
        assert id(tool_context) not in _EMITTERS_BY_CONTEXT_ID
        # ``off`` stopped the mounted scheduler; a later start restarts it.
        assert emitter.scheduler.start() is True
        emitter.stop()


# ---------------------------------------------------------------------------
# drain_proactive_prompts unit behaviour
# ---------------------------------------------------------------------------


def test_drain_pops_typed_and_dict_events_keeps_others() -> None:
    blank = ProactivePromptEvent(prompt="   ")
    outbox: list = [
        CronPromptEvent(prompt="cron job", task_id="t1", run_id="r1"),
        ProactivePromptEvent(prompt="<tick>10:00:00</tick>", source="tick"),
        {"type": "proactive_prompt", "prompt": "sleep wake-up", "source": "sleep"},
        {"type": "cron_prompt", "prompt": "dict cron", "task_id": "t2", "run_id": "r2"},
        {"type": "some_other_event", "prompt": "unknown"},
        blank,
    ]

    drained = drain_proactive_prompts(outbox)

    assert drained == ["<tick>10:00:00</tick>", "sleep wake-up"]
    # Cron events, unknown events and blank proactive prompts stay put.
    assert len(outbox) == 4
    assert outbox[0].get("type") == "cron_prompt"
    assert outbox[1].get("type") == "cron_prompt"
    assert outbox[2].get("type") == "some_other_event"
    assert outbox[3] is blank


# ---------------------------------------------------------------------------
# REPL wake-loop drain: tick-due events are injected as prompts
# ---------------------------------------------------------------------------


def test_repl_drain_injects_tick_and_wake_prompts(tmp_path: Path) -> None:
    _enable_proactive()
    repl = _make_repl(tmp_path)
    tick_text = "<tick>10:00:00</tick>"
    wake_text = "focus check"
    repl.tool_context.outbox = [
        ProactivePromptEvent(prompt=tick_text, source="tick"),
        {"type": "proactive_prompt", "prompt": wake_text, "source": "sleep"},
        CronPromptEvent(prompt="cron job", task_id="t1", run_id="r1"),
    ]

    repl._drain_proactive_outbox()

    # Proactive entries drained; the cron entry is left for its own consumer.
    assert [e.get("type") for e in repl.tool_context.outbox] == ["cron_prompt"]
    # Injected through the cron queue -- same echo / permission semantics as
    # a cron fire (lower priority than user input, source "cron").
    first = repl._pop_queued_prompt()
    second = repl._pop_queued_prompt()
    assert first == (tick_text, "cron")
    assert second == (wake_text, "cron")
    assert repl._pop_queued_prompt() is None


def test_repl_drain_enqueues_nothing_when_feature_off(tmp_path: Path) -> None:
    _force_features_off()
    repl = _make_repl(tmp_path)
    tick = ProactivePromptEvent(prompt="<tick>10:00:00</tick>", source="tick")
    cron = CronPromptEvent(prompt="cron job", task_id="t1", run_id="r1")
    repl.tool_context.outbox = [tick, cron]

    repl._drain_proactive_outbox()

    # With the feature off nothing is injected; the stale tick is drained
    # (dropped) so it can never keep the 1s outbox watcher in a wake loop,
    # and cron entries are untouched.
    assert repl._pop_queued_prompt() is None
    assert [e.get("type") for e in repl.tool_context.outbox] == ["cron_prompt"]


def test_repl_drain_noop_on_empty_outbox(tmp_path: Path) -> None:
    _enable_proactive()
    repl = _make_repl(tmp_path)
    repl.tool_context.outbox = []

    repl._drain_proactive_outbox()

    assert repl._pop_queued_prompt() is None


# ---------------------------------------------------------------------------
# RuntimeContext.build: the two cron mirror sites in the unified factory
# ---------------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, model: str | None) -> None:
        self.model = model or "fake"


class _FakeRegistry:
    def __init__(self, provider: _FakeProvider) -> None:
        self.provider = provider
        self._tools = []

    def register(self, tool: object) -> None:
        self._tools.append(tool)

    def unregister(self, name: str) -> None:
        self._tools = [t for t in self._tools if getattr(t, "name", "") != name]

    def list_tools(self) -> list:
        return list(self._tools)


def _stub_runtime_build_seams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bypass provider/config/registry resolution so build runs offline.

    Mirrors the scaffolding of ``tests/multimodel/test_cli_and_runtime.py``:
    only the kairos attach under test stays real.
    """
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.resolve",
        lambda **kwargs: SimpleNamespace(provider="glm", model="zai/glm-4"),
    )
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.attach_cron_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.replace_cron_tools",
        lambda registry: None,
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: _FakeProvider(model),
    )
    monkeypatch.setattr(
        "src.tool_system.defaults.build_default_registry",
        lambda provider, **kwargs: _FakeRegistry(provider),
    )


def test_runtime_context_build_attaches_when_feature_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable_proactive()
    _stub_runtime_build_seams(monkeypatch, tmp_path)
    from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions

    runtime = RuntimeContext.build(RuntimeOptions(workspace_root=tmp_path))

    tc_emitter = getattr(runtime.tool_context, "proactive_emitter", None)
    rt_emitter = getattr(runtime, "proactive_emitter", None)
    assert isinstance(tc_emitter, TickEmitter)
    assert isinstance(rt_emitter, TickEmitter)
    try:
        # tool-context emitter is the live one (autostart=True in build)...
        assert tc_emitter._outbox is runtime.tool_context.outbox
        assert tc_emitter.scheduler.start() is False
        # ...while the ctx-level mirror on the runtime object stays dormant,
        # exactly like cron's second attach in RuntimeContext.build.
        assert rt_emitter.scheduler.is_running is False
    finally:
        tc_emitter.stop()
        rt_emitter.stop()


def test_runtime_context_build_noop_when_feature_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _force_features_off()
    _stub_runtime_build_seams(monkeypatch, tmp_path)
    from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions

    runtime = RuntimeContext.build(RuntimeOptions(workspace_root=tmp_path))

    assert getattr(runtime, "proactive_emitter", None) is None
    assert getattr(runtime.tool_context, "proactive_emitter", None) is None
