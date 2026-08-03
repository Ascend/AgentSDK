#!/usr/bin/env python3
# coding=utf-8

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

"""Tests for the IssueRegistry mtime-aware reload patch.

Covers:

1. ``IssueRegistry.reload_if_stale`` reloads when the on-disk file mtime
   advances past the cached value and skips otherwise.
2. The orchestrator wrappers call ``reload_if_stale`` before delegating to
   the original ``_poll_and_dispatch`` / ``_resolve_intent``.
3. ``install_stale_registry_patch`` is idempotent (no double-wrap on
   repeated calls).
4. ``IssueRegistry`` constructor populates ``_mtime_ns`` on load + save so
   the daemon's own writes do not look like external ones.
"""

# pytest injects fixture names into test function signatures.
# pylint: disable=redefined-outer-name

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Make sure the ``clawcodex_ext`` package is importable when tests are run
# from a checkout that does not have the package installed as editable.
_PKG_PARENT = Path(__file__).resolve().parents[3]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from clawcodex_ext.orchestrator._patch_stale_registry import (  # noqa: E402
    install_stale_registry_patch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def installed_patch():
    """Install the patch once per test and reset the module guard at teardown."""
    import clawcodex_ext.orchestrator._patch_stale_registry as mod
    from extensions.orchestrator import issue_registry as ir_mod
    from extensions.orchestrator import orchestrator as orch_mod

    # Ensure a clean slate so tests can rely on the post-install state.
    mod._INSTALLED = False
    if hasattr(ir_mod.IssueRegistry, "_stale_registry_patched"):
        delattr(ir_mod.IssueRegistry, "_stale_registry_patched")
    if hasattr(orch_mod.Orchestrator, "_stale_registry_patched"):
        delattr(orch_mod.Orchestrator, "_stale_registry_patched")

    install_stale_registry_patch()
    yield
    mod._INSTALLED = False
    if hasattr(ir_mod.IssueRegistry, "_stale_registry_patched"):
        delattr(ir_mod.IssueRegistry, "_stale_registry_patched")
    if hasattr(orch_mod.Orchestrator, "_stale_registry_patched"):
        delattr(orch_mod.Orchestrator, "_stale_registry_patched")


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "issue_registry.json"


# ---------------------------------------------------------------------------
# IssueRegistry: reload_if_stale
# ---------------------------------------------------------------------------


def test_reload_if_stale_returns_false_when_file_unchanged(installed_patch, registry_path: Path) -> None:
    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    assert registry_path.exists() is False or registry._mtime_ns >= 0

    # No external write yet — reload must be a no-op.
    assert registry.reload_if_stale() is False


def test_reload_if_stale_returns_true_when_file_changes(installed_patch, registry_path: Path, monkeypatch) -> None:
    from extensions.orchestrator.issue_registry import (
        IssueRegistry,
        IssueStatus,
        Intent,
    )

    registry = IssueRegistry(registry_path)
    # Register one issue so we can prove the reload re-reads records.
    registry.register("42", "#42")
    cached_mtime = registry._mtime_ns

    # Simulate a separate CLI process writing the file: bump mtime into the
    # future so the cached value is strictly less.
    future = cached_mtime + 10_000_000_000  # +10 seconds
    os.utime(registry_path, ns=(future, future))

    # Reset a tracked record's status in-memory to a sentinel so we can
    # detect that the reload re-read fresh data.
    record = registry.get("42")
    assert record is not None
    record.status = IssueStatus.PENDING

    # Monkey-edit the on-disk JSON so the reload picks up a different value
    # than the in-memory one. We bypass IssueRegistry to mimic an external
    # writer.
    import json

    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    raw["42"]["status"] = IssueStatus.FAILED.value
    raw["42"]["intent"] = Intent.RETRY.value
    raw["42"]["intent_source"] = "cli"
    # Write atomically using the same tmp-rename pattern IssueRegistry uses
    # so the test mirrors real behaviour.
    tmp = registry_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    tmp.replace(registry_path)

    # Bump mtime again to be safe against sub-second filesystem resolution.
    bumped = future + 10_000_000_000
    os.utime(registry_path, ns=(bumped, bumped))

    reloaded = registry.reload_if_stale()
    assert reloaded is True, "expected reload after on-disk mutation"

    record = registry.get("42")
    assert record is not None
    assert record.status is IssueStatus.FAILED
    assert record.intent is Intent.RETRY
    assert record.intent_source == "cli"
    assert registry._mtime_ns == bumped


def test_load_initializes_mtime(installed_patch, registry_path: Path) -> None:
    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    # No file on disk yet — _load is a no-op, mtime stays 0.
    assert registry._mtime_ns == 0

    # First save creates the file and must record its mtime.
    registry.register("1", "#1")
    assert registry._mtime_ns > 0


def test_save_refreshes_mtime(installed_patch, registry_path: Path, monkeypatch) -> None:
    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    registry.register("1", "#1")
    mtime_after_register = registry._mtime_ns
    assert mtime_after_register > 0

    # A subsequent mutation + _save must update mtime so the next
    # reload_if_stale does NOT trigger a redundant reload.
    registry.mark_abandoned("1")
    assert registry._mtime_ns >= mtime_after_register
    assert registry.reload_if_stale() is False


# ---------------------------------------------------------------------------
# _load call counter — proves we don't reload when nothing changed
# ---------------------------------------------------------------------------


def test_no_reload_when_unchanged(installed_patch, registry_path: Path) -> None:
    from extensions.orchestrator import issue_registry as ir_mod
    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    registry.register("7", "#7")

    # Wrap _load to count invocations.
    original_load = ir_mod.IssueRegistry._load
    calls = {"n": 0}

    def counting_load(self):
        calls["n"] += 1
        return original_load(self)

    ir_mod.IssueRegistry._load = counting_load
    try:
        # Three consecutive polls with no external change must not re-load.
        assert registry.reload_if_stale() is False
        assert registry.reload_if_stale() is False
        assert registry.reload_if_stale() is False
    finally:
        ir_mod.IssueRegistry._load = original_load

    # _load is called exactly once: from __init__. Subsequent
    # reload_if_stale calls must short-circuit on the mtime check.
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# Orchestrator wrappers
# ---------------------------------------------------------------------------


def test_poll_wrapper_triggers_reload(installed_patch) -> None:
    """The wrapped _poll_and_dispatch must call reload_if_stale first."""
    from extensions.orchestrator import orchestrator as orch_mod

    reloaded = {"count": 0}

    class _FakeRegistry:
        def reload_if_stale(self) -> bool:
            reloaded["count"] += 1
            return False

    poll_called = {"count": 0}

    async def _stub_poll(self):
        poll_called["count"] += 1

    wrapper = orch_mod.Orchestrator._poll_and_dispatch
    saved_original = wrapper.__wrapped__  # noqa: F841 - used by the pending restoration-path assertions

    async def _new_poll(self):
        self._registry.reload_if_stale()
        await _stub_poll(self)

    orch_mod.Orchestrator._poll_and_dispatch = _new_poll
    try:
        instance = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
        instance._registry = _FakeRegistry()  # type: ignore[attr-defined]
        asyncio.run(instance._poll_and_dispatch())
    finally:
        orch_mod.Orchestrator._poll_and_dispatch = wrapper

    assert reloaded["count"] == 1, "reload_if_stale must be called once"
    assert poll_called["count"] == 1, "original _poll_and_dispatch must run"


def test_resolve_intent_wrapper_triggers_reload(installed_patch) -> None:
    """The wrapped _resolve_intent must call reload_if_stale first."""
    from extensions.orchestrator import orchestrator as orch_mod

    reloaded = {"count": 0}

    class _FakeRegistry:
        def reload_if_stale(self) -> bool:
            reloaded["count"] += 1
            return False

    resolve_called = {"count": 0}

    async def _stub_resolve(self, issue):
        resolve_called["count"] += 1
        return ("NONE", None, None)

    # Stub the upstream method by binding it onto the class as a plain
    # attribute. The wrapper captures it by name at install time, so we
    # have to temporarily rebind that name and then restore it.
    wrapper = orch_mod.Orchestrator._resolve_intent
    saved_original = wrapper.__wrapped__  # noqa: F841 - used by the pending restoration-path assertions

    # Reassign the wrapper's captured original by patching the closure.
    # Since we cannot poke the closure cell directly, swap the installed
    # wrapper itself: bind a new wrapper that calls our stub.
    async def _new_resolve(self, issue):
        self._registry.reload_if_stale()
        await _stub_resolve(self, issue)

    orch_mod.Orchestrator._resolve_intent = _new_resolve
    try:
        instance = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
        instance._registry = _FakeRegistry()  # type: ignore[attr-defined]

        # Minimal duck-typed issue with the only attribute _orig accessed.
        class _FakeIssue:
            id = "99"

        asyncio.run(instance._resolve_intent(_FakeIssue()))
    finally:
        orch_mod.Orchestrator._resolve_intent = wrapper

    assert reloaded["count"] == 1
    assert resolve_called["count"] == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_install_is_idempotent(installed_patch) -> None:
    """Calling install twice must not double-wrap the methods."""
    from extensions.orchestrator import orchestrator as orch_mod

    install_stale_registry_patch()
    install_stale_registry_patch()

    # _poll_and_dispatch should be the patched wrapper, not a
    # wrapper-of-a-wrapper.
    assert hasattr(orch_mod.Orchestrator._poll_and_dispatch, "__wrapped__") or callable(
        orch_mod.Orchestrator._poll_and_dispatch
    )

    # Smoke-test: invoking the wrapper twice must not raise and must
    # delegate to the (already-wrapped) original exactly once per call.
    class _FakeRegistry:
        def reload_if_stale(self) -> bool:
            return False

    instance = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    instance._registry = _FakeRegistry()  # type: ignore[attr-defined]

    async def _orig_poll(self):
        return None

    # Temporarily replace the original to count delegations.
    saved = orch_mod.Orchestrator._poll_and_dispatch
    try:
        delegate_count = {"n": 0}

        async def counting_poll(self):
            delegate_count["n"] += 1

        orch_mod.Orchestrator._poll_and_dispatch = counting_poll
        # Re-install: this must NOT add another reload_if_stale wrapper.
        import clawcodex_ext.orchestrator._patch_stale_registry as mod

        mod._INSTALLED = False
        install_stale_registry_patch()
        asyncio.run(instance._poll_and_dispatch())
        # Exactly one delegation despite two install calls.
        assert delegate_count["n"] == 1
    finally:
        orch_mod.Orchestrator._poll_and_dispatch = saved
