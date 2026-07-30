"""Tests for :mod:`clawcodex_ext.permissions.runtime`.

Five cases covering the contract the unified runtime permission
controller must hold for both REPL and TUI:

1. ``cycle`` always goes through :func:`apply_permission_update` (no
   raw attribute mutation; the ``ToolPermissionContext`` reference
   changes).
2. ``set_mode`` fires the AppState listener chain so downstream
   observers (CCR bridge, SDK status stream) stay in sync.
3. The :class:`threading.Lock` serializes concurrent cycles — 100
   threads each calling ``cycle()`` advance the mode by exactly 100
   steps, never less.
4. Cycling out of ``bypassPermissions`` restores the snapshotted
   default handler — the controller doesn't reach into the REPL/TUI
   internals to remember it.
5. The AppState write happens under the lock — a slow ``set_state``
   blocks other mutators so the listener can't race with another
   cycle.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from clawcodex_ext.permissions.runtime import (
    RuntimePermissionController,
    apply_permission_mode_runtime,
)
from clawcodex_ext.permissions.types import ToolPermissionContext
from src.state.app_state import (
    create_app_state_store,
    set_permission_mode_listener,
)
from src.tool_system.context import ToolContext

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def default_handler() -> Any:
    """A sentinel permission handler that records the call signature."""
    recorded: list[tuple[str, str, Any]] = []

    def _handler(tool_name: str, message: str, suggestion: Any) -> tuple[bool, bool]:
        recorded.append((tool_name, message, suggestion))
        return (True, False)

    _handler.recorded = recorded  # type: ignore[attr-defined]
    return _handler


@pytest.fixture(autouse=True)
def _reset_permission_mode_listener() -> None:
    """Each test gets a clean listener slot — clear any previous registration."""
    set_permission_mode_listener(None)
    yield
    set_permission_mode_listener(None)


def _build_tool_context(
    *,
    mode: str = "default",
    is_bypass_available: bool = True,
    default_handler: Any = None,
) -> ToolContext:
    """Build a fresh :class:`ToolContext` with the given permission mode."""
    return ToolContext(
        workspace_root="/tmp",
        permission_context=ToolPermissionContext(
            mode=mode,
            is_bypass_permissions_mode_available=is_bypass_available,
        ),
        permission_handler=default_handler,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cycle_uses_apply_permission_update(default_handler: Any) -> None:
    """``cycle()`` must go through ``apply_permission_update`` and
    replace the ``ToolPermissionContext`` reference (not mutate in
    place) so the on_change tick sees a fresh object.
    """
    ctx = _build_tool_context(
        mode="default",
        is_bypass_available=True,
        default_handler=default_handler,
    )
    original_perm_ctx = ctx.permission_context
    controller = RuntimePermissionController(
        tool_context_factory=lambda: ctx,
        default_handler=default_handler,
    )

    next_mode = controller.cycle()

    assert next_mode == "acceptEdits"
    assert ctx.permission_context.mode == "acceptEdits"
    # Identity check — the new context is a fresh object, not a
    # mutation of the old one. This is what ``apply_permission_update``
    # guarantees and what the AppState identity-skip relies on.
    assert ctx.permission_context is not original_perm_ctx


def test_set_mode_fires_listener_chain(default_handler: Any) -> None:
    """``set_mode`` must fire the AppState listener chain
    (``_on_permission_mode_change``) so the CCR bridge and SDK
    status stream see the change. This is the regression test for
    the silent-bypass bug where the runtime Shift+Tab path skipped
    the listener.
    """
    ctx = _build_tool_context(
        mode="default",
        is_bypass_available=True,
        default_handler=default_handler,
    )
    store = create_app_state_store()
    captured: list[str] = []
    set_permission_mode_listener(captured.append)

    controller = RuntimePermissionController(
        tool_context_factory=lambda: ctx,
        default_handler=default_handler,
        app_state_store=store,
    )

    controller.set_mode("plan")

    # Listener fired exactly once with the new mode.
    assert captured == ["plan"]
    # AppState itself reflects the new mode.
    assert store.get_state().permission_mode == "plan"
    # ToolContext was updated.
    assert ctx.permission_context.mode == "plan"
    # Non-bypass mode: handler restored, allow_docs cleared.
    assert ctx.permission_handler is default_handler
    assert ctx.allow_docs is False


def test_lock_serializes_concurrent_cycle(default_handler: Any) -> None:
    """100 threads each calling ``cycle()`` advance the mode by
    exactly 100 steps. The lock prevents the TOCTOU race where two
    threads both read ``"default"`` and both compute ``"acceptEdits"``,
    wasting one cycle. With ``is_bypass_permissions_mode_available=True``
    the cycle length is 5 (default → acceptEdits → plan →
    bypassPermissions → dontAsk → default), so 100 % 5 = 0 lands on
    ``"default"`` (the same starting mode).
    """
    ctx = _build_tool_context(
        mode="default",
        is_bypass_available=True,
        default_handler=default_handler,
    )
    controller = RuntimePermissionController(
        tool_context_factory=lambda: ctx,
        default_handler=default_handler,
    )

    valid_modes = {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"}
    results: list[str] = []
    results_lock = threading.Lock()
    start = threading.Event()

    def _worker() -> None:
        start.wait()
        returned = controller.cycle()
        with results_lock:
            results.append(returned)

    threads = [threading.Thread(target=_worker) for _ in range(100)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == 100
    # Every returned mode is in the valid set — never a torn state.
    for r in results:
        assert r in valid_modes, f"unexpected mode: {r!r}"
    # Final state is exactly (initial + 100) % 5 = 0 → "default".
    assert controller.current_mode() == "default"
    assert ctx.permission_context.mode == "default"


def test_default_handler_is_restored_on_cycle_out_of_bypass(
    default_handler: Any,
) -> None:
    """Starting in ``bypassPermissions`` and cycling once should
    restore the snapshotted default handler so subsequent permission
    requests route through the normal ``_handle_permission_request``
    flow (UI modal), not the always-allow lambda.
    """
    ctx = _build_tool_context(
        mode="bypassPermissions",
        is_bypass_available=True,
        default_handler=default_handler,
    )
    # Simulate the runtime bypass state — the controller would have
    # installed the always-allow lambda as a side effect of a prior
    # cycle into bypassPermissions.
    ctx.permission_handler = lambda _tn, _msg, _sug: (True, False)
    ctx.allow_docs = True
    assert ctx.default_permission_handler is default_handler

    controller = RuntimePermissionController(
        tool_context_factory=lambda: ctx,
        default_handler=default_handler,
    )

    next_mode = controller.cycle()

    # After bypassPermissions, the cycle advances to dontAsk.
    assert next_mode == "dontAsk"
    assert ctx.permission_context.mode == "dontAsk"
    # The snapshotted default handler is back in place — the
    # controller didn't reach into ``self._handle_permission_request``
    # to find it; it read ``tool_context.default_permission_handler``
    # (which ``__post_init__`` defaults to the constructed handler).
    assert ctx.permission_handler is default_handler
    assert ctx.allow_docs is False

    # Cycle once more — from dontAsk → default. Handler is still
    # the default (not the lambda).
    next_mode2 = controller.cycle()
    assert next_mode2 == "default"
    assert ctx.permission_handler is default_handler
    assert ctx.allow_docs is False

    # Full cycle with bypass available is 5 steps:
    # default → acceptEdits → plan → bypassPermissions → dontAsk → default.
    # So from default we need 3 more cycles to reach bypassPermissions
    # (default → acceptEdits → plan → bypassPermissions).
    next_mode3 = controller.cycle()
    assert next_mode3 == "acceptEdits"
    assert ctx.permission_handler is default_handler
    assert ctx.allow_docs is False

    next_mode4 = controller.cycle()
    assert next_mode4 == "plan"
    assert ctx.permission_handler is default_handler
    assert ctx.allow_docs is False

    # And cycling back into bypassPermissions installs the lambda
    # again, but the default handler is still snapshotted.
    next_mode5 = controller.cycle()
    assert next_mode5 == "bypassPermissions"
    assert ctx.permission_handler is not default_handler
    assert ctx.allow_docs is True
    # ``default_permission_handler`` is the snapshot, not the lambda.
    assert ctx.default_permission_handler is default_handler


def test_set_state_is_held_under_lock(default_handler: Any) -> None:
    """The AppState write happens under the controller's lock.

    Monkeypatch ``store.set_state`` to sleep briefly, start a
    ``cycle()`` in a thread, and verify the lock is held for the
    duration. A second thread (or ``lock.acquire(timeout=...)`` from
    the main thread) cannot acquire it during the sleep.
    """
    ctx = _build_tool_context(
        mode="default",
        is_bypass_available=True,
        default_handler=default_handler,
    )

    sleep_seconds = 0.10
    set_state_entered = threading.Event()
    set_state_release = threading.Event()
    real_set_state_calls: list[Any] = []

    def _slow_set_state(updater: Any) -> None:
        set_state_entered.set()
        # Hold the lock (we are inside ``_apply_locked`` which holds
        # ``controller._lock``) for ``sleep_seconds``.
        time.sleep(sleep_seconds)
        real_set_state_calls.append(updater)
        set_state_release.set()

    store = create_app_state_store()
    store.set_state = _slow_set_state  # type: ignore[method-assign]

    controller = RuntimePermissionController(
        tool_context_factory=lambda: ctx,
        default_handler=default_handler,
        app_state_store=store,
    )

    cycle_done = threading.Event()

    def _cycle_worker() -> None:
        controller.cycle()
        cycle_done.set()

    worker = threading.Thread(target=_cycle_worker)
    worker.start()

    try:
        # Wait for the worker to enter the lock + start the sleep.
        assert set_state_entered.wait(timeout=2.0), "worker did not enter set_state"

        # The lock is held right now. ``acquire(blocking=False)`` returns
        # False immediately.
        assert controller.lock.acquire(blocking=False) is False, (
            "controller lock was free while set_state was sleeping — the swap is not properly serialized"
        )
        # ``acquire(timeout=...)`` with a short timeout also fails
        # because the worker is mid-sleep.
        assert controller.lock.acquire(timeout=0.02) is False
    finally:
        # Unblock the worker so it can finish and the thread can join.
        set_state_release.set()
        # Defensive: if the worker is somehow past the sleep already
        # (timing flake), the release event is a no-op.
        assert cycle_done.wait(timeout=2.0)
        worker.join(timeout=2.0)

    # The set_state was actually called with the right updater.
    assert len(real_set_state_calls) == 1


# ---------------------------------------------------------------------------
# Module-level convenience helper
# ---------------------------------------------------------------------------


def test_apply_permission_mode_runtime_cycle_shortcut(
    default_handler: Any,
) -> None:
    """``apply_permission_mode_runtime(controller)`` (no mode) calls
    ``cycle()``; passing a mode calls ``set_mode()``.
    """
    ctx = _build_tool_context(
        mode="default",
        is_bypass_available=True,
        default_handler=default_handler,
    )
    controller = RuntimePermissionController(
        tool_context_factory=lambda: ctx,
        default_handler=default_handler,
    )

    # No arg → cycle
    result = apply_permission_mode_runtime(controller)
    assert result == "acceptEdits"
    assert ctx.permission_context.mode == "acceptEdits"

    # Specific mode → set_mode
    result2 = apply_permission_mode_runtime(controller, "plan")
    assert result2 == "plan"
    assert ctx.permission_context.mode == "plan"
