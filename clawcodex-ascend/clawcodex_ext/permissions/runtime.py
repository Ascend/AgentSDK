"""Unified runtime permission controller.

Both the REPL (``ClawcodexREPL``) and the TUI (``ClawCodexTUI``) build one
of these in their ``__init__`` and pass a reference to whichever UI
surface needs to mutate. The controller is the **single chokepoint** for
runtime permission mode switching: it serializes the multi-field write
under a lock, always goes through :func:`apply_permission_update` (never
raw attribute assignment), fires the AppState listener chain
(``_on_permission_mode_change``) so the CCR bridge / SDK status stream
stay in sync, and notifies the surface so the status bar / transcript
update.

**Why a lock?** A runtime Shift+Tab can fire from one thread (the
prompt_toolkit background thread in REPL, the Textual app thread in TUI)
while the agent worker thread is reading
``tool_context.permission_context.mode`` inside
:meth:`has_permissions_to_use_tool_inner` (``src/permissions/check.py:209``).
The single-attribute read is GIL-safe on its own, but the swap spans
THREE fields (``permission_context``, ``permission_handler``,
``allow_docs``) and a torn write would let the worker see a fresh
``permission_context.mode`` paired with a stale ``permission_handler``.
The lock holds for the full multi-field swap so the worker sees the
pre-cycle or post-cycle state, never a half-swapped one.

**Why a controller (not a free function)?** Surface lifecycle owns the
``threading.Lock`` — the lock is created when the surface is created and
discarded when the surface exits. Two surfaces must never share a lock
(they have independent lifecycles); two surfaces sharing a free function
can't tell when to recreate the lock. The controller object makes the
ownership explicit.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from clawcodex_ext.permissions.cycle import get_next_permission_mode
from clawcodex_ext.permissions.modes import has_allow_bypass_permissions_mode
from clawcodex_ext.permissions.types import PermissionMode, PermissionUpdateSetMode
from clawcodex_ext.permissions.updates import apply_permission_update
from src.state.app_state import replace_state
from src.tool_system.context import ToolContext
from src.utils.store import Store

logger = logging.getLogger(__name__)

# ``Store[AppState]`` is generic over the state type; importing the
# concrete ``AppState`` here would create a cycle with ``app_state.py``
# (which imports back into the permissions stack via
# ``set_permission_mode_listener``). Use ``Any`` for the type alias and
# cast at the call site.
AppStateStore = Store[Any]


class RuntimePermissionController:
    """Single chokepoint for runtime permission mode switching.

    Constructed once per surface (REPL or TUI). Holds:

    * A :class:`threading.Lock` that serializes the multi-field swap and
      the AppState write.
    * A :class:`Callable` (``tool_context_factory``) that returns the
      live :class:`ToolContext` (or ``None`` if the surface hasn't built
      one yet). The callable is used rather than a direct reference so
      tests that reconstruct the ``ToolContext`` mid-run keep working.
    * An optional :class:`AppStateStore` whose ``set_state`` triggers
      ``_on_permission_mode_change`` on the next dispatch tick.
    * A ``default_handler`` callable captured at construction — restored
      on the ``ToolContext.permission_handler`` slot whenever the cycle
      leaves ``bypassPermissions`` mode.
    * A ``notify`` hook the surface uses to update UI (REPL:
      ``LiveStatus.update``; TUI: ``post_message(PermissionModeChanged)``).
    """

    def __init__(
        self,
        *,
        tool_context_factory: Callable[[], ToolContext | None],
        default_handler: Callable[..., Any] | None,
        app_state_store: AppStateStore | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self._tool_context_factory = tool_context_factory
        self._default_handler = default_handler
        self._store = app_state_store
        self._notify = notify
        self._lock = threading.Lock()

    # ---- public read-only accessors (used by tests; not by surfaces) ----
    @property
    def lock(self) -> threading.Lock:
        """The :class:`threading.Lock` guarding the multi-field swap.

        Exposed for tests; production code should call :meth:`cycle` or
        :meth:`set_mode` rather than acquiring the lock directly.
        """
        return self._lock

    @property
    def default_handler(self) -> Callable[..., Any] | None:
        """The non-bypass handler snapshotted at construction time."""
        return self._default_handler

    @property
    def app_state_store(self) -> AppStateStore | None:
        """The reactive :class:`AppState` store, if wired."""
        return self._store

    def current_mode(self) -> PermissionMode:
        """Return the current ``ToolContext.permission_context.mode``.

        Reads outside the lock — the single-attribute access is
        GIL-safe and the snapshot is "good enough" for display. The
        authoritative read inside a tool dispatch happens under the
        lock via :meth:`cycle` / :meth:`set_mode`.
        """
        ctx = self._tool_context_factory()
        if ctx is None or ctx.permission_context is None:
            return "default"
        return ctx.permission_context.mode

    # ---- mutators (all hold the lock for the full swap) ----
    def cycle(self) -> PermissionMode:
        """Advance to the next permission mode and return it.

        Cycle order is the canonical Shift+Tab order
        (``default → acceptEdits → plan → bypassPermissions → default``);
        the underlying :func:`cycle_permission_mode` handles the
        ``is_bypass_permissions_mode_available=False`` fallback to
        ``default`` automatically.

        The lock is held for the full read → compute → write
        sequence. Reading outside the lock (e.g. the first version of
        this method) creates a TOCTOU window: thread A reads
        ``"default"`` and computes ``"acceptEdits"``; thread B reads
        ``"default"`` and also computes ``"acceptEdits"``; both
        ``_apply`` calls succeed, so a cycle is "lost" — the final
        state advances by 1, not 2. Keeping the read under the lock
        serializes the entire decision.
        """
        with self._lock:
            return self._cycle_locked()

    def set_mode(self, mode: PermissionMode) -> PermissionMode:
        """Set a specific permission mode and return it.

        Used by the picker (``/permissions``) where the user picks a
        specific mode rather than cycling. ``mode`` must be a valid
        external mode string (``"default"`` / ``"acceptEdits"`` /
        ``"plan"`` / ``"bypassPermissions"`` / ``"dontAsk"``).
        """
        with self._lock:
            return self._apply_locked(mode)

    # ---- internals ----
    def _resolve_bypass_available(self, perm_ctx: Any) -> bool:
        """Resolve whether the cycle may advance to ``bypassPermissions``.

        Prefers the context's stored value (the user opted in via
        ``--dangerously-skip-permissions``); falls back to the global
        :func:`has_allow_bypass_permissions_mode` helper for legacy
        contexts that pre-date the field.
        """
        stored = getattr(perm_ctx, "is_bypass_permissions_mode_available", None)
        if stored is not None:
            return bool(stored)
        try:
            return has_allow_bypass_permissions_mode()
        except Exception:
            return False

    def _cycle_locked(self) -> PermissionMode:
        """Compute the next mode and apply it. Caller must hold ``_lock``."""
        ctx = self._tool_context_factory()
        if ctx is None or ctx.permission_context is None:
            logger.debug("_cycle_locked() called before ToolContext was built; no-op")
            return "default"
        current_mode = ctx.permission_context.mode
        is_bypass_available = self._resolve_bypass_available(ctx.permission_context)
        next_mode = get_next_permission_mode(
            ToolPermissionContextFactory(  # type: ignore[arg-type]
                mode=current_mode,
                is_bypass_permissions_mode_available=is_bypass_available,
            )
        )
        return self._apply_locked(next_mode)

    def _apply_locked(self, target_mode: PermissionMode) -> PermissionMode:
        """Atomically apply ``target_mode``. Caller must hold ``_lock``.

        Performs: (1) :func:`apply_permission_update` to construct a
        fresh :class:`ToolPermissionContext`; (2) the
        ``permission_handler`` / ``allow_docs`` swap; (3) the
        :meth:`AppStateStore.set_state` write that fires
        ``_on_permission_mode_change``; (4) the surface's notify hook.

        Listeners are short-running and synchronous
        (``src/state/app_state.py:220-229`` wraps each in
        ``try/except``); the surface ``notify`` hook should be the same
        — it must not block on user input.
        """
        ctx = self._tool_context_factory()
        if ctx is None or ctx.permission_context is None:
            logger.debug("_apply_locked(%r) aborted: no ToolContext", target_mode)
            return target_mode

        # 1. Apply the canonical mutation — produces a new context.
        new_perm_ctx = apply_permission_update(
            ctx.permission_context,
            PermissionUpdateSetMode(
                type="setMode",
                destination="session",
                mode=target_mode,
            ),
        )
        ctx.permission_context = new_perm_ctx

        # 2. Swap permission_handler + allow_docs based on the new mode.
        # ``bypassPermissions`` gets the always-allow lambda; every
        # other mode restores the snapshotted default handler so the
        # cycle can return cleanly.
        if target_mode == "bypassPermissions":
            ctx.permission_handler = lambda _tn, _msg, _sug: (True, False)
            ctx.allow_docs = True
        else:
            ctx.permission_handler = self._default_handler
            ctx.allow_docs = False

        # 3. Fire the AppState listener chain. The store's set_state
        # invokes ``on_change_app_state`` → ``_on_permission_mode_change``,
        # which notifies the CCR bridge / SDK status stream.
        if self._store is not None:
            try:
                self._store.set_state(lambda s: replace_state(s, permission_mode=target_mode))
            except Exception:
                logger.exception(
                    "AppState.set_state raised during permission mode swap; "
                    "ToolContext was updated but the listener did not fire"
                )

        # 4. Notify the surface so the status bar / transcript
        # update. The hook is fire-and-forget — failures here must
        # not unwind the swap.
        if self._notify is not None:
            try:
                self._notify(target_mode)
            except Exception:
                logger.exception("notify hook raised during permission mode swap")

        return target_mode


# Tiny local helper to construct a :class:`ToolPermissionContext` for
# ``get_next_permission_mode`` without pulling in a top-level import
# (avoids a cycle: ``src.permissions.types`` → ``app_state`` → …). The
# result is consumed read-only.
def ToolPermissionContextFactory(  # type: ignore[no-redef]
    *,
    mode: str,
    is_bypass_permissions_mode_available: bool,
) -> Any:
    from clawcodex_ext.permissions.types import ToolPermissionContext

    return ToolPermissionContext(
        mode=mode,
        is_bypass_permissions_mode_available=is_bypass_permissions_mode_available,
    )


def apply_permission_mode_runtime(
    controller: RuntimePermissionController,
    target_mode: PermissionMode | None = None,
) -> PermissionMode:
    """Module-level convenience helper.

    ``target_mode=None`` → :meth:`RuntimePermissionController.cycle`
    (used by the Shift+Tab keybinding). A specific mode → :meth:`set_mode`
    (used by the picker).

    The picker call site stays readable::

        next_mode = apply_permission_mode_runtime(self._runtime_permission_controller, mode)
    """
    if target_mode is None:
        return controller.cycle()
    return controller.set_mode(target_mode)


__all__ = [
    "RuntimePermissionController",
    "apply_permission_mode_runtime",
]
