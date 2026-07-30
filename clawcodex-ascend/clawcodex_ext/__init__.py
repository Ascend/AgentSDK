"""Downstream ClawCodex extension layer.

This package's ``__init__.py`` is intentionally kept **thin** — it does
NOT import any subpackage at module load time.  Every heavy subpackage
(``permissions``, ``providers``, ``models``, ``agent``, ``orchestrator``,
…) is imported lazily on first access.  This reduces cold-start import
overhead from ~3.5 s to ~0.05 s for callers that never need the full
extension layer (e.g. CLI ``--help``).

Wiring rationale
----------------
Four downstream extensions must be registered before they can be used:

1. ``install_permission_extensions`` — registers the ``bypassPermissions →
   dontAsk`` cycle step and the LLM auto-mode classifier.
2. ``install_memory_extension`` — registers a scope-aware memory section
   builder with the prompt-assembly system.
3. ``install_provider_patches`` — warms the model registry discovery
   cache.
4. ``install_stale_registry_patch`` — monkey-patches the orchestrator
   daemon's issue registry for live-reload.

These were historically called at package-import time (lines 29-44 of the
original file), which pulled in ~100 submodules on any ``import
clawcodex_ext``.  The calls are now deferred to
:func:`ensure_eager_extensions_installed`, invoked from the per-process
bootstrap in ``clawcodex_ext/init.py:init()`` — the same canonical hook
that already registers the nested-session transcript resolver.  By the
time ``init()`` runs, every ``src/`` module is fully loaded, so the
circular-import constraints documented in the original file (``src.tool_system
build_tool`` vs ``src.agent.transcript`` partial-init cycle) are no longer
a concern.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("clawcodex-cli")
except PackageNotFoundError:
    __version__ = "1.2.1"
__author__ = "Claw Codex Team"


def __getattr__(name: str):
    """Expose the upstream root API without defeating the thin import."""
    if name in {"load_config", "get_provider_config"}:
        from clawcodex_ext import config  # pylint: disable=import-self,no-name-in-module

        value = getattr(config, name)
        globals()[name] = value
        return value
    if name == "BaseProvider":
        try:
            from clawcodex_ext.providers.base import BaseProvider  # pylint: disable=no-name-in-module
        except Exception:  # pragma: no cover - optional provider deps
            BaseProvider = None  # type: ignore[assignment,misc]
        globals()[name] = BaseProvider
        return BaseProvider
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Lazy extension installations
# ---------------------------------------------------------------------------

_eager_extensions_installed: bool = False


def ensure_eager_extensions_installed() -> None:
    """Install the four downstream extensions once.

    Idempotent — safe to call more than once.  Invoked from the canonical
    per-process bootstrap (``clawcodex_ext/init.py:init()``) after all
    ``src/`` modules are fully loaded, sidestepping the partial-init
    circular-import cycle that would occur if these were called at package
    import time.
    """
    global _eager_extensions_installed
    if _eager_extensions_installed:
        return
    _eager_extensions_installed = True

    from clawcodex_ext.memory.scope_aware_prompt import install_memory_extension  # pylint: disable=no-name-in-module
    from clawcodex_ext.orchestrator import install_stale_registry_patch  # pylint: disable=no-name-in-module
    from clawcodex_ext.permissions import install_permission_extensions  # pylint: disable=no-name-in-module
    from clawcodex_ext.providers.patches import install as _install_provider_patches  # pylint: disable=no-name-in-module

    install_permission_extensions()
    install_memory_extension()
    _install_provider_patches()
    install_stale_registry_patch()

    # F-94 BG_SESSIONS — wrap launch_background_runner to upsert the global
    # index after the per-session marker is written. No-op when
    # CLAWCODEX_BG_SESSIONS=off (验收标准 1).
    try:
        from clawcodex_ext.tasks.bg_session_hook import install_bg_session_index_hook  # pylint: disable=no-name-in-module

        install_bg_session_index_hook()
    except Exception:  # nosec B110
        pass

    # F-84 P84-H — register ``daemon`` subcommand behind the
    # DAEMON feature gate. No-op when the flag is disabled.
    try:
        from clawcodex_ext.daemon import install_daemon_gate  # pylint: disable=no-name-in-module

        install_daemon_gate()
    except Exception:  # nosec B110
        pass

    # Provider registrations (model extensions, downstream providers,
    # cancel-latency overrides, media registry).
    from clawcodex_ext.providers import _init_provider_extensions  # pylint: disable=no-name-in-module

    _init_provider_extensions()

    # F-88 Monitor — install runtime hooks (TUI keybinding, stall watchdog
    # exemption) after upstream modules are fully loaded.
    try:
        from clawcodex_ext.services.monitor.install import install_monitor_extensions

        install_monitor_extensions()
    except Exception:  # nosec B110
        pass


# ---------------------------------------------------------------------------
# Nested-transcript lazy initializer (unchanged semantics)
# ---------------------------------------------------------------------------

_nested_transcript_initialized: bool = False


def ensure_nested_transcript_initialized() -> None:
    """Register the nested-session transcript path resolver.

    Safe to call multiple times. Must be called from a context where
    ``src/`` modules are fully loaded (e.g. the CLI entry point or
    the REPL launcher) — invoking it during ``clawcodex_ext`` package
    import would re-trigger the circular import that this lazy
    wrapper exists to avoid.
    """
    global _nested_transcript_initialized
    if _nested_transcript_initialized:
        return
    from clawcodex_ext.agent.transcript import init as _init_nested_transcript  # pylint: disable=no-name-in-module

    _init_nested_transcript()
    _nested_transcript_initialized = True


__all__ = [
    "BaseProvider",  # pylint: disable=undefined-all-variable
    "__author__",
    "__version__",
    "ensure_eager_extensions_installed",
    "ensure_nested_transcript_initialized",
    "get_provider_config",  # pylint: disable=undefined-all-variable
    "load_config",  # pylint: disable=undefined-all-variable
]
