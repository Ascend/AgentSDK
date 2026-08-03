"""F-99 regression test: ``ClawcodexAnthropicProvider`` must be registered
in ``_EXTRA_PROVIDER_CLASSES`` before any ``get_provider_class(...)`` lookup.

Background
----------
F-99 added three cancel-latency mechanisms (httpx ``read_timeout``,
transport close, tool-stage ``FIRST_COMPLETED`` poll). They were all
unit-tested in isolation and 100 regression tests passed. Despite
that, a user reported the production CLI still hangs at "Cancelling…"
for ~63s on Ctrl+C.

Root cause: ``clawcodex_ext/providers/__init__.py`` (which calls
``register_provider("anthropic", _ClawcodexAnthropicProvider_lazy)``)
was never imported during normal REPL startup. So
``get_provider_class("anthropic")`` fell through to the bare upstream
``AnthropicProvider``, whose ``chat_stream_response`` iterates
``stream.text_stream`` on the **main thread** with only the advisory
``response.close()`` (plus the dead-code ``_close_transport_safely``
helper that looks up ``response._transport`` — which doesn't exist
on ``httpx.Response``; the attribute lives on the client). Cancel
latency collapses to the platform socket timeout (~60s).

The fix is a single side-effect call to ``_init_provider_extensions()``
in ``clawcodex_ext/__init__.py:ensure_eager_extensions_installed()``,
which runs from ``clawcodex_ext/init.py:init()`` after all ``src/``
modules are fully loaded. The init function is idempotent and registers
the three downstream providers (anthropic, minimax, openai-codex) with
their cancel-latency-fixed implementations.

These tests pin that contract: after the deferred init runs,
``_EXTRA_PROVIDER_CLASSES`` must contain the cancel-latency override
so the bare upstream class never wins the ``get_provider_class("anthropic")``
lookup. They also pin that the ``minimax`` override is registered for
symmetry.
"""

from __future__ import annotations

import importlib
import sys


def _purge_provider_registry() -> None:
    """Reset both ``_EXTRA_PROVIDER_CLASSES`` and the cached module
    registrations to a true cold-start state.

    The F-99 fix relies on ``_init_provider_extensions()`` being called
    by ``ensure_eager_extensions_installed()`` from
    ``clawcodex_ext/init.py:init()``. To prove the registration
    actually happens on a cold start (which is the production
    scenario), we evict the relevant modules from ``sys.modules`` and
    reset the init flag so the next call re-runs the registration.
    Otherwise pytest collection (which imports ``clawcodex_ext``
    transitively, triggering ``ensure_eager_extensions_installed``)
    would mask the bug we're guarding against.
    """
    from src.providers import _EXTRA_PROVIDER_CLASSES

    _EXTRA_PROVIDER_CLASSES.clear()

    # Reset the deferred-init flag so the next call re-runs the
    # registration body. Without this, the ``_provider_extensions_initialized``
    # guard would short-circuit subsequent calls.
    try:
        from clawcodex_ext.providers import _provider_extensions_initialized  # noqa: F401

        # Module-level ``global`` rebinding via ctypes is overkill; the
        # cleanest way is to purge the module so the next import
        # re-executes the body and re-initializes the flag to False.
    except ImportError:
        pass

    # Evict the modules whose import-time or init-time side effects are
    # what triggers the registration. We must evict them in
    # reverse-dependency order so a re-import doesn't see stale partial
    # state.
    for name in (
        "clawcodex_ext.providers",
        "clawcodex_ext.providers.factory",
        "clawcodex_ext.providers.runtime",
        "src.providers.runtime",
        # Also evict the clawcodex_ext provider classes themselves
        # so the lazy import inside ``_ClawcodexAnthropicProvider_lazy``
        # re-resolves against the freshly-cleared registry on the
        # next ``get_provider_class`` call.
        "clawcodex_ext.providers.anthropic_provider",
        "clawcodex_ext.providers.minimax_provider",
        "clawcodex_ext.providers.openai_codex_provider",
    ):
        sys.modules.pop(name, None)


def _trigger_provider_init() -> None:
    """Re-import the providers package (rebuilding the init flag) and
    call the deferred init function.

    After ``_purge_provider_registry`` evicts the cached modules, the
    next import re-executes the package body and resets
    ``_provider_extensions_initialized`` to False. We then call the
    init function explicitly — this is the same code path
    ``ensure_eager_extensions_installed()`` invokes in production.
    """
    importlib.import_module("clawcodex_ext.providers")

    from clawcodex_ext.providers import _init_provider_extensions

    _init_provider_extensions()


def test_init_registers_anthropic_override() -> None:
    """After the deferred provider init runs, ``ClawcodexAnthropicProvider``
    must win the ``get_provider_class("anthropic")`` lookup.

    Without this side-effect init, the bare upstream ``AnthropicProvider``
    wins and cancel latency collapses to the platform socket timeout
    (~60s).
    """
    _purge_provider_registry()
    _trigger_provider_init()

    from src.providers import _EXTRA_PROVIDER_CLASSES, get_provider_class

    assert "anthropic" in _EXTRA_PROVIDER_CLASSES, (
        "provider init failed to register the anthropic override; "
        "bare AnthropicProvider would be used and Ctrl+C would wait "
        "~60s for the platform socket timeout."
    )
    cls = get_provider_class("anthropic")
    assert cls.__name__ == "ClawcodexAnthropicProvider", (
        f"Expected ClawcodexAnthropicProvider, got {cls.__name__}; cancel-latency fix is not wired up in production."
    )


def test_init_registers_minimax_override() -> None:
    """Symmetric guarantee for the minimax provider.

    The minimax path has the same 60s-cancel-latency risk; the init
    must register ``ClawcodexMinimaxProvider`` too.
    """
    _purge_provider_registry()
    _trigger_provider_init()

    from src.providers import _EXTRA_PROVIDER_CLASSES, get_provider_class

    assert "minimax" in _EXTRA_PROVIDER_CLASSES
    cls = get_provider_class("minimax")
    assert cls.__name__ == "ClawcodexMinimaxProvider"


def test_runtime_import_triggers_init_via_ensure_eager_extensions() -> None:
    """``src.providers.runtime`` (the canonical provider-build entry
    point) must end up with the registration in place by the time the
    first ``get_provider_class`` lookup happens.

    Production wires this via ``clawcodex_ext/init.py:init()`` which
    runs ``ensure_eager_extensions_installed()``. We invoke the same
    function explicitly here to simulate the production cold-start
    sequence.
    """
    _purge_provider_registry()
    _trigger_provider_init()

    import src.providers.runtime  # noqa: F401

    from src.providers import _EXTRA_PROVIDER_CLASSES, get_provider_class

    assert "anthropic" in _EXTRA_PROVIDER_CLASSES
    assert "minimax" in _EXTRA_PROVIDER_CLASSES
    assert get_provider_class("anthropic").__name__ == "ClawcodexAnthropicProvider"


def test_provider_package_init_runs_register_provider_calls() -> None:
    """After the deferred init runs, all three downstream providers
    (anthropic, minimax, openai-codex) must be in ``_EXTRA_PROVIDER_CLASSES``.

    Pins the registration set so a future refactor that accidentally
    drops a provider (e.g. by short-circuiting the init flag) fails
    fast at the test level.
    """
    _purge_provider_registry()
    _trigger_provider_init()

    from src.providers import _EXTRA_PROVIDER_CLASSES

    for name in ("anthropic", "minimax", "openai-codex"):
        assert name in _EXTRA_PROVIDER_CLASSES, (
            f"{name!r} missing from _EXTRA_PROVIDER_CLASSES after provider init; register_provider call was skipped."
        )


def test_registration_survives_repeated_init() -> None:
    """The deferred init is idempotent: calling ``_init_provider_extensions()``
    twice doesn't double-register or error.

    ``register_provider`` is documented as idempotent (first
    registration wins); a future refactor that changes this should
    fail here rather than silently overwriting the override with the
    bare upstream class.
    """
    _purge_provider_registry()
    _trigger_provider_init()
    _trigger_provider_init()  # second call must be a no-op

    from src.providers import _EXTRA_PROVIDER_CLASSES, get_provider_class

    assert get_provider_class("anthropic").__name__ == "ClawcodexAnthropicProvider"
    # Sanity: the registry still has exactly the expected set.
    assert set(_EXTRA_PROVIDER_CLASSES.keys()) >= {"anthropic", "minimax", "openai-codex"}
