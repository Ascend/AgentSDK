"""Bridge enablement and entitlement gates.

Ports ``typescript/src/bridge/bridgeEnabled.ts``.

Per refactoring plan §0.1 Q2, GrowthBook is not wired in the Python build,
so every gate is stubbed to a static value matching the v2-priority decision
(§0.1 Q3 — v2-first). When/if GrowthBook integration lands in Phase 10, the
stubs swap to real evaluators; callers see no API change.

Returning ``True`` for ``is_bridge_enabled*`` is intentional — orchestrators
(Phase 5+) need to be exercisable end-to-end without a runtime gate refusing
them. The real entitlement decision belongs to the auth subsystem, which is
out of scope for the Python port (per §0.1 Q6).

Originally lived at ``src/bridge/bridge_enabled.py``; moved to
``clawcodex_ext/`` per the Decoupling Mandate. The ``src/`` shim is a
``sys.modules`` facade so ``from src.bridge.bridge_enabled import ...``
continues to work for the small number of legacy callers in
``extensions/`` and tests.
"""

from __future__ import annotations


def is_bridge_enabled() -> bool:
    """Cached, non-blocking entitlement check.

    Mirrors TS ``isBridgeEnabled`` on ``bridgeEnabled.ts:28-36``. Always
    True in the Python build — see module docstring.
    """
    return True


async def is_bridge_enabled_blocking() -> bool:
    """Blocking entitlement check (awaits GrowthBook in TS).

    Mirrors TS ``isBridgeEnabledBlocking`` on ``bridgeEnabled.ts:50-55``.
    Returns ``True`` synchronously — there's no GB to await.
    """
    return True


async def get_bridge_disabled_reason() -> str | None:
    """Diagnostic reason for "bridge not available", or None if enabled.

    Mirrors TS ``getBridgeDisabledReason`` on ``bridgeEnabled.ts:70-87``.
    Always None in the Python build (bridge always enabled per stubs).
    """
    return None


def is_env_less_bridge_enabled() -> bool:
    """Whether the env-less (v2) REPL bridge path is enabled.

    Mirrors TS ``isEnvLessBridgeEnabled`` on ``bridgeEnabled.ts:126-130``.
    Always True (v2-first per refactoring plan §0.1 Q3).
    """
    return True


def is_cse_shim_enabled() -> bool:
    """Whether the ``cse_*`` → ``session_*`` retag shim is active.

    Mirrors TS ``isCseShimEnabled`` on ``bridgeEnabled.ts:141-148``. Always
    True — matches the TS default and `session_id_compat.py`'s default-active
    behavior when ``set_cse_shim_gate`` is never called.
    """
    return True


def check_bridge_min_version() -> str | None:
    """Returns an error message if CLI version is below the v1 min, else None.

    Mirrors TS ``checkBridgeMinVersion`` on ``bridgeEnabled.ts:160-173``. The
    Python build has no GrowthBook version floor, so always None.
    """
    return None


def get_ccr_auto_connect_default() -> bool:
    """Default for ``remote_control_at_startup``.

    Mirrors TS ``getCcrAutoConnectDefault`` on ``bridgeEnabled.ts:185-189``.
    Always False — the user must opt in.
    """
    return False


def is_ccr_mirror_enabled() -> bool:
    """Whether CCR mirror mode is enabled.

    Mirrors TS ``isCcrMirrorEnabled`` on ``bridgeEnabled.ts:197-202``. Always
    False in the Python build — mirror mode is a niche internal feature.
    """
    return False


__all__ = [
    "check_bridge_min_version",
    "get_bridge_disabled_reason",
    "get_ccr_auto_connect_default",
    "is_bridge_enabled",
    "is_bridge_enabled_blocking",
    "is_ccr_mirror_enabled",
    "is_cse_shim_enabled",
    "is_env_less_bridge_enabled",
]
