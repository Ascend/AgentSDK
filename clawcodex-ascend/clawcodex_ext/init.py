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

"""Bootstrap init — chapter phase 2 of the bootstrap pipeline.

Mirrors ``typescript/src/entrypoints/init.ts``. The memoize property is
load-bearing: multiple entry points (REPL, headless, MCP fast-path if
it grows to need init) may each call ``init()``, and each call must
observe an already-initialized environment without re-running the
substeps. ``functools.cache`` is the canonical memoize.

See:
  - ``claude-code-from-source/book/ch02-bootstrap.md`` §"Phase 2"
  - ``my-docs/ch02-bootstrap-gap-analysis.md`` §2 Phase 2
  - ``my-docs/ch02-bootstrap-refactoring-plan.md`` P1.3-P1.6

Architectural split
-------------------

``init()`` is the per-process idempotent setup (memoized, no args).
``run_pre_action(args)`` is the per-invocation hook (called from
``cli.main()`` after argparse, takes ``args`` so future flag-handling
like ``--bare`` can branch on CLI flags without leaking into init).
This split mirrors TS's separation of ``init()`` (called from
Commander's ``preAction`` hook) and the action-handler body that runs
per-command.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import cache

from src.permissions.trust_boundary import (
    apply_safe_config_environment_variables,
    establish_session_trust,
    extract_mdm_env,
)
from src.prefetch import (
    get_or_start_keychain_prefetch,
    get_or_start_mdm_raw_read,
    wait_and_read_keychain,
    wait_and_read_mdm,
)
from src.utils.api_preconnect import start_api_preconnect
from src.utils.keychain_stash import stash_keychain_credentials
from src.utils.graceful_shutdown import (
    register_cleanup,  # noqa: F401 — exposed for callers to register cleanups
    setup_graceful_shutdown,
)
from src.utils.startup_profiler import profile_checkpoint

__all__ = [
    "init",
    "run_pre_action",
    "reset_init_for_test_only",
]


_logger = logging.getLogger("clawcodex.init")


@cache
def init() -> None:
    """Bootstrap initialization. Idempotent across multiple callers.

    Substeps (each profiled):
    1. ``apply_safe_config_environment_variables`` — pre-trust env subset.
    2. ``setup_graceful_shutdown`` — SIGINT/SIGTERM handlers.
    3. ``start_api_preconnect`` — DNS+TLS warmup (moved from cli.main).
    4. Placeholder for plan-phase-2: remote-managed-settings init.
    5. Placeholder for plan-phase-2: policy-limits init.
    6. ``ensure_nested_transcript_initialized`` — register the
       nested-session transcript path resolver so sub-agent JSONL
       files land under
       ``<parent_session_id>/subagents/agent-<id>.jsonl`` instead
       of the flat ``~/.clawcodex/transcripts/`` fallback. Wrapped
       in a flag-guarded idempotent helper; safe under repeated
       invocations. Local import — ``clawcodex_ext`` is loaded
       only after ``src/`` is fully initialized, sidestepping the
       ``src.permissions.cycle`` ↔ ``clawcodex_ext`` circular
       import that prevents registering at package import time
       (see ``clawcodex_ext/__init__.py`` for the full rationale).

    The placeholder substeps are no-ops for plan phase 1; they exist
    to mark the seam where future work plugs in.
    """
    profile_checkpoint("init_function_start")

    # ch02 round-2 PR-1 (G1): consume the keychain + MDM prefetches that
    # ``src/cli.py`` fires at module import. The Popens are already
    # running; ``wait_and_read_*`` blocks for at most the supplied
    # timeout. Both helpers return ``None`` on any failure — silent
    # degrade is intentional (TS behavior).
    _logger.info("init: consuming keychain + MDM prefetches")
    stash_keychain_credentials(wait_and_read_keychain(get_or_start_keychain_prefetch(), timeout=5.0))
    mdm_payload = wait_and_read_mdm(get_or_start_mdm_raw_read(), timeout=2.0)
    mdm_env = extract_mdm_env(mdm_payload)
    profile_checkpoint("init_after_prefetch_consumption")

    # ch02 round-3: the pre-trust pass applies the TRUSTED tiers in full
    # (global config env — including stored API keys like TAVILY_API_KEY —
    # and user settings env), the SAFE subset of the project-scoped tiers,
    # and the MDM policy env last. Project/local tiers apply in full only
    # after trust (run_pre_action seeding or a trust-gate accept). The old
    # secret_store startup applier is gone — it copied the MERGED env
    # (committable project tiers included) with no trust gate.
    _logger.info("init: applying pre-trust env (trusted tiers + safe subset)")
    apply_safe_config_environment_variables(extra_env=mdm_env)
    profile_checkpoint("init_safe_env_vars_applied")

    _logger.info("init: setting up graceful shutdown")
    setup_graceful_shutdown()
    profile_checkpoint("init_after_graceful_shutdown")

    _logger.info("init: starting API preconnect")
    start_api_preconnect()
    profile_checkpoint("init_after_api_preconnect")

    _placeholder_initialize_remote_managed_settings()
    _placeholder_initialize_policy_limits()

    # Substep 6: register the nested-session transcript path resolver
    # so sub-agent JSONL files land under
    # ``<parent_session_id>/subagents/agent-<id>.jsonl`` instead of
    # the flat ``~/.clawcodex/transcripts/`` fallback. The local
    # import keeps ``clawcodex_ext`` out of ``src.init``'s module
    # surface — by the time ``init()`` runs, every ``src/`` module
    # is fully loaded, so the lazy wrapper sees the world in a
    # consistent state and can safely register the resolver. The
    # helper is flag-guarded, so re-entry from a second ``init()``
    # caller (memoize or test) is a no-op.
    _logger.info("init: registering nested-session transcript resolver")
    from clawcodex_ext import ensure_nested_transcript_initialized

    ensure_nested_transcript_initialized()
    profile_checkpoint("init_after_nested_transcript")

    # Substep 7: install the four downstream extensions (permission cycle,
    # memory section builder, provider patches, stale-registry reload).
    # Deferred from package-import time to here so that the circular-import
    # constraint documented in ``clawcodex_ext/__init__.py`` is side-stepped.
    _logger.info("init: installing downstream extensions")
    from clawcodex_ext import ensure_eager_extensions_installed

    ensure_eager_extensions_installed()
    profile_checkpoint("init_after_eager_extensions")

    # initialize the feature-gate registry so that
    # ``@feature_gated`` decorators and ``registry.is_enabled()``
    # calls work from the earliest possible point in the agent loop.
    _logger.info("init: initializing feature-gate registry")
    try:
        from clawcodex_ext.feature_gate import get_registry
        from clawcodex_ext.feature_gate.types import FeatureFlag

        reg = get_registry()
        # Register a small set of built-in feature flags that map
        # closely to the CCB macro-defines.  These serve as examples
        # and can be extended by downstream modules.
        _builtin_features: list[FeatureFlag] = [
            FeatureFlag(
                name="AGENTIC_MODE",
                default=False,
                deps=[],
                mutex_with=["PLAIN_CHAT"],
                description="Enable agentic multi-step planning mode",
            ),
            FeatureFlag(
                name="PLAIN_CHAT",
                default=True,
                deps=[],
                mutex_with=["AGENTIC_MODE"],
                description="Plain chat mode (opposite of agentic planning)",
            ),
            FeatureFlag(
                name="EXPERIMENTAL_TOOLS",
                default=False,
                deps=[],
                mutex_with=[],
                description="Expose experimental tool integrations",
            ),
            FeatureFlag(
                name="REMOTE_SESSIONS",
                default=False,
                deps=["EXPERIMENTAL_TOOLS"],
                mutex_with=[],
                description="Allow remote session connections",
            ),
            FeatureFlag(
                name="VOICE_MODE",
                default=False,
                deps=["EXPERIMENTAL_TOOLS"],
                mutex_with=[],
                description="Enable voice input/output mode",
            ),
            FeatureFlag(
                name="SANDBOX_EXECUTION",
                default=False,
                deps=[],
                mutex_with=[],
                description="Run tool executions in an isolated sandbox",
            ),
            FeatureFlag(
                name="KAIROS",
                default=False,
                deps=[],
                mutex_with=[],
                description="Enable Kairos tick scheduling primitives",
            ),
            FeatureFlag(
                name="PROACTIVE",
                default=False,
                deps=[],
                mutex_with=[],
                description="Enable proactive tick-driven autonomous mode",
            ),
            FeatureFlag(
                name="goals",
                default=True,
                deps=[],
                mutex_with=[],
                description="Enable upstream-compatible /goal mode",
            ),
        ]
        for feat in _builtin_features:
            try:
                reg.register(feat)
            except ValueError:  # nosec B110
                pass  # Already registered (idempotent).
        profile_checkpoint("init_feature_gate_registered")
    except Exception as exc:  # noqa: BLE001
        _logger.debug("init: feature-gate registry init failed: %s", exc)

    # install global exception hooks. Idempotent and best-effort —
    # a misconfigured telemetry config must never block init. Local
    # import avoids pinning telemetry in ``src.init``'s module surface.
    _logger.info("init: installing telemetry exception hooks")
    try:
        from telemetry.hooks import install_exception_hooks

        install_exception_hooks()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("init: telemetry hook install failed: %s", exc)
    profile_checkpoint("init_after_telemetry_hooks")

    # register telemetry shutdown flush so the aggregator and
    # reporter emit run on SIGINT/SIGTERM/atexit. Idempotent, best-effort.
    _logger.info("init: registering telemetry shutdown flush")
    try:
        from clawcodex_ext.telemetry_lifecycle import install_telemetry_shutdown_flush

        install_telemetry_shutdown_flush()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("init: telemetry shutdown flush install failed: %s", exc)
    profile_checkpoint("init_after_telemetry_shutdown_flush")

    # opt-in freeze-detection watchdog. ``maybe_start_from_env``
    # is a no-op when ``CLAWCODEX_FREEZE_DIAG`` is unset, so this adds no
    # overhead to normal runs. When the env var is set it spawns the daemon
    # thread that dumps thread stacks after ``threshold_s`` of silence.
    # Wired here so every entry point that calls the canonical ``init()``
    # (CLI, TUI, REPL, headless via dispatch) gets the watchdog for free.
    _logger.info("init: enabling freeze-detection watchdog if requested")
    try:
        from clawcodex_ext.diagnostics import FreezeDetector

        FreezeDetector.maybe_start_from_env()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("init: freeze-detection watchdog setup failed: %s", exc)
    profile_checkpoint("init_after_freeze_detector")

    profile_checkpoint("init_function_end")


def _placeholder_initialize_remote_managed_settings() -> None:
    """No-op. Plan phase 2 will wire remote-managed-settings here."""


def _placeholder_initialize_policy_limits() -> None:
    """No-op. Plan phase 2 will wire policy-limits service here."""


def run_pre_action(args: object) -> None:
    """Python analog of Commander's ``preAction`` hook.

    Called from ``cli.main()`` after argparse parses, before
    ``_resolve_permission_state`` and mode dispatch. Mirrors the
    ``program.hook('preAction', ...)`` pattern at
    ``typescript/src/main.tsx:911``.

    The split between ``init()`` and ``run_pre_action()`` exists
    because:

    * ``init()`` is meant to be called from multiple entry points
      (each one calls it once; memoize handles dedup).
    * ``run_pre_action`` takes ``args`` so per-invocation flag handling
      (e.g., future ``--bare`` env injection) can branch on CLI flags
      without leaking into ``init()``.
    """
    profile_checkpoint("pre_action_start")
    init()
    profile_checkpoint("pre_action_after_init")

    # P1.6: interactive bootstrap state mutators move into preAction
    # so subsystems that read ``get_is_interactive()`` during init or
    # setup see the right value. Lazy import to avoid bootstrap state
    # being pulled in by ``init.py``-importers that only want ``init``.
    from src.bootstrap.state import (
        set_client_type,
        set_is_interactive,
    )

    is_interactive = _determine_is_interactive(args)
    set_is_interactive(is_interactive)
    set_client_type(_determine_client_type())

    # ch02 round-3 trust seeding (closes the C8 follow-up TODO). TS mapping
    # (main.tsx:1955-1967, interactiveHelpers.tsx:139-194):
    #   * non-interactive (-p / non-TTY stdout): trust is implicit — grant
    #     it and apply the full env immediately so git spawns and tools see
    #     project PATH/GIT_DIR etc.
    #   * interactive + previously accepted for this folder (parent-walk in
    #     check_trust_accepted): same.
    #   * interactive + not yet accepted: stay untrusted (bootstrap default
    #     is False); the parent's folder-trust gate decides — the plain-text
    #     prompt in cli.py (_gate_folder_trust / _prompt_folder_trust) — and
    #     its accept path calls establish_session_trust().
    from src.services.startup_gates import check_trust_accepted

    if not is_interactive or check_trust_accepted():
        establish_session_trust()

    profile_checkpoint("pre_action_end")


def _determine_is_interactive(args: object) -> bool:
    """Mirrors TS ``isInteractiveSession`` (main.tsx:803-816)."""
    if getattr(args, "print", False):
        return False
    if not sys.stdout.isatty():
        return False
    return True


# Recognized values of ``CLAUDE_CODE_ENTRYPOINT``. Mirrors TS
# main.tsx:822-838. Unknown values fall back to ``cli`` (defensive
# default — an attacker setting this env var to a random string
# shouldn't change client-type-gated behavior).
_KNOWN_CLIENT_TYPES = frozenset(
    {
        "sdk-py",
        "sdk-ts",
        "sdk-cli",
        "cli",
        "claude-vscode",
    }
)


def _determine_client_type() -> str:
    """Mirrors TS main.tsx:822-838 — read CLAUDE_CODE_ENTRYPOINT."""
    entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT", "")
    if entrypoint in _KNOWN_CLIENT_TYPES:
        return entrypoint
    return "cli"


def reset_init_for_test_only() -> None:
    """Reset the memoize cache. Test-only.

    Gated by ``PYTEST_CURRENT_TEST`` so production callers cannot
    accidentally re-run init mid-session. Matches the discipline used
    by ``bootstrap.state.reset_state_for_tests``.
    """
    if os.environ.get("PYTEST_CURRENT_TEST") is None:
        raise RuntimeError("reset_init_for_test_only can only be called in tests")
    init.cache_clear()
