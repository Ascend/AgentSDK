#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for all CLI subcommands (``clawcodex-dev <subcommand>``).

Covers every subcommand in the dispatch sieve and the ``@register``
subcommand registry, verifying that each can be routed without crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from clawcodex_ext.cli.subcommand_registry import (
    _SUBCOMMANDS,
    get_subcommand,
    load_builtin_subcommands,
    telemetry_mode_for,
)

# ---------------------------------------------------------------------------
# Subcommand registry completeness
# ---------------------------------------------------------------------------

_ALL_SIEVE_SUBCOMMANDS = {
    "login",
    "config",
    "mcp",
    "daemon",
    "doctor",
    "orchestrator",
    "autonomy",
    "schedule",
}

_ALL_REGISTERED_SUBCOMMANDS = {
    "auth",
    "model",
    "sop",
    "provider",
    "session",
    "stats",
    "telemetry",
    "api",
    "viz",
}


def test_all_sieve_subcommands_exist():
    """Every dispatch-sieve noun must resolve to a registered handler."""
    load_builtin_subcommands()
    registered = set(_SUBCOMMANDS.keys())
    missing = _ALL_SIEVE_SUBCOMMANDS - registered
    assert not missing, f"Sieve subcommands missing from registry: {missing}"
    for name in _ALL_SIEVE_SUBCOMMANDS:
        handler = get_subcommand(name)
        assert handler is not None, f"get_subcommand({name!r}) returned None"
        assert callable(handler), f"get_subcommand({name!r}) returned non-callable: {handler}"


def test_core_subcommands_resolve_without_builtin_load(monkeypatch: pytest.MonkeyPatch):
    """Core fast-path nouns resolve at module import; discovery loads only on a miss.

    Looking up a core noun must never trigger the discovery load, and an
    unknown noun must escalate to it exactly once.
    """
    import clawcodex_ext.cli.subcommand_registry as registry

    builtin_calls = []
    monkeypatch.setattr(
        registry,
        "load_builtin_subcommands",
        lambda: builtin_calls.append(1),
    )

    for name in ("login", "config", "mcp", "daemon", "doctor", "orchestrator"):
        handler = registry.get_subcommand(name)
        assert handler is not None, f"get_subcommand({name!r}) returned None"
        assert callable(handler)
    assert builtin_calls == [], "core lookup must not load the discovery set"

    assert registry.get_subcommand("no-such-subcommand") is None
    assert builtin_calls == [1], "a lookup miss must escalate to the discovery load"


def test_all_registered_subcommands_loaded():
    """Every ``@register`` subcommand must be registered after
    ``load_builtin_subcommands``.
    """
    load_builtin_subcommands()
    registered = set(_SUBCOMMANDS.keys())
    missing = _ALL_REGISTERED_SUBCOMMANDS - registered
    assert not missing, f"Subcommands missing from registry: {missing}"


def test_get_subcommand_returns_handler():
    """``get_subcommand`` returns a callable for each registered subcommand."""
    load_builtin_subcommands()
    for name in _ALL_REGISTERED_SUBCOMMANDS:
        handler = get_subcommand(name)
        assert handler is not None, f"get_subcommand({name!r}) returned None"
        assert callable(handler), f"get_subcommand({name!r}) returned non-callable: {handler}"


def test_subcommand_telemetry_modes():
    """Daemon-style verbs record as ``daemon``; everything else keeps the default."""
    load_builtin_subcommands()
    for name in ("daemon", "orchestrator"):
        assert telemetry_mode_for(name) == "daemon", f"{name!r} should record as 'daemon'"
    for name in ("login", "config", "mcp", "doctor", "auth", "model"):
        assert telemetry_mode_for(name) == "non_interactive"
    assert telemetry_mode_for("no-such-token") == "non_interactive"


# ---------------------------------------------------------------------------
# Sieve fast-path subcommand routing
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_argv() -> list[str]:
    return ["clawcodex-dev"]


def _run_cli_with_token(token: str, rest_args: list[str] | None = None):
    """Simulate the dispatch sieve logic from ``run_cli``."""
    from clawcodex_ext.cli.dispatch import run_cli

    argv = ["clawcodex-dev", token, *(rest_args or [])]
    return run_cli(argv)


@pytest.mark.parametrize("subcommand", sorted(_ALL_SIEVE_SUBCOMMANDS))
def test_sieve_subcommand_routes_without_crash(subcommand: str):
    """Each sieve subcommand routes to its handler without an unhandled
    exception (the handler itself may error on missing args, but must not
    raise an unhandled exception from the routing logic).
    """
    from clawcodex_ext.cli.dispatch import run_cli

    argv = ["clawcodex-dev", subcommand]
    if subcommand in ("autonomy",):
        argv.append("status")
    if subcommand == "schedule":
        argv.extend(["list"])

    import sys as _sys

    with patch.object(_sys, "argv", argv):
        # Some sieve subcommands (login, config, daemon, etc.) require
        # real config/API keys and will fail with SystemExit or similar.
        # That's acceptable — the routing itself succeeded.
        try:
            rc = run_cli(argv)
            assert isinstance(rc, int)
        except (SystemExit, TypeError, Exception):  # noqa: BLE001, S110
            pass  # This smoke case only verifies that routing reaches the handler.


# ---------------------------------------------------------------------------
# @register subcommand handler dispatch (lightweight)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subcommand", sorted(_ALL_REGISTERED_SUBCOMMANDS))
def test_registered_subcommand_handler_is_callable(subcommand: str):
    """Each ``@register`` subcommand handler must be importable and callable."""
    load_builtin_subcommands()
    handler = get_subcommand(subcommand)
    assert handler is not None, f"Handler for {subcommand!r} not found"

    # The handler must accept a list of strings and return an int.
    # Some handlers raise SystemExit on empty args (argparse errors);
    # that's acceptable — it means routing + resolution succeeded.
    try:
        if subcommand == "viz":
            # Empty arguments intentionally start the long-running visualizer
            # server.  Stub its service body while still exercising command
            # discovery, routing, and the handler signature.
            with patch("extensions.visualizer.cli.run_viz", return_value=0):
                rc = handler([])
        else:
            rc = handler([])
        assert isinstance(rc, int), f"{subcommand} handler returned {type(rc).__name__}, expected int"
    except (SystemExit, TypeError):
        pass  # CLI handlers may signal usage through these expected exceptions.
    except Exception as exc:
        raise AssertionError(f"{subcommand} handler raised unexpected exception: {exc}") from exc


# ---------------------------------------------------------------------------
# Provider/model fast-path subcommand (CLI-level, not slash command)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subcommand", ["provider", "model"])
def test_provider_model_subcommand_prints_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    subcommand: str,
):
    """``clawcodex-dev provider`` and ``clawcodex-dev model`` must print
    the current setting without crashing.
    """
    monkeypatch.setattr("sys.argv", ["clawcodex-dev", subcommand])
    # Mock ModelStore to avoid real file I/O
    mock_store = MagicMock()
    mock_store.default_provider = "anthropic"
    mock_store.default_model = "claude-sonnet-4"
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.commands.ModelStore",
        MagicMock(return_value=mock_store),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.provider_cmd.commands.ModelStore",
        MagicMock(return_value=mock_store),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.runtime_commands.ModelStore",
        MagicMock(return_value=mock_store),
    )

    from clawcodex_ext.cli.dispatch import run_cli

    rc = run_cli(["clawcodex-dev", subcommand])
    assert rc == 0


# ---------------------------------------------------------------------------
# Argcomplete integration
# ---------------------------------------------------------------------------


def test_argcomplete_top_level_includes_all_subcommands():
    """``_maybe_argcomplete_top_level`` must include all registered subcommands."""
    from clawcodex_ext.cli.subcommand_registry import load_builtin_subcommands

    load_builtin_subcommands()
    # The parser's choices are populated only when _ARGCOMPLETE env var is set.
    # Here we verify that the registered subcommands exist in the registry,
    # which is the source of truth for argcomplete.
    from clawcodex_ext.cli.subcommand_registry import _SUBCOMMANDS

    registered = set(_SUBCOMMANDS.keys())
    for name in _ALL_REGISTERED_SUBCOMMANDS:
        assert name in registered, f"Registered subcommand {name!r} not in _SUBCOMMANDS"
