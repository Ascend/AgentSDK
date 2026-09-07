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

"""Tests for the shell tab completion (argcomplete) hook in dispatch.

The hook is intentionally lazy and no-op when ``_ARGCOMPLETE`` is unset,
so the heavy-module load contract enforced by
``test_stage2_cli::test_cli_help_does_not_load_heavy_modules`` and
``test_downstream_cli_entrypoint::test_downstream_cli_main_import_is_lightweight``
is preserved.
"""

from __future__ import annotations

import sys
import time


def test_argcomplete_no_env_var_is_noop(monkeypatch):
    """Without ``_ARGCOMPLETE=1`` the hook is a true no-op (no argcomplete import)."""
    monkeypatch.delenv("_ARGCOMPLETE", raising=False)

    # Purge any pre-existing argcomplete import so the assertion is meaningful.
    sys.modules.pop("argcomplete", None)

    from clawcodex_ext.cli.dispatch import _maybe_argcomplete_top_level

    _maybe_argcomplete_top_level(["clawcodex-dev", "provider"])

    assert "argcomplete" not in sys.modules, "Hook must not import argcomplete when _ARGCOMPLETE is unset"


def test_argcomplete_helper_invokes_autocomplete(monkeypatch):
    """With ``_ARGCOMPLETE=1`` the hook calls ``argcomplete.autocomplete(parser)``
    after attaching the sieve-mirror noun set to the ``prompt`` action.
    """
    monkeypatch.setenv("_ARGCOMPLETE", "1")

    captured: dict[str, object] = {}

    class _FakeAction:
        def __init__(self, dest: str) -> None:
            self.dest = dest
            self.choices = None  # populated by hook

    class _FakeParser:
        def __init__(self) -> None:
            self._actions = [_FakeAction("prompt"), _FakeAction("stream")]

    fake_parser = _FakeParser()

    import argcomplete as real_argcomplete

    def _fake_autocomplete(parser, **kwargs):
        captured["parser"] = parser
        captured["kwargs"] = kwargs

    monkeypatch.setattr(real_argcomplete, "autocomplete", _fake_autocomplete)
    monkeypatch.setattr("clawcodex_ext.cli.parser.build_parser", lambda: fake_parser)

    # Reload the hook's view of argcomplete so monkeypatch takes effect.
    from clawcodex_ext.cli import dispatch

    dispatch._maybe_argcomplete_top_level(["clawcodex-dev"])

    assert captured.get("parser") is fake_parser
    # The hook should have attached the full noun set to the prompt action.
    assert fake_parser._actions[0].choices is not None
    nouns = set(fake_parser._actions[0].choices)
    expected = {
        "login",
        "config",
        "mcp",
        "daemon",
        "doctor",
        "orchestrator",
        "autonomy",
        "schedule",
        # Registry-loaded subcommands (load_builtin_subcommands runs lazily).
        "provider",
        "model",
        "sop",
        "viz",
    }
    assert expected.issubset(nouns), f"Missing nouns: {expected - nouns}"


def test_argcomplete_run_cli_does_not_break_help(monkeypatch):
    """``--help`` still exits cleanly and stays under 5s when argcomplete hook is in place.

    Mirrors the existing ``test_stage2_cli::test_cli_help_does_not_load_heavy_modules``
    contract: --help must return in < 5s and not import heavy modules.
    """
    monkeypatch.delenv("_ARGCOMPLETE", raising=False)

    from clawcodex_ext.cli.dispatch import run_cli

    # Purge any pre-existing argcomplete import so the assertion is meaningful.
    sys.modules.pop("argcomplete", None)

    start = time.monotonic()
    # ``--help`` raises SystemExit (argparse convention); catch it.
    try:
        rc = run_cli(["clawcodex-dev", "--help"])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 0
    elapsed = time.monotonic() - start

    assert rc == 0
    assert elapsed < 5.0, f"--help took {elapsed:.2f}s, expected < 5s"
    # Sanity: argcomplete must still not be imported.
    assert "argcomplete" not in sys.modules, "Lazy hook violated: argcomplete was imported without _ARGCOMPLETE=1"


def test_argcomplete_orchestrator_noun_completion(monkeypatch):
    """The orchestrator entrypoint calls ``argcomplete.autocomplete(parser)``
    when ``_ARGCOMPLETE=1`` is set.

    Uses a real ``argparse.ArgumentParser`` (so the subparser machinery is
    intact) and patches only ``argcomplete.autocomplete`` to capture the
    parser instance.
    """
    monkeypatch.setenv("_ARGCOMPLETE", "1")

    captured: dict[str, object] = {}

    import argcomplete as real_argcomplete

    def _fake_autocomplete(parser, **kwargs):
        captured["parser"] = parser
        captured["kwargs"] = kwargs

    monkeypatch.setattr(real_argcomplete, "autocomplete", _fake_autocomplete)

    from clawcodex_ext.entrypoints import orchestrator

    # Patch the heavy subparser builders to no-ops so we don't drag in
    # extensions.orchestrator.cli.* machinery. We only need to observe
    # that argcomplete.autocomplete is invoked with the parser.
    monkeypatch.setattr(
        "extensions.orchestrator.cli.dashboard.add_dashboard_parser",
        lambda _sub: None,
        raising=False,
    )
    monkeypatch.setattr(
        "extensions.orchestrator.cli.issue.add_issue_parser",
        lambda _sub: None,
        raising=False,
    )
    monkeypatch.setattr(
        "extensions.orchestrator.cli.server.add_server_parser",
        lambda _sub: None,
        raising=False,
    )

    try:
        orchestrator.run_orchestrator_subcommand(["server", "status"])
    except SystemExit:
        # argparse may call sys.exit on parse error in the test sandbox.
        pass
    except Exception:  # noqa: BLE001, S110
        # We only care that argcomplete was engaged; the actual dispatch
        # outcome is irrelevant for this unit test.
        pass

    assert "parser" in captured, "argcomplete.autocomplete was not called when _ARGCOMPLETE=1"


def test_argcomplete_noun_set_matches_registry(monkeypatch):
    """The hook's prompt choices derive from the two-tier registry.

    ``_maybe_argcomplete_top_level`` no longer carries a hardcoded noun
    tuple — the choice list comes from ``_SUBCOMMANDS`` after the
    discovery load. Every noun the registry exposes must be
    offered, and every dispatched noun must resolve to a handler.
    """
    monkeypatch.setenv("_ARGCOMPLETE", "1")

    captured: dict[str, object] = {}

    class _FakeAction:
        def __init__(self, dest: str) -> None:
            self.dest = dest
            self.choices = None  # populated by hook

    class _FakeParser:
        def __init__(self) -> None:
            self._actions = [_FakeAction("prompt")]

    fake_parser = _FakeParser()

    import argcomplete as real_argcomplete

    def _fake_autocomplete(parser, **kwargs):
        captured["parser"] = parser

    monkeypatch.setattr(real_argcomplete, "autocomplete", _fake_autocomplete)
    monkeypatch.setattr("clawcodex_ext.cli.parser.build_parser", lambda: fake_parser)

    from clawcodex_ext.cli import dispatch
    from clawcodex_ext.cli import subcommand_registry as registry

    dispatch._maybe_argcomplete_top_level(["clawcodex-dev"])

    choices = fake_parser._actions[0].choices
    assert choices is not None
    offered = set(choices)
    assert offered == set(registry._SUBCOMMANDS.keys()), "prompt choices must mirror the loaded registry exactly"
    for noun in (
        "login",
        "config",
        "mcp",
        "daemon",
        "doctor",
        "orchestrator",
        "autonomy",
        "schedule",
    ):
        assert noun in offered, f"Noun {noun!r} missing from argcomplete choices"
        assert registry.get_subcommand(noun) is not None, f"Noun {noun!r} has no handler"
