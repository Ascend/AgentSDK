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

"""Tests for stage3d runtime commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _build_context(*, runtime_context: bool = False):
    """Build a minimal ``CommandContext`` for testing.

    Parameters
    ----------
    runtime_context : bool
        When *True*, attach a ``runtime_context`` namespace so the command
        shows the current-state header.  When *False* (default), omit it so
        the command must degrade gracefully.
    """
    from clawcodex_ext.command_system.engine import create_command_context

    provider = SimpleNamespace(
        provider_name="anthropic",
        model="test-model",
        get_available_models=lambda: ["test-model"],
    )
    kwargs: dict = {
        "workspace_root": Path("/tmp"),
        "provider": provider,
    }
    if runtime_context:
        kwargs["runtime_context"] = SimpleNamespace(
            provider_name="anthropic",
            options=SimpleNamespace(model="test-model"),
            provider=provider,
            tool_registry=None,
            tool_context=None,
            swap_provider=lambda p, m=None: None,
        )
    return create_command_context(**kwargs)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestRuntimeCommandsRegistration:
    """Tests for TestRuntimeCommandsRegistration."""

    def test_register_runtime_commands_adds_model(self):
        """Verify register runtime commands adds model."""
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.types import CommandType

        get_command_registry().clear()
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands

        register_runtime_commands(None)
        cmd = get_command_registry().get("model")
        assert cmd is not None, "model command should be registered"
        assert cmd.command_type == CommandType.LOCAL, f"expected LOCAL, got {cmd.command_type}"

    def test_register_runtime_commands_adds_provider(self):
        """Verify register runtime commands adds provider."""
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.types import CommandType

        get_command_registry().clear()
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands

        register_runtime_commands(None)
        cmd = get_command_registry().get("provider")
        assert cmd is not None, "provider command should be registered"
        assert cmd.command_type == CommandType.LOCAL, f"expected LOCAL, got {cmd.command_type}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestRuntimeCommandsWithoutRuntimeContext:
    """Tests for TestRuntimeCommandsWithoutRuntimeContext."""

    def _ensure_registered(self):
        from clawcodex_ext import ensure_eager_extensions_installed
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        # Downstream provider registrations (e.g. kimi-coding) are deferred
        # from package import time to a lazy init function.  Ensure they
        # are in place before exercising /model or /provider commands.
        ensure_eager_extensions_installed()

        get_command_registry().clear()
        register_runtime_commands(None)

    def test_model_no_args_without_context(self):
        """Verify model no args without context."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "Models:" in result_text, "output should contain 'Models:'"
        assert "anthropic:" in result_text, "output should list providers"

    def test_provider_no_args_without_context(self):
        """Verify provider no args without context."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "Providers:" in result_text, "output should contain 'Providers:'"
        assert "anthropic" in result_text, "output should list providers"

    def test_model_no_args_without_context_no_unknown_command(self):
        """Verify model no args without context no unknown command."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, _, error = execute_command_sync("model", "", _build_context(runtime_context=False))
        assert success is True, f"should not return Unknown command; got error={error!r}"

    def test_provider_no_args_without_context_no_unknown_command(self):
        """Verify provider no args without context no unknown command."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, _, error = execute_command_sync("provider", "", _build_context(runtime_context=False))
        assert success is True, f"should not return Unknown command; got error={error!r}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestRuntimeCommandsWithRuntimeContext:
    """Tests for TestRuntimeCommandsWithRuntimeContext."""

    def _ensure_registered(self):
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        get_command_registry().clear()
        register_runtime_commands(None)

    def test_model_no_args_with_context(self):
        """Verify model no args with context."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "", _build_context(runtime_context=True))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "[primary]provider[/primary]" in result_text, "output should show current provider"
        assert "test-model" in result_text, "output should show current model"
        assert "Models:" in result_text, "output should contain 'Models:'"

    def test_provider_no_args_with_context(self):
        """Verify provider no args with context."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "", _build_context(runtime_context=True))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "[primary]provider[/primary]" in result_text, "output should show current provider"
        assert "test-model" in result_text, "output should show current model"
        assert "Providers:" in result_text, "output should contain 'Providers:'"


# ---------------------------------------------------------------------------
# /dream slash skill
# ---------------------------------------------------------------------------


class TestDreamCommandRegistration:
    """``register_dream_skill`` wires ``/dream`` as a LocalCommand in the
    global command registry.
    """

    def test_register_dream_skill_adds_dream(self):
        """register_dream_skill adds a LocalCommand named ``dream``."""
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.types import CommandType
        from extensions.skills_ext.bundled.dream import register_dream_skill

        get_command_registry().clear()
        register_dream_skill()

        cmd = get_command_registry().get("dream")
        assert cmd is not None, "dream command should be registered"
        assert cmd.command_type == CommandType.LOCAL, f"expected LOCAL, got {cmd.command_type}"


class TestDreamCommandExecution:
    """``/dream`` subcommands run via execute_command_sync ."""

    @pytest.fixture(autouse=True)
    def _isolate_dream_service(self):
        """Reset the dream service's closure-scoped runner state.

        The ``_service._runner`` module-level singleton carries a
        :class:`RuntimeTaskRegistry` reference. Tests in
        ``tests/dreaming/`` populate that registry; the stage-3d
        ``/dream status`` test must observe an empty registry even
        when the full suite runs ``tests/dreaming/`` first.
        """
        from clawcodex_ext.dreaming import service as _service

        _service._runner = None
        yield
        _service._runner = None

    def _ensure_registered(self):
        from clawcodex_ext.command_system import get_command_registry
        from extensions.skills_ext.bundled.dream import register_dream_skill

        get_command_registry().clear()
        register_dream_skill()

    def test_dream_no_args_shows_help(self):
        """``/dream`` with no args returns the usage help text."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("dream", "", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "Usage:" in result_text
        assert "run" in result_text
        assert "status" in result_text

    def test_dream_help_subcommand(self):
        """``/dream help`` returns the same usage."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, _ = execute_command_sync("dream", "help", _build_context(runtime_context=False))
        assert success is True
        assert "Usage:" in result_text

    def test_dream_status_no_init(self):
        """``/dream status`` works even when the dream service was not
        initialized (returns the empty-state message).
        """
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, _ = execute_command_sync("dream", "status", _build_context(runtime_context=False))
        assert success is True
        assert "No dream tasks in flight" in result_text

    def test_dream_unknown_subcommand_does_not_crash(self):
        """``/dream frobnicate`` returns a clean warning, not a stack trace."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, _ = execute_command_sync("dream", "frobnicate", _build_context(runtime_context=False))
        # Engine treats unknown-subcommand as a successful help render.
        assert success is True
        assert "Unknown subcommand" in result_text
        assert "frobnicate" in result_text

    def test_dream_command_no_unknown_command(self):
        """Verify dream command no unknown command."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("dream", "", _build_context(runtime_context=False))
        assert success is True, f"should not return Unknown command; got error={error!r}, result_text={result_text!r}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# "Command not implemented for sync execution"。
# ---------------------------------------------------------------------------


class TestRuntimeCommandsRaceCondition:
    """Tests for TestRuntimeCommandsRaceCondition."""

    def test_build_command_suggestions_does_not_overwrite_model(self):
        """Verify build command suggestions does not overwrite model."""

        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import register_builtin_commands
        from clawcodex_ext.command_system.types import CommandType
        from clawcodex_ext.tui.commands import build_command_suggestions

        reg = get_command_registry()
        reg.clear()

        register_builtin_commands(None)
        register_runtime_commands(None)

        assert reg.get("model").command_type == CommandType.LOCAL

        build_command_suggestions(Path("/tmp"))

        cmd = reg.get("model")
        assert cmd is not None, "model command must survive build_command_suggestions"
        assert cmd.command_type == CommandType.LOCAL, (
            f"expected LOCAL after build_command_suggestions, got {cmd.command_type}"
        )

    def test_build_command_suggestions_does_not_overwrite_provider(self):
        """Verify build command suggestions does not overwrite provider."""

        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import register_builtin_commands
        from clawcodex_ext.command_system.types import CommandType
        from clawcodex_ext.tui.commands import build_command_suggestions

        reg = get_command_registry()
        reg.clear()

        register_builtin_commands(None)
        register_runtime_commands(None)

        assert reg.get("provider").command_type == CommandType.LOCAL

        build_command_suggestions(Path("/tmp"))

        cmd = reg.get("provider")
        assert cmd is not None, "provider command must survive build_command_suggestions"
        assert cmd.command_type == CommandType.LOCAL, (
            f"expected LOCAL after build_command_suggestions, got {cmd.command_type}"
        )

    def test_model_executable_after_build_command_suggestions(self):
        """Verify model executable after build command suggestions."""

        from clawcodex_ext import ensure_eager_extensions_installed
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import (
            execute_command_sync,
            register_builtin_commands,
        )
        from clawcodex_ext.tui.commands import build_command_suggestions

        ensure_eager_extensions_installed()

        reg = get_command_registry()
        reg.clear()
        register_builtin_commands(None)
        register_runtime_commands(None)

        build_command_suggestions(Path("/tmp"))

        success, result_text, error = execute_command_sync("model", "", _build_context(runtime_context=False))
        assert success is True, f"should be executable after build_command_suggestions; got error={error!r}"
        assert "Models:" in (result_text or "")

    def test_provider_executable_after_build_command_suggestions(self):
        """Verify provider executable after build command suggestions."""

        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import (
            execute_command_sync,
            register_builtin_commands,
        )
        from clawcodex_ext.tui.commands import build_command_suggestions

        reg = get_command_registry()
        reg.clear()
        register_builtin_commands(None)
        register_runtime_commands(None)

        build_command_suggestions(Path("/tmp"))

        success, result_text, error = execute_command_sync("provider", "", _build_context(runtime_context=False))
        assert success is True, f"should be executable after build_command_suggestions; got error={error!r}"
        assert "Providers:" in (result_text or "")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class TestRuntimeCommandCompletion:
    """Tests for TestRuntimeCommandCompletion."""

    def test_build_command_suggestions_includes_model(self):
        """Verify build command suggestions includes model."""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        names = [s.name for s in suggestions]
        assert "model" in names, f"build_command_suggestions must include 'model'; got {names}"

    def test_build_command_suggestions_includes_provider(self):
        """Verify build command suggestions includes provider."""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        names = [s.name for s in suggestions]
        assert "provider" in names, f"build_command_suggestions must include 'provider'; got {names}"

    def test_build_command_suggestions_model_entry_is_slash_completable(self):
        """Verify build command suggestions model entry is slash completable."""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        model_entry = next((s for s in suggestions if s.name == "model"), None)
        assert model_entry is not None, "model entry must exist"
        assert model_entry.slash == "/model", f"expected slash='/model', got {model_entry.slash!r}"

    def test_build_command_suggestions_provider_entry_is_slash_completable(self):
        """Verify build command suggestions provider entry is slash completable."""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        provider_entry = next((s for s in suggestions if s.name == "provider"), None)
        assert provider_entry is not None, "provider entry must exist"
        assert provider_entry.slash == "/provider", f"expected slash='/provider', got {provider_entry.slash!r}"

    def test_provider_appears_in_slash_only_completer_flat_words(self):
        """Verify provider appears in slash only completer flat words."""
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import register_builtin_commands
        from clawcodex_ext.tui.commands import build_command_words

        reg = get_command_registry()
        reg.clear()
        register_builtin_commands(None)
        register_runtime_commands(None)

        words = build_command_words(Path("/tmp"))
        assert "/provider" in words, f"flat words must include '/provider'; got {words}"
        assert "/model" in words, f"flat words must include '/model'; got {words}"


class TestModelProviderFallback:
    """Tests for TestModelProviderFallback."""

    def _ensure_registered(self):
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        get_command_registry().clear()
        register_runtime_commands(None)

    def test_unknown_model_falls_back_to_runtime_provider(self):
        """Verify unknown model falls back to runtime provider."""
        self._ensure_registered()

        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(
            model="gpt-4",
            get_available_models=lambda: ["gpt-4"],
        )
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "truly-unknown-model-xyz", context)
        assert success is True, f"expected success, got error={error!r}"
        assert "provider: openai" in (result_text or ""), f"expected 'provider: openai' in output, got {result_text!r}"
        assert "anthropic" not in (result_text or "").lower(), f"should not fall back to anthropic, got {result_text!r}"
        assert "unknown model" in (result_text or "").lower(), f"expected unknown model warning, got {result_text!r}"

    def test_known_model_stays_on_current_provider(self):
        """Verify known model stays on current provider."""
        self._ensure_registered()

        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(
            model="gpt-4",
            get_available_models=lambda: ["gpt-4", "another-model"],
        )
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "another-model", context)
        assert success is True, f"expected success, got error={error!r}"
        assert "provider: openai" in (result_text or ""), f"expected 'provider: openai', got {result_text!r}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestRuntimeCommandsWithArgs:
    """Tests for TestRuntimeCommandsWithArgs."""

    @pytest.fixture(autouse=True)
    def _no_config_persistence(self):
        """Test helper for no config persistence."""
        from unittest.mock import patch

        with patch("src.config.set_default_provider"):
            yield

    def _ensure_registered(self):
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        get_command_registry().clear()
        register_runtime_commands(None)

    # ── /model <name> ─────────────────────────────────────────────────

    def test_model_known_arg_without_context(self):
        """Verify model known arg without context."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, _result_text, error = execute_command_sync(
            "model", "claude-sonnet-4-6", _build_context(runtime_context=False)
        )
        assert success is True, f"expected success, got error={error!r}"

    def test_model_unknown_arg_does_not_crash(self):
        """Verify model unknown arg does not crash."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync(
            "model",
            "truly-nonexistent-model-xyz-12345",
            _build_context(runtime_context=False),
        )
        assert success is True, f"should not crash on unknown model; got error={error!r}"
        assert result_text is not None

    # ── /provider <name> ─────────────────────────────────────────────

    def test_provider_known_arg_without_context(self):
        """Verify provider known arg without context."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "openai", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None

    def test_provider_unknown_arg_does_not_crash(self):
        """Verify provider unknown arg does not crash."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync(
            "provider",
            "nonexistent-provider-xyz",
            _build_context(runtime_context=False),
        )
        assert success is True, f"should not crash on unknown provider; got error={error!r}"
        assert result_text is not None
        assert "provider" in result_text.lower(), f"expected provider-related output, got {result_text!r}"

    def test_provider_arg_with_runtime_context_does_not_crash(self):
        """Verify provider arg with runtime context does not crash."""
        self._ensure_registered()

        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(model="gpt-4", get_available_models=lambda: ["gpt-4"])
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "openai", context)
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None

    def test_provider_unknown_arg_with_runtime_context_does_not_crash(self):
        """Verify provider unknown arg with runtime context does not crash."""
        self._ensure_registered()

        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(model="gpt-4", get_available_models=lambda: ["gpt-4"])
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "nonexistent-provider-xyz", context)
        assert success is True, f"should not crash; got error={error!r}"
        assert result_text is not None
