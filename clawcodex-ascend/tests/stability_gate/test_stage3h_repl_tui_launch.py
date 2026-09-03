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

"""Tests for stage3h repl tui launch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# =========================================================================
# =========================================================================


class TestStage3hModuleImport:
    """Tests for TestStage3hModuleImport."""

    def test_src_entrypoints_tui_facade_importable(self):
        """Verify src entrypoints tui facade importable."""
        import src.entrypoints.tui as tui_proxy

        assert hasattr(tui_proxy, "TUIOptions"), "missing TUIOptions"
        assert hasattr(tui_proxy, "run_tui"), "missing run_tui"
        assert hasattr(tui_proxy, "should_use_tui"), "missing should_use_tui"

    def test_ext_entrypoints_tui_importable(self):
        """Verify ext entrypoints tui importable."""
        import clawcodex_ext.entrypoints.tui as ext_tui

        assert hasattr(ext_tui, "TUIOptions")
        assert hasattr(ext_tui, "run_tui")
        assert hasattr(ext_tui, "should_use_tui")
        assert callable(ext_tui.run_tui)

    def test_ext_tui_entrypoint_run_tui_importable(self):
        """Verify ext tui entrypoint run tui importable."""
        from clawcodex_ext.tui.entrypoint import run_tui

        assert callable(run_tui)

    def test_ext_repl_core_importable(self):
        """Verify ext repl core importable."""
        import clawcodex_ext.repl.core as ext_repl

        assert ext_repl is not None
        assert hasattr(ext_repl, "ClawcodexREPL")

    def test_src_repl_core_facade_importable(self):
        """Verify src repl core facade importable."""
        import src.repl.core as repl_facade

        assert hasattr(repl_facade, "ClawcodexREPL")
        cls = repl_facade.ClawcodexREPL
        assert cls is not None

    def test_ext_repl_app_importable(self):
        """Verify ext repl app importable."""
        from clawcodex_ext.repl.app import ClawCodexExtREPL

        assert ClawCodexExtREPL is not None

    def test_repl_frontend_registered(self):
        """Verify repl frontend registered."""
        from clawcodex_ext.frontend.registry import get_frontend

        repl = get_frontend("repl")
        assert repl is not None, "REPLFrontend should be registered"
        assert repl.name == "repl"
        assert callable(repl.run)

    def test_tui_frontend_registered(self):
        """Verify tui frontend registered."""
        from clawcodex_ext.frontend.registry import get_frontend

        tui = get_frontend("tui")
        assert tui is not None, "TUIFrontend should be registered"
        assert tui.name == "tui"
        assert callable(tui.run)

    def test_cli_dispatch_can_import_tui_should_use(self):
        """Verify cli dispatch can import tui should use."""
        from src.entrypoints.tui import should_use_tui

        assert callable(should_use_tui)

    def test_frontend_plugins_module_imports_cleanly(self):
        """Verify frontend plugins module imports cleanly."""
        import clawcodex_ext.frontend as f

        assert hasattr(f, "get_frontend")
        assert hasattr(f, "register_frontend")
        assert hasattr(f, "list_frontends")


# =========================================================================
# Section 2 — TUIOptions / should_use_tui
# =========================================================================


class TestStage3hTuiOptions:
    """Tests for TestStage3hTuiOptions."""

    def test_TUIOptions_dataclass(self):
        """Verify TUIOptions dataclass."""
        from src.entrypoints.tui import TUIOptions

        opts = TUIOptions(
            provider_name="anthropic",
            max_turns=10,
            stream=True,
        )
        assert opts.provider_name == "anthropic"
        assert opts.max_turns == 10
        assert opts.stream is True
        assert opts.permission_mode == "default"
        assert opts.workspace_root is None

    def test_TUIOptions_defaults(self):
        """Verify TUIOptions defaults."""
        from src.entrypoints.tui import TUIOptions

        opts = TUIOptions()
        assert opts.max_turns == 20  # default from dataclass
        assert opts.stream is True
        assert opts.permission_mode == "default"
        assert opts.is_bypass_permissions_mode_available is False

    def test_should_use_tui_explicit_false(self):
        """Verify should use tui explicit false."""
        from src.entrypoints.tui import should_use_tui

        assert should_use_tui(False) is False

    def test_should_use_tui_explicit_none_no_env(self):
        """Verify should use tui explicit none no env."""
        old = os.environ.pop("CLAWCODEX_TUI", None)
        try:
            from src.entrypoints.tui import should_use_tui

            result = should_use_tui(None)
            assert result is False
        finally:
            if old is not None:
                os.environ["CLAWCODEX_TUI"] = old

    def test_should_use_tui_handles_env_var(self):
        """Verify should use tui handles env var."""
        old = os.environ.get("CLAWCODEX_TUI")
        os.environ["CLAWCODEX_TUI"] = "1"
        try:
            from src.entrypoints.tui import should_use_tui

            result = should_use_tui(None)
            assert result is False  # non-TTY
        finally:
            if old is not None:
                os.environ["CLAWCODEX_TUI"] = old
            else:
                os.environ.pop("CLAWCODEX_TUI", None)

    def test_should_use_tui_handles_legacy_repl(self):
        """Verify should use tui handles legacy repl."""
        old = os.environ.get("CLAWCODEX_LEGACY_REPL")
        os.environ["CLAWCODEX_LEGACY_REPL"] = "1"
        try:
            from src.entrypoints.tui import should_use_tui

            assert should_use_tui(True) is False
        finally:
            if old is not None:
                os.environ["CLAWCODEX_LEGACY_REPL"] = old
            else:
                os.environ.pop("CLAWCODEX_LEGACY_REPL", None)

    def test_should_use_tui_handles_tui_0(self):
        """Verify should use tui handles tui 0."""
        old = os.environ.get("CLAWCODEX_TUI")
        os.environ["CLAWCODEX_TUI"] = "0"
        try:
            from src.entrypoints.tui import should_use_tui

            assert should_use_tui(None) is False
        finally:
            if old is not None:
                os.environ["CLAWCODEX_TUI"] = old
            else:
                os.environ.pop("CLAWCODEX_TUI", None)


# =========================================================================
# =========================================================================


class TestStage3hTuiAppImport:
    """Tests for TestStage3hTuiAppImport."""

    def test_tui_app_importable_when_textual_available(self):
        """Verify tui app importable when textual available."""
        try:
            import textual  # noqa: F401
        except ImportError:
            pytest.skip("textual not installed — cannot test TUI App import")

        from clawcodex_ext.tui.app import ClawCodexExtTUI
        from src.tui.app import ClawCodexTUI

        assert ClawCodexTUI is not None
        assert ClawCodexExtTUI is not None

    def test_tui_ext_entrypoint_importable_when_textual_available(self):
        """Verify tui ext entrypoint importable when textual available."""
        try:
            import textual  # noqa: F401
        except ImportError:
            pytest.skip("textual not installed — cannot test TUI entrypoint")

        import clawcodex_ext.tui.entrypoint as tui_ep

        assert hasattr(tui_ep, "run_tui")
        assert callable(tui_ep.run_tui)
        assert tui_ep.__name__ == "clawcodex_ext.tui.entrypoint"

    def test_tui_app_inheritance_chain(self):
        """Verify tui app inheritance chain."""
        try:
            import textual  # noqa: F401
        except ImportError:
            pytest.skip("textual not installed")

        from clawcodex_ext.tui.app import ClawCodexExtTUI
        from src.tui.app import ClawCodexTUI

        assert issubclass(ClawCodexExtTUI, ClawCodexTUI)
        assert hasattr(ClawCodexExtTUI, "compose")
        assert hasattr(ClawCodexExtTUI, "on_mount")


# =========================================================================
# =========================================================================


class TestStage3hReplTuiHandoffReplay:
    """Regression guard for REPL -> TUI -> REPL history replay."""

    pytestmark = pytest.mark.asyncio

    async def test_handoff_exit_snapshot_excludes_replayed_repl_history(self, tmp_path):
        """Returning from TUI must not print pre-handoff REPL history again."""
        pytest.importorskip("textual")

        from io import StringIO

        from rich.console import Console
        from src.agent.conversation import Conversation
        from src.tool_system.context import ToolContext
        from src.tool_system.registry import ToolRegistry
        from src.tui.app import ClawCodexTUI
        from src.types.messages import Message

        class _StubProvider:
            model = "stub-model"
            completions = []  # noqa: RUF012

            def generate(self, *args, **kwargs):  # pragma: no cover - unused
                raise RuntimeError("provider should not be called in UI test")

        class _Session:
            session_id = "stability-handoff"

            def __init__(self) -> None:
                self.conversation = Conversation()
                self.conversation.messages = [
                    Message(role="user", content="old repl prompt"),
                    Message(role="assistant", content="old repl answer"),
                ]

            def save(self) -> None:
                return None

        app = ClawCodexTUI(
            provider=_StubProvider(),
            provider_name="stub",
            workspace_root=tmp_path,
            tool_registry=ToolRegistry(),
            tool_context=ToolContext(workspace_root=tmp_path),
            session=_Session(),
            max_turns=1,
            stream=False,
            replay_exit_snapshot_from_start=False,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app._repl_screen is not None  # type: ignore[attr-defined]
            transcript = app._repl_screen.transcript  # type: ignore[attr-defined]
            transcript.append_user("new tui prompt")
            transcript.append_assistant("new tui answer")
            await pilot.pause()
            app.exit()
            await pilot.pause()

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        for piece in app.exit_snapshot:
            console.print(piece)
        rendered = buf.getvalue()

        assert "new tui prompt" in rendered
        assert "new tui answer" in rendered
        assert "old repl prompt" not in rendered
        assert "old repl answer" not in rendered


class TestStage3hCliSubprocess:
    """Tests for TestStage3hCliSubprocess."""

    def _check_no_crash_startup(self, *args: str, config_dir: Path) -> None:
        """Test helper for check no crash startup."""
        import time

        child_env = {
            **os.environ,
            "HOME": str(config_dir.parent),
            "CLAWCODEX_CONFIG_DIR": str(config_dir),
            "CLAWCODEX_HOME": str(config_dir),
            "CLAW_TELEMETRY_STORAGE_DIR": str(config_dir / "telemetry"),
        }
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.cli", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=child_env,
        )
        time.sleep(3)
        returncode = proc.poll()
        if returncode is not None:
            _out, err = proc.communicate()
            assert returncode != -6, f"SIGABRT, stderr={err!r}"
            assert returncode != -11, f"SIGSEGV, stderr={err!r}"
            assert "Traceback" not in err, f"Traceback in stderr: {err}"
            assert "ImportError" not in err, f"ImportError in stderr: {err}"
            assert "AttributeError" not in err, f"AttributeError in stderr: {err}"
            return

        proc.terminate()
        try:
            _, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _out, err = proc.communicate(timeout=3)

        assert "Traceback" not in err, f"Traceback in stderr: {err}"
        assert "ImportError" not in err, f"ImportError in stderr: {err}"
        assert "AttributeError" not in err, f"AttributeError in stderr: {err}"
        if err:
            pass

    def test_cli_tui_help_works(self):
        """Verify cli tui help works."""
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "src.cli", "--tui", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        output = (proc.stdout + proc.stderr).lower()
        assert "usage:" in output, "expected usage in --tui --help output"

    def test_cli_no_tui_help_works(self):
        """Verify cli no tui help works."""
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "src.cli", "--no-tui", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        output = (proc.stdout + proc.stderr).lower()
        assert "usage:" in output

    def test_cli_tui_version_works(self):
        """Verify cli tui version works."""
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "src.cli", "--tui", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        assert len(proc.stdout.strip()) > 0

    def test_cli_tui_flag_no_traceback(self, tmp_path: Path):
        """Verify cli tui flag no traceback."""
        self._check_no_crash_startup("--tui", config_dir=tmp_path / ".clawcodex")

    def test_cli_no_tui_flag_no_traceback(self, tmp_path: Path):
        """Verify cli no tui flag no traceback."""
        self._check_no_crash_startup("--no-tui", config_dir=tmp_path / ".clawcodex")

    def test_cli_legacy_repl_flag_no_traceback(self):
        """Verify cli legacy repl flag no traceback."""
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "src.cli", "--legacy-repl", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"

    def test_cli_tui_resume_no_traceback(self, tmp_path: Path):
        """Verify cli tui resume no traceback."""
        self._check_no_crash_startup("--tui", "--resume", "browse", config_dir=tmp_path / ".clawcodex")

    def test_cli_tui_remembers_slash_commands_importable(self):
        """Verify cli tui remembers slash commands importable."""
        from clawcodex_ext.command_system.builtins import register_builtin_commands

        assert callable(register_builtin_commands)


# =========================================================================
# =========================================================================


class TestStage3hImportChain:
    """Tests for TestStage3hImportChain."""

    def test_cli_dispatch_imports_work_individually(self):
        """Verify cli dispatch imports work individually."""
        # dispatch.py:635 — from src.entrypoints.tui import should_use_tui
        from src.entrypoints.tui import TUIOptions, run_tui, should_use_tui

        assert callable(should_use_tui)
        assert callable(run_tui)
        assert TUIOptions is not None

    def test_frontend_tui_plugin_imports_cleanly(self):
        """Verify frontend tui plugin imports cleanly."""

        from clawcodex_ext.frontend.registry import get_frontend

        tui = get_frontend("tui")
        assert tui is not None
        assert callable(tui.run)

    def test_frontend_repl_plugin_imports_cleanly(self):
        """Verify frontend repl plugin imports cleanly."""
        import clawcodex_ext.frontend.repl as f_repl

        assert f_repl is not None
        from clawcodex_ext.frontend.registry import get_frontend

        repl = get_frontend("repl")
        assert repl is not None
        assert callable(repl.run)

    def test_ext_repl_app_initialization_without_provider(self):
        """Verify ext repl app initialization without provider."""
        try:
            from clawcodex_ext.repl.app import ClawCodexExtREPL
        except ImportError:
            pytest.skip("ClawCodexExtREPL import failed (is prompt_toolkit installed?)")

        from unittest.mock import patch

        class _PromptSessionStub:
            def __init__(self, *args, **kwargs):
                pass

            def prompt(self, *args, **kwargs):
                return "/exit"

        try:
            with patch("prompt_toolkit.PromptSession", _PromptSessionStub):
                repl = ClawCodexExtREPL(
                    provider_name="anthropic",
                    stream=False,
                    permission_mode="default",
                )
            assert isinstance(repl._api_key_missing, bool), (
                f"_api_key_missing should be bool, got {type(repl._api_key_missing)}"
            )
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"ClawCodexExtREPL() raised unexpected exception: {exc}")

    def test_repl_inheritance_chain(self):
        """Verify repl inheritance chain."""
        import clawcodex_ext.repl.core as ext_repl
        from clawcodex_ext.repl.app import ClawCodexExtREPL

        assert issubclass(ClawCodexExtREPL, ext_repl.ClawcodexREPL)
        assert hasattr(ClawCodexExtREPL, "run")
        assert hasattr(ClawCodexExtREPL, "chat")
        assert hasattr(ClawCodexExtREPL, "handle_command")

    def test_tui_model_command_dispatched_locally(self):
        """Verify tui model command dispatched locally."""
        from pathlib import Path

        from clawcodex_ext.tui.commands import (
            LOCAL_BUILTINS,
            CommandDispatchResult,
            dispatch_local_command,
        )

        assert "/model" in LOCAL_BUILTINS, "/model 不在 LOCAL_BUILTINS 中"

        result = dispatch_local_command(
            "/model",
            session=MagicMock(),
            workspace_root=Path("/tmp"),
            tool_registry=MagicMock(),
        )
        assert isinstance(result, CommandDispatchResult)
        assert result.handled is True
        assert result.open_dialog == "model", f"/model 应映射为 open_dialog='model', 实际得到 {result.open_dialog!r}"

        assert "/models" not in LOCAL_BUILTINS
        result_plural = dispatch_local_command(
            "/models",
            session=MagicMock(),
            workspace_root=Path("/tmp"),
            tool_registry=MagicMock(),
        )
        assert result_plural.handled is False
        assert result_plural.open_dialog is None

        from clawcodex_ext.tui.screens.model_picker import ModelPickerScreen

        assert ModelPickerScreen is not None


# =========================================================================
# =========================================================================


class TestStage3hTextualAvailability:
    """Tests for TestStage3hTextualAvailability."""

    def test_textual_available(self):
        """Verify textual available."""
        from src.entrypoints.tui import _textual_available

        result = _textual_available()
        assert isinstance(result, bool)
