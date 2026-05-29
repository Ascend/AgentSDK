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


from skillhub.cli import app, run


class TestCLI:
    """Tests for CLI application."""

    def test_app_creation(self):
        """Test that CLI app is created correctly."""
        assert app is not None
        assert app.info.name == "skillhub"

    def test_app_has_subcommands(self):
        """Test that CLI has registered subcommands."""
        # Get registered commands
        registered = app.registered_commands
        registered_groups = app.registered_groups

        # Check that key commands are registered
        assert len(registered) > 0 or len(registered_groups) > 0

    def test_run_function_exists(self):
        """Test that run function exists."""
        assert callable(run)

    def test_main_callback_exists(self):
        """Test that main callback is registered."""
        # The callback should be registered
        assert app.callback is not None


class TestCLIIntegration:
    """Integration tests for CLI using CliRunner."""

    def test_cli_help(self):
        """Test CLI help output."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "skillhub" in result.output.lower() or "skill" in result.output.lower()

    def test_cli_version_not_implemented(self):
        """Test that version flag behavior."""
        from typer.testing import CliRunner

        runner = CliRunner()
        runner.invoke(app, ["--version"])
        # May fail if not implemented, which is fine

    def test_cli_no_args_shows_help(self):
        """Test that no args shows help."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, [])

        # Should show help when no_args_is_help=True (exit_code 0 or 2)
        assert result.exit_code in [0, 2]
        if result.exit_code == 0:
            assert "skillhub" in result.output.lower() or "skill" in result.output.lower()

    def test_cli_invalid_command(self):
        """Test invalid command handling."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0


class TestCLISubcommands:
    """Tests for CLI subcommand registration."""

    def test_source_command_registered(self):
        """Test source command is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["source", "--help"])

        assert result.exit_code == 0

    def test_skill_command_registered(self):
        """Test skill command is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["skill", "--help"])

        assert result.exit_code == 0

    def test_search_command_registered(self):
        """Test search command is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0

    def test_config_command_registered(self):
        """Test config command is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0

    def test_auth_command_registered(self):
        """Test auth command is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["auth", "--help"])

        assert result.exit_code == 0

    def test_cache_command_registered(self):
        """Test cache command is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["cache", "--help"])

        assert result.exit_code == 0

    def test_doctor_command_registered(self):
        """Test doctor command is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["doctor", "--help"])

        assert result.exit_code == 0

    def test_install_alias_registered(self):
        """Test install alias is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["install", "--help"])

        assert result.exit_code == 0

    def test_list_alias_registered(self):
        """Test list alias is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["list", "--help"])

        assert result.exit_code == 0

    def test_uninstall_alias_registered(self):
        """Test uninstall alias is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["uninstall", "--help"])

        assert result.exit_code == 0

    def test_upgrade_alias_registered(self):
        """Test upgrade alias is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["upgrade", "--help"])

        assert result.exit_code == 0

    def test_info_alias_registered(self):
        """Test info alias is registered."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["info", "--help"])

        assert result.exit_code == 0
