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


import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from typer.testing import CliRunner

from skillhub.commands.list import app as list_app
from skillhub.commands.skill import app as skill_app
from skillhub.commands.source import app as source_app
from skillhub.commands.auth import app as auth_app
from skillhub.commands.cache import app as cache_app
from skillhub.commands.config import app as config_app
from skillhub.commands.doctor import app as doctor_app
from skillhub.commands.install import app as install_app
from skillhub.commands.uninstall import app as uninstall_app
from skillhub.commands.upgrade import app as upgrade_app
from skillhub.commands.info import app as info_app
from skillhub.commands.search import app as search_app


runner = CliRunner()


class TestSearchCommand:
    """Tests for search command."""

    def test_search_help(self):
        """Test search command help."""
        result = runner.invoke(search_app, ["--help"])
        assert result.exit_code == 0

    def test_search_function_exists(self):
        """Test that search_skills function exists."""
        from skillhub.commands.search import search_skills

        assert callable(search_skills)


class TestListCommand:
    """Tests for list command."""

    def test_list_help(self):
        """Test list command help."""
        result = runner.invoke(list_app, ["--help"])
        assert result.exit_code == 0

    def test_list_installed_help(self):
        """Test list installed subcommand help."""
        result = runner.invoke(list_app, ["installed", "--help"])
        assert result.exit_code == 0


class TestSkillCommand:
    """Tests for skill command group."""

    def test_skill_help(self):
        """Test skill command help."""
        result = runner.invoke(skill_app, ["--help"])
        assert result.exit_code == 0

    def test_skill_list_help(self):
        """Test skill list subcommand."""
        result = runner.invoke(skill_app, ["list", "--help"])
        assert result.exit_code == 0

    def test_skill_install_help(self):
        """Test skill install subcommand."""
        result = runner.invoke(skill_app, ["install", "--help"])
        assert result.exit_code == 0

    def test_skill_uninstall_help(self):
        """Test skill uninstall subcommand."""
        result = runner.invoke(skill_app, ["uninstall", "--help"])
        assert result.exit_code == 0

    def test_skill_upgrade_help(self):
        """Test skill upgrade subcommand."""
        result = runner.invoke(skill_app, ["upgrade", "--help"])
        assert result.exit_code == 0

    def test_skill_info_help(self):
        """Test skill info subcommand."""
        result = runner.invoke(skill_app, ["info", "--help"])
        assert result.exit_code == 0

    def test_is_local_path_function(self):
        """Test _is_local_path helper."""
        from skillhub.commands.skill import _is_local_path

        # Test with non-existent path
        assert _is_local_path("/nonexistent") is False


class TestSourceCommand:
    """Tests for source command."""

    def test_source_help(self):
        """Test source command help."""
        result = runner.invoke(source_app, ["--help"])
        assert result.exit_code == 0

    def test_source_add_help(self):
        """Test source add subcommand."""
        result = runner.invoke(source_app, ["add", "--help"])
        assert result.exit_code == 0

    def test_source_list_help(self):
        """Test source list subcommand."""
        result = runner.invoke(source_app, ["list", "--help"])
        assert result.exit_code == 0

    def test_source_remove_help(self):
        """Test source remove subcommand."""
        result = runner.invoke(source_app, ["remove", "--help"])
        assert result.exit_code == 0

    def test_source_test_help(self):
        """Test source test subcommand."""
        result = runner.invoke(source_app, ["test", "--help"])
        assert result.exit_code == 0


class TestAuthCommand:
    """Tests for auth command."""

    def test_auth_help(self):
        """Test auth command help."""
        result = runner.invoke(auth_app, ["--help"])
        assert result.exit_code == 0

    def test_auth_login_help(self):
        """Test auth login subcommand."""
        result = runner.invoke(auth_app, ["login", "--help"])
        assert result.exit_code == 0

    def test_auth_logout_help(self):
        """Test auth logout subcommand."""
        result = runner.invoke(auth_app, ["logout", "--help"])
        assert result.exit_code == 0

    def test_auth_status_help(self):
        """Test auth status subcommand."""
        result = runner.invoke(auth_app, ["status", "--help"])
        assert result.exit_code == 0


class TestCacheCommand:
    """Tests for cache command."""

    def test_cache_help(self):
        """Test cache command help."""
        result = runner.invoke(cache_app, ["--help"])
        assert result.exit_code == 0

    def test_cache_clear_help(self):
        """Test cache clear subcommand."""
        result = runner.invoke(cache_app, ["clear", "--help"])
        assert result.exit_code == 0

    def test_cache_stats_help(self):
        """Test cache stats subcommand."""
        # cache.py has 'info' command, not 'stats'
        result = runner.invoke(cache_app, ["info", "--help"])
        assert result.exit_code == 0


class TestConfigCommand:
    """Tests for config command."""

    def test_config_help(self):
        """Test config command help."""
        result = runner.invoke(config_app, ["--help"])
        assert result.exit_code == 0

    def test_config_show_help(self):
        """Test config show subcommand."""
        # config.py has 'get' and 'list', not 'show'
        result = runner.invoke(config_app, ["list", "--help"])
        assert result.exit_code == 0

    def test_config_set_help(self):
        """Test config set subcommand."""
        result = runner.invoke(config_app, ["set", "--help"])
        assert result.exit_code == 0


class TestDoctorCommand:
    """Tests for doctor command."""

    def test_doctor_help(self):
        """Test doctor command help."""
        result = runner.invoke(doctor_app, ["--help"])
        assert result.exit_code == 0

    def test_doctor_run_help(self):
        """Test doctor run subcommand."""
        result = runner.invoke(doctor_app, ["run", "--help"])
        assert result.exit_code == 0


class TestInstallCommand:
    """Tests for install command (alias)."""

    def test_install_help(self):
        """Test install command help."""
        result = runner.invoke(install_app, ["--help"])
        assert result.exit_code == 0


class TestUninstallCommand:
    """Tests for uninstall command (alias)."""

    def test_uninstall_help(self):
        """Test uninstall command help."""
        result = runner.invoke(uninstall_app, ["--help"])
        assert result.exit_code == 0


class TestUpgradeCommand:
    """Tests for upgrade command (alias)."""

    def test_upgrade_help(self):
        """Test upgrade command help."""
        result = runner.invoke(upgrade_app, ["--help"])
        assert result.exit_code == 0


class TestInfoCommand:
    """Tests for info command (alias)."""

    def test_info_help(self):
        """Test info command help."""
        result = runner.invoke(info_app, ["--help"])
        assert result.exit_code == 0


class TestConfigGetCommand:
    """Tests for config get command with real config."""

    def test_config_get_existing_key(self):
        """Test config get with existing key."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(config_app, ["get", "cache.enabled"])
                assert result.exit_code == 0
                assert "true" in result.output.lower() or "false" in result.output.lower()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_config_get_nested_key(self):
        """Test config get with nested key."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(config_app, ["get", "cache.ttl_metadata"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_config_get_invalid_key(self):
        """Test config get with invalid key."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(config_app, ["get", "nonexistent_key"])
                assert result.exit_code == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestConfigListCommand:
    """Tests for config list command."""

    def test_config_list_success(self):
        """Test config list command."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(config_app, ["list"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestConfigResetCommand:
    """Tests for config reset command."""

    def test_config_reset_with_force(self):
        """Test config reset with force."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            config_file = temp_dir / "config.json"
            config_file.write_text("{\"test\": \"value\"}")

            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(config_app, ["reset", "--force"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSourceListCommand:
    """Tests for source list command with real config."""

    def test_source_list_empty(self):
        """Test source list when empty."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(source_app, ["list"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestAuthStatusCommand:
    """Tests for auth status command with real config."""

    def test_auth_status(self):
        """Test auth status command."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(auth_app, ["status"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCacheInfoCommand:
    """Tests for cache info command with real config."""

    def test_cache_info(self):
        """Test cache info command."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(cache_app, ["info"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCacheClearCommand:
    """Tests for cache clear command."""

    def test_cache_clear_with_force(self):
        """Test cache clear command with force flag."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(cache_app, ["clear", "--force"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDoctorCommandWithConfig:
    """Tests for doctor command with real config."""

    def test_doctor_basic(self):
        """Test doctor command."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(doctor_app, [])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_doctor_verbose(self):
        """Test doctor with verbose flag."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(doctor_app, ["--verbose"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestListInstalledCommand:
    """Tests for list installed command with real config."""

    def test_list_installed_empty(self):
        """Test list installed when empty."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(list_app, ["installed"])
                assert result.exit_code in (0, 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSearchCommandWithConfig:
    """Tests for search command with real config."""

    def test_search_empty_results(self):
        """Test search with empty results."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(search_app, ["test"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSkillListCommand:
    """Tests for skill list command."""

    def test_skill_list_empty(self):
        """Test skill list when empty."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(skill_app, ["list"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSkillUninstallCommand:
    """Tests for skill uninstall command."""

    def test_skill_uninstall_nonexistent(self):
        """Test skill uninstall nonexistent."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(skill_app, ["uninstall", "nonexistent"])
                assert result.exit_code == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestUninstallCommandWithConfig:
    """Tests for uninstall command."""

    def test_uninstall_nonexistent(self):
        """Test uninstall nonexistent skill."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(uninstall_app, ["nonexistent"])
                assert result.exit_code == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestUpgradeCommandWithConfig:
    """Tests for upgrade command."""

    def test_upgrade_nonexistent(self):
        """Test upgrade nonexistent skill."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(upgrade_app, ["nonexistent"])
                assert result.exit_code == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestInfoCommandWithConfig:
    """Tests for info command."""

    def test_info_nonexistent(self):
        """Test info nonexistent skill."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(info_app, ["nonexistent"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestInstallCommandWithConfig:
    """Tests for install command."""

    def test_install_nonexistent(self):
        """Test install nonexistent skill."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(install_app, ["nonexistent"])
                assert result.exit_code == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_install_local_nonexistent(self):
        """Test install from nonexistent local path."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(install_app, ["./nonexistent_path"])
                assert result.exit_code == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestConfigSetCommand:
    """Tests for config set command."""

    def test_config_set_invalid_key(self):
        """Test config set with invalid key."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(config_app, ["set", "invalid.key", "value"])
                assert result.exit_code == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSourceAddCommand:
    """Tests for source add command."""

    def test_source_add_github(self):
        """Test source add github."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(source_app, ["add", "github", "https://github.com/test/skills"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_source_add_gitee(self):
        """Test source add gitee."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(source_app, ["add", "gitee", "https://gitee.com/test/skills"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestAuthLoginCommand:
    """Tests for auth login command."""

    def test_auth_login_with_token(self):
        """Test auth login with token."""
        from skillhub.models.credential import TokenValidation

        temp_dir = Path(tempfile.mkdtemp())
        try:
            mock_validation = TokenValidation(valid=True, scopes=["repo"], message="OK")

            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                with patch(
                    "skillhub.commands.auth.CredentialManagerImpl.validate_token",
                    new_callable=AsyncMock,
                    return_value=mock_validation,
                ):
                    with patch("skillhub.commands.auth.CredentialManagerImpl.store_token", new_callable=AsyncMock):
                        result = runner.invoke(auth_app, ["login", "github", "--token", "test_token"])
                        assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDoctorFixCommand:
    """Tests for doctor --fix command."""

    def test_doctor_fix(self):
        """Test doctor with fix flag."""

        temp_dir = Path(tempfile.mkdtemp())
        fresh_dir = temp_dir / "fresh"
        fresh_dir.mkdir(parents=True, exist_ok=True)

        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(fresh_dir)}):
                result = runner.invoke(doctor_app, ["--fix"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSearchJsonCommand:
    """Tests for search with JSON output."""

    def test_search_json_output(self):
        """Test search with JSON output."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(search_app, ["test", "--json"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestListJsonCommand:
    """Tests for list with JSON output."""

    def test_list_json_output(self):
        """Test list with JSON output."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(list_app, ["installed", "--json"])
                assert result.exit_code == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSourceRemoveCommand:
    """Tests for source remove command."""

    def test_source_remove_nonexistent(self):
        """Test source remove nonexistent."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(source_app, ["remove", "nonexistent"])
                assert result.exit_code == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestAuthLogoutCommand:
    """Tests for auth logout command."""

    def test_auth_logout(self):
        """Test auth logout."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(auth_app, ["logout", "github"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSkillInfoCommand:
    """Tests for skill info command."""

    def test_skill_info_nonexistent(self):
        """Test skill info nonexistent."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(skill_app, ["info", "nonexistent"])
                assert result.exit_code == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSkillUpgradeCommand:
    """Tests for skill upgrade command."""

    def test_skill_upgrade_nonexistent(self):
        """Test skill upgrade nonexistent."""

        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(temp_dir)}):
                result = runner.invoke(skill_app, ["upgrade", "nonexistent"])
                assert result.exit_code == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSourceCommandAdvanced:
    """Advanced tests for source command."""

    def test_source_list_with_mock(self):
        """Test source list with mock."""
        with patch("skillhub.commands.source.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            with patch("skillhub.commands.source.SourceManagerImpl") as mock_manager:
                mock_instance = MagicMock()
                mock_instance.list_sources = AsyncMock(return_value=[])
                mock_manager.return_value = mock_instance

                result = runner.invoke(source_app, ["list"])
                assert result.exit_code == 0


class TestAuthCommandAdvanced:
    """Advanced tests for auth command."""

    def test_auth_status_with_mock(self):
        """Test auth status."""
        with patch("skillhub.commands.auth.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            with patch("skillhub.commands.auth.CredentialManagerImpl") as mock_cred:
                mock_instance = MagicMock()
                mock_instance.list_tokens = AsyncMock(return_value=[])
                mock_cred.return_value = mock_instance

                result = runner.invoke(auth_app, ["status"])
                assert result.exit_code == 0


class TestCacheCommandAdvanced:
    """Advanced tests for cache command."""

    def test_cache_stats_with_mock(self):
        """Test cache stats."""
        from skillhub.models.cache import CacheStats

        mock_stats = CacheStats(
            size=100,
            hit_rate=0.5,
            miss_rate=0.5,
            total_size=1024,
            oldest_entry=None,
            newest_entry=None,
        )

        with patch("skillhub.commands.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            with patch("skillhub.commands.cache.CacheManagerImpl") as mock_cache:
                mock_instance = MagicMock()
                mock_instance.get_stats = AsyncMock(return_value=mock_stats)
                mock_cache.return_value = mock_instance

                result = runner.invoke(cache_app, ["info"])
                assert result.exit_code == 0


class TestConfigCommandAdvanced:
    """Advanced tests for config command."""

    def test_config_show_with_mock(self):
        """Test config show."""
        with patch("skillhub.commands.config.get_config") as mock_get:
            mock_get.return_value = MagicMock()

            result = runner.invoke(config_app, ["list"])
            assert result.exit_code == 0
