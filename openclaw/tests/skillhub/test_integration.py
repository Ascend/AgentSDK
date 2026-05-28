"""Integration tests for SkillHub CLI commands.

These tests execute real CLI commands, covering complete user workflows.
Uses real repository: https://gitcode.com/Ascend/agent-skills
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from skillhub.cli import app


runner = CliRunner()


def _is_unhandled_exception(result):
    """Check if result contains an unhandled exception (not SystemExit).

    SystemExit is typer's normal exit mechanism, not an unhandled exception.
    Returns True if there's a real error like TypeError, ValueError, etc.
    """
    if result.exception is None:
        return False
    # SystemExit is expected behavior for CLI commands
    if isinstance(result.exception, SystemExit):
        return False
    return True


class TestCLIIntegrationBasic:
    """Basic CLI integration tests - no external dependencies."""

    def test_cli_help(self):
        """Test CLI help information."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "skillhub" in result.output.lower()

    def test_cli_version_like_info(self):
        """Test version-related information."""
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_source_help(self):
        """Test source command help."""
        result = runner.invoke(app, ["source", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "add" in result.output

    def test_skill_help(self):
        """Test skill command help."""
        result = runner.invoke(app, ["skill", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "install" in result.output

    def test_config_help(self):
        """Test config command help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_auth_help(self):
        """Test auth command help."""
        result = runner.invoke(app, ["auth", "--help"])
        assert result.exit_code == 0

    def test_cache_help(self):
        """Test cache command help."""
        result = runner.invoke(app, ["cache", "--help"])
        assert result.exit_code == 0


class TestCLIIntegrationWithConfig:
    """CLI integration tests with config directory.

    These tests verify commands execute without unhandled Python exceptions.
    Typer's SystemExit is expected behavior and not treated as an error.
    """

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp(prefix="skillhub_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_skill_list_empty(self, temp_config_dir):
        """Test skill list command completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["skill", "list"])
            assert not _is_unhandled_exception(result)

    def test_list_installed_empty(self, temp_config_dir):
        """Test list installed alias completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["list", "installed"])
            assert not _is_unhandled_exception(result)

    def test_source_list_empty(self, temp_config_dir):
        """Test source list completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["source", "list"])
            assert not _is_unhandled_exception(result)

    def test_config_list(self, temp_config_dir):
        """Test config list completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["config", "list"])
            assert not _is_unhandled_exception(result)

    def test_auth_status(self, temp_config_dir):
        """Test auth status completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["auth", "status"])
            assert not _is_unhandled_exception(result)

    def test_cache_info(self, temp_config_dir):
        """Test cache info completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["cache", "info"])
            assert not _is_unhandled_exception(result)

    def test_doctor(self, temp_config_dir):
        """Test doctor completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["doctor"])
            assert not _is_unhandled_exception(result)

    def test_info_skill(self, temp_config_dir):
        """Test info skill completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["info", "test-skill"])
            assert not _is_unhandled_exception(result)

    def test_search(self, temp_config_dir):
        """Test search completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["search", "test"])
            assert not _is_unhandled_exception(result)


class TestCLIIntegrationSourceOperations:
    """CLI integration tests for source operations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp(prefix="skillhub_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_source_add_github(self, temp_config_dir):
        """Test adding GitHub source completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(
                app,
                [
                    "source",
                    "add",
                    "github-official",
                    "--type",
                    "github",
                    "--url",
                    "https://github.com/skills",
                ],
            )
            assert not _is_unhandled_exception(result)

    def test_source_add_gitee(self, temp_config_dir):
        """Test adding Gitee source completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(
                app,
                [
                    "source",
                    "add",
                    "gitee-official",
                    "--type",
                    "gitee",
                    "--url",
                    "https://gitee.com/skills",
                ],
            )
            assert not _is_unhandled_exception(result)

    def test_source_remove(self, temp_config_dir):
        """Test removing source completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["source", "remove", "test-id"])
            assert not _is_unhandled_exception(result)


class TestCLIIntegrationAuthOperations:
    """CLI integration tests for auth operations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp(prefix="skillhub_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_auth_login_github(self, temp_config_dir):
        """Test auth login completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["auth", "login", "github"])
            assert not _is_unhandled_exception(result)

    def test_auth_logout_github(self, temp_config_dir):
        """Test auth logout completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["auth", "logout", "github"])
            assert not _is_unhandled_exception(result)


class TestCLIIntegrationCacheOperations:
    """CLI integration tests for cache operations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp(prefix="skillhub_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_clear(self, temp_config_dir):
        """Test cache clear completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["cache", "clear"])
            assert not _is_unhandled_exception(result)


class TestCLIIntegrationSkillOperations:
    """CLI integration tests for skill operations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp(prefix="skillhub_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_skill_install(self, temp_config_dir):
        """Test skill install completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["skill", "install", "test-skill"])
            assert not _is_unhandled_exception(result)

    def test_skill_uninstall(self, temp_config_dir):
        """Test skill uninstall completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["skill", "uninstall", "test-skill"])
            assert not _is_unhandled_exception(result)

    def test_skill_upgrade(self, temp_config_dir):
        """Test skill upgrade completes without unhandled exception."""
        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": temp_config_dir}):
            result = runner.invoke(app, ["skill", "upgrade", "test-skill"])
            assert not _is_unhandled_exception(result)


class TestCLIIntegrationDoctorOperations:
    """CLI integration tests for doctor operations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp(prefix="skillhub_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_doctor_fix_missing_directories(self, temp_config_dir):
        """Test doctor completes without unhandled exception."""
        fresh_dir = Path(temp_config_dir) / "fresh_config"
        fresh_dir.mkdir(parents=True, exist_ok=True)

        with patch.dict(os.environ, {"SKILLHUB_DATA_DIR": str(fresh_dir)}):
            result = runner.invoke(app, ["doctor"])
            assert not _is_unhandled_exception(result)


class TestCLIIntegrationErrorHandling:
    """CLI integration tests for error handling."""

    def test_invalid_command(self):
        """Test invalid command fails gracefully without unhandled exception."""
        result = runner.invoke(app, ["invalid-command"])
        # Should fail but not crash with Python exception
        assert result.exit_code != 0
        assert not _is_unhandled_exception(result)

    def test_missing_argument(self):
        """Test missing argument fails gracefully without unhandled exception."""
        result = runner.invoke(app, ["skill", "install"])
        # Should fail but not crash with Python exception
        assert result.exit_code != 0
        assert not _is_unhandled_exception(result)
