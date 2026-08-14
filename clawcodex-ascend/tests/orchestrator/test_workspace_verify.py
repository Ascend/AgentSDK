#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for workspace verification script generation (Phase 3).

Tests the verify.sh and README.md generation for preserved workspaces:
- generate_verify_script() creates executable verification script
- generate_workspace_readme() creates documentation
- Script content includes correct commands from agent config
"""

import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from extensions.orchestrator.workspace_verify import (
    generate_verify_script,
    generate_workspace_readme,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_agent_config():
    """Create a mock agent config with test commands."""
    config = MagicMock()
    config.test_command = "pytest tests/"
    config.build_command = "python -m build"
    config.lint_command = "ruff check ."
    return config


@pytest.fixture
def mock_issue_record():
    """Create a mock issue record."""
    record = MagicMock()
    record.issue_id = "issue-123"
    record.identifier = "issue-123"
    record.branch_name = "fix/issue-123"
    record.commit_sha = "abc123def456"
    record.status = "completed"
    record.pr_url = "https://github.com/org/repo/pull/123"
    record.verification_status = "passed"
    return record


class TestGenerateVerifyScript:
    """Tests for generate_verify_script function."""

    def test_generates_executable_script(self, temp_workspace, mock_agent_config, mock_issue_record):
        """Test that verify.sh is generated and is executable."""
        generate_verify_script(temp_workspace, mock_agent_config, mock_issue_record)

        verify_path = temp_workspace / ".orchestrator_workspace" / "verify.sh"
        assert verify_path.exists()

        # Check executable permission
        file_stat = verify_path.stat()
        assert file_stat.st_mode & stat.S_IXUSR, "verify.sh should be executable"

    def test_script_contains_all_commands(self, temp_workspace, mock_agent_config, mock_issue_record):
        """Test that verify.sh contains all configured commands."""
        generate_verify_script(temp_workspace, mock_agent_config, mock_issue_record)

        verify_path = temp_workspace / ".orchestrator_workspace" / "verify.sh"
        content = verify_path.read_text()

        assert "pytest tests/" in content
        assert "python -m build" in content
        assert "ruff check ." in content

    def test_script_contains_metadata(self, temp_workspace, mock_agent_config, mock_issue_record):
        """Test that verify.sh contains issue metadata."""
        generate_verify_script(temp_workspace, mock_agent_config, mock_issue_record)

        verify_path = temp_workspace / ".orchestrator_workspace" / "verify.sh"
        content = verify_path.read_text()

        assert "issue-123" in content
        assert "fix/issue-123" in content
        assert "abc123def456" in content

    def test_script_has_proper_structure(self, temp_workspace, mock_agent_config, mock_issue_record):
        """Test that verify.sh has proper bash structure."""
        generate_verify_script(temp_workspace, mock_agent_config, mock_issue_record)

        verify_path = temp_workspace / ".orchestrator_workspace" / "verify.sh"
        content = verify_path.read_text()

        # Should start with shebang
        assert content.startswith("#!/bin/bash") or content.startswith("#!/usr/bin/env bash")

        # Should have set -e for error handling
        assert "set -e" in content

        # Should echo status messages
        assert "Running tests" in content or "pytest" in content

    def test_script_with_missing_commands(self, temp_workspace, mock_issue_record):
        """Test script generation when some commands are missing."""
        agent_config = MagicMock()
        agent_config.test_command = "pytest"
        agent_config.build_command = None
        agent_config.lint_command = None

        generate_verify_script(temp_workspace, agent_config, mock_issue_record)

        verify_path = temp_workspace / ".orchestrator_workspace" / "verify.sh"
        assert verify_path.exists()

        content = verify_path.read_text()
        assert "pytest" in content
        # Should not have build or lint sections
        assert "python -m build" not in content
        assert "ruff check" not in content

    def test_script_handles_empty_commands(self, temp_workspace, mock_issue_record):
        """Test script generation when all commands are empty."""
        agent_config = MagicMock()
        agent_config.test_command = ""
        agent_config.build_command = ""
        agent_config.lint_command = ""

        # Should not create script if no commands
        generate_verify_script(temp_workspace, agent_config, mock_issue_record)

        verify_path = temp_workspace / ".orchestrator_workspace" / "verify.sh"
        # Script may or may not be created depending on implementation
        if verify_path.exists():
            content = verify_path.read_text()
            # Should still have basic structure
            assert "#!/bin/bash" in content or "#!/usr/bin/env bash" in content


class TestGenerateWorkspaceReadme:
    """Tests for generate_workspace_readme function."""

    def test_generates_readme_file(self, temp_workspace, mock_issue_record):
        """Test that README.md is generated."""
        generate_workspace_readme(temp_workspace, mock_issue_record)

        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"
        assert readme_path.exists()

    def test_readme_contains_issue_info(self, temp_workspace, mock_issue_record):
        """Test that README.md contains issue information."""
        generate_workspace_readme(temp_workspace, mock_issue_record)

        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"
        content = readme_path.read_text()

        assert "issue-123" in content
        assert "fix/issue-123" in content
        assert "abc123def456" in content
        assert "completed" in content

    def test_readme_contains_pr_link(self, temp_workspace, mock_issue_record):
        """Test that README.md contains PR link."""
        generate_workspace_readme(temp_workspace, mock_issue_record)

        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"
        content = readme_path.read_text()

        assert "https://github.com/org/repo/pull/123" in content

    def test_readme_contains_verification_instructions(self, temp_workspace, mock_issue_record):
        """Test that README.md contains verification instructions."""
        generate_workspace_readme(temp_workspace, mock_issue_record)

        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"
        content = readme_path.read_text()

        assert "verify.sh" in content
        assert "./.orchestrator_workspace/verify.sh" in content

    def test_readme_without_pr_url(self, temp_workspace):
        """Test README generation without PR URL."""
        issue_record = MagicMock()
        issue_record.issue_id = "issue-456"
        issue_record.identifier = "issue-456"
        issue_record.branch_name = "feat/issue-456"
        issue_record.commit_sha = "def789"
        issue_record.status = "failed"
        issue_record.pr_url = None
        issue_record.verification_status = "failed"

        generate_workspace_readme(temp_workspace, issue_record)

        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"
        assert readme_path.exists()

        content = readme_path.read_text()
        assert "issue-456" in content
        assert "failed" in content

    def test_readme_lists_workspace_files(self, temp_workspace, mock_issue_record):
        """Test that README.md lists workspace files."""
        # Create some test files
        (temp_workspace / "src").mkdir()
        (temp_workspace / "src" / "main.py").write_text("print('hello')")
        (temp_workspace / "tests").mkdir()
        (temp_workspace / "tests" / "test_main.py").write_text("def test_main(): pass")

        generate_workspace_readme(temp_workspace, mock_issue_record)

        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"
        content = readme_path.read_text()

        # Should mention the directories
        assert "src" in content
        assert "tests" in content


class TestIntegration:
    """Integration tests for verify.sh and README.md together."""

    def test_both_files_generated_together(self, temp_workspace, mock_agent_config, mock_issue_record):
        """Test that both verify.sh and README.md can be generated together."""
        generate_verify_script(temp_workspace, mock_agent_config, mock_issue_record)
        generate_workspace_readme(temp_workspace, mock_issue_record)

        verify_path = temp_workspace / ".orchestrator_workspace" / "verify.sh"
        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"

        assert verify_path.exists()
        assert readme_path.exists()

    def test_readme_references_verify_script(self, temp_workspace, mock_agent_config, mock_issue_record):
        """Test that README.md references the verify.sh script."""
        generate_verify_script(temp_workspace, mock_agent_config, mock_issue_record)
        generate_workspace_readme(temp_workspace, mock_issue_record)

        readme_path = temp_workspace / ".orchestrator_workspace" / "README.md"
        content = readme_path.read_text()

        # README should mention how to run verification
        assert "verify.sh" in content

    def test_scripts_are_consistent(self, temp_workspace, mock_agent_config, mock_issue_record):
        """Test that verify.sh and README.md have consistent information."""
        generate_verify_script(temp_workspace, mock_agent_config, mock_issue_record)
        generate_workspace_readme(temp_workspace, mock_issue_record)

        verify_content = (temp_workspace / ".orchestrator_workspace" / "verify.sh").read_text()
        readme_content = (temp_workspace / ".orchestrator_workspace" / "README.md").read_text()

        # Both should mention the issue ID
        assert "issue-123" in verify_content
        assert "issue-123" in readme_content

        # Both should mention the branch
        assert "fix/issue-123" in verify_content
        assert "fix/issue-123" in readme_content
