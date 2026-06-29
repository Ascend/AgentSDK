"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """Get the project root directory path."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_openclaw_scripts_dir() -> Path:
    """Get the openclaw scripts directory path."""
    return get_project_root() / "openclaw" / "scripts"


@dataclass
class CLIResult:
    """Result of a CLI command execution."""

    exit_code: int
    stdout: str
    stderr: str
    combined_output: str

    @property
    def succeeded(self) -> bool:
        """Check if the command succeeded (exit code 0)."""
        return self.exit_code == 0

    @property
    def failed(self) -> bool:
        """Check if the command failed (non-zero exit code)."""
        return self.exit_code != 0


class OpenClawRunner:
    """
    Utility for running OpenClaw deploy.sh from source code.
    """

    def __init__(self, timeout: int = 120):
        """
        Initialize the OpenClaw runner.

        Args:
            timeout: Maximum time in seconds to wait for command completion.
        """
        self.timeout = timeout
        self.project_root = get_project_root()
        self.deploy_script = get_openclaw_scripts_dir() / "deploy.sh"

    def run_config(
        self,
        config_dir: str,
        base_port: int = 18789,
        instance_count: int = 1,
        extra_args: Optional[List[str]] = None,
    ) -> CLIResult:
        """
        Run 'deploy.sh config' to generate configuration files.

        Args:
            config_dir: Absolute path to the output config directory.
            base_port: Base port number for Gateway.
            instance_count: Number of instances to configure.
            extra_args: Optional list of additional CLI arguments.

        Returns:
            CLIResult containing exit code and captured output.
        """
        if not self.deploy_script.exists():
            raise RuntimeError(
                f"deploy.sh not found at: {self.deploy_script}"
            )

        cmd = [
            "bash",
            str(self.deploy_script),
            "config",
            "-n", str(instance_count),
            "-p", str(base_port),
            "-c", config_dir,
        ]

        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout,
                cwd=str(self.project_root),
            )
            combined = result.stdout + "\n" + result.stderr
            return CLIResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                combined_output=combined,
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
            stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
            return CLIResult(
                exit_code=-1,
                stdout=stdout,
                stderr=stderr + f"\n[TIMEOUT] Command timed out after {self.timeout}s",
                combined_output=stdout + "\n" + stderr,
            )


class SystemTestBase(unittest.TestCase):
    """Base class for OpenClaw system tests."""

    cli_timeout: int = 120

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.project_root = get_project_root()
        cls.scripts_dir = get_openclaw_scripts_dir()
        cls.cli_runner = OpenClawRunner(timeout=cls.cli_timeout)

        # Create a temporary directory for config output
        cls._temp_dir = tempfile.mkdtemp(prefix="openclaw_presmoke_")

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        if hasattr(cls, '_temp_dir') and os.path.exists(cls._temp_dir):
            shutil.rmtree(cls._temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

    def assertExitSuccess(self, result: CLIResult, msg: Optional[str] = None):
        """Assert that the CLI command succeeded (exit code 0)."""
        if result.exit_code != 0:
            failure_msg = msg or f"Expected exit code 0, got {result.exit_code}"
            failure_msg += f"\n\nOutput:\n{result.combined_output}"
            self.fail(failure_msg)

    def assertExitFailure(self, result: CLIResult, msg: Optional[str] = None):
        """Assert that the CLI command failed (non-zero exit code)."""
        if result.exit_code == 0:
            failure_msg = msg or "Expected non-zero exit code, got 0"
            failure_msg += f"\n\nOutput:\n{result.combined_output}"
            self.fail(failure_msg)

    def assertFileExists(self, file_path: str, msg: Optional[str] = None):
        """Assert that a file or directory exists."""
        if not os.path.exists(file_path):
            failure_msg = msg or f"Expected file/directory not found: {file_path}"
            self.fail(failure_msg)

    def assertFileExecutable(self, file_path: str, msg: Optional[str] = None):
        """Assert that a file is executable."""
        self.assertFileExists(file_path, msg)
        if not os.access(file_path, os.X_OK):
            failure_msg = msg or f"Expected file to be executable: {file_path}"
            self.fail(failure_msg)

    def assertJsonValid(self, file_path: str, msg: Optional[str] = None):
        """Assert that a file contains valid JSON."""
        self.assertFileExists(file_path, msg)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            failure_msg = msg or f"Expected valid JSON in: {file_path}"
            failure_msg += f"\n\nError: {e}"
            self.fail(failure_msg)

    def assertFileContains(
        self, file_path: str, pattern: str, msg: Optional[str] = None
    ):
        """Assert that a file contains the given pattern."""
        self.assertFileExists(file_path, msg)
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if pattern not in content:
            failure_msg = msg or f"Expected pattern not found in {file_path}: '{pattern}'"
            self.fail(failure_msg)

    def assertFileContainsAny(
        self, file_path: str, patterns: List[str], msg: Optional[str] = None
    ):
        """Assert that a file contains any of the given patterns."""
        self.assertFileExists(file_path, msg)
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if not any(pattern in content for pattern in patterns):
            failure_msg = (
                msg or f"Expected any of patterns not found in {file_path}: {patterns}"
            )
            self.fail(failure_msg)
