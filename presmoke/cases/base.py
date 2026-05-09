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
import os
import subprocess
import unittest
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """Get the project root directory path."""
    return Path(__file__).resolve().parent.parent.parent


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return get_project_root() / "configs"


def get_presmoke_configs_dir() -> Path:
    """Get the presmoke test configs directory path."""
    return get_project_root() / "presmoke" / "configs"


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


class SourceRunner:
    """
    Utility for running AgenticRL from source code via run_start_in_local.sh script.
    """

    def __init__(self, timeout: int = 300):
        """
        Initialize the source runner.

        Args:
            timeout: Maximum time in seconds to wait for command completion.
        """
        self.timeout = timeout
        self.project_root = get_project_root()
        self.run_script = self.project_root / "aura" / "run_start_in_local.sh"
        self.presmoke_configs_dir = get_presmoke_configs_dir()

    def run(self, config_name: str, extra_args: Optional[List[str]] = None, expect_error: bool = False):
        """
        Run the training script with the specified config name.

        Args:
            config_name: Name of the YAML configuration file (will look in presmoke/configs/).
            extra_args: Optional list of additional CLI arguments.
            expect_error: If True, don't raise exception on script not found.

        Returns:
            CLIResult containing exit code and captured output.
        """
        if not self.run_script.exists():
            if expect_error:
                return CLIResult(
                    exit_code=1,
                    stdout="",
                    stderr=f"run_start_in_local.sh not found at: {self.run_script}",
                    combined_output=f"run_start_in_local.sh not found at: {self.run_script}"
                )
            raise RuntimeError(f"run_start_in_local.sh not found at: {self.run_script}")

        # Check if config file exists in presmoke/configs/
        config_path = self.presmoke_configs_dir / config_name
        if config_path.exists():
            # Use absolute path for presmoke configs
            full_config_path = str(config_path.absolute())
            cmd = ["bash", str(self.run_script), "--config-name", full_config_path]
        else:
            # Use config name directly (will look in configs/ directory)
            cmd = ["bash", str(self.run_script), "--config-name", config_name]

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

    def run_without_args(self) -> CLIResult:
        """Run the script without any arguments."""
        cmd = ["bash", str(self.run_script)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
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
                stderr=stderr + "\n[TIMEOUT]",
                combined_output=stdout + "\n" + stderr,
            )


class LogAssertions:
    """Utility class for asserting log content patterns."""

    @staticmethod
    def contains(output: str, pattern: str) -> bool:
        """Check if output contains the given pattern (case-sensitive)."""
        return pattern in output

    @staticmethod
    def contains_any(output: str, patterns: List[str]) -> bool:
        """Check if output contains any of the given patterns."""
        return any(pattern in output for pattern in patterns)


class SystemTestBase(unittest.TestCase):
    """Base class for AgenticRL system tests."""

    cli_timeout: int = 300

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.project_root = get_project_root()
        cls.configs_dir = get_configs_dir()
        cls.presmoke_configs_dir = get_presmoke_configs_dir()
        cls.cli_runner = SourceRunner(timeout=cls.cli_timeout)
        cls.log_assert = LogAssertions()
        cls._temp_files: List[str] = []

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        for temp_file in cls._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

    def setUp(self):
        """Set up test-level fixtures."""
        self._test_temp_files: List[str] = []

    def tearDown(self):
        """Clean up test-level fixtures."""
        for temp_file in self._test_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

    def get_config_name(self, filename: str) -> str:
        """Get the config name for use with run_start_in_local.sh."""
        return filename

    def run_cli(self, config_name: str) -> CLIResult:
        """
        Run the CLI with the given config name.

        Args:
            config_name: Name of the configuration file.

        Returns:
            CLIResult with exit code and captured output.
        """
        return self.cli_runner.run(config_name)

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

    def assertLogContains(self, result: CLIResult, pattern: str, msg: Optional[str] = None):
        """Assert that the output contains the given pattern."""
        if not self.log_assert.contains(result.combined_output, pattern):
            failure_msg = msg or f"Expected pattern not found in output: '{pattern}'"
            failure_msg += f"\n\nOutput:\n{result.combined_output}"
            self.fail(failure_msg)

    def assertLogContainsAny(self, result: CLIResult, patterns: List[str], msg: Optional[str] = None):
        """Assert that the output contains any of the given patterns."""
        if not self.log_assert.contains_any(result.combined_output, patterns):
            failure_msg = msg or f"Expected any of patterns not found in output: {patterns}"
            failure_msg += f"\n\nOutput:\n{result.combined_output}"
            self.fail(failure_msg)

    def assertLogContainsAll(self, result: CLIResult, patterns: List[str], msg: Optional[str] = None):
        """Assert that the output contains all of the given patterns."""
        missing = [p for p in patterns if not self.log_assert.contains(result.combined_output, p)]
        if missing:
            failure_msg = msg or f"Missing expected patterns in output: {missing}"
            failure_msg += f"\n\nOutput:\n{result.combined_output}"
            self.fail(failure_msg)
