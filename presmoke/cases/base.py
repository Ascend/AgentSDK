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


def get_referenceapps_dir() -> Path:
    """Get the reference applications directory path."""
    return Path("/home/presmoke_data/AgentSDK/referenceapps")

def get_fixtures_dir() -> Path:
    """Get the fixtures directory path."""
    return Path("/home/presmoke_data/AgentSDK/configs")


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


class CLIRunner:
    """
    Utility for running the AgenticRL CLI via the 'agentic_rl' command.
    """

    def __init__(self, timeout: int = 600):
        """
        Initialize the CLI runner.

        Args:
            timeout: Maximum time in seconds to wait for command completion.
        """
        self.timeout = timeout
        self.cli_command = self._find_cli_command()

    def _find_cli_command(self) -> str:
        """
        Locate the 'agentic_rl' executable in the system PATH.

        Returns:
            Full path to the 'agentic_rl' command.

        Raises:
            RuntimeError: If the command is not found.
        """
        raw_path = os.environ.get("PATH", "")
        # Expand any tilde (~) in each PATH component
        expanded_paths = [os.path.expanduser(p) for p in raw_path.split(os.pathsep)]
        expanded_path = os.pathsep.join(expanded_paths)

        cmd_path = shutil.which("agentic_rl", path=expanded_path)
        if cmd_path is None:
            raise RuntimeError(
                f"agentic_rl command not found in PATH after expansion: {expanded_path}"
            )
        return cmd_path

    def run(self, config_path: str, extra_args: Optional[List[str]] = None):
        """
        Run the training CLI with the specified config.

        Args:
            config_path: Path to the YAML configuration file.
            extra_args: Optional list of additional CLI arguments.

        Returns:
            CLIResult containing exit code and captured output.
        """
        cmd = [self.cli_command, "--config-path", config_path]
        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout,
                cwd=str(get_referenceapps_dir()),
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


class LogAssertions:
    """Utility class for asserting log content patterns."""

    @staticmethod
    def contains(output: str, pattern: str) -> bool:
        """Check if output contains the given pattern (case-sensitive)."""
        return pattern in output


class SystemTestBase(unittest.TestCase):
    """Base class for AgenticRL system tests."""

    cli_timeout: int = 1200  # 20 minutes default timeout

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.fixtures_dir = get_fixtures_dir()
        cls.cli_runner = CLIRunner(timeout=cls.cli_timeout)
        cls.log_assert = LogAssertions()
        cls._temp_files: List[str] = []
        cls.ray_path = shutil.which("ray")
        if cls.ray_path is None:
            raise RuntimeError("ray executable not found in PATH")

        try:
            subprocess.run([cls.ray_path, "stop"], check=False, capture_output=True)
            subprocess.run(
                [cls.ray_path, "start", "--head", "--port=6379"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start Ray: {e.stderr}") from e


    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        for temp_file in cls._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

        # Stop Ray after all tests
        try:
            subprocess.run([cls.ray_path, "stop"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
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

        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")

    def get_fixture_path(self, filename: str) -> str:
        """Get the full path to a fixture file."""
        return str(self.fixtures_dir / filename)


    def run_cli(self, config_path: str) -> CLIResult:
        """
        Run the CLI with the given config path.

        Args:
            config_path: Path to the configuration file.

        Returns:
            CLIResult with exit code and captured output.
        """
        return self.cli_runner.run(config_path)

    # Assertion helpers
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


    def assertLogContainsAll(self, result: CLIResult, patterns: List[str], msg: Optional[str] = None):
        """Assert that the output contains all of the given patterns."""
        missing = [p for p in patterns if not self.log_assert.contains(result.combined_output, p)]
        if missing:
            failure_msg = msg or f"Missing expected patterns in output: {missing}"
            failure_msg += f"\n\nOutput:\n{result.combined_output}"
            self.fail(failure_msg)
