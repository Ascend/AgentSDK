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
import re
import shutil
import subprocess
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union


def get_project_root() -> Path:
    """Get the project root directory path."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return get_project_root() / "configs"


def get_presmoke_configs_dir() -> Path:
    """Get the presmoke test configs directory path."""
    return get_project_root() / "presmoke" / "aura" / "configs"


def get_train_configs_dir() -> Path:
    """Get the aura train configs directory path."""
    return get_project_root() / "aura" / "configs" / "train"


def get_local_ip() -> str:
    """
    Get the local IPv4 address.

    Resolution order:
    1. LOCAL_IP environment variable (injected by presmoke.sh via
       ``hostname -I | awk '{print $1}'``).
    2. The IPv4 address of the network interface specified by
       DEFAULT_SOCKET_IFNAME (default eth0), mirroring the logic in
       aura/scripts/base/utils.sh.
    """
    env_ip = os.environ.get("LOCAL_IP", "").strip()
    if env_ip:
        return env_ip

    ifname = os.environ.get("DEFAULT_SOCKET_IFNAME", "eth0")
    try:
        output = subprocess.run(
            ["ifconfig", ifname],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""

    for line in output.splitlines():
        if "inet " in line:
            match = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
            if match:
                return match.group(1)
    return ""


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
    Utility for running AgenticRL from source code via start_rl_with_verl_vllm.sh script.
    """

    def __init__(self, timeout: int = 300):
        """
        Initialize the source runner.

        Args:
            timeout: Maximum time in seconds to wait for command completion.
        """
        self.timeout = timeout
        self.project_root = get_project_root()
        self.aura_dir = self.project_root / "aura"
        self.run_script = self.aura_dir / "scripts" / "start_rl_with_verl_vllm.sh"

    def run(self):
        """
        Run the training script.

        Returns:
            CLIResult containing exit code and captured output.

        Raises:
            RuntimeError: If start_rl_with_verl_vllm.sh does not exist.
        """
        if not self.run_script.exists():
            raise RuntimeError(f"start_rl_with_verl_vllm.sh not found at: {self.run_script}")

        cmd = ["bash", "scripts/start_rl_with_verl_vllm.sh"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout,
                cwd=str(self.aura_dir),
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

    @staticmethod
    def contains_any(output: str, patterns: List[str]) -> bool:
        """Check if output contains any of the given patterns."""
        return any(pattern in output for pattern in patterns)


class FileOps:
    """
    Utility class for file operations during tests.

    Supports modifying existing config files (with automatic restoration)
    and copying config files to the train configs directory (with automatic
    cleanup). All operations are tracked so they can be rolled back in
    tearDown.
    """

    def __init__(self):
        self._modified_files: List[Tuple[Path, str]] = []
        self._copied_files: List[Path] = []
        self._deleted_files: List[Tuple[Path, str]] = []

    def modify_file(self, file_path: Union[str, Path], new_content: str) -> Path:
        """
        Overwrite an existing file with new content. The original content is
        backed up so it can be restored during cleanup.

        Args:
            file_path: Path to the file to modify.
            new_content: New content to write into the file.

        Returns:
            The resolved Path of the modified file.

        Raises:
            FileNotFoundError: If the target file does not exist.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Cannot modify non-existent file: {path}")

        original_content = path.read_text(encoding="utf-8")
        self._modified_files.append((path, original_content))
        path.write_text(new_content, encoding="utf-8")
        return path

    def replace_in_file(
        self,
        file_path: Union[str, Path],
        old: str,
        new: str,
        count: int = -1,
    ) -> Path:
        """
        Replace occurrences of ``old`` with ``new`` in a file. The original
        content is backed up for restoration during cleanup.

        Args:
            file_path: Path to the file to modify.
            old: Substring to be replaced.
            new: Substring to replace with.
            count: Maximum number of occurrences to replace. -1 means all.

        Returns:
            The resolved Path of the modified file.

        Raises:
            FileNotFoundError: If the target file does not exist.
            ValueError: If ``old`` is not found in the file.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Cannot modify non-existent file: {path}")

        original_content = path.read_text(encoding="utf-8")
        if old not in original_content:
            raise ValueError(f"Pattern not found in {path}: {old!r}")

        self._modified_files.append((path, original_content))
        new_content = original_content.replace(old, new, count)
        path.write_text(new_content, encoding="utf-8")
        return path

    def copy_to_train_configs(
        self,
        src: Union[str, Path],
        dest_name: Optional[str] = None,
    ) -> Path:
        """
        Copy a config file into the aura train configs directory. The copied
        file is tracked and will be removed during cleanup.

        Args:
            src: Path to the source config file. If a relative name is given,
                it is resolved against the presmoke configs directory.
            dest_name: Optional name for the destination file. If omitted, the
                source file name is used.

        Returns:
            The resolved Path of the copied file in the train configs dir.

        Raises:
            FileNotFoundError: If the source file does not exist.
        """
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = get_presmoke_configs_dir() / src_path
        src_path = src_path.resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Source config not found: {src_path}")

        dest_dir = get_train_configs_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = (dest_dir / (dest_name or src_path.name)).resolve()

        shutil.copy2(src_path, dest_path)
        self._copied_files.append(dest_path)
        return dest_path

    def delete_file(self, file_path: Union[str, Path]) -> Path:
        """
        Delete a file. The original content is backed up so the file can be
        restored during cleanup.

        Args:
            file_path: Path to the file to delete.

        Returns:
            The resolved Path of the deleted file.

        Raises:
            FileNotFoundError: If the target file does not exist.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Cannot delete non-existent file: {path}")

        original_content = path.read_text(encoding="utf-8")
        self._deleted_files.append((path, original_content))
        path.unlink()
        return path

    def cleanup(self):
        """Restore modified/deleted files and remove copied files."""
        for path, original_content in reversed(self._modified_files):
            try:
                path.write_text(original_content, encoding="utf-8")
            except OSError:
                pass
        self._modified_files.clear()

        for path, original_content in reversed(self._deleted_files):
            try:
                if not path.exists():
                    path.write_text(original_content, encoding="utf-8")
            except OSError:
                pass
        self._deleted_files.clear()

        for path in self._copied_files:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        self._copied_files.clear()


class SystemTestBase(unittest.TestCase):
    """Base class for AgenticRL system tests."""

    cli_timeout: int = 300

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.project_root = get_project_root()
        cls.configs_dir = get_configs_dir()
        cls.presmoke_configs_dir = get_presmoke_configs_dir()
        cls.train_configs_dir = get_train_configs_dir()
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
        self.file_ops = FileOps()

    def tearDown(self):
        """Clean up test-level fixtures."""
        self.file_ops.cleanup()
        for temp_file in self._test_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

    def run_cli(self) -> CLIResult:
        """
        Run the CLI.

        Returns:
            CLIResult with exit code and captured output.
        """
        return self.cli_runner.run()

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

    def modify_file(self, file_path: Union[str, Path], new_content: str) -> Path:
        """Overwrite a file with new content (original is restored on tearDown)."""
        return self.file_ops.modify_file(file_path, new_content)

    def replace_in_file(
        self,
        file_path: Union[str, Path],
        old: str,
        new: str,
        count: int = -1,
    ) -> Path:
        """Replace occurrences of ``old`` with ``new`` in a file (restored on tearDown)."""
        return self.file_ops.replace_in_file(file_path, old, new, count)

    def copy_to_train_configs(
        self,
        src: Union[str, Path],
        dest_name: Optional[str] = None,
    ) -> Path:
        """Copy a config file into the aura train configs directory (removed on tearDown)."""
        return self.file_ops.copy_to_train_configs(src, dest_name)

    def delete_file(self, file_path: Union[str, Path]) -> Path:
        """Delete a file (restored on tearDown)."""
        return self.file_ops.delete_file(file_path)
