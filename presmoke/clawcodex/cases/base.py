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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPSTREAM_PATCHES_BASE = Path("clawcodex-ascend") / "patches" / "upstream"
"""Path segment from project root to the upstream patches directory."""


def get_project_root() -> Path:
    """Get the project root directory path."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_clawcodex_dir() -> Path:
    """Get the clawcodex-ascend directory path."""
    return get_project_root() / "clawcodex-ascend"


def get_patches_dir() -> Path:
    """Auto-discover the upstream snapshot directory under patches/upstream/.

    Scans ``<project_root>/clawcodex-ascend/patches/upstream/`` for
    sub-directories that contain a ``series`` file.  If multiple candidates
    exist the lexicographically-last name is used (newer commit hashes
    sort later), avoiding a hard-coded upstream snapshot hash.

    Returns:
        Path to the discovered upstream patches directory.

    Raises:
        FileNotFoundError: If no sub-directory containing a ``series``
            file is found under the upstream patches base directory.
    """
    upstream_base = get_project_root() / UPSTREAM_PATCHES_BASE
    if not upstream_base.is_dir():
        raise FileNotFoundError(
            f"Upstream patches directory not found: {upstream_base}"
        )

    candidates = sorted(
        d for d in upstream_base.iterdir()
        if d.is_dir() and (d / "series").is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            f"No upstream snapshot with a 'series' file found under {upstream_base}"
        )

    return candidates[-1]


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


class ClawCodexRunner:
    """Utility for running ClawCodex CLI commands."""

    def __init__(self, timeout: int = 60):
        """
        Initialize the ClawCodex runner.

        Args:
            timeout: Maximum time in seconds to wait for command completion.
        """
        self.timeout = timeout
        self.project_root = get_project_root()

    def run(self, args: Optional[List[str]] = None) -> CLIResult:
        """
        Run the ClawCodex CLI with the given arguments.

        Args:
            args: Optional list of CLI arguments to append.

        Returns:
            CLIResult containing exit code and captured output.
        """
        cmd = ["python3", "-c", "import clawcodex_ext; clawcodex_ext.cli.main()"]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=str(self.project_root),
            )
            combined = result.stdout + result.stderr
            return CLIResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                combined_output=combined,
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            stderr_with_timeout = stderr + f"\n[TIMEOUT] Command timed out after {self.timeout}s"
            return CLIResult(
                exit_code=-1,
                stdout=stdout,
                stderr=stderr_with_timeout,
                combined_output=stdout + stderr_with_timeout,
            )


class FileOps:
    """
    Utility class for file operations during tests.

    Supports modifying existing files (with automatic restoration),
    deleting files (with automatic restoration), and replacing
    content in files. All operations are tracked so they can be
    rolled back in tearDown.
    """

    def __init__(self):
        self._modified_files: List[Tuple[Path, str]] = []
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
        """Restore modified/deleted files."""
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


class PatchValidator:
    """Validate clawcodex-ascend patch series file integrity."""

    @staticmethod
    def _check_series_exists() -> Path:
        """Check that the ``series`` file exists.

        Returns:
            The resolved Path to the series file.

        Raises:
            FileNotFoundError: If the series file is not found.
        """
        series_file = get_patches_dir() / "series"
        if not series_file.exists():
            raise FileNotFoundError(
                f"Patch series file not found: {series_file}"
            )
        return series_file

    @staticmethod
    def validate_series_integrity() -> List[str]:
        """
        Check every patch file referenced in ``series`` actually exists on disk.

        Returns:
            List of missing patch file names (empty if all present).

        Raises:
            FileNotFoundError: If the series file does not exist.
        """
        series_file = PatchValidator._check_series_exists()
        patches_dir = series_file.parent

        missing = []
        with open(series_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patch_path = patches_dir / line
                if not patch_path.exists():
                    missing.append(line)
        return missing

    @staticmethod
    def validate_patch_count() -> Tuple[int, int, Optional[str]]:
        """
        Return (declared_count, actual_count, error_message).

        The series file has a header comment ``# Total patches: N``.
        If it doesn't, returns (-1, actual_count, None).

        Raises:
            FileNotFoundError: If the series file does not exist.
        """
        series_file = PatchValidator._check_series_exists()
        patches_dir = series_file.parent

        actual_count = 0
        declared = -1
        with open(series_file, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("# Total patches:"):
                    try:
                        declared = int(stripped.split(":")[1].strip())
                    except (IndexError, ValueError):
                        pass
                if stripped and not stripped.startswith("#"):
                    actual_count += 1

        error = None
        if declared > 0 and declared != actual_count:
            error = f"Declared {declared} patches but found {actual_count}"
        return (declared, actual_count, error)

    @staticmethod
    def check_duplicates() -> List[str]:
        """Return list of duplicate patch filenames found in the series file.

        Raises:
            FileNotFoundError: If the series file does not exist.
        """
        series_file = PatchValidator._check_series_exists()
        patches_dir = series_file.parent

        seen = set()
        duplicates = []
        with open(series_file, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    if stripped in seen:
                        duplicates.append(stripped)
                    seen.add(stripped)
        return duplicates


class SystemTestBase(unittest.TestCase):
    """Base class for ClawCodex system tests."""

    cli_timeout: int = 60

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.project_root = get_project_root()
        cls.clawcodex_dir = get_clawcodex_dir()
        cls.patches_dir = get_patches_dir()
        cls.cli_runner = ClawCodexRunner(timeout=cls.cli_timeout)
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
        self.file_ops = FileOps()
        self._test_temp_files: List[str] = []

    def tearDown(self):
        """Clean up test-level fixtures."""
        self.file_ops.cleanup()
        for temp_file in self._test_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

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

    def delete_file(self, file_path: Union[str, Path]) -> Path:
        """Delete a file (restored on tearDown)."""
        return self.file_ops.delete_file(file_path)
