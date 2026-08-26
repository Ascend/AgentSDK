#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0402,W0706

"""Atomic JSON write primitives for LKB board storage.

Implements the crash-safe atomic-write protocol from spec §7.5:
  1. Build candidate envelope + payload hash in memory
  2. Exclusive-create unique temp file in ``.tmp/`` (same dir as target)
  3. Write UTF-8 JSON (sorted keys, compact separators), flush, fsync
  4. If backup_path is given: copy current → temp backup, fsync, replace .bak
  5. ``os.replace`` temp → target (atomic rename)
  6. POSIX: fsync the directory
  7. Readback verify header + payload hash
  8. Cleanup temp on success; on failure, cleanup temp and leave old target

Crash semantics (spec §7.5):
  * Before os.replace → old target is authoritative
  * After os.replace → new target is authoritative
  * Temp files are never authoritative

This module imports nothing from ToolContext or Task-v2 (spec §11.4 inv 12).
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .error_codes import LkbErrorCode
from .ir_hash import canonical_hash


_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_INITIAL_DELAY_SECONDS = 0.005
_REPLACE_RETRY_MAX_DELAY_SECONDS = 0.05

__all__ = [
    "BoardStoreDiskFullError",
    "BoardStoreHashMismatchError",
    "BoardStoreIOError",
    "atomic_replace_with_backup",
    "atomic_write_json",
    "dir_fsync",
]


# ── error types ───────────────────────────────────────────────────────


class BoardStoreIOError(Exception):
    """Base class for board-store I/O failures.

    The old target file is guaranteed to still be readable when this is
    raised (spec §7.12 last paragraph).
    """

    code = LkbErrorCode.BOARD_STORE_IO_ERROR


class BoardStoreHashMismatchError(BoardStoreIOError):
    """Raised when readback verification finds a hash mismatch.

    This indicates either on-disk corruption between fsync and readback,
    or a serialization bug.  The caller must treat the write as failed.
    """

    code = LkbErrorCode.BOARD_STORE_HASH_MISMATCH


class BoardStoreDiskFullError(BoardStoreIOError):
    """Raised when the filesystem reports ENOSPC or EDQUOT during write."""

    code = LkbErrorCode.BOARD_STORE_DISK_FULL


# ── public API ────────────────────────────────────────────────────────


def atomic_write_json(
    target: Path,
    data: dict[str, Any],
    *,
    backup_path: Path | None = None,
    fsync_dir: bool = True,
    failpoint: Any | None = None,
    payload_hash_key: str = "payload_hash",
) -> None:
    """Atomically write *data* as JSON to *target*.

    Follows spec §7.5 steps 5-12, adapted to a generic JSON envelope:
      1. Serialize candidate and compute payload hash
      2. Exclusive-create temp in ``.tmp/`` sibling of target
      3. Write UTF-8 JSON, flush, fsync
      4. (optional) Rotate current target into *backup_path*
      5. ``os.replace`` temp → target
      6. (POSIX) fsync directory
      7. Readback and verify header/hash
      8. Cleanup

    Parameters
    ----------
    target:
      Final destination path.  Parent directory must exist.
    data:
      Dict to serialize.  If *payload_hash_key* is present in *data*, its
      value is used for readback verification; otherwise a hash is computed
      from *data* itself and readback is a byte-for-byte equality check.
    backup_path:
      If provided, the current *target* is atomically rotated into this
      path before the new candidate is installed.
    fsync_dir:
      If True (default), fsync the parent directory on POSIX.  Skipped on
      Windows (directory fsync is not meaningful).
    failpoint:
      Optional ``Failpoint`` instance for crash injection.  Named hit
      points:
        * ``"after_fsync_before_backup"``
        * ``"after_backup_before_replace"``
        * ``"after_fsync_before_replace"`` (alias, hit before backup step)
        * ``"after_replace_before_dirfsync"``
        * ``"after_dirfsync_before_readback"``
    payload_hash_key:
      Top-level key in *data* that holds the pre-computed payload hash
      string (``"sha256:..."``).  Readback verifies the on-disk file's
      payload matches this hash.
    """
    target = Path(target)
    parent = target.parent
    tmp_dir = parent / ".tmp"

    # -- step 1: perform every fallible serialization before filesystem mutation
    try:
        serialized = (
            json.dumps(
                data,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        expected_hash = str(data[payload_hash_key]) if payload_hash_key in data else canonical_hash(data)
    except (TypeError, ValueError) as exc:
        raise BoardStoreIOError(f"JSON serialization failed for {target}: {exc}") from exc

    tmp_dir.mkdir(parents=True, exist_ok=True)
    previous_content: bytes | None = None
    target_existed = target.exists()
    if target_existed:
        try:
            previous_content = target.read_bytes()
        except OSError as exc:
            raise _wrap_os_error(exc, "read current target before atomic replace") from exc

    # -- step 2: exclusive-create unique temp in .tmp -------------------
    fd = -1
    tmp_path: Path | None = None
    replaced = False
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(tmp_dir),
        )
        tmp_path = Path(tmp_name)

        # -- step 3: write UTF-8 JSON, flush, fsync --------------------
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1  # owned by the with-block; fd is closed now

        # failpoint: after fsync, before any file-system mutation of
        # the authoritative target/backup files
        if failpoint is not None:
            failpoint.hit("after_fsync_before_replace")
            failpoint.hit("after_fsync_before_backup")

        # -- step 4: rotate backup (current target → backup_path) ------
        if backup_path is not None and target.exists():
            _rotate_backup(target, Path(backup_path), tmp_dir, failpoint=failpoint)

        if failpoint is not None:
            failpoint.hit("after_backup_before_replace")

        # -- step 5: atomic replace temp → target ----------------------
        try:
            _replace_with_retry(tmp_path, target)
        except OSError as exc:
            raise _wrap_os_error(exc, "atomic replace") from exc

        tmp_path = None  # no longer exists after replace
        replaced = True

        if failpoint is not None:
            failpoint.hit("after_replace_before_dirfsync")

        # -- step 6: POSIX directory fsync -----------------------------
        if fsync_dir:
            _dir_fsync(parent)

        if failpoint is not None:
            failpoint.hit("after_dirfsync_before_readback")

        # -- step 7: readback verify -----------------------------------
        _verify_readback(target, expected_hash, payload_hash_key)

    except (BoardStoreIOError, OSError) as exc:
        failure = exc if isinstance(exc, BoardStoreIOError) else _wrap_os_error(exc, "atomic write")
        if replaced:
            try:
                _restore_previous_target(
                    target,
                    previous_content,
                    target_existed=target_existed,
                    tmp_dir=tmp_dir,
                    fsync_directory=fsync_dir,
                )
            except (BoardStoreIOError, OSError) as rollback_exc:
                raise BoardStoreIOError(
                    f"{failure}; rollback of {target} also failed: {rollback_exc}"
                ) from rollback_exc
        if failure is exc:
            raise failure
        raise failure from exc
    finally:
        # -- step 8: cleanup temp on any outcome -----------------------
        if tmp_path is not None and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def atomic_replace_with_backup(
    target: Path,
    new_content: dict[str, Any],
    backup_path: Path,
    *,
    fsync_dir: bool = True,
    failpoint: Any | None = None,
    payload_hash_key: str = "payload_hash",
) -> None:
    """Convenience wrapper: atomic write with mandatory .bak rotation.

    Equivalent to ``atomic_write_json(..., backup_path=backup_path)``.
    Exposed as a separate entrypoint so callers can express intent more
    clearly at the call site, and so the .bak rotation has its own test
    surface (spec §7.5 step 8 + §7.12).
    """
    atomic_write_json(
        target,
        new_content,
        backup_path=backup_path,
        fsync_dir=fsync_dir,
        failpoint=failpoint,
        payload_hash_key=payload_hash_key,
    )


def dir_fsync(path: Path) -> None:
    """Fsync a directory on POSIX; no-op on Windows.

    Directory fsync is needed after ``os.replace`` to guarantee the
    directory entry (not just the file contents) is durable.  On Windows
    this is not meaningful / not exposed through the C runtime.
    """
    _dir_fsync(Path(path))


# ── internals ─────────────────────────────────────────────────────────


def _rotate_backup(
    target: Path,
    backup_path: Path,
    tmp_dir: Path,
    *,
    failpoint: Any | None = None,
) -> None:
    """Atomically rotate the current *target* into *backup_path*.

    Uses a temp-in-.tmp + fsync + os.replace pattern for the backup itself,
    so a crash during backup rotation never leaves the .bak half-written.

    Spec §7.5 step 8.
    """
    # Copy current target to a temp file in .tmp, fsync, then atomically
    # replace backup_path.  This way the backup is always a full file.
    fd, tmp_backup_name = tempfile.mkstemp(
        prefix=f".{backup_path.name}.",
        suffix=".bak.tmp",
        dir=str(tmp_dir),
    )
    tmp_backup = Path(tmp_backup_name)
    try:
        with os.fdopen(fd, "wb") as out_f:
            with open(target, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)
            out_f.flush()
            os.fsync(out_f.fileno())

        try:
            _replace_with_retry(tmp_backup, backup_path)
        except OSError as exc:
            raise _wrap_os_error(exc, "backup rotation") from exc
        tmp_backup = None  # type: ignore[assignment]
    finally:
        if tmp_backup is not None:
            try:
                os.unlink(tmp_backup)
            except OSError:
                pass


def _dir_fsync(path: Path) -> None:
    """Internal dir fsync — no-op on Windows."""
    if os.name == "nt":
        return
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        raise _wrap_os_error(exc, "dir open for fsync") from exc
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise _wrap_os_error(exc, "dir fsync") from exc
    finally:
        os.close(dir_fd)


def _restore_previous_target(
    target: Path,
    previous_content: bytes | None,
    *,
    target_existed: bool,
    tmp_dir: Path,
    fsync_directory: bool,
) -> None:
    """Restore the pre-replace target after a post-commit verification failure."""
    if not target_existed:
        target.unlink(missing_ok=True)
        if fsync_directory:
            _dir_fsync(target.parent)
        return
    if previous_content is None:
        raise OSError("missing captured target content for rollback")

    fd, rollback_name = tempfile.mkstemp(prefix=f".{target.name}.rollback.", suffix=".tmp", dir=str(tmp_dir))
    rollback_path = Path(rollback_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(previous_content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(rollback_path, target)
        rollback_path = None  # type: ignore[assignment]
        if fsync_directory:
            _dir_fsync(target.parent)
    finally:
        if rollback_path is not None:
            try:
                os.unlink(rollback_path)
            except OSError:
                pass


def _verify_readback(
    target: Path,
    expected_hash: str,
    payload_hash_key: str,
) -> None:
    """Read back *target* and verify its payload hash matches *expected_hash*.

    Raises ``BoardStoreHashMismatchError`` on mismatch.  On any I/O error
    during readback, raises ``BoardStoreIOError``.
    """
    try:
        with open(target, "r", encoding="utf-8") as f:
            readback_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise BoardStoreIOError(f"Readback verification failed for {target}: {exc}") from exc

    if not isinstance(readback_data, dict):
        raise BoardStoreHashMismatchError(
            f"Readback of {target} is not a JSON object (got {type(readback_data).__name__})"
        )

    # Compute the hash of the readback payload (stripping the hash field
    # if present, to mirror the canonical payload-hash convention).
    if payload_hash_key in readback_data:
        payload = {k: v for k, v in readback_data.items() if k != payload_hash_key}
    else:
        payload = readback_data

    actual_hash = canonical_hash(payload)
    if actual_hash != expected_hash:
        raise BoardStoreHashMismatchError(
            f"Readback hash mismatch for {target}: expected {expected_hash}, got {actual_hash}"
        )


def _replace_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = _REPLACE_ATTEMPTS,
) -> None:
    """Bounded retry for transient Windows sharing violations."""
    delay = _REPLACE_RETRY_INITIAL_DELAY_SECONDS
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            transient = getattr(exc, "winerror", None) in {5, 32, 33} or getattr(exc, "errno", None) in {
                errno.EACCES,
                errno.EBUSY,
            }
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _REPLACE_RETRY_MAX_DELAY_SECONDS)


def _wrap_os_error(exc: OSError, operation: str) -> BoardStoreIOError:
    """Map a raw OSError to the appropriate BoardStore error subtype.

    ENOSPC / EDQUOT → BoardStoreDiskFullError
    everything else  → BoardStoreIOError
    """
    msg = f"{operation} failed: {exc.strerror or exc}"
    if getattr(exc, "errno", None) in (errno.ENOSPC, getattr(errno, "EDQUOT", None)):
        return BoardStoreDiskFullError(msg)
    return BoardStoreIOError(msg)
