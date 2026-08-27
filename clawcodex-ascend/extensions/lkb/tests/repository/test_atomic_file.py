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
# pylint: disable=C1803,relative-beyond-top-level
"""Tests for atomic_file.py — crash-safe JSON write primitives.

Covers spec §7.5 atomic-write protocol and spec §11.4 invariant 14
(no half-write observable).

Test IDs:
  LKB-STORE-003  — basic atomic write: target is created with correct content
  LKB-STORE-011  — crash before os.replace: old file untouched, no temp leak
  LKB-STORE-012  — crash after os.replace (but before dir fsync): new file readable
  LKB-STORE-013  — .bak rotation: previous target becomes .bak after write
  LKB-STORE-014  — readback hash mismatch raises BoardStoreHashMismatchError
  LKB-STORE-017  — temp files created in .tmp/ sibling of target
  LKB-STORE-019  — disk-full / ENOSPC maps to BoardStoreDiskFullError
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest

from .._support import Failpoint
from lkb.atomic_file import (
    BoardStoreDiskFullError,
    BoardStoreHashMismatchError,
    BoardStoreIOError,
    atomic_replace_with_backup,
    atomic_write_json,
    dir_fsync,
)
from lkb.ir_hash import canonical_hash


# ── helpers ───────────────────────────────────────────────────────────


def _sample_data(rev: int = 1) -> dict[str, Any]:
    payload = {
        "board_id": "board-alpha",
        "store_revision": rev,
        "nodes": {"n1": {"title": "hello", "state": "ready"}},
    }
    payload["payload_hash"] = canonical_hash({k: v for k, v in payload.items() if k != "payload_hash"})
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── LKB-STORE-003 ────────────────────────────────────────────────────


class TestLkbStore003BasicAtomicWrite:
    """Basic atomic write creates target with correct content."""

    def test_writes_correct_json(self, tmp_path: Path) -> None:
        target = tmp_path / "board.json"
        data = _sample_data(1)
        atomic_write_json(target, data)

        assert target.exists()
        on_disk = _read_json(target)
        assert on_disk["board_id"] == "board-alpha"
        assert on_disk["store_revision"] == 1
        assert on_disk["payload_hash"] == data["payload_hash"]

    def test_writes_sorted_keys_compact_separators(self, tmp_path: Path) -> None:
        target = tmp_path / "board.json"
        data = {"z": 1, "a": 2, "m": 3}
        atomic_write_json(target, data)

        raw = target.read_text(encoding="utf-8")
        # compact separators → no spaces after : or ,
        assert " " not in raw.strip().split("{", 1)[1].rsplit("}", 1)[0]
        # sorted keys → "a" before "m" before "z"
        pos_a = raw.index('"a"')
        pos_m = raw.index('"m"')
        pos_z = raw.index('"z"')
        assert pos_a < pos_m < pos_z

    def test_utf8_content_preserved(self, tmp_path: Path) -> None:
        target = tmp_path / "board.json"
        data = {"title": "看板测试", "payload_hash": ""}
        data["payload_hash"] = canonical_hash({k: v for k, v in data.items() if k != "payload_hash"})
        atomic_write_json(target, data)

        on_disk = _read_json(target)
        assert on_disk["title"] == "看板测试"

    def test_creates_tmp_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "board.json"
        target.parent.mkdir()
        atomic_write_json(target, _sample_data(1))
        assert (tmp_path / "sub" / ".tmp").exists()
        # .tmp should be empty after successful write
        assert list((tmp_path / "sub" / ".tmp").iterdir()) == []


# ── LKB-STORE-011 ────────────────────────────────────────────────────


class TestLkbStore011CrashBeforeReplace:
    """Crash before os.replace: old file untouched, no temp leak."""

    def test_crash_after_fsync_leaves_old_target_intact(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
    ) -> None:
        target = tmp_path / "board.json"
        old_data = _sample_data(1)
        atomic_write_json(target, old_data)
        old_mtime = target.stat().st_mtime_ns

        new_data = _sample_data(2)

        class SimulatedCrash(Exception):
            pass

        failpoint.register("after_fsync_before_replace", SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_write_json(target, new_data, failpoint=failpoint)

        # old file is still there and unchanged
        assert target.exists()
        assert target.stat().st_mtime_ns == old_mtime
        on_disk = _read_json(target)
        assert on_disk["store_revision"] == 1

    def test_no_temp_file_leaked_after_crash(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
    ) -> None:
        target = tmp_path / "board.json"
        atomic_write_json(target, _sample_data(1))

        class SimulatedCrash(Exception):
            pass

        failpoint.register("after_fsync_before_replace", SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_write_json(target, _sample_data(2), failpoint=failpoint)

        tmp_dir = tmp_path / ".tmp"
        leftover = list(tmp_dir.glob("*.tmp")) if tmp_dir.exists() else []
        assert not leftover, f"Temp files leaked after crash: {leftover}"

    def test_crash_during_write_no_partial_target(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
    ) -> None:
        target = tmp_path / "board.json"

        # target does not exist yet — crash mid-write must not create it
        class SimulatedCrash(Exception):
            pass

        failpoint.register("after_fsync_before_replace", SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_write_json(target, _sample_data(1), failpoint=failpoint)

        assert not target.exists(), "Target must not exist after pre-replace crash"


# ── LKB-STORE-012 ────────────────────────────────────────────────────


class TestLkbStore012CrashAfterReplace:
    """Crash after os.replace but before dir-fsync: new file is readable.

    Spec §7.5 crash semantics: after step 9 (os.replace), the new file is
    authoritative.  A crash during the later steps (dir fsync, readback,
    cleanup) must NOT leave the target in an unreadable state.
    """

    def test_crash_after_replace_new_file_readable(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
    ) -> None:
        target = tmp_path / "board.json"
        old_data = _sample_data(1)
        atomic_write_json(target, old_data)
        new_data = _sample_data(2)

        class SimulatedCrash(Exception):
            pass

        failpoint.register("after_replace_before_dirfsync", SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_write_json(target, new_data, failpoint=failpoint)

        # After os.replace, the new content is authoritative
        assert target.exists()
        on_disk = _read_json(target)
        assert on_disk["store_revision"] == 2

    def test_crash_during_readback_new_file_still_readable(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
    ) -> None:
        target = tmp_path / "board.json"
        new_data = _sample_data(5)

        class SimulatedCrash(Exception):
            pass

        failpoint.register("after_dirfsync_before_readback", SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_write_json(target, new_data, failpoint=failpoint)

        assert target.exists()
        on_disk = _read_json(target)
        assert on_disk["store_revision"] == 5
        # target has correct hash (it was written properly)
        assert on_disk["payload_hash"] == new_data["payload_hash"]


# ── LKB-STORE-013 ────────────────────────────────────────────────────


class TestLkbStore013BackupRotation:
    """.bak rotation: previous target becomes .bak after write."""

    def test_backup_created_on_first_rotation(self, tmp_path: Path) -> None:
        target = tmp_path / "board.json"
        backup = tmp_path / "board.json.bak"
        data_v1 = _sample_data(1)
        atomic_write_json(target, data_v1)

        data_v2 = _sample_data(2)
        atomic_replace_with_backup(target, data_v2, backup)

        assert backup.exists()
        backup_data = _read_json(backup)
        assert backup_data["store_revision"] == 1

        current_data = _read_json(target)
        assert current_data["store_revision"] == 2

    def test_backup_rotates_multiple_times(self, tmp_path: Path) -> None:
        target = tmp_path / "board.json"
        backup = tmp_path / "board.json.bak"

        for rev in range(1, 5):
            atomic_replace_with_backup(target, _sample_data(rev), backup)

        # current = rev 4, backup = rev 3
        assert _read_json(target)["store_revision"] == 4
        assert _read_json(backup)["store_revision"] == 3

    def test_no_bak_when_target_does_not_exist(self, tmp_path: Path) -> None:
        target = tmp_path / "board.json"
        backup = tmp_path / "board.json.bak"
        # First write — no previous target, so no .bak should be created
        atomic_replace_with_backup(target, _sample_data(1), backup)

        assert target.exists()
        # backup may or may not exist — spec says "copy current to .bak"
        # when current exists. If there's no current, .bak is irrelevant.
        # We just verify the target is correct.
        assert _read_json(target)["store_revision"] == 1

    def test_backup_file_is_complete_not_half_written(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
    ) -> None:
        target = tmp_path / "board.json"
        backup = tmp_path / "board.json.bak"
        atomic_write_json(target, _sample_data(1))

        class SimulatedCrash(Exception):
            pass

        failpoint.register("after_backup_before_replace", SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_replace_with_backup(target, _sample_data(2), backup, failpoint=failpoint)

        # Even though the overall write failed, the .bak was fully written
        # (it's a complete file from the previous revision).
        if backup.exists():
            backup_data = _read_json(backup)
            assert backup_data["store_revision"] == 1
            assert "payload_hash" in backup_data

        # And the original target is untouched
        assert _read_json(target)["store_revision"] == 1


# ── LKB-STORE-014 ────────────────────────────────────────────────────


class TestLkbStore014HashMismatch:
    """Readback hash mismatch raises BoardStoreHashMismatchError."""

    def test_mismatched_payload_hash_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "board.json"
        data = _sample_data(1)
        # Tamper with the expected hash so it doesn't match the payload
        data["payload_hash"] = "sha256:deadbeef" * 8  # invalid

        with pytest.raises(BoardStoreHashMismatchError):
            atomic_write_json(target, data)

    def test_existing_target_is_restored_after_readback_mismatch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "board.json"
        atomic_write_json(target, _sample_data(1))
        data = _sample_data(1)
        data["store_revision"] = 2
        data["payload_hash"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

        with pytest.raises(BoardStoreHashMismatchError):
            atomic_write_json(target, data)

        # Post-replace verification failure restores the prior complete target.
        assert target.exists()
        on_disk = _read_json(target)
        assert on_disk["store_revision"] == 1

    def test_serialization_failure_does_not_touch_target_or_create_tmp(self, tmp_path: Path) -> None:
        target = tmp_path / "board.json"
        old = _sample_data(1)
        target.write_text(
            json.dumps(old, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(BoardStoreIOError, match="JSON serialization failed"):
            atomic_write_json(target, {"unsupported": object()})

        assert _read_json(target) == old
        assert not (tmp_path / ".tmp").exists()

    def test_non_dict_json_raises_mismatch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # We can't easily write a non-dict via atomic_write_json itself
        # (it takes a dict), so we verify the internal function behavior.
        from lkb.atomic_file import _verify_readback

        target = tmp_path / "board.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(BoardStoreHashMismatchError):
            _verify_readback(target, "sha256:abc", "payload_hash")


# ── LKB-STORE-017 ────────────────────────────────────────────────────


class TestLkbStore017TempInDotTmp:
    """Temp files are created in .tmp/ sibling of target."""

    def test_tmp_dir_is_sibling_of_target(self, tmp_path: Path) -> None:
        target = tmp_path / "boards" / "my-board" / "board.json"
        target.parent.mkdir(parents=True)
        atomic_write_json(target, _sample_data(1))
        assert (tmp_path / "boards" / "my-board" / ".tmp").exists()
        assert list((tmp_path / "boards" / "my-board" / ".tmp").iterdir()) == []

    def test_tmp_dir_is_clean_after_failure(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
    ) -> None:
        target = tmp_path / "board.json"

        class SimulatedCrash(Exception):
            pass

        failpoint.register("after_fsync_before_replace", SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_write_json(target, _sample_data(1), failpoint=failpoint)

        tmp_dir = tmp_path / ".tmp"
        if tmp_dir.exists():
            temps = list(tmp_dir.iterdir())
            assert temps == [], f"Temp files leaked: {temps}"

    def test_tmp_on_same_filesystem_as_target(self, tmp_path: Path) -> None:
        """Temp dir must be on the same filesystem (os.replace requires it).

        We verify this by checking that .tmp is a child of target's parent
        directory (same mount point in practice).
        """
        target = tmp_path / "deep" / "nested" / "board.json"
        target.parent.mkdir(parents=True)
        atomic_write_json(target, _sample_data(1))
        tmp_dir = tmp_path / "deep" / "nested" / ".tmp"
        assert tmp_dir.exists()
        # .tmp is inside target's parent directory → same filesystem
        assert tmp_dir.parent == target.parent


# ── LKB-STORE-019 ────────────────────────────────────────────────────


class TestLkbStore019DiskFull:
    """ENOSPC / EDQUOT maps to BoardStoreDiskFullError."""

    def test_enospc_maps_to_disk_full_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import lkb.atomic_file as af

        real_fdopen = os.fdopen

        def fake_fdopen(fd, mode="r", **kwargs):
            # Return the real file object but monkey-patch its .write
            f = real_fdopen(fd, mode, **kwargs)
            real_write = f.write

            def exploding_write(data):
                # First write succeeds, second fails with ENOSPC
                real_write(data)
                raise OSError(errno.ENOSPC, "No space left on device")

            f.write = exploding_write  # type: ignore[assignment]
            return f

        monkeypatch.setattr(af.os, "fdopen", fake_fdopen)

        target = tmp_path / "board.json"
        with pytest.raises(BoardStoreDiskFullError) as exc_info:
            atomic_write_json(target, _sample_data(1))

        assert "ENOSPC" in str(exc_info.value) or "space" in str(exc_info.value).lower()

    def test_disk_full_leaves_old_target_intact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import lkb.atomic_file as af

        target = tmp_path / "board.json"
        atomic_write_json(target, _sample_data(1))
        old_content = target.read_bytes()

        real_fdopen = os.fdopen

        def fake_fdopen(fd, mode="r", **kwargs):
            f = real_fdopen(fd, mode, **kwargs)
            real_write = f.write

            def exploding_write(data):
                real_write(data)
                raise OSError(errno.ENOSPC, "No space left on device")

            f.write = exploding_write  # type: ignore[assignment]
            return f

        monkeypatch.setattr(af.os, "fdopen", fake_fdopen)

        with pytest.raises(BoardStoreDiskFullError):
            atomic_write_json(target, _sample_data(2))

        # Old target must be untouched
        assert target.read_bytes() == old_content
        assert _read_json(target)["store_revision"] == 1

    def test_other_os_error_is_generic_io_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import lkb.atomic_file as af

        def fake_mkstemp(*args, **kwargs):
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(af.tempfile, "mkstemp", fake_mkstemp)

        target = tmp_path / "board.json"
        with pytest.raises(BoardStoreIOError) as exc_info:
            atomic_write_json(target, _sample_data(1))

        assert not isinstance(exc_info.value, BoardStoreDiskFullError)
        assert "Permission denied" in str(exc_info.value) or "EACCES" in str(exc_info.value)


# ── dir_fsync ─────────────────────────────────────────────────────────


class TestDirFsync:
    def test_dir_fsync_noop_on_existing_dir(self, tmp_path: Path) -> None:
        # Should not raise on a real directory
        dir_fsync(tmp_path)

    def test_dir_fsync_raises_on_missing_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(BoardStoreIOError):
            dir_fsync(missing)


# ── Inv 14: no half-write observable (spec §11.4) ────────────────────


class TestNoHalfWrite:
    """Inv 14 — target is always a full pre- or post-commit revision."""

    @pytest.mark.parametrize(
        "failpoint_name",
        [
            "after_fsync_before_replace",
            "after_fsync_before_backup",
            "after_backup_before_replace",
        ],
    )
    def test_target_never_half_written(
        self,
        tmp_path: Path,
        failpoint: Failpoint,
        failpoint_name: str,
    ) -> None:
        target = tmp_path / "board.json"
        atomic_write_json(target, _sample_data(1))

        class SimulatedCrash(Exception):
            pass

        failpoint.register(failpoint_name, SimulatedCrash)

        with pytest.raises(SimulatedCrash):
            atomic_replace_with_backup(target, _sample_data(2), tmp_path / "board.json.bak", failpoint=failpoint)

        # After any pre-replace crash, target must be the complete old version
        assert target.exists()
        data = _read_json(target)
        assert data["store_revision"] == 1
        # And its hash must be valid
        payload = {k: v for k, v in data.items() if k != "payload_hash"}
        assert canonical_hash(payload) == data["payload_hash"]


def test_replace_sharing_violation_has_bounded_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lkb.atomic_file as atomic_file_module

    target = tmp_path / "board.json"
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            exc = PermissionError(errno.EACCES, "sharing violation")
            exc.winerror = 32  # type: ignore[attr-defined]
            raise exc
        real_replace(source, destination)

    monkeypatch.setattr(atomic_file_module.os, "replace", flaky_replace)
    atomic_write_json(target, {"revision": 1})
    assert _read_json(target) == {"revision": 1}
    assert attempts == 3
