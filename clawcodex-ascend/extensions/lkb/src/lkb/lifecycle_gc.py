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

"""Conservative garbage collection for managed LKB storage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_store import validate_board_envelope
from .file_lock import BoardStoreBusyError
from .lifecycle_core import (
    GC_QUARANTINE_AGE_SECONDS,
    GC_SESSION_ORPHAN_AGE_SECONDS,
    GC_TEMP_AGE_SECONDS,
)
from .lifecycle_paths import _is_reparse, _lexical_absolute, _safe_chain


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GcCandidate:
    path: Path
    kind: str
    age_seconds: float
    reason: str
    size_bytes: int = 0
    action: str = "report"
    root: Path | None = None
    board_dir: Path | None = None
    observed_path: Path | None = None
    observed_mtime_ns: int | None = None
    observed_size: int | None = None
    observed_inode: int | None = None
    observed_hash: str = ""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return f"sha256:{digest.hexdigest()}"


def _observe(path: Path) -> tuple[int, int, int, str] | None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return None
    value_hash = _file_hash(path) if stat.S_ISREG(info.st_mode) else ""
    return info.st_mtime_ns, info.st_size, info.st_ino, value_hash


def _candidate(
    path: Path,
    kind: str,
    age: float,
    reason: str,
    *,
    action: str,
    root: Path,
    board_dir: Path,
    observed_path: Path | None = None,
) -> GcCandidate:
    subject = observed_path or path
    observed = _observe(subject)
    if observed is None:
        return GcCandidate(path, "unsafe_path", 0, "stat failed; retained", root=root)
    mtime, size, inode, value_hash = observed
    return GcCandidate(
        path=path,
        kind=kind,
        age_seconds=age,
        reason=reason,
        size_bytes=size,
        action=action,
        root=root,
        board_dir=board_dir,
        observed_path=subject,
        observed_mtime_ns=mtime,
        observed_size=size,
        observed_inode=inode,
        observed_hash=value_hash,
    )


def _age(path: Path, now: float) -> float | None:
    observed = _observe(path)
    if observed is None:
        return None
    age = now - observed[0] / 1_000_000_000
    return age if age >= 0 else None


def _read_board_header(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        board = data.get("board")
        lifecycle = data.get("lifecycle")
        if not isinstance(board, dict) or not isinstance(lifecycle, dict):
            return None
        board_id = board.get("board_id")
        if not isinstance(board_id, str) or not board_id:
            return None
        from .board_resolver import safe_board_id

        if path.parent.name != safe_board_id(board_id):
            return None
        validate_board_envelope(data, board_id=board_id, verify_hash=True)
        project_uri = str(board.get("project_uri", ""))
        scope = lifecycle.get("scope")
        if (scope == "session") != project_uri.startswith("session:"):
            return None
    except (OSError, json.JSONDecodeError, ValueError, AssertionError):
        return None
    return data


def _claims_active(data: dict[str, Any]) -> bool:
    claims = data.get("claims", {})
    return isinstance(claims, dict) and any(
        isinstance(value, dict) and value.get("status") == "active" for value in claims.values()
    )


def _managed_temp(path: Path) -> bool:
    return path.is_file() and path.name.startswith(".") and path.name.endswith(".tmp")


def _collect_board_candidates(
    root: Path,
    board_dir: Path,
    candidates: list[GcCandidate],
    *,
    now: float,
    open_board_ids: Collection[str],
) -> None:
    if not _safe_chain(root, board_dir):
        candidates.append(
            GcCandidate(
                board_dir,
                "unsafe_path",
                0,
                "symlink/junction/reparse or escape refused",
                root=root,
                board_dir=board_dir,
            )
        )
        return
    board_json = board_dir / "board.json"
    data = _read_board_header(board_json) if board_json.is_file() else None
    if board_json.is_file() and data is None:
        candidates.append(
            GcCandidate(
                board_json,
                "invalid_board",
                0,
                "schema, identity, lifecycle scope, or payload hash invalid; retained",
                action="report",
                root=root,
                board_dir=board_dir,
            )
        )
        return
    lifecycle = (data or {}).get("lifecycle", {})
    state = str(lifecycle.get("state", "unknown")) if isinstance(lifecycle, dict) else "unknown"
    board_id = str((data or {}).get("board", {}).get("board_id", ""))
    busy = (
        state in {"archiving", "purging"} or (data is not None and _claims_active(data)) or board_id in open_board_ids
    )

    tmp_dir = board_dir / ".tmp"
    if tmp_dir.is_dir() and _safe_chain(root, tmp_dir) and not busy:
        for item in tmp_dir.iterdir():
            if not _safe_chain(root, item):
                candidates.append(
                    GcCandidate(
                        item,
                        "unsafe_path",
                        0,
                        "unsafe temp entry refused",
                        root=root,
                        board_dir=board_dir,
                    )
                )
                continue
            age = _age(item, now)
            if age is not None and age >= GC_TEMP_AGE_SECONDS:
                kind = "temp" if _managed_temp(item) else "temp_suspicious"
                candidates.append(
                    _candidate(
                        item,
                        kind,
                        age,
                        "expired atomic temp" if kind == "temp" else "unrecognized old temp",
                        action="delete" if kind == "temp" else "quarantine",
                        root=root,
                        board_dir=board_dir,
                    )
                )

    quarantine = board_dir / "quarantine"
    if quarantine.is_dir() and _safe_chain(root, quarantine):
        for item in quarantine.iterdir():
            if not _safe_chain(root, item):
                continue
            age = _age(item, now)
            if age is not None and age >= GC_QUARANTINE_AGE_SECONDS:
                candidates.append(
                    _candidate(
                        item,
                        "quarantine",
                        age,
                        "expired quarantine requires explicit confirmation",
                        action="report",
                        root=root,
                        board_dir=board_dir,
                    )
                )

    scope = lifecycle.get("scope") if isinstance(lifecycle, dict) else None
    project_uri = str((data or {}).get("board", {}).get("project_uri", ""))
    if (
        data is not None
        and (scope == "session" or project_uri.startswith("session:"))
        and state == "active"
        and not busy
    ):
        age = _age(board_json, now)
        if age is not None and age >= GC_SESSION_ORPHAN_AGE_SECONDS:
            candidates.append(
                _candidate(
                    board_dir,
                    "session_orphan",
                    age,
                    "expired inactive session board",
                    action="delete",
                    root=root,
                    board_dir=board_dir,
                    observed_path=board_json,
                )
            )


def _observation_matches(candidate: GcCandidate) -> bool:
    subject = candidate.observed_path or candidate.path
    observed = _observe(subject)
    if observed is None:
        return False
    return observed == (
        candidate.observed_mtime_ns,
        candidate.observed_size,
        candidate.observed_inode,
        candidate.observed_hash,
    )


def _execute_gc_candidate(
    candidate: GcCandidate,
    *,
    now: float,
    open_board_ids: Collection[str],
) -> None:
    if candidate.action not in {"delete", "quarantine"}:
        return
    root = candidate.root
    board_dir = candidate.board_dir
    if root is None or board_dir is None:
        return
    from .file_lock import BoardFileLock

    with BoardFileLock(board_dir, timeout=0.25):
        if (
            not _safe_chain(root, board_dir, descendants=True)
            or not _safe_chain(root, candidate.path, descendants=candidate.path.is_dir())
            or not _observation_matches(candidate)
        ):
            return
        subject = candidate.observed_path or candidate.path
        age = _age(subject, now)
        threshold = GC_SESSION_ORPHAN_AGE_SECONDS if candidate.kind == "session_orphan" else GC_TEMP_AGE_SECONDS
        if age is None or age < threshold:
            return
        board_json = board_dir / "board.json"
        data = _read_board_header(board_json) if board_json.is_file() else None
        if board_json.is_file() and data is None:
            return
        lifecycle = (data or {}).get("lifecycle", {})
        state = str(lifecycle.get("state", "unknown")) if isinstance(lifecycle, dict) else "unknown"
        board_id = str((data or {}).get("board", {}).get("board_id", ""))
        if (
            state in {"archiving", "purging"}
            or (data is not None and _claims_active(data))
            or board_id in open_board_ids
        ):
            return
        if candidate.kind == "temp":
            if candidate.path.parent != board_dir / ".tmp" or not _managed_temp(candidate.path):
                return
            candidate.path.unlink(missing_ok=True)
        elif candidate.kind == "temp_suspicious":
            if candidate.path.parent != board_dir / ".tmp":
                return
            quarantine = board_dir / "quarantine"
            quarantine.mkdir(parents=True, exist_ok=True)
            if not _safe_chain(root, quarantine):
                return
            destination = quarantine / candidate.path.name
            # Never replace an earlier quarantine artifact implicitly.
            if destination.exists() or not _safe_chain(root, destination):
                return
            os.replace(candidate.path, destination)
        elif candidate.kind == "session_orphan":
            scope = lifecycle.get("scope") if isinstance(lifecycle, dict) else None
            if data is None or scope != "session" or state != "active":
                return
            for entry in list(board_dir.iterdir()):
                if entry.name in {".lock", ".lock.owner.json"}:
                    continue
                if not _safe_chain(root, entry, descendants=entry.is_dir()):
                    return
            for entry in list(board_dir.iterdir()):
                if entry.name in {".lock", ".lock.owner.json"}:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink(missing_ok=True)
    # Never unlink the permanent lock anchor after releasing its OS lock.


def gc_scan(
    root: Path | str,
    *,
    dry_run: bool = True,
    now: float | None = None,
    open_board_ids: Collection[str] = (),
) -> list[GcCandidate]:
    """Conservative GC with observed fingerprints and in-lock TOCTOU revalidation."""
    lkb_root = _lexical_absolute(Path(root))
    current = time.time() if now is None else now
    if not lkb_root.is_dir() or not _safe_chain(lkb_root, lkb_root):
        return []
    candidates: list[GcCandidate] = []
    boards = lkb_root / "boards"
    if boards.is_dir() and _safe_chain(lkb_root, boards):
        for board_dir in boards.iterdir():
            if board_dir.is_dir() or _is_reparse(board_dir):
                _collect_board_candidates(
                    lkb_root,
                    board_dir,
                    candidates,
                    now=current,
                    open_board_ids=open_board_ids,
                )
    # Project boards, archives, tombstones, and exports are never inferred as
    # automatic deletion candidates.
    candidates.sort(key=lambda item: item.age_seconds, reverse=True)
    if not dry_run:
        gc_apply(candidates, now=current, open_board_ids=open_board_ids)
    return candidates


def gc_apply(
    candidates: Collection[GcCandidate],
    *,
    now: float | None = None,
    open_board_ids: Collection[str] = (),
) -> None:
    """Apply previously observed candidates with full in-lock revalidation."""
    current = time.time() if now is None else now
    for candidate in candidates:
        try:
            _execute_gc_candidate(candidate, now=current, open_board_ids=open_board_ids)
        except BoardStoreBusyError:
            _LOGGER.debug("Skipped busy LKB GC candidate", extra={"candidate_kind": candidate.kind})
