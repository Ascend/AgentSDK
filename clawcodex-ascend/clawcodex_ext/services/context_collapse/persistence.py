#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Atomic file-based persistence for the context collapse state.

The :class:`ContextCollapseStore` ships with ``to_dict`` / ``from_dict``
helpers but does not handle disk I/O. This module bridges that gap:

* :class:`CollapseStateFile` — atomic ``save`` / ``load`` of a
  collapse state JSON file with corruption detection.
* :func:`load_store` / :func:`save_store` — convenience helpers
  that wrap a path and return / accept a
  :class:`ContextCollapseStore`.

The state file is independent of the in-process global store (which
the existing ``src/services/compact/context_collapse.py`` module
manages); an application can persist the global store on shutdown
and reload it on startup, or it can maintain multiple stores keyed
by session id.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import (
    CollapseStateCorruptError,
    CollapseStateNotFoundError,
)

if TYPE_CHECKING:
    from ...services.compact.context_collapse import ContextCollapseStore


def _load_store_cls() -> type[ContextCollapseStore]:
    """Late import to avoid a hard dependency on the compact package."""
    from ...services.compact.context_collapse import ContextCollapseStore

    return ContextCollapseStore


class CollapseStateFile:
    """Atomic on-disk persistence for a :class:`ContextCollapseStore`."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        with self._lock:
            return self._path.exists()

    def save(self, store: ContextCollapseStore) -> Path:
        cls = _load_store_cls()
        if not isinstance(store, cls):
            raise TypeError("CollapseStateFile.save expects a ContextCollapseStore")
        with self._lock:
            data = store.to_dict()
            self._atomic_write(self._path, data)
            return self._path

    def load(self) -> ContextCollapseStore:
        cls = _load_store_cls()
        with self._lock:
            if not self._path.exists():
                raise CollapseStateNotFoundError(f"collapse state file does not exist: {self._path}")
            try:
                raw = self._path.read_text(encoding="utf-8")
                data: Any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CollapseStateCorruptError(f"collapse state file is not valid JSON: {exc}") from exc
            except OSError as exc:
                raise CollapseStateCorruptError(f"collapse state file could not be read: {exc}") from exc
        try:
            return cls.from_dict(data)
        except (ValueError, TypeError) as exc:
            raise CollapseStateCorruptError(f"collapse state file failed validation: {exc}") from exc

    def delete(self) -> None:
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                return

    def _atomic_write(self, target: Path, payload: dict[str, Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def save_store(path: Path | str, store: ContextCollapseStore) -> Path:
    """Convenience: write ``store`` to ``path`` atomically."""
    return CollapseStateFile(path).save(store)


def load_store(path: Path | str) -> ContextCollapseStore:
    """Convenience: read a store from ``path``."""
    return CollapseStateFile(path).load()


def merge_stores(
    target: ContextCollapseStore,
    source: ContextCollapseStore,
    *,
    prefer_target: bool = True,
) -> int:
    """Append any commits in ``source`` missing from ``target``'s archive.

    Two commits are considered the same when their ``archived`` UUID
    sets are equal. The function returns the number of commits
    appended to ``target``. Useful for merging a disk-loaded store
    into an in-memory one without duplicating already-known
    archives.
    """
    cls = _load_store_cls()
    if not isinstance(target, cls) or not isinstance(source, cls):
        raise TypeError("merge_stores expects two ContextCollapseStore instances")
    existing = {frozenset(c.archived): c for c in target.commits}
    added = 0
    for commit in source.commits:
        key = frozenset(commit.archived)
        if key in existing:
            if not prefer_target:
                # Caller wants source's version to win; overwrite.
                target.commits = [c for c in target.commits if frozenset(c.archived) != key]
                target.commits.append(commit)
            continue
        target.commits.append(commit)
        existing[key] = commit
        added += 1
    return added


def safe_filename_suffix() -> str:
    """Return a short, low-collision suffix used in temp file names."""
    return uuid.uuid4().hex[:8]
