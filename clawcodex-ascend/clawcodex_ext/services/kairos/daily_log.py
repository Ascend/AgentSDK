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

"""Daily log writer: append Markdown entries to ``logs/YYYY/MM/YYYY-MM-DD.md``.

The KAIROS daily log is a single Markdown file per calendar day. Entries
are appended in order, separated by blank lines. The file is created on
first append for the day. Concurrent appends from multiple threads are
serialized through a per-path ``RLock`` so two threads writing to the
same day's file do not interleave bytes.

Path resolution:

* The writer accepts any :class:`pathlib.Path` supplied by the caller.
* Callers that want the canonical auto-memory daily log path should
  use :func:`src.memdir.paths.get_auto_mem_daily_log_path`, which
  produces the ``logs/YYYY/MM/YYYY-MM-DD.md`` shape relative to the
  auto-memory directory. This module deliberately does not duplicate
  that helper.

This module does **not** depend on the rest of the agent runtime —
the path is supplied by the caller.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .exceptions import DailyLogError
from .models import DailyLogEntry


class DailyLogWriter:
    """Append-only writer for a single daily log file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        # Ensure the parent directory exists. Fail loudly if the path is
        # unsafe (e.g. caller hands us something with no parent).
        parent = self._path.parent
        if str(parent) and not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise DailyLogError(f"cannot create daily log directory {parent}: {exc}") from exc

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        with self._lock:
            return self._path.exists()

    def append(self, entry: DailyLogEntry) -> int:
        """Append an entry to the file. Returns the number of bytes written."""
        if not isinstance(entry, DailyLogEntry):
            raise TypeError("append() expects a DailyLogEntry")
        rendered = entry.render()
        data = rendered.encode("utf-8")
        with self._lock:
            try:
                with open(self._path, "ab") as fh:
                    fh.write(data)
            except OSError as exc:
                raise DailyLogError(f"failed to append to {self._path}: {exc}") from exc
        return len(data)

    def read(self) -> str:
        """Read the full file contents. Empty string if the file does not exist."""
        with self._lock:
            if not self._path.exists():
                return ""
            try:
                return self._path.read_text(encoding="utf-8")
            except OSError as exc:
                raise DailyLogError(f"failed to read {self._path}: {exc}") from exc

    def delete(self) -> None:
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                return


__all__ = [
    "DailyLogWriter",
]
