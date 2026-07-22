#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.parse
from typing import Any

logger = logging.getLogger(__name__)

# Valid cache behaviours. See README/design for semantics.
CACHE_MODE_OFF = "off"
CACHE_MODE_READ_WRITE = "read_write"
CACHE_MODE_READ_ONLY = "read_only"
CACHE_MODE_STRICT = "strict"
VALID_CACHE_MODES = (
    CACHE_MODE_OFF,
    CACHE_MODE_READ_WRITE,
    CACHE_MODE_READ_ONLY,
    CACHE_MODE_STRICT,
)

CRAWLER_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url           TEXT PRIMARY KEY,
    root_url      TEXT,
    status        TEXT NOT NULL,
    html          TEXT,
    markdown      TEXT,
    buttons_json  TEXT,
    content_hash  TEXT,
    fetched_at    INTEGER,
    crawler_ver   INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_pages_root ON pages(root_url);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status);
"""


def normalize_cache_url(url: str) -> str:
    """Normalize a URL into a stable cache key.

    Strips the fragment, lowercases scheme/host, sorts query parameters, and
    normalizes trailing slashes. The real URL is still used for fetching and
    button parsing; only the cache key is normalized to maximise hit rate.
    """
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError as exc:
        logger.warning(f"[PageCacheStore] Invalid cache URL {text!r}: {exc}")
        return ""

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if not scheme or not netloc:
        logger.warning(f"[PageCacheStore] Skip non-absolute cache URL: {text!r}")
        return ""
    path = parsed.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    query = urllib.parse.urlencode(
        sorted(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)),
        doseq=True,
    )
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def _content_hash(markdown: str) -> str:
    return hashlib.sha1(str(markdown or "").encode("utf-8", errors="ignore")).hexdigest()


class PageCacheStore:
    """Thread-safe SQLite-backed page cache.

    A single instance is meant to be shared across env instances in a process
    (use :func:`get_page_cache_store`). The connection uses
    ``check_same_thread=False`` and is guarded by a lock so it can be touched
    from the env's worker threads.
    """

    def __init__(self, db_path: str, *, read_only: bool = False) -> None:
        self.db_path = str(db_path)
        self.read_only = bool(read_only)
        self._lock = threading.RLock()
        self._thread_local = threading.local()

        if not self.read_only:
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        if self.read_only and not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"[PageCacheStore] read-only cache file not found: {self.db_path}"
            )

        if self.read_only:
            uri = f"file:{urllib.parse.quote(os.path.abspath(self.db_path))}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout=5000;")
            if not self.read_only:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn.executescript(_SCHEMA)
                self._conn.commit()

    def _read_conn(self) -> sqlite3.Connection:
        """Per-thread read connection avoids serializing SELECTs behind writes."""
        attr = "ro_conn" if self.read_only else "rw_conn"
        conn = getattr(self._thread_local, attr, None)
        if conn is None:
            if self.read_only:
                uri = f"file:{urllib.parse.quote(os.path.abspath(self.db_path))}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000;")
            setattr(self._thread_local, attr, conn)
        return conn

    def _get_record_nolock(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM pages WHERE url = ?",
            (key,),
        ).fetchone()
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------ reads
    def get(self, url: str) -> tuple[str, str] | None:
        """Return ``(html, markdown)`` for a cached *ok* page, else ``None``."""
        key = normalize_cache_url(url)
        if not key:
            return None
        row = self._read_conn().execute(
            "SELECT status, html, markdown FROM pages WHERE url = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["status"] != "ok":
            # A recorded failure/empty page is not a usable hit for fetching.
            return None
        return row["html"] or "", row["markdown"] or ""

    def get_record(self, url: str) -> dict[str, Any] | None:
        key = normalize_cache_url(url)
        if not key:
            return None
        row = self._read_conn().execute(
            "SELECT * FROM pages WHERE url = ?",
            (key,),
        ).fetchone()
        return dict(row) if row is not None else None

    def has(self, url: str) -> bool:
        key = normalize_cache_url(url)
        if not key:
            return False
        row = self._read_conn().execute(
            "SELECT 1 FROM pages WHERE url = ?",
            (key,),
        ).fetchone()
        return row is not None

    def stats(self) -> dict[str, int]:
        rows = self._read_conn().execute(
            "SELECT status, COUNT(*) AS n FROM pages GROUP BY status",
        ).fetchall()
        out = {"total": sum(int(row["n"]) for row in rows)}
        for row in rows:
            out[str(row["status"])] = int(row["n"])
        return out

    # ----------------------------------------------------------------- writes
    @staticmethod
    def _status_rank(status: str) -> int:
        """Higher rank = better cached page. Used to avoid downgrading ok -> failed."""
        return {"ok": 3, "empty": 2, "failed": 1}.get(str(status or ""), 0)

    def put(
        self,
        url: str,
        html: str | None,
        markdown: str | None,
        *,
        status: str = "ok",
        root_url: str = "",
        buttons: list[dict] | None = None,
        crawler_ver: int = CRAWLER_VERSION,
        force: bool = False,
    ) -> bool:
        """Upsert a page record.

        Returns True if the row was inserted/updated, False if skipped.

        Unless ``force=True``, a new ``failed``/``empty`` record will **not**
        overwrite an existing ``ok`` row. That way multi-pass crawling can
        retry flaky URLs without clobbering pages that succeeded on an earlier
        pass (your A-fail/B-ok then A-ok/B-fail scenario).
        """
        if self.read_only:
            return False
        key = normalize_cache_url(url)
        if not key:
            return False
        new_status = str(status or "ok")
        buttons_json = json.dumps(buttons or [], ensure_ascii=False)
        with self._lock:
            if not force:
                existing = self._get_record_nolock(key)
                if existing is not None:
                    old_status = str(existing.get("status") or "")
                    if self._status_rank(new_status) < self._status_rank(old_status):
                        logger.debug(
                            f"[PageCacheStore] Skip downgrade write for {key}: "
                            f"keeping {old_status}, not writing {new_status}"
                        )
                        return False
            self._conn.execute(
                """
                INSERT INTO pages
                    (url, root_url, status, html, markdown, buttons_json,
                     content_hash, fetched_at, crawler_ver)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    root_url=excluded.root_url,
                    status=excluded.status,
                    html=excluded.html,
                    markdown=excluded.markdown,
                    buttons_json=excluded.buttons_json,
                    content_hash=excluded.content_hash,
                    fetched_at=excluded.fetched_at,
                    crawler_ver=excluded.crawler_ver
                """,
                (
                    key,
                    str(root_url or ""),
                    str(status or "ok"),
                    html if html is not None else "",
                    markdown if markdown is not None else "",
                    buttons_json,
                    _content_hash(markdown or ""),
                    int(time.time()),
                    int(crawler_ver),
                ),
            )
            self._conn.commit()
        return True

    def list_urls_by_status(self, statuses: tuple[str, ...] = ("failed", "empty")) -> list[str]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        rows = self._read_conn().execute(
            f"SELECT url FROM pages WHERE status IN ({placeholders})",
            tuple(statuses),
        ).fetchall()
        return [str(row["url"]) for row in rows]

    def checkpoint(self) -> None:
        """Flush the WAL into the main DB file.

        Important before shipping the file to read-only training nodes: a
        ``mode=ro`` connection may not pick up data still sitting in the ``-wal``
        sidecar, so we truncate it into the main file.
        """
        if self.read_only:
            return
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.debug(f"[PageCacheStore] wal_checkpoint failed: {exc}")

    def close(self) -> None:
        with self._lock:
            try:
                if not self.read_only:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    self._conn.commit()
            except Exception:
                pass
            for attr in ("ro_conn", "rw_conn"):
                conn = getattr(self._thread_local, attr, None)
                if conn is None:
                    continue
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    delattr(self._thread_local, attr)
                except AttributeError:
                    pass
            try:
                self._conn.close()
            except Exception:
                pass


# --------------------------------------------------------- process singletons
_STORE_REGISTRY: dict[tuple[str, bool], PageCacheStore] = {}
_REGISTRY_LOCK = threading.Lock()


def get_page_cache_store(db_path: str, *, read_only: bool = False) -> PageCacheStore | None:
    """Return a process-shared :class:`PageCacheStore` for ``db_path``.

    Returns ``None`` (and logs) if the store cannot be opened, so callers can
    gracefully fall back to live fetching instead of crashing the rollout.
    """
    if not db_path:
        return None
    abspath = os.path.abspath(str(db_path))
    registry_key = (abspath, bool(read_only))
    with _REGISTRY_LOCK:
        store = _STORE_REGISTRY.get(registry_key)
        if store is not None:
            return store
        try:
            store = PageCacheStore(abspath, read_only=read_only)
        except (OSError, sqlite3.Error) as exc:
            logger.warning(f"[PageCacheStore] Failed to open cache store ({abspath}): {exc}")
            return None
        _STORE_REGISTRY[registry_key] = store
        logger.info(
            f"[PageCacheStore] Opened cache store path={abspath} read_only={read_only}"
        )
        return store
