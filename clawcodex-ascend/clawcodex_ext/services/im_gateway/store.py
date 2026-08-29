#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""File-based reliability store.

Persists the gateway's durable state under ``state_dir`` using the
project's existing file-based conventions (small state → JSON, append
logs → NDJSON). Each file has a single-writer lock; JSON state files
are written via tmp + atomic ``os.replace``; NDJSON logs are
append-only. v1 keeps this single-process/single-account; the backend
interface is reserved so SQLite/Postgres can slot in later (P6+).

P1 ships functional basics (dedupe, outbox, dead-letter, context
tokens, audit). P4 adds compaction/rotation, the retry loop,
storm aggregation, and cross-process audit/redaction hardening.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import ReliabilityConfig

logger = logging.getLogger(__name__)

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    if os.name != "nt":
        path.chmod(_PRIVATE_DIRECTORY_MODE)


def _secure_existing_file(path: Path) -> None:
    if path.exists() and os.name != "nt":
        path.chmod(_PRIVATE_FILE_MODE)


def _open_private_append(path: Path):
    _ensure_private_directory(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _PRIVATE_FILE_MODE)
    if os.name != "nt":
        os.fchmod(fd, _PRIVATE_FILE_MODE)
    return os.fdopen(fd, "a", encoding="utf-8", newline="\n")


def _private_temp_file(path: Path) -> tuple[int, Path]:
    _ensure_private_directory(path.parent)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    if os.name != "nt":
        os.fchmod(fd, _PRIVATE_FILE_MODE)
    return fd, Path(raw_path)


def _atomic_write_json(path: Path, data: Any) -> None:
    fd, tmp = _private_temp_file(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(tmp, path)
        _secure_existing_file(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass  # The resource is already absent; cleanup is complete.


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    _secure_existing_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        logger.warning("im_gateway store: invalid JSON file=%s; using default", path.name)
        return dict(default)
    if not isinstance(value, dict):
        logger.warning(
            "im_gateway store: expected JSON object file=%s actual=%s; using default",
            path.name,
            type(value).__name__,
        )
        return dict(default)
    return value


def _read_string_map(path: Path) -> dict[str, str]:
    value = _read_json(path, {})
    result = {key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, str)}
    invalid_count = len(value) - len(result)
    if invalid_count:
        logger.warning(
            "im_gateway store: skipped invalid string-map entries file=%s count=%d",
            path.name,
            invalid_count,
        )
    return result


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read_wechat_accounts(path: Path) -> dict[str, dict[str, Any]]:
    value = _read_json(path, {})
    result: dict[str, dict[str, Any]] = {}
    for account_id, entry in value.items():
        if not isinstance(account_id, str) or not isinstance(entry, dict):
            continue
        cursor = entry.get("get_updates_buf")
        updated_at = entry.get("updated_at")
        if cursor is not None and not isinstance(cursor, str):
            continue
        if updated_at is not None and not _is_finite_number(updated_at):
            continue
        result[account_id] = entry
    invalid_count = len(value) - len(result)
    if invalid_count:
        logger.warning(
            "im_gateway store: skipped invalid account entries file=%s count=%d",
            path.name,
            invalid_count,
        )
    return result


def _append_ndjson(path: Path, entry: dict[str, Any]) -> None:
    with _open_private_append(path) as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_ndjson_state(
    path: Path,
    *,
    validator: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    if not path.exists():
        return [], 0
    try:
        _secure_existing_file(path)
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        logger.warning("im_gateway store: unreadable NDJSON file=%s; using empty state", path.name)
        return [], None
    raw_entry_count = sum(1 for line in lines if line.strip())
    out: list[dict[str, Any]] = []
    invalid_json = 0
    non_object = 0
    invalid_schema = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            invalid_json += 1
            continue
        if not isinstance(entry, dict):
            non_object += 1
            continue
        if validator is not None and not validator(entry):
            invalid_schema += 1
            continue
        out.append(entry)
    if invalid_json or non_object or invalid_schema:
        logger.warning(
            "im_gateway store: skipped invalid NDJSON lines file=%s json=%d non_object=%d schema=%d",
            path.name,
            invalid_json,
            non_object,
            invalid_schema,
        )
    return out, raw_entry_count


def _read_ndjson(
    path: Path,
    *,
    validator: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    entries, _ = _read_ndjson_state(path, validator=validator)
    return entries


def _is_valid_processed_entry(entry: dict[str, Any]) -> bool:
    key = entry.get("key")
    return isinstance(key, str) and bool(key) and _is_finite_number(entry.get("seen_at"))


def _has_valid_optional_timestamp(entry: dict[str, Any], fields: tuple[str, ...]) -> bool:
    values = [entry[field] for field in fields if field in entry]
    return not values or any(_is_finite_number(value) for value in values)


def _rewrite_ndjson(path: Path, entries: list[dict[str, Any]]) -> None:
    """Rewrite an NDJSON file with selected records."""
    fd, tmp = _private_temp_file(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for e in entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        _secure_existing_file(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass  # The resource is already absent; cleanup is complete.


def _rotate_ndjson(path: Path, max_bytes: int, backup_count: int) -> None:
    """Rotate an NDJSON file when it exceeds its size limit."""
    _ensure_private_directory(path.parent)
    for candidate in [
        path,
        *(path.with_suffix(path.suffix + f".{i}") for i in range(1, backup_count + 1)),
    ]:
        _secure_existing_file(candidate)
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    oldest = path.with_suffix(path.suffix + f".{backup_count}")
    if oldest.exists():
        oldest.unlink()
    for i in range(backup_count - 1, 0, -1):
        src = path.with_suffix(path.suffix + f".{i}")
        dst = path.with_suffix(path.suffix + f".{i + 1}")
        if src.exists():
            src.rename(dst)
    path.rename(path.with_suffix(path.suffix + ".1"))


class ReliabilityStore:
    def __init__(
        self,
        state_dir: str | Path,
        reliability: ReliabilityConfig | None = None,
    ) -> None:
        self._dir = Path(state_dir).expanduser()
        _ensure_private_directory(self._dir)
        self._reliability = reliability or ReliabilityConfig()
        self._lock = threading.RLock()
        # in-memory dedupe index: key -> first_seen_ts
        self._dedupe: dict[str, float] = {}
        self._load_dedupe()

    # -- paths -----------------------------------------------------------
    @property
    def state_dir(self) -> Path:
        return self._dir

    def _p(self, name: str) -> Path:
        return self._dir / name

    # -- inbound dedupe --------------------------------------------------
    def _load_dedupe(self) -> None:
        cutoff = time.time() - self._reliability.inbound_dedupe_ttl_seconds
        for entry in _read_ndjson(self._p("processed_inbound.ndjson"), validator=_is_valid_processed_entry):
            key = str(entry["key"])
            ts = float(entry["seen_at"])
            if ts >= cutoff:
                self._dedupe[key] = ts

    def is_duplicate(self, key: str) -> bool:
        with self._lock:
            self._purge_expired()
            return key in self._dedupe

    def record_processed(self, key: str, *, message_id: str | None = None) -> None:
        with self._lock:
            ts = time.time()
            self._dedupe[key] = ts
            _append_ndjson(
                self._p("processed_inbound.ndjson"),
                {"key": key, "message_id": message_id, "seen_at": ts},
            )

    def check_and_record(self, key: str, *, message_id: str | None = None) -> bool:
        """Return True if newly seen (and recorded); False if duplicate."""
        with self._lock:
            self._purge_expired()
            if key in self._dedupe:
                logger.debug("im_gateway dedupe hit: key=%s", key[:32])
                return False
            self.record_processed(key, message_id=message_id)
            return True

    def _purge_expired(self) -> None:
        cutoff = time.time() - self._reliability.inbound_dedupe_ttl_seconds
        self._dedupe = {k: v for k, v in self._dedupe.items() if v >= cutoff}

    # -- outbox ----------------------------------------------------------
    def append_outbox(self, entry: dict[str, Any]) -> None:
        with self._lock:
            _append_ndjson(self._p("outbox.ndjson"), entry)

    def outbox_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_ndjson(self._p("outbox.ndjson"))

    def outbox_pending(self) -> list[dict[str, Any]]:
        """Latest status per idempotency_key where status != delivered."""
        with self._lock:
            latest: dict[str, dict[str, Any]] = {}
            for e in _read_ndjson(self._p("outbox.ndjson")):
                key = e.get("idempotency_key")
                if not key:
                    continue
                latest[key] = e
            terminal = {"delivered", "dead", "failed"}
            return [e for e in latest.values() if e.get("status") not in terminal]

    # -- dead letter -----------------------------------------------------
    def append_dead_letter(self, entry: dict[str, Any]) -> None:
        with self._lock:
            path = self._p("dead_letter.ndjson")
            _rotate_ndjson(
                path,
                self._reliability.dead_letter_max_bytes,
                self._reliability.dead_letter_backup_count,
            )
            _append_ndjson(path, entry)
        logger.warning(
            "im_gateway dead-letter appended: channel=%s idem=%s category=%s",
            entry.get("channel"),
            str(entry.get("idempotency_key"))[:16],
            entry.get("error_category"),
        )

    def dead_letter_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_ndjson(self._p("dead_letter.ndjson"))

    # -- context tokens --------------------------------------------------
    def get_context_token(self, account_id: str, user_id: str) -> str | None:
        data = _read_string_map(self._p("wechat_context_tokens.json"))
        return data.get(f"{account_id}:{user_id}")

    def set_context_token(self, account_id: str, user_id: str, token: str | None) -> None:
        with self._lock:
            data = _read_string_map(self._p("wechat_context_tokens.json"))
            key = f"{account_id}:{user_id}"
            if token is None:
                data.pop(key, None)
            else:
                data[key] = token
            _atomic_write_json(self._p("wechat_context_tokens.json"), data)

    def wechat_context_users(self, account_id: str) -> list[str]:
        """Return user_ids that have a persisted context token for ``account_id``.

        Used to resolve a wildcard WeChat OUTBOUND origin right after a
        gateway restart, before any new inbound arrives: the context-token
        store already survives restarts (it backs the ``context_reply``
        capability), so it doubles as the durable record of known senders
        without a separate persistence file.
        """
        data = _read_string_map(self._p("wechat_context_tokens.json"))
        prefix = f"{account_id}:"
        return [k[len(prefix) :] for k in data if isinstance(k, str) and k.startswith(prefix)]

    def get_feishu_last_sender(self, channel_id: str) -> str | None:
        data = _read_string_map(self._p("feishu_last_senders.json"))
        sender = data.get(channel_id)
        return str(sender) if sender else None

    def set_feishu_last_sender(self, channel_id: str, sender: str | None) -> None:
        with self._lock:
            data = _read_string_map(self._p("feishu_last_senders.json"))
            if sender:
                data[channel_id] = sender
            else:
                data.pop(channel_id, None)
            _atomic_write_json(self._p("feishu_last_senders.json"), data)

    def get_wechat_cursor(self, account_id: str) -> str:
        """Return the saved iLink ``get_updates_buf`` cursor, or ``""``.

        iLink expects the cursor as a string on every ``getupdates`` POST;
        sending JSON ``null`` (the previous default) can cause the server to
        never deliver messages even though the session is valid. The two
        reference clients (hermes-agent, AstrBot) both default to ``""``.
        """
        data = _read_wechat_accounts(self._p("wechat_accounts.json"))
        entry = data.get(account_id)
        if not isinstance(entry, dict):
            return ""
        cursor = entry.get("get_updates_buf")
        return cursor if isinstance(cursor, str) else ""

    def set_wechat_cursor(self, account_id: str, get_updates_buf: str | None) -> None:
        with self._lock:
            data = _read_wechat_accounts(self._p("wechat_accounts.json"))
            entry = data.get(account_id)
            if not isinstance(entry, dict):
                entry = {}
            entry["get_updates_buf"] = get_updates_buf
            entry["updated_at"] = time.time()
            data[account_id] = entry
            _atomic_write_json(self._p("wechat_accounts.json"), data)

    # -- unsupported inbound (P2) ----------------------------------------
    def record_unsupported_media(self, entry: dict[str, Any]) -> None:
        with self._lock:
            _append_ndjson(self._p("unsupported_inbound.ndjson"), entry)

    # -- audit -----------------------------------------------------------
    def audit(self, event_type: str, **fields: Any) -> None:
        from .audit import redact

        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            **redact(fields),
        }
        with self._lock:
            path = self._p("audit.ndjson")
            _rotate_ndjson(
                path,
                self._reliability.audit_max_bytes,
                self._reliability.audit_backup_count,
            )
            _append_ndjson(path, entry)

    def audit_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_ndjson(self._p("audit.ndjson"))

    # -- cron retention --------------------------------------------------
    def purge_processed_inbound(self, ttl_seconds: int, max_entries: int) -> int:
        """Delete processed inbound records older than the retention limit."""
        return self._purge_ndjson_ttl_cap(
            "processed_inbound.ndjson",
            ttl_seconds,
            max_entries,
            ts_fields=("seen_at",),
            entry_validator=_is_valid_processed_entry,
        )

    def purge_outbox(self, ttl_seconds: int, max_entries: int) -> int:
        return self._purge_ndjson_ttl_cap(
            "outbox.ndjson",
            ttl_seconds,
            max_entries,
            ts_fields=("at", "timestamp"),
            entry_validator=lambda entry: _has_valid_optional_timestamp(entry, ("at", "timestamp")),
        )

    def purge_unsupported_inbound(self, ttl_seconds: int, max_entries: int) -> int:
        return self._purge_ndjson_ttl_cap(
            "unsupported_inbound.ndjson",
            ttl_seconds,
            max_entries,
            ts_fields=("received_at", "at", "timestamp"),
            entry_validator=lambda entry: _has_valid_optional_timestamp(entry, ("received_at", "at", "timestamp")),
        )

    def purge_all(self, reliability: ReliabilityConfig) -> dict[str, int]:
        """Return cron cleanup counts for bounded append-style files only."""
        return {
            "processed_inbound.ndjson": self.purge_processed_inbound(
                reliability.retention_processed_inbound_ttl_seconds,
                reliability.retention_processed_inbound_max_entries,
            ),
            "outbox.ndjson": self.purge_outbox(
                reliability.retention_outbox_ttl_seconds,
                reliability.retention_outbox_max_entries,
            ),
            "unsupported_inbound.ndjson": self.purge_unsupported_inbound(
                reliability.retention_unsupported_inbound_ttl_seconds,
                reliability.retention_unsupported_inbound_max_entries,
            ),
        }

    # -- internal helpers ------------------------------------------------
    def _purge_ndjson_ttl_cap(
        self,
        name: str,
        ttl_seconds: int,
        max_entries: int,
        ts_fields: tuple[str, ...],
        entry_validator: Callable[[dict[str, Any]], bool] | None = None,
    ) -> int:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            path = self._p(name)
            entries, raw_entry_count = _read_ndjson_state(path, validator=entry_validator)
            if raw_entry_count is None:
                return 0
            if not entries:
                if raw_entry_count:
                    _rewrite_ndjson(path, [])
                return 0
            indexed = list(enumerate(entries))
            survivors = [
                (index, entry)
                for index, entry in indexed
                if (ts := self._entry_timestamp(entry, ts_fields)) is None or ts >= cutoff
            ]
            if len(survivors) > max_entries:
                survivors = sorted(
                    survivors,
                    key=lambda item: self._entry_timestamp(item[1], ts_fields) or item[0],
                    reverse=True,
                )[:max_entries]
                survivors.sort(key=lambda item: item[0])
            kept = [entry for _, entry in survivors]
            removed = len(entries) - len(kept)
            if removed or raw_entry_count != len(entries):
                _rewrite_ndjson(path, kept)
            return removed

    @staticmethod
    def _entry_timestamp(entry: dict[str, Any], fields: tuple[str, ...]) -> float | None:
        for field in fields:
            value = entry.get(field)
            if _is_finite_number(value):
                return float(value)
        return None


__all__ = ["ReliabilityStore"]
