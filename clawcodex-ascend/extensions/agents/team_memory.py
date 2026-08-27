#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""TeamMem — Team shared memory service layer.

Provides a persistent, searchable, auditable, team-isolated long-term
collaboration knowledge base for the Team / Coordinator / Agent modes.

This module is the **new independent subsystem** (Layer 2).
It reuses the existing path-defense primitives from
``clawcodex_ext/memdir/team_mem_paths.py`` (Layer 1) and the
``TeamFile`` roster model from ``clawcodex_ext/services/swarm/team_file.py``
but does not modify any ``src/`` file.

Layout on disk (per team workspace, under the auto-memory dir)::

    <auto_mem>/team/
      ├─ MEMORY.md          # human-readable entrypoint (rebuilt from entries)
      ├─ entries.jsonl      # append-only structured entry log
      ├─ index.json         # rebuildable cache (id → entry summary)
      ├─ audit.jsonl        # who-wrote-what mutation log
      └─ archive/*.jsonl    # TeamDelete / compact archives

All writes are atomic (tmp + ``os.replace``). The JSONL store is
append-only with tombstones for ``delete`` — readers skip tombstoned
ids. Corrupt single lines are skipped with a WARN log, never aborting
the whole store (§1.10 / acceptance #6).

Score model (§1.8)::

    score = lexical * tag_boost * source_weight * recency_decay * confidence

    source_weight = {manual: 1.2, task_result: 1.1, review: 1.1,
                     send_message: 0.9, system: 0.8}

Public surface:
    - :class:`TeamMemoryEntry` / :class:`TeamMemoryQuery` /
      :class:`TeamMemoryResult` / :class:`TeamMemoryConfig`
    - :class:`TeamMemoryDisabledError` / :class:`TeamNotFoundError` /
      :class:`TeamMemoryPermissionError` / :class:`TeamMemoryCorruptError` /
      :class:`TeamMemoryTooLargeError`
    - :class:`TeamMemoryAuditLog`
    - :class:`TeamMemoryStore`
    - :class:`TeamMemoryIndex`
    - :class:`TeamMemoryService`
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from clawcodex_ext.memdir.team_mem_paths import (
    get_team_mem_path,
    is_team_memory_enabled,
)
from clawcodex_ext.services.swarm.team_file import (
    TeamFile,
    read_team_file,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EntrySource",
    "EntryScope",
    "TeamMemoryEntry",
    "TeamMemoryQuery",
    "TeamMemoryResult",
    "TeamMemoryConfig",
    "TeamMemoryDisabledError",
    "TeamNotFoundError",
    "TeamMemoryPermissionError",
    "TeamMemoryCorruptError",
    "TeamMemoryTooLargeError",
    "TeamMemoryAuditLog",
    "TeamMemoryStore",
    "TeamMemoryIndex",
    "TeamMemoryService",
    "SOURCE_WEIGHTS",
    "make_iso_timestamp",
]


EntrySource = Literal["manual", "send_message", "task_result", "review", "system"]
EntryScope = Literal["team", "lead_only", "agent_pair"]

SOURCE_WEIGHTS: dict[str, float] = {
    "manual": 1.2,
    "task_result": 1.1,
    "review": 1.1,
    "send_message": 0.9,
    "system": 0.8,
}

_VALID_SOURCES = frozenset(SOURCE_WEIGHTS.keys())
_VALID_SCOPES = frozenset({"team", "lead_only", "agent_pair"})


def make_iso_timestamp() -> str:
    """UTC ISO 8601 timestamp with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _entry_id(team_id: str, created_at: str, author: str, content: str) -> str:
    """Stable id: ``hash(team_id + created_at + author + content)`` sha1 hex."""
    h = hashlib.sha1(usedforsecurity=False)
    h.update(team_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(created_at.encode("utf-8"))
    h.update(b"\x00")
    h.update(author.encode("utf-8"))
    h.update(b"\x00")
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:24]


class TeamMemoryDisabledError(RuntimeError):
    """Team memory is off (flag or auto-memory disabled)."""


class TeamNotFoundError(RuntimeError):
    """No ``.clawcodex/team.json`` in the workspace."""


class TeamMemoryPermissionError(PermissionError):
    """Requester is not a team member or violated scope rules."""


class TeamMemoryCorruptError(RuntimeError):
    """The JSONL store is corrupt beyond single-line tolerance."""


class TeamMemoryTooLargeError(ValueError):
    """A single entry exceeds ``max_entry_bytes``."""


@dataclass(frozen=True)
class TeamMemoryEntry:
    """One append-only team-memory record."""

    id: str
    team_id: str
    content: str
    summary: str
    author_agent_id: str
    created_at: str
    author_name: str | None = None
    source: EntrySource = "manual"
    scope: EntryScope = "team"
    tags: tuple[str, ...] = ()
    related_agents: tuple[str, ...] = ()
    updated_at: str | None = None
    expires_at: str | None = None
    confidence: float = 1.0
    deleted: bool = False
    deleted_by: str | None = None
    deleted_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "team_id": self.team_id,
            "content": self.content,
            "summary": self.summary,
            "author_agent_id": self.author_agent_id,
            "created_at": self.created_at,
            "source": self.source,
            "scope": self.scope,
            "tags": list(self.tags),
            "related_agents": list(self.related_agents),
            "confidence": self.confidence,
            "deleted": self.deleted,
        }
        if self.author_name is not None:
            d["author_name"] = self.author_name
        if self.updated_at is not None:
            d["updated_at"] = self.updated_at
        if self.expires_at is not None:
            d["expires_at"] = self.expires_at
        if self.deleted_by is not None:
            d["deleted_by"] = self.deleted_by
        if self.deleted_reason is not None:
            d["deleted_reason"] = self.deleted_reason
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TeamMemoryEntry":
        return cls(
            id=str(raw["id"]),
            team_id=str(raw["team_id"]),
            content=str(raw["content"]),
            summary=str(raw.get("summary", "")),
            author_agent_id=str(raw["author_agent_id"]),
            created_at=str(raw["created_at"]),
            author_name=raw.get("author_name"),
            source=raw.get("source", "manual"),  # type: ignore[arg-type]
            scope=raw.get("scope", "team"),  # type: ignore[arg-type]
            tags=tuple(raw.get("tags", ()) or ()),
            related_agents=tuple(raw.get("related_agents", ()) or ()),
            updated_at=raw.get("updated_at"),
            expires_at=raw.get("expires_at"),
            confidence=float(raw.get("confidence", 1.0)),
            deleted=bool(raw.get("deleted", False)),
            deleted_by=raw.get("deleted_by"),
            deleted_reason=raw.get("deleted_reason"),
        )


@dataclass(frozen=True)
class TeamMemoryQuery:
    team_id: str
    query: str
    requester_agent_id: str
    top_k: int = 8
    tags: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    include_expired: bool = False


@dataclass(frozen=True)
class TeamMemoryResult:
    entry: TeamMemoryEntry
    score: float
    matched_terms: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class TeamMemoryConfig:
    enabled: bool = False
    max_entries: int = 2000
    max_entry_bytes: int = 16_384
    prompt_top_k: int = 8
    index_path: Path | None = None
    allow_agent_writes: bool = True
    require_lead_approval_for_lead_only: bool = True


class TeamMemoryAuditLog:
    """Append-only JSONL of mutations: who wrote/deleted what, when."""

    AUDIT_NAME = "audit.jsonl"

    def __init__(self, root: Path, *, lock: threading.RLock | None = None) -> None:
        self._root = root
        self._lock = lock or threading.RLock()

    def _audit_path(self) -> Path:
        return self._root / self.AUDIT_NAME

    def record(self, *, action: str, actor: str, entry_id: str, **detail: Any) -> None:
        """Append one audit line. Never raises on I/O failure — audit is
        best-effort; a missing audit line must not block a successful
        user-visible write (§3 risk: JSONL 写冲突).
        """
        line = {
            "ts": make_iso_timestamp(),
            "action": action,
            "actor": actor,
            "entry_id": entry_id,
            **detail,
        }
        try:
            with self._lock:
                path = self._audit_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("team_memory audit write failed: %s", exc)


class TeamMemoryStore:
    """Append-only JSONL persistence for team-memory entries.

    Layout under ``root`` (must be the validated team-memory dir):

      - ``entries.jsonl`` — append-only entry log (one JSON object per line)
      - ``MEMORY.md`` — human-readable entrypoint (rebuilt on append)
      - ``archive/<ts>.jsonl`` — snapshot taken on ``archive()``

    Single corrupt lines are skipped with a WARN log (acceptance #6).
    Writes are atomic per line: ``open("a")`` is POSIX-atomic for
    line-sized appends under a process-level lock.
    """

    ENTRIES_NAME = "entries.jsonl"
    ENTRYPOINT_NAME = "MEMORY.md"
    ARCHIVE_DIR = "archive"

    def __init__(
        self,
        *,
        team_id: str,
        root: Path,
        config: TeamMemoryConfig,
        audit: TeamMemoryAuditLog | None = None,
        lock: threading.RLock | None = None,
    ) -> None:
        self._team_id = team_id
        self._root = Path(root)
        self._config = config
        self._audit = audit or TeamMemoryAuditLog(self._root, lock=lock)
        self._lock = lock or threading.RLock()

    def _entries_path(self) -> Path:
        return self._root / self.ENTRIES_NAME

    def _entrypoint_path(self) -> Path:
        return self._root / self.ENTRYPOINT_NAME

    def _archive_dir(self) -> Path:
        return self._root / self.ARCHIVE_DIR

    def append(self, entry: TeamMemoryEntry) -> TeamMemoryEntry:
        """Append ``entry`` to the JSONL store, rebuild MEMORY.md, audit."""
        payload_bytes = len(entry.content.encode("utf-8"))
        if payload_bytes > self._config.max_entry_bytes:
            raise TeamMemoryTooLargeError(
                f"entry content {payload_bytes}B exceeds limit "
                f"{self._config.max_entry_bytes}B; compact or summarize first."
            )
        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            path = self._entries_path()
            max_entries = self._config.max_entries
            if max_entries > 0:
                live = self.list_entries(include_expired=False)
                if len(live) >= max_entries:
                    self.compact(actor=entry.author_agent_id)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            self._audit.record(
                action="append",
                actor=entry.author_agent_id,
                entry_id=entry.id,
                source=entry.source,
                scope=entry.scope,
            )
            self._rebuild_entrypoint_unlocked()
        return entry

    def delete(self, entry_id: str, *, actor: str, reason: str) -> bool:
        """Tombstone an entry. Returns ``True`` if a live entry was found."""
        with self._lock:
            existing = self.get(entry_id)
            if existing is None or existing.deleted:
                return False
            tomb = replace(
                existing,
                deleted=True,
                deleted_by=actor,
                deleted_reason=reason,
                updated_at=make_iso_timestamp(),
            )
            line = json.dumps(tomb.to_dict(), ensure_ascii=False) + "\n"
            with self._entries_path().open("a", encoding="utf-8") as fh:
                fh.write(line)
            self._audit.record(
                action="delete",
                actor=actor,
                entry_id=entry_id,
                reason=reason,
            )
            self._rebuild_entrypoint_unlocked()
            return True

    def compact(self, *, actor: str) -> TeamMemoryEntry:
        """Collapse all live entries into a single summary entry.

        The compact summary content is a JSON-encoded list of
        ``{id, summary, tags}`` for every live entry. Archive snapshot
        is written first so no information is lost (acceptance #9).
        """
        with self._lock:
            live = [e for e in self.list_entries(include_expired=False) if not e.deleted]
            summary_payload = json.dumps(
                [{"id": e.id, "summary": e.summary, "tags": list(e.tags)} for e in live],
                ensure_ascii=False,
                indent=2,
            )
            ts = make_iso_timestamp()
            compact_entry = TeamMemoryEntry(
                id=_entry_id(self._team_id, ts, actor, "compact:" + summary_payload[:256]),
                team_id=self._team_id,
                content=summary_payload,
                summary=f"compact of {len(live)} entries",
                author_agent_id=actor,
                created_at=ts,
                source="system",
                scope="team",
                tags=("compact",),
            )
            self.archive(reason=f"compact by {actor} at {ts}")
            for e in live:
                tomb = replace(
                    e,
                    deleted=True,
                    deleted_by=actor,
                    deleted_reason="compacted",
                    updated_at=ts,
                )
                with self._entries_path().open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(tomb.to_dict(), ensure_ascii=False) + "\n")
            with self._entries_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(compact_entry.to_dict(), ensure_ascii=False) + "\n")
            self._audit.record(
                action="compact",
                actor=actor,
                entry_id=compact_entry.id,
                collapsed_count=len(live),
            )
            self._rebuild_entrypoint_unlocked()
            return compact_entry

    def archive(self, *, reason: str) -> Path:
        """Snapshot the current entries.jsonl into ``archive/<ts>.jsonl``."""
        with self._lock:
            ts = make_iso_timestamp().replace(":", "").replace("+", "")
            safe_ts = re.sub(r"[^A-Za-z0-9_.-]", "_", ts)
            archive_dir = self._archive_dir()
            archive_dir.mkdir(parents=True, exist_ok=True)
            src = self._entries_path()
            dst = archive_dir / f"{safe_ts}.jsonl"
            if src.exists():
                data = src.read_bytes()
                dst.write_bytes(data)
            else:
                dst.write_text("", encoding="utf-8")
            self._audit.record(action="archive", actor="system", entry_id="", reason=reason, archive=str(dst))
            return dst

    def list_entries(self, *, include_expired: bool = False) -> list[TeamMemoryEntry]:
        """Return live entries, newest-last. Tombstones override earlier
        live records with the same id. Corrupt lines are skipped (WARN).
        """
        path = self._entries_path()
        if not path.exists():
            return []
        latest: dict[str, TeamMemoryEntry] = {}
        seen_tomb: set[str] = set()
        with self._lock:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "team_memory corrupt line %d in %s: %s (skipped)",
                            lineno,
                            path,
                            exc,
                        )
                        continue
                    if not isinstance(obj, dict) or "id" not in obj:
                        logger.warning("team_memory non-entry line %d in %s (skipped)", lineno, path)
                        continue
                    try:
                        entry = TeamMemoryEntry.from_dict(obj)
                    except (KeyError, TypeError, ValueError) as exc:
                        logger.warning(
                            "team_memory malformed entry line %d in %s: %s (skipped)",
                            lineno,
                            path,
                            exc,
                        )
                        continue
                    if entry.deleted:
                        seen_tomb.add(entry.id)
                        latest[entry.id] = entry
                    else:
                        latest[entry.id] = entry
        out: list[TeamMemoryEntry] = []
        for eid, entry in latest.items():
            if entry.deleted and eid in seen_tomb:
                continue
            if not include_expired and entry.expires_at is not None:
                if _is_expired(entry.expires_at):
                    continue
            out.append(entry)
        out.sort(key=lambda e: e.created_at)
        return out

    def get(self, entry_id: str) -> TeamMemoryEntry | None:
        for entry in self.list_entries(include_expired=True):
            if entry.id == entry_id and not entry.deleted:
                return entry
        return None

    def _rebuild_entrypoint_unlocked(self) -> None:
        """Rewrite ``MEMORY.md`` from current live entries (atomic replace)."""
        entries = [e for e in self.list_entries(include_expired=False) if not e.deleted]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        lines: list[str] = [
            "# Team Memory",
            "",
            f"Team: `{self._team_id}` — {len(entries)} live entries.",
            "",
        ]
        for e in entries[: self._config.prompt_top_k * 4]:
            tag_str = (" [" + ", ".join(e.tags) + "]") if e.tags else ""
            lines.append(f"- **{e.summary}**{tag_str} — `{e.author_agent_id}` ({e.created_at})")
            lines.append(f"  `{e.id}` · source={e.source} scope={e.scope}")
        content = "\n".join(lines) + "\n"
        self._atomic_write_text(self._entrypoint_path(), content)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        """Write ``content`` to ``path`` via tmp + ``os.replace`` (atomic)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tm-", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def _is_expired(expires_at: str) -> bool:
    """True if ``expires_at`` (ISO 8601) is in the past."""
    try:
        exp = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return exp <= datetime.now(timezone.utc)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-/.]+")


def _tokenize(text: str) -> list[str]:
    """Cheap lexical tokenizer — lowercased word tokens.

    §1.8 notes that once Skill Search lands, this can be
    swapped for the shared TF-IDF tokenizer. For now a regex split is
    sufficient for the acceptance criterion (#8: 1000 entries top8
    < 50ms).
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _lexical_score(query_terms: list[str], entry: TeamMemoryEntry) -> tuple[float, tuple[str, ...]]:
    """Return ``(score, matched_terms)`` for lexical overlap.

    Score is the fraction of query terms present in the entry's
    content+summary+tags, weighted by tf (count capped at 3).
    """
    if not query_terms:
        return 0.0, ()
    hay = " ".join(
        [
            entry.content,
            entry.summary,
            " ".join(entry.tags),
            " ".join(entry.related_agents),
        ]
    ).lower()
    matched: list[str] = []
    total = 0.0
    for term in query_terms:
        if term in hay:
            tf = min(hay.count(term), 3)
            total += tf / 3.0
            matched.append(term)
    return total / len(query_terms), tuple(matched)


def _recency_decay(created_at: str, *, now: float | None = None) -> float:
    """Exponential decay: 1.0 at t=0, ~0.5 at 30d, ~0.1 at 100d."""
    try:
        ts = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    n = now if now is not None else time.time()
    age_days = max(0.0, (n - ts.timestamp()) / 86400.0)
    return 0.5 ** (age_days / 30.0)


class TeamMemoryIndex:
    """In-memory retrieval index over a :class:`TeamMemoryStore`.

    Rebuilt from the JSONL on every :meth:`search` call — acceptable
    for the #8 budget (1000 entries, top8, <50ms). A persistent
    ``index.json`` cache (§1.4) is left as a follow-up: the
    JSONL scan is already fast and the cache would add a staleness
    failure mode (§1.10 ``TeamMemoryIndexStaleError``) without
    a measurable win at current scale.
    """

    def __init__(self, store: TeamMemoryStore) -> None:
        self._store = store

    def search(
        self,
        query: TeamMemoryQuery,
        *,
        now: float | None = None,
    ) -> list[TeamMemoryResult]:
        """Return ranked :class:`TeamMemoryResult` for ``query``.

        Honors ``tags``, ``sources``, ``include_expired`` and ``top_k``.
        Filtering by scope/permission is the caller's responsibility
        (the :class:`TeamMemoryService` applies
        :class:`~extensions.agents.team_memory_policy.TeamMemoryPolicy`
        before calling here).
        """
        entries = self._store.list_entries(include_expired=query.include_expired)
        query_terms = _tokenize(query.query)
        if not query_terms:
            return []
        results: list[TeamMemoryResult] = []
        for entry in entries:
            if query.tags and not any(t in entry.tags for t in query.tags):
                continue
            if query.sources and entry.source not in query.sources:
                continue
            lex, matched = _lexical_score(query_terms, entry)
            if lex <= 0.0:
                continue
            tag_boost = 1.0 + (0.1 * len([t for t in query.tags if t in entry.tags])) if query.tags else 1.0
            source_w = SOURCE_WEIGHTS.get(entry.source, 1.0)
            recency = _recency_decay(entry.created_at, now=now)
            score = lex * tag_boost * source_w * recency * float(entry.confidence)
            reason = (
                f"lexical={lex:.2f} tag_boost={tag_boost:.2f} "
                f"source_w={source_w:.2f} recency={recency:.2f} conf={entry.confidence:.2f}"
            )
            results.append(
                TeamMemoryResult(
                    entry=entry,
                    score=score,
                    matched_terms=matched,
                    reason=reason,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: max(0, query.top_k)]


class TeamMemoryService:
    """High-level facade for Team / Agent / Tool callers.

    Wires :class:`TeamMemoryStore` + :class:`TeamMemoryIndex` +
    :class:`~extensions.agents.team_memory_policy.TeamMemoryPolicy`.

    Construction is cheap; callers (tool, command, integration hooks)
    build one per request. The store's process-level RLock serializes
    concurrent appends; the policy is stateless.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        team_file: TeamFile | None = None,
        config: TeamMemoryConfig | None = None,
        store: TeamMemoryStore | None = None,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._config = config or TeamMemoryConfig()
        self._team_file = team_file if team_file is not None else read_team_file(self._workspace_root)
        if self._team_file is None:
            raise TeamNotFoundError(f"No .clawcodex/team.json in {self._workspace_root}; TeamCreate must run first.")
        self._team_id = self._team_file.team_name or self._workspace_root.name
        if store is not None:
            self._store = store
        else:
            team_dir_str = get_team_mem_path(str(self._workspace_root))
            team_dir = Path(team_dir_str)
            self._store = TeamMemoryStore(
                team_id=self._team_id,
                root=team_dir,
                config=self._config,
            )
        self._index = TeamMemoryIndex(self._store)
        from .team_memory_policy import TeamMemoryPolicy

        self._policy = TeamMemoryPolicy(team_file=self._team_file)

    @property
    def team_id(self) -> str:
        return self._team_id

    @property
    def team_file(self) -> TeamFile:
        assert self._team_file is not None
        return self._team_file

    @property
    def config(self) -> TeamMemoryConfig:
        return self._config

    @property
    def store(self) -> TeamMemoryStore:
        return self._store

    @property
    def policy(self):
        return self._policy

    def remember(
        self,
        content: str,
        *,
        author_agent_id: str,
        author_name: str | None = None,
        tags: Iterable[str] = (),
        source: str = "manual",
        scope: str = "team",
        related_agents: Iterable[str] = (),
        summary: str | None = None,
        confidence: float = 1.0,
        expires_at: str | None = None,
    ) -> TeamMemoryEntry:
        if not self._config.enabled or not is_team_memory_enabled():
            raise TeamMemoryDisabledError(
                "Team memory is disabled (set CLAUDE_CODE_TEAM_MEMORY=1 and enable auto-memory)."
            )
        if source not in _VALID_SOURCES:
            raise ValueError(f"invalid source {source!r}; expected one of {sorted(_VALID_SOURCES)}")
        if scope not in _VALID_SCOPES:
            raise ValueError(f"invalid scope {scope!r}; expected one of {sorted(_VALID_SCOPES)}")
        if not self._config.allow_agent_writes and source != "system":
            raise TeamMemoryPermissionError("agent writes disabled by config")
        self._policy.authorize_write(
            author_agent_id=author_agent_id,
            scope=scope,  # type: ignore[arg-type]
            related_agents=tuple(related_agents),
            require_lead_approval=self._config.require_lead_approval_for_lead_only,
        )
        ts = make_iso_timestamp()
        summary_text = summary if summary is not None else _auto_summary(content)
        entry = TeamMemoryEntry(
            id=_entry_id(self._team_id, ts, author_agent_id, content),
            team_id=self._team_id,
            content=content,
            summary=summary_text,
            author_agent_id=author_agent_id,
            author_name=author_name,
            source=source,  # type: ignore[arg-type]
            scope=scope,  # type: ignore[arg-type]
            tags=tuple(tags),
            related_agents=tuple(related_agents),
            created_at=ts,
            confidence=confidence,
            expires_at=expires_at,
        )
        return self._store.append(entry)

    def delete(self, entry_id: str, *, actor: str, reason: str) -> bool:
        if not self._config.enabled or not is_team_memory_enabled():
            raise TeamMemoryDisabledError("Team memory is disabled.")
        entry = self._store.get(entry_id)
        if entry is None:
            return False
        self._policy.authorize_delete(
            actor=actor,
            entry=entry,
        )
        return self._store.delete(entry_id, actor=actor, reason=reason)

    def compact(self, *, actor: str) -> TeamMemoryEntry:
        if not self._config.enabled or not is_team_memory_enabled():
            raise TeamMemoryDisabledError("Team memory is disabled.")
        self._policy.authorize_compact(actor)
        return self._store.compact(actor=actor)

    def archive(self, *, reason: str) -> Path:
        """Snapshot the store. Used by TeamDelete integration (P93-E)."""
        return self._store.archive(reason=reason)

    def recall(self, query: TeamMemoryQuery) -> list[TeamMemoryResult]:
        if not self._config.enabled or not is_team_memory_enabled():
            return []
        if query.team_id != self._team_id:
            raise TeamMemoryPermissionError(
                f"query team_id {query.team_id!r} does not match service team {self._team_id!r}"
            )
        self._policy.authorize_read(requester_agent_id=query.requester_agent_id)
        raw = self._index.search(query)
        out: list[TeamMemoryResult] = []
        for r in raw:
            if self._policy.can_see(requester_agent_id=query.requester_agent_id, entry=r.entry):
                out.append(r)
        return out

    def list_entries(
        self,
        *,
        requester_agent_id: str,
        limit: int = 50,
        tags: Iterable[str] = (),
        sources: Iterable[str] = (),
        include_expired: bool = False,
    ) -> list[TeamMemoryEntry]:
        if not self._config.enabled or not is_team_memory_enabled():
            return []
        self._policy.authorize_read(requester_agent_id=requester_agent_id)
        entries = self._store.list_entries(include_expired=include_expired)
        tag_set = set(tags)
        src_set = set(sources)
        out: list[TeamMemoryEntry] = []
        for e in entries:
            if tag_set and not (tag_set & set(e.tags)):
                continue
            if src_set and e.source not in src_set:
                continue
            if not self._policy.can_see(requester_agent_id=requester_agent_id, entry=e):
                continue
            out.append(e)
        out.sort(key=lambda e: e.created_at, reverse=True)
        return out[: max(0, limit)]

    def build_prompt_section(self, *, requester_agent_id: str, task: str) -> str:
        """Return the ``<team_memory>...</team_memory>`` prompt section.

        Empty string when disabled or no results — callers should drop
        the section entirely on empty (§1.8 / acceptance #1).
        """
        if not self._config.enabled or not is_team_memory_enabled():
            return ""
        results = self.recall(
            TeamMemoryQuery(
                team_id=self._team_id,
                query=task,
                requester_agent_id=requester_agent_id,
                top_k=self._config.prompt_top_k,
            )
        )
        if not results:
            return ""
        lines = [
            "<team_memory>",
            f'You are working in team "{self._team_id}". Relevant shared memories:',
        ]
        for i, r in enumerate(results, 1):
            tag_str = (" [" + ", ".join(r.entry.tags) + "]") if r.entry.tags else ""
            lines.append(f"{i}.{tag_str} {r.entry.summary}")
        lines.append("")
        lines.append(
            "Only rely on team memory when it matches the current repo state. "
            "If memory conflicts with observed files, trust the files."
        )
        lines.append("</team_memory>")
        return "\n".join(lines)

    def record_message_summary(
        self,
        *,
        sender: str,
        recipients: Iterable[str],
        summary: str,
        message: str,
        sender_agent_id: str | None = None,
    ) -> TeamMemoryEntry | None:
        """Sink a SendMessage exchange into team memory.

        Returns ``None`` when disabled (the integration hook treats
        ``None`` as "skip silently" — §1.10 ``TeamMemoryDisabledError``
        is for explicit tool calls, not background hooks).
        """
        if not self._config.enabled or not is_team_memory_enabled():
            return None
        recipient_tuple = tuple(recipients)
        if not recipient_tuple:
            return None
        content = f"[{sender} → {', '.join(recipient_tuple)}] {message}"
        author_id = sender_agent_id or sender
        try:
            return self.remember(
                content,
                author_agent_id=author_id,
                author_name=sender,
                tags=("send_message",),
                source="send_message",
                scope="team",
                related_agents=recipient_tuple,
                summary=summary or _auto_summary(content),
            )
        except (TeamMemoryPermissionError, TeamMemoryTooLargeError, ValueError) as exc:
            logger.info("team_memory message summary rejected: %s", exc)
            return None


def _auto_summary(content: str, *, max_len: int = 120) -> str:
    """Derive a one-line summary from ``content`` (first non-empty line)."""
    for line in content.splitlines():
        s = line.strip()
        if s:
            return s[:max_len]
    return content[:max_len].strip()
