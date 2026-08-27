"""Session management with persistence.

The session ID is authoritative-from-bootstrap: ``Session.create`` reads
``get_session_id()`` rather than generating its own. This fixes the
strftime-collision bug (sessions started in the same second would have
overlapped IDs) and unifies session identity across the codebase — the
bootstrap singleton is the single source of truth, exactly per Chapter 3.

``Session.load(sid)`` continues to read from disk by ID; the resume path
should call ``switch_session(SessionId(sid))`` first (or via a wrapping
helper) to update the bootstrap singleton, then call ``Session.load(sid)``
to reconstruct the per-conversation Persistence record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from src.bootstrap.state import (
    get_session_id,
)
from src.utils.clawcodex_dirs import get_sessions_dir

from .conversation import Conversation


def _get_sessions_dir() -> Path:
    override = os.environ.get("CLAWCODEX_SESSIONS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return get_sessions_dir()


@dataclass
class Session:
    """Session manager with persistence."""

    session_id: str
    provider: str
    model: str
    conversation: Conversation = field(default_factory=Conversation)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self):
        """Save session to disk, appending a ``session_snapshot`` line.

        F-49 P5-A: ``session.json`` is no longer written. Instead, a
        ``session_snapshot`` line carrying the cost block is appended
        to ``transcript.jsonl`` (via the ``save_to_session_storage``
        hook). ``Session.load()`` and ``cost_restore`` read this line
        via ``tail -1`` rather than parsing a separate snapshot file.

        The JSONL transcript remains the single source of truth:
        ``save_to_session_storage`` flushes the current conversation
        and writes a ``session_init`` line on the first call; this
        method then appends the trailing ``session_snapshot`` so the
        file ends with the latest cost state.

        Backward compat: if an old ``session.json`` exists from a
        pre-P5 save, it is left untouched on disk. The migration
        script (``clawcodex-dev session migrate --from-3-file``)
        converts it to the new format on demand. ``Session.load()``
        auto-detects legacy ``session.json`` and falls back to reading
        it when ``transcript.jsonl`` lacks a ``session_init`` line.
        """
        session_dir = _get_sessions_dir() / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        cost_block = _snapshot_cost_block()
        self.updated_at = datetime.now().isoformat()

        # Persist the conversation via SessionStorage (JSONL transcript)
        # so ``--resume`` can attach a :class:`TailFollower` to watch
        # for new lines during a backgrounded agent. Implementation
        # lives in clawcodex_ext/agent/session_persist.py so the upstream
        # Session stays free of orchestrator-specific persistence
        # concerns.
        try:
            from clawcodex_ext.agent.session_persist import save_to_session_storage

            save_to_session_storage(self)
        except ImportError:
            pass

        # F-49 P5-A: append a ``session_snapshot`` line carrying the
        # cost block. ``cost_restore`` reads ``tail -1`` from the
        # transcript to recover counters, so the snapshot MUST be the
        # final line of the file at the time of save. We append after
        # ``save_to_session_storage`` (which writes messages) so the
        # snapshot is trailing. Multiple snapshots may coexist across
        # successive saves; ``tail -1`` picks the latest one.
        self._append_session_snapshot(cost_block)

    def _append_session_snapshot(self, cost_block: dict) -> None:
        """Append a ``session_snapshot`` line to ``transcript.jsonl``.

        The line carries the full cost block (matching the shape
        produced by :func:`_snapshot_cost_block`) so cost_restore can
        rebuild bootstrap counters via ``tail -1``.

        Errors are swallowed: a transcript that fails to write the
        snapshot is not fatal — ``cost_restore`` will simply fall back
        to the prior snapshot or return False (no cost recovered).
        """
        try:
            transcript_path = _get_sessions_dir() / self.session_id / "transcript.jsonl"
            payload: dict = {
                "type": "session_snapshot",
                "cost": cost_block,
                "updated_at": datetime.now().isoformat(),
                "provider": self.provider,
                "model": self.model,
            }
            # F-125 C13: serialise snapshot appends across processes
            # via flock so two concurrent ``--resume <sid>`` runs
            # don't interleave their snapshot lines with message
            # flushes from SessionStorage. ``_locked_append`` is a
            # no-op on Windows (no fcntl).
            from clawcodex_ext.services.session_storage import _locked_append

            with _locked_append(transcript_path) as f:
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError:
            # Best-effort: the in-memory session state is still valid
            # even if the snapshot cannot be written to disk.
            pass

    def save_transcript(self):
        """Lightweight per-turn save: JSONL transcript only.

        Skips the full JSON snapshot (``self.save()`` writes that) so
        each call costs O(new messages) rather than O(conversation size).
        The full snapshot with cost block is written once at session exit
        via ``save()``.

        ``--resume`` can reconstruct the conversation from the JSONL
        transcript alone (via ``SessionStorage.read_messages()``), so
        intermediate snapshots are unnecessary for correctness.
        """
        try:
            from clawcodex_ext.agent.session_persist import save_to_session_storage

            save_to_session_storage(self)
        except ImportError:
            pass

    @classmethod
    def load(cls, session_id: str) -> Optional["Session"]:
        """Load session from disk.

        F-49 P5-B: primary source is the **enhanced transcript JSONL**
        introduced by Phase 5. The transcript's structure is::

            line 1      {"type": "session_init", "session_id": ..., "provider": ..., "model": ..., "cwd": ..., "created_at": ...}
            lines 2..N  message entries (one per agent turn)
            last line   {"type": "session_snapshot", "cost": {...}, "updated_at": ...}

        ``Session.load()`` walks the transcript once, pulling provider /
        model / created_at from the init line and building the
        conversation from the message lines. The cost block is left to
        ``cost_restore`` (which reads ``tail -1`` at resume time).

        Backward compat: when ``transcript.jsonl`` does not start with
        a ``session_init`` line, fall back to the legacy ``session.json``
        snapshot. ``session.json`` is left on disk untouched so the
        migration script can convert it on demand. The same fallback
        covers orchestrator/cron sessions that only write metadata +
        plain message JSONL (no session_init marker).
        """
        session_dir = _get_sessions_dir() / session_id
        transcript_path = session_dir / "transcript.jsonl"
        session_file = session_dir / "session.json"

        # F-49 P5-B: prefer the new enhanced-transcript format.
        # ``_load_from_enhanced_transcript`` returns None when the
        # transcript lacks a session_init marker — in that case the
        # file is either legacy (no marker by design) or empty, and
        # we fall back to session.json.
        if transcript_path.exists():
            new_format = _load_from_enhanced_transcript(session_id, transcript_path)
            if new_format is not None:
                return new_format

        # Legacy fallback: pre-P5 ``session.json`` snapshot.
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                return None
            # If the snapshot has real messages, use it. If messages is
            # empty (e.g. orchestrator's _save_json_snapshot couldn't load
            # messages from storage, or the snapshot was written before the
            # first turn flushed the transcript), fall through to the
            # transcript-reading branch below so the REPL can display the
            # actual conversation history from transcript.jsonl.
            if data.get("conversation", {}).get("messages"):
                return cls(
                    session_id=data["session_id"],
                    provider=data["provider"],
                    model=data["model"],
                    conversation=Conversation.from_dict(data["conversation"]),
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                )

        # Final fallback: orchestrator/cron sessions that only wrote
        # metadata + plain message JSONL. Read metadata for model and
        # created_at; scan transcript for messages. Provider is left
        # empty because the file does not record it.
        if not transcript_path.exists():
            return None
        metadata_path = session_dir / "metadata.json"
        provider = ""
        model = ""
        created_at = ""
        updated_at = ""
        if metadata_path.exists():
            try:
                md = json.loads(metadata_path.read_text())
                model = md.get("model", "")
                created_at = str(md.get("start_time", ""))
                updated_at = str(md.get("last_updated", ""))
            except (OSError, json.JSONDecodeError):
                pass

        from src.services.session_storage import SessionStorage

        # Pass the parent of ``session_dir`` (the sessions root) so the
        # storage reads from the directory Session.save() actually wrote
        # to. The dynamic default inside SessionStorage would also pick
        # this up via the patched ``Path.home()``, but threading the
        # explicit path keeps the read site obvious and removes any
        # dependency on import-time ``SESSIONS_DIR`` resolution.
        storage = SessionStorage(
            session_id=session_id,
            sessions_dir=session_dir.parent,
        )
        entries = storage.read_transcript()
        messages = []
        for entry in entries:
            if entry.get("role") == "system" and entry.get("content") == "__background_complete__":
                continue
            try:
                from clawcodex_ext.types.messages import message_from_dict

                messages.append(message_from_dict(entry))
            except Exception:  # nosec
                pass

        return cls(
            session_id=session_id,
            provider=provider,
            model=model,
            conversation=Conversation(messages=messages),
            created_at=created_at,
            updated_at=updated_at,
        )

    @classmethod
    def create(cls, provider: str, model: str) -> "Session":
        """Create a new session using the bootstrap singleton's session ID.

        Previously this generated its own strftime-based ID, producing
        collisions when two sessions started in the same second and
        diverging from the rest of the codebase. Now reads
        ``get_session_id()`` — a UUID-based ID generated at bootstrap
        import time — so every consumer that talks about "the current
        session" agrees on the identifier.
        """
        return cls(
            session_id=get_session_id(),
            provider=provider,
            model=model,
        )

    @classmethod
    def resume(cls, session_id: str) -> Optional["Session"]:
        """Resume a session: update bootstrap identity, restore cost,
        reconstruct the per-conversation record from disk.

        Ch03 round-2 (R2.2): single entry point that keeps the three
        operations in lockstep (CC-34 single-setter discipline at the
        resume layer). Callers (REPL ``/resume``, headless / SDK)
        should use this rather than calling ``Session.load`` plus
        ``switch_session`` plus ``restore_cost_state_for_session``
        independently.

        Order matters: ``switch_session`` fires BEFORE
        ``restore_cost_state_for_session`` so any subscriber that reads
        ``get_session_id()`` during the cost restore sees the loaded id.

        F-49 Phase 0.2: also accepts sessions stored in the
        :class:`SessionStorage` directory format
        (``~/.clawcodex/sessions/<sid>/transcript.jsonl`` + ``metadata.json``).
        This is the on-disk shape the orchestrator's
        :class:`AgentRunner` writes — the headless run is keyed by
        ``run_id`` and persists there so ``clawcodex --resume <run_id>``
        works for orchestrator sessions without a second flat-file
        write. Provider is left as ``""`` because SessionStorage does
        not record it; the resume target's provider should be
        supplied by the caller (REPL config, env, etc.).
        """
        from src.bootstrap.state import SessionId, switch_session
        from clawcodex_ext.services.cost_restore import restore_cost_state_for_session

        loaded = cls.load(session_id)
        if loaded is None:
            # F-49 P5-D: Session.load() now reads transcript.jsonl
            # directly (P5-B), so it returns a fully-populated Session
            # with messages for both new-format transcripts AND legacy
            # session.json / metadata.json + JSONL combinations. The
            # old ``load_from_session_storage`` fallback has been
            # removed — the only way to reach None here is "no session
            # exists on disk at all", which is a real failure.
            return None
        switch_session(SessionId(session_id))
        restore_cost_state_for_session(session_id)
        return loaded


def _snapshot_cost_block() -> dict:
    """Build the cost block written by ``Session.save``.

    The schema owner lives beside the restore path so reader and writer
    cannot drift.
    """
    from src.services.cost_restore import build_cost_block

    return build_cost_block()


def _load_from_enhanced_transcript(
    session_id: str,
    transcript_path: Path,
    *,
    chain_filter: bool = True,
) -> Optional[Session]:
    """F-49 P5-B: load a session from the enhanced transcript JSONL format.

    Reads ``transcript_path`` and reconstructs a :class:`Session` from:

    * Line 1: ``session_init`` — provider / model / created_at /
      cwd. The init marker signals "this transcript is in the new
      format"; absence of it triggers the legacy ``session.json``
      fallback in :meth:`Session.load`.
    * Middle / tail lines: ``message`` entries — the conversation.

    ``cost_block`` (legacy) and ``session_snapshot`` (new) tail lines
    are skipped here; their cost block is consumed by ``cost_restore``
    at resume time so this method stays a pure conversation reader.

    F-103 P103-D: when ``chain_filter`` is True (default), the
    transcript is first run through
    :func:`clawcodex_ext.agent.chain_filter.walk_chain_before_parse`,
    which byte-level prunes any dead-branch messages left over from
    ``/rewind`` / fork. Legacy transcripts (no ``parentUuid``
    fields) skip the filter automatically — see the chain_filter
    docstring for the gating rules. Set ``chain_filter=False`` to
    load the full transcript including dead branches (used by
    Visualizer / Telemetry consumers that need the entire history).

    Returns ``None`` when the transcript is empty OR when its first
    non-blank line is not a ``session_init`` marker — signalling the
    file is legacy format (or not yet initialized) and the caller
    should fall back to ``session.json`` / metadata.

    An empty transcript is treated the same as a non-init first line
    because both cases mean "this transcript doesn't carry the
    new-format anchor" — falling back to ``session.json`` is the only
    way to recover the conversation in that scenario (legacy pre-P5
    sessions often wrote a ``session.json`` alongside an empty
    ``transcript.jsonl``).
    """
    provider = ""
    model = ""
    created_at = ""

    from clawcodex_ext.types.messages import message_from_dict

    # F-103: byte-level chain pruning. ``walk_chain_before_parse``
    # short-circuits on legacy transcripts (no parentUuid tokens)
    # and on small / low-dead-branch-ratio files, in which case
    # ``result.raw_bytes`` equals the input and parsing cost is
    # unchanged. When pruning fires, we save JSON-parse work on
    # every dead-branch line.
    try:
        raw_bytes = transcript_path.read_bytes()
    except OSError:
        return None

    if chain_filter:
        from clawcodex_ext.agent.chain_filter import walk_chain_before_parse

        filter_result = walk_chain_before_parse(raw_bytes)
        parse_source = filter_result.raw_bytes
    else:
        parse_source = raw_bytes

    messages: list = []
    found_init = False
    try:
        for raw_line in parse_source.split(b"\n"):
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines, mirroring
                # ``SessionStorage.read_transcript``.
                continue
            if not isinstance(entry, dict):
                continue

            # ---- First non-blank line: must be session_init ----
            if not found_init:
                if entry.get("type") != "session_init":
                    # Not the new format — caller falls back to
                    # ``session.json`` or metadata-based reconstruction.
                    return None
                found_init = True
                provider = entry.get("provider", "") or ""
                model = entry.get("model", "") or ""
                created_at = entry.get("created_at", "") or ""
                continue

            # ---- Middle / tail lines ----
            entry_type = entry.get("type")
            if entry_type in (
                "cost_block",
                "session_snapshot",
                "session_init",
            ):
                # cost_block: legacy cost entry written by
                # pre-P5-E session_persist (kept for backward
                # compat with already-existing transcripts).
                # session_snapshot: tail line from Session.save();
                # cost_restore owns reading the cost block from
                # this line so this method stays pure.
                # session_init: defensive — a second init line
                # should not happen but we tolerate it.
                continue
            if entry.get("role") == "system" and entry.get("content") == "__background_complete__":
                continue

            try:
                messages.append(message_from_dict(entry))
            except Exception:  # nosec
                # Skip unparseable message entries; the transcript
                # remains valid overall.
                continue
    except Exception:  # nosec
        # F-103: any failure while iterating filtered bytes is
        # treated as "transcript unreadable" so the caller falls
        # back to ``session.json`` rather than silently returning
        # a partially-reconstructed Session.
        return None

    # No session_init seen — the transcript is empty or contains only
    # legacy markers. Treat it as "not in the new format" so the caller
    # falls back to ``session.json`` / metadata. Without this guard an
    # empty transcript would yield an empty Session with no provider/
    # model, which masks the legacy session.json entirely.
    if not found_init:
        return None

    # F-103 P103-C: rebuild the conversation chain from the leaf so
    # the returned messages are guaranteed to follow the
    # ``parentUuid`` topology (rather than the on-disk append
    # order, which may include dead-branch lines if the gate did
    # not fire). This is a no-op on transcripts where every line
    # is on the active chain — the leaf walk collapses to a
    # single in-order pass.
    if chain_filter:
        from clawcodex_ext.agent.chain_filter import build_conversation_chain
        from src.types.messages import message_to_dict

        serialised = [message_to_dict(m) if not isinstance(m, dict) else m for m in messages]
        chained = build_conversation_chain(serialised)
        rebuilt: list = []
        # ``build_conversation_chain`` returns the original dict
        # objects (not copies), so we map them back to typed
        # Message instances by re-parsing. This is cheap relative
        # to the file read and keeps the public surface (typed
        # messages) intact.
        for d in chained:
            try:
                rebuilt.append(message_from_dict(d))
            except Exception:  # nosec
                continue
        messages = rebuilt

    return Session(
        session_id=session_id,
        provider=provider,
        model=model,
        conversation=Conversation(messages=messages),
        created_at=created_at,
        updated_at=created_at,
    )
