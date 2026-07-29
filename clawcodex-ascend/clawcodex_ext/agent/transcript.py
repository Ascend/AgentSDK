# pylint: disable=redefined-outer-name,reimported,useless-return
"""Sidechain JSONL transcript writer + reader — Chunk C / WI-2.2.

**This is gate-zero for Chunk D and Chunk F.** Without on-disk transcript
persistence:

* Chapter §"Background: Three Channels" can't expose ``outputFile`` /
  ``outputOffset`` to ``TaskOutput`` (Chunk D / WI-3.1 / gap #7).
* Chapter §"Auto-Resume Pattern" can't reconstruct the conversation
  history from disk (Chunk F / WI-7.4 / gap #10).

Both depend on a JSONL file at a stable path containing one JSON object
per ``Message`` the agent produced. This module is the single source of
truth for that path and that file format.

Format
------

JSONL — one JSON object per line, encoded as UTF-8, terminated with
``\\n``. Per assumption A4 (gap-analysis ambiguity #4): the trade-offs
are atomic appends without an external lock library at the cost of
losing "read whole inbox in one parse." The reader is tolerant of a
trailing partial line (writer-crashed-mid-write case) — it logs once
and skips.

Concurrency
-----------

Writes go through ``os.write`` on an ``O_APPEND`` file descriptor. POSIX
guarantees ``write()`` calls of size ≤ ``PIPE_BUF`` (typically 4096
bytes) are atomic with respect to other writers; for a single agent
producing one line per Message, lines stay well under that and
interleaving cannot occur. Lines >4 KiB *can* interleave under heavy
concurrency; the reader's tolerant parser absorbs the resulting partial
line. Documented as a known limitation; a future revision could move to
``filelock`` if the >4 KiB regime becomes routine.

Out of scope (per critic concern C4 / Phase 11 follow-up)
---------------------------------------------------------

GC / rotation / age-based eviction. Transcripts grow unbounded under
this WI; the chapter-11 follow-up ticket will land a cleanup policy.
This module DOES expose ``get_agent_transcript_path`` / ``ensure_transcript_dir``
so the future GC has a stable target to walk.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from src.utils.clawcodex_dirs import get_transcripts_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers — stable across Writer / Reader / future GC
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Path helper — overridable via extension hook
# ---------------------------------------------------------------------------

#: Optional resolver registered by extensions (see :func:`register_transcript_path_resolver`).
#: Signature: ``(agent_id: str, parent_session_id: str | None) -> str | None``.
#: Return ``None`` to fall through to the default flat path.
_transcript_path_resolver: Callable[[str, Optional[str]], Optional[str]] | None = None


def register_transcript_path_resolver(
    resolver: Callable[[str, Optional[str]], Optional[str]],
) -> None:
    """Register a custom transcript path resolver.

    The resolver receives ``(agent_id, parent_session_id)`` and should
    return an absolute path (as ``str``) or ``None`` to fall through to
    the default ``~/.clawcodex/transcripts/<agent_id>.jsonl``.

    Provided so extensions in ``clawcodex_ext/`` can nest sub-agent
    transcripts under the parent session's directory tree without
    patching core code.
    """
    global _transcript_path_resolver
    _transcript_path_resolver = resolver


def _transcripts_root() -> Path:
    """Return the transcripts dir (created if absent).

    Resolved through ``get_transcripts_dir()`` so it honors
    ``$CLAWCODEX_CONFIG_DIR`` (default ``~/.clawcodex/transcripts``). Mirrors
    the directory convention that ``typescript/src/utils/sessionStorage.ts``
    uses for sidechain transcripts; per-user, not per-workspace.
    """
    root = get_transcripts_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


#: One-shot guard so a long-lived process doesn't spam the log once
#: per sub-agent spawn when the resolver is missing. The flag flips
#: on the first fallback hit; subsequent hits are silent.
_flat_fallback_warned: bool = False


def _warn_flat_fallback(parent_session_id: Optional[str]) -> None:
    """Emit a single warning when no nested resolver is registered.

    Tells the operator that a sub-agent transcript is about to land in
    the flat ``~/.clawcodex/transcripts/`` fallback rather than the
    designed
    ``~/.clawcodex/sessions/<parent_session_id>/subagents/`` tree.
    In practice this means the entry point skipped
    ``src.init.init()`` — every documented entry point (REPL,
    headless, bridge, TUI, SDK) is supposed to call ``init()`` first
    so the resolver gets registered. The warning silences itself
    under ``PYTEST_CURRENT_TEST`` because the test suite routinely
    exercises the fallback path with stub resolvers cleared
    afterwards (see ``tests/misc/test_transcript.py``).
    """
    global _flat_fallback_warned
    if _flat_fallback_warned:
        return
    if os.environ.get("PYTEST_CURRENT_TEST") is not None:
        return
    _flat_fallback_warned = True
    logger.warning(
        "sub-agent transcript fell back to flat path "
        "~/.clawcodex/transcripts/<id>.jsonl; no nested resolver "
        "registered. This usually means the entry point skipped "
        "src.init.init() — confirm init() runs before the agent "
        "loop. parent_session_id=%r",
        parent_session_id,
    )


def get_agent_transcript_path(
    agent_id: str,
    parent_session_id: str | None = None,
) -> str:
    """Absolute path to the agent's JSONL transcript.

    Delegates to a registered extension resolver first; when none is
    registered (or the resolver returns ``None``), falls back to the
    default flat path ``~/.clawcodex/transcripts/<safe_id>.jsonl``
    and emits a one-shot warning so the operator can spot entry
    points that bypassed ``src.init.init()``.

    Returns a string (not a ``Path``) because ``LocalAgentTaskState.output_file``
    is typed as ``str`` for serializability. Callers that prefer
    ``Path(get_agent_transcript_path(...))`` can still wrap it.
    """
    safe_id = _sanitize_agent_id(agent_id)
    if _transcript_path_resolver is not None:
        override = _transcript_path_resolver(agent_id, parent_session_id)
        if override is not None:
            return override
    _warn_flat_fallback(parent_session_id)
    return str(_transcripts_root() / f"{safe_id}.jsonl")


def get_main_transcript_path(session_id: str) -> str:
    """Absolute path to the main conversation's JSONL transcript.

    The main conversation is the user's primary session — distinct
    from sub-agent sidechains. Layout matches the convention the
    session-analysis viewer expects at
    ``clawcodex_sessions_analysis/lib/adapters/clawcodex.ts``.

    Path: ``$CLAWCODEX_SESSIONS_DIR/<session_id>/transcript.jsonl`` when the
    override is set, otherwise ``~/.clawcodex/sessions/<session_id>/transcript.jsonl``.

    The directory is created on demand by ``TranscriptWriter``'s
    constructor (parents=True). No resolver hook is consulted — the
    main path is a fixed convention; routing it through the
    sub-agent resolver would only complicate the call sites.
    """
    safe_id = _sanitize_agent_id(session_id)
    return str(_sessions_root() / safe_id / "transcript.jsonl")


def _sessions_root() -> Path:
    """Return the configured session root shared by main and child transcripts."""
    override = os.environ.get("CLAWCODEX_SESSIONS_DIR", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".clawcodex" / "sessions"


def get_workflow_run_path(run_id: str) -> str:
    """Absolute path to a workflow run's journal file.

    Layout: ``~/.clawcodex/transcripts/workflows/<run_id>.json``. The
    Workflow tool persists per-run journals here so that resumed runs
    can replay completed ``agent()`` calls from disk.
    """
    safe_id = _sanitize_agent_id(run_id)
    root = _transcripts_root() / "workflows"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"{safe_id}.json")


def _sanitize_agent_id(agent_id: str) -> str:
    """Reject path-traversing agent_ids before we touch the filesystem.

    ``agent_id`` is internally generated by ``generate_task_id`` (CSPRNG
    base36), so this is defense-in-depth — a future caller passing a
    user-supplied id would otherwise be able to write to arbitrary
    paths via ``../../etc/passwd``. Mirrors the chapter's symlink-attack
    rationale on TS Task.ts:96.
    """
    if not agent_id or not all(c.isalnum() or c in "_-" for c in agent_id):
        raise ValueError(f"invalid agent_id for transcript path: {agent_id!r} (allowed: alphanumeric + '_' + '-')")
    if len(agent_id) > 64:
        raise ValueError(f"agent_id too long ({len(agent_id)} > 64 chars)")
    return agent_id


# ---------------------------------------------------------------------------
# JSON serialization helper — tolerant of Message dataclasses
# ---------------------------------------------------------------------------


def _serialize_message(
    message: Any,
    parent_session_id: str | None = None,
    *,
    parent_uuid: str | None = None,
    include_parent_uuid: bool = False,
) -> str:
    """Convert a ``Message`` (or any JSON-shaped object) to a single
    UTF-8 JSON line. Falls back to ``repr`` for non-serializable objects
    so a malformed message can't bring the writer down — corrupt lines
    are reader-tolerant per the format design.

    When *parent_session_id* is provided the string ``"parent_session_id"``
    key is injected into the serialized dict so downstream consumers
    (reader, auto-resume) can correlate the message back to the parent
    session without inspecting the file path.

    String ``content`` payloads are wrapped as a single text block at
    the disk boundary so the on-disk shape is uniform across main and
    sub-agent transcripts (``content`` is always a list of blocks).
    The API boundary (``normalize_message_for_api`` in
    ``src.types.messages``) keeps its string-or-list tolerance
    unchanged — only the on-disk representation is normalized.
    """
    if is_dataclass(message) and not isinstance(message, type):
        # Chunk-D N1 fold-in (widened from `(TypeError, ValueError)`):
        # a pathological dataclass __reduce__/property could leak any
        # exception class. The transcript writer is a non-essential
        # persistence layer — no failure mode justifies bringing the
        # whole agent run down. Catch broadly and fall through to repr.
        try:
            payload = asdict(message)
        except Exception:  # nosec
            payload = {"_unserializable": repr(message)}
    elif isinstance(message, dict):
        payload = dict(message)
    else:
        # Last-resort: try to serialize a str() of it.
        payload = {"_unserializable": repr(message)}
    if parent_session_id is not None:
        payload["parent_session_id"] = parent_session_id
    if include_parent_uuid:
        payload["parentUuid"] = parent_uuid
    # Normalize string content → single text block. Done after
    # ``asdict`` (so the asdict call doesn't need to know about the
    # shape) and before ``json.dumps`` (so the on-disk JSONL is
    # uniform). The API boundary still accepts both shapes.
    content = payload.get("content")
    if isinstance(content, str):
        payload["content"] = [{"type": "text", "text": content}]
    try:
        # ``ensure_ascii=False`` keeps unicode readable in transcripts;
        # ``separators`` with no spaces keeps lines compact.
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception as exc:
        return json.dumps({"_unserializable": repr(message), "_error": str(exc)})


def _extract_ts_epoch(message: Any) -> float | None:
    """Best-effort extract a comparable epoch-seconds float from a message.

    Used by :class:`TranscriptWriter`'s ts-order flush buffer to sort
    records before writing. Returns ``None`` when no parseable ts is
    available — the caller treats ``None`` as "flush immediately" so
    the buffer never loses data.

    Accepts dataclass messages (looks for ``.timestamp``), top-level
    dicts (looks for ``"timestamp"`` or ``"ts"``), and nested
    ``{message: {...}}`` dicts (looks for ``"timestamp"`` inside the
    inner message). Falls back to ``None`` on parse failure rather
    than raising — a malformed ts must not abort the write path.
    """
    candidate: Any = None
    if is_dataclass(message) and not isinstance(message, type):
        candidate = getattr(message, "timestamp", None)
    elif isinstance(message, dict):
        candidate = message.get("timestamp") or message.get("ts")
        if candidate is None:
            inner = message.get("message")
            if isinstance(inner, dict):
                candidate = inner.get("timestamp") or inner.get("ts")
    if not isinstance(candidate, str) or not candidate:
        return None
    # Accept both ``...Z`` and ``...+00:00`` ISO shapes. ``fromisoformat``
    # handles both from Python 3.11+; for the trailing-Z case we
    # normalize to ``+00:00`` so older interpreters don't choke.
    iso = candidate
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    try:
        return dt.timestamp()
    except (OSError, OverflowError, ValueError):
        # ``OSError`` on platforms where ``timestamp()`` fails for
        # pre-1970 or far-future dates; ``OverflowError`` for
        # pre-epoch years that overflow the C time_t. Both signal
        # "unusable for ordering" — fall back to None.
        return None


# ---------------------------------------------------------------------------
# Writer — append-only, O_APPEND atomic line writes
# ---------------------------------------------------------------------------


class TranscriptWriter:
    """Append-only writer for the JSONL transcript at ``path``.

    Opens an ``O_APPEND`` file descriptor on construction. Each
    ``append(message)`` call serializes one line and emits it via a
    single ``os.write`` so concurrent writers (e.g. the same agent
    across crash-restart, or a future multi-writer scenario) cannot
    interleave bytes at sub-PIPE_BUF line sizes.

    The writer is **synchronous** (file IO, not asyncio). Per the A6/C5
    contract, callers must NOT hold ``RuntimeTaskRegistry``'s RLock
    across an ``append()`` call: while file IO is fast, blocking under
    the registry's lock would deadlock the asyncio scheduler against
    bash worker threads.

    ts-order flushing (added for the session-analysis viewer): the
    writer holds a small in-memory buffer keyed on the message
    timestamp. When the buffer hits a size cap (entries or bytes,
    whichever first) it's flushed in ascending ts order. Records
    without a parseable ISO ts flush immediately — opportunistic
    ordering, no data loss. POSIX line atomicity is preserved
    because each individual ``os.write`` is still ≤ PIPE_BUF.
    """

    # Buffer caps for ts-order flushing. 64 entries is enough to
    # absorb an asyncio scheduler reorder of one agent turn's worth
    # of messages (user prompt + assistant tool_use + tool_result +
    # next assistant) without growing unboundedly; 8 KiB keeps the
    # buffered payload under 2× PIPE_BUF so the sort + flush is
    # bounded. Both are class constants so tests can override.
    _SORT_BUFFER_MAX_ENTRIES: int = 64
    _SORT_BUFFER_MAX_BYTES: int = 8 * 1024

    def __init__(
        self,
        path: str | Path,
        parent_session_id: str | None = None,
        *,
        chain_messages: bool = False,
        initial_parent_uuid: str | None = None,
    ) -> None:
        self._path = str(path)
        self._parent_session_id = parent_session_id
        self._chain_messages = chain_messages
        self._chain_parent_uuid = initial_parent_uuid
        # Ensure parent dir exists (transcript root may not have been
        # created yet if the caller bypassed ``get_agent_transcript_path``).
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # ``O_APPEND`` makes every write atomic at the file-position
        # level. ``O_CLOEXEC`` keeps the fd from leaking to bash
        # subprocesses. ``0o600`` because transcripts can contain
        # sensitive prompt content — readable by the user only.
        # ``O_CLOEXEC`` is POSIX-only; on Windows it is absent (and fds
        # are non-inheritable by default since PEP 446), so fall back to 0.
        self._fd: int | None = os.open(
            self._path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        self._closed = False
        # ts-order flush buffer. List of ``(ts_epoch_float, encoded_line)``
        # pairs; flushed in ascending ts when full or on close.
        # ``None`` ts means "flush immediately" — we never buffer those.
        self._sort_buffer: list[tuple[float, bytes]] = []
        self._sort_buffer_bytes: int = 0

    @property
    def path(self) -> str:
        return self._path

    def append(self, message: Any) -> None:
        """Append one message as a UTF-8 JSON line, terminated with ``\\n``.

        Crash-safety: a single ``os.write`` of the line ensures the
        writer either appends the whole line or nothing. POSIX
        guarantees this for sizes ≤ ``PIPE_BUF`` (≥4096 bytes on every
        modern Unix); for larger lines the reader is tolerant of
        partial trailing content per the JSONL format design.

        ts ordering: a small in-memory buffer is held and flushed in
        ascending-timestamp order once it hits a size cap. Records
        without a parseable ISO timestamp bypass the buffer — they
        land on disk in arrival order, which is the safest fallback.
        """
        if self._closed or self._fd is None:
            raise RuntimeError("TranscriptWriter is closed")
        line = (
            _serialize_message(
                message,
                self._parent_session_id,
                parent_uuid=self._chain_parent_uuid,
                include_parent_uuid=self._chain_messages,
            )
            + "\n"
        )
        encoded = line.encode("utf-8")
        if self._chain_messages:
            # Parent links describe append order, so timestamp sorting would
            # be self-contradictory when a child happens to carry an earlier
            # timestamp than its parent. Chained transcripts are written in
            # call order and Session.load() uses parentUuid for branch repair.
            self._write_raw(encoded)
            self._advance_message_chain(message)
            return
        ts = _extract_ts_epoch(message)
        if ts is None:
            # Unparseable ts → write now, don't buffer. Also flush
            # any pending buffered records first so the file is in
            # roughly-ts-order up to this point.
            self._flush_sort_buffer()
            self._write_raw(encoded)
            self._advance_message_chain(message)
            return
        self._sort_buffer.append((ts, encoded))
        self._sort_buffer_bytes += len(encoded)
        if (
            len(self._sort_buffer) >= self._SORT_BUFFER_MAX_ENTRIES
            or self._sort_buffer_bytes >= self._SORT_BUFFER_MAX_BYTES
        ):
            self._flush_sort_buffer()
        self._advance_message_chain(message)

    def _advance_message_chain(self, message: Any) -> None:
        """Advance chained-writer state after accepting one message."""
        if not self._chain_messages:
            return
        if isinstance(message, dict):
            message_uuid = message.get("uuid")
        else:
            message_uuid = getattr(message, "uuid", None)
        if isinstance(message_uuid, str) and message_uuid:
            self._chain_parent_uuid = message_uuid

    def write_session_init(
        self,
        session_id: str,
        provider: str = "",
        model: str = "",
        created_at: str | None = None,
    ) -> None:
        """F-49 P5-G: write a ``session_init`` line as the first transcript entry.

        The line carries session-level metadata (provider, model, created_at)
        so ``Session.load()`` can reconstruct these fields from the JSONL
        transcript alone, without a separate ``session.json`` snapshot.

        Written directly (not through the sort buffer) so it lands on disk
        immediately as the first line.

        Example line written to disk::

            {"type":"session_init","session_id":"abc...","provider":"anthropic",
             "model":"claude-sonnet-4-20250514","created_at":"2026-06-19T09:03:02"}
        """
        if self._closed or self._fd is None:
            raise RuntimeError("TranscriptWriter is closed")
        from datetime import datetime

        payload: dict[str, Any] = {
            "type": "session_init",
            "session_id": session_id,
        }
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        payload["created_at"] = created_at or datetime.now().isoformat()
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._write_raw(line.encode("utf-8"))

    def _flush_sort_buffer(self) -> None:
        """Sort the buffered (ts, line) pairs ascending and write them out.

        Called when the buffer hits its size cap or on ``close``.
        Stable sort so equal-ts records retain insertion order
        (Python's sort is stable, per the language spec).
        """
        if not self._sort_buffer:
            return
        # ``key=lambda r: r[0]`` is faster than ``itemgetter`` for a
        # tuple because it avoids the per-element attribute lookup.
        self._sort_buffer.sort(key=lambda r: r[0])
        for _ts, encoded in self._sort_buffer:
            self._write_raw(encoded)
        self._sort_buffer.clear()
        self._sort_buffer_bytes = 0

    def _write_raw(self, encoded: bytes) -> None:
        # ``os.write`` may short-write under specific OS conditions;
        # loop until the whole buffer is on disk. For O_APPEND files
        # the returned ``n`` is byte count of THIS write, so the loop
        # is straightforward.
        view = memoryview(encoded)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                # Defensive: 0-byte returns shouldn't happen on regular
                # files but the loop would spin forever otherwise.
                raise OSError(f"transcript write returned {written}")
            view = view[written:]

    def close(self) -> None:
        if self._closed:
            return
        # Drain any buffered records BEFORE marking closed so the
        # final state of the file is in ts order. ``_flush_sort_buffer``
        # is a no-op when the buffer is empty.
        try:
            self._flush_sort_buffer()
        except OSError:
            logger.exception("transcript flush-on-close failed for %s", self._path)
        self._closed = True
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                logger.exception("transcript close failed for %s", self._path)

    # Context-manager support so callers can write `with TranscriptWriter(...) as w:`
    def __enter__(self) -> "TranscriptWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader — tolerant of trailing partial lines and missing files
# ---------------------------------------------------------------------------


class TranscriptReader:
    """Replay companion for ``resume_agent_background`` (Chunk F / WI-7.4).

    Reads the JSONL transcript line-by-line, parsing each into a Python
    object (typically a dict mirroring the original ``asdict(Message)``
    payload). Callers with a typed Message hierarchy can hydrate the
    dicts back into ``AssistantMessage``/``UserMessage`` etc. via their
    own factories — this reader stays loose-typed to avoid a cycle
    between the agent module and the transcript module.

    Tolerant of:
    * **Missing file** — yields nothing rather than raising.
    * **Trailing partial line** — log-once-and-skip rather than poison
      the entire history. Mirrors the chapter §"Mailbox" approach.
    * **Embedded blank lines** — skipped.

    Defining the reader here (per critic concern C6) so Chunk F's
    ``resume_agent_background`` SOLID DIP claim has a real interface
    to depend on rather than the writer's IO layer.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._logged_partial = False

    @property
    def path(self) -> str:
        return self._path

    def __iter__(self) -> Iterator[Any]:
        return self._iterate()

    def _iterate(self) -> Iterator[Any]:
        """Yield one parsed object per line; skip blank/unparseable lines."""
        try:
            with open(self._path, "rb") as handle:
                for raw_line in handle:
                    # raw_line still has its trailing newline; strip and skip
                    # blanks. Decode is utf-8 with replacement so a corrupt
                    # byte doesn't crash the iterator.
                    #
                    # N2 caveat (Chunk-D fold-in): ``errors="replace"`` maps
                    # corrupt bytes to U+FFFD. In theory a corrupted line
                    # could still parse as JSON if the U+FFFD substitutions
                    # land inside string literals — yielding "garbage but
                    # technically valid JSON". For chapter-10 transcripts
                    # the writer is the only producer and uses utf-8
                    # round-trip, so the regime is safe; downstream replay
                    # consumers should validate message shape post-parse.
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        if not self._logged_partial:
                            # Log once per reader instance — a corrupt
                            # transcript shouldn't spam the log.
                            logger.warning(
                                "skipping unparseable transcript line in %s (file may have a trailing partial-write)",
                                self._path,
                            )
                            self._logged_partial = True
        except FileNotFoundError:
            return

    def read_all(self) -> list[Any]:
        """Materialize every parseable line into a list."""
        return list(self._iterate())


def ensure_transcript_dir() -> str:
    """Create (if needed) and return the transcripts root path.

    Exposed so future Phase-11 GC / rotation logic has a stable target
    to walk; today's writers also call ``_transcripts_root`` indirectly
    through ``get_agent_transcript_path``.
    """
    return str(_transcripts_root())


__all__ = [
    "TranscriptWriter",
    "TranscriptReader",
    "get_agent_transcript_path",
    "get_main_transcript_path",
    "get_workflow_run_path",
    "ensure_transcript_dir",
    "register_transcript_path_resolver",
    "nested_session_path_resolver",
    "init",
]


# ---------------------------------------------------------------------------
# Absorbed from clawcodex_ext/transcript/nested_path.py — sub-agent nested
# path resolver. Previously split out as a separate module; merged here so
# the transcript extension lives in one canonical place. The legacy import
# path ``clawcodex_ext.transcript.nested_path`` is no longer supported —
# callers should use ``clawcodex_ext.agent.transcript`` instead. See also
# :func:`init` which is the entry point called by ``clawcodex_ext/__init__.py``.
# ---------------------------------------------------------------------------


def nested_session_path_resolver(
    agent_id: str,
    parent_session_id: str | None = None,
) -> str | None:
    """当有 *parent_session_id* 时，嵌套到 sessions/<id>/subagents/ 下。

    返回绝对路径字符串，或 ``None`` 回退到核心默认 flat 目录。
    """
    if not parent_session_id:
        return None  # 回退到 flat ~/.clawcodex/transcripts/

    _safe_session = _sanitize_for_path(parent_session_id)
    root = _sessions_root() / _safe_session / "subagents"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"agent-{agent_id}.jsonl")


def _sanitize_for_path(name: str) -> str:
    """轻量 sanitize — 只允许字母数字、连字符、下划线。"""
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"invalid component for session path: {name!r} (allowed: alphanumeric + '_' + '-')")
    if len(name) > 128:
        raise ValueError(f"session_id too long ({len(name)} > 128 chars)")
    return name


def init() -> None:
    """在扩展加载入口点注册嵌套路径解析器。

    调用方式（在 ``clawcodex_ext/__init__.py`` 中）::

        from clawcodex_ext.agent.transcript import init
        init()
    """
    register_transcript_path_resolver(nested_session_path_resolver)
    import logging

    logging.getLogger(__name__).info("registered nested-session transcript path resolver")
