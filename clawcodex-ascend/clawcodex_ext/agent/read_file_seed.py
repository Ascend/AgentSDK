"""F-125 C9: read-file-state seed for headless resume.

When a session is resumed via ``--resume`` / ``--fork-session``, the
conversation history carries ``tool_use`` blocks for prior ``Read``
calls. Without re-seeding ``tool_context.read_file_fingerprints``, the
``Edit`` / ``Write`` / ``NotebookEdit`` staleness checks
(``was_file_read_and_unchanged``) cannot tell that the model already
"knows" a file — they will reject edits with "file must be read first",
and the dedup path in ``Read`` cannot collapse re-reads to
``file_unchanged``.

Upstream CCB solves this with ``extractReadFilesFromMessages`` at
``print.ts:1173-1176``. This module is the Python analog: it walks the
resumed conversation's content blocks, finds every ``tool_use`` whose
``name == "Read"`` with a ``file_path`` argument that still exists on
disk, and calls ``context.mark_file_read(path)`` so the fingerprint
cache reflects the file's current mtime/size.

Design notes
------------
* **Pairing not required**: a ``tool_use`` alone is sufficient evidence
  the model read the file — the corresponding ``tool_result`` only
  carries the content snapshot, not the disk fingerprint. We seed from
  ``tool_use`` blocks directly.
* **Partial reads**: when the historical Read used ``offset`` / ``limit``
  we mark the file partial so the dedup path won't falsely collapse a
  later full read.
* **Missing files**: silently skipped. The file may have been deleted
  between runs; re-reading it would have failed anyway.
* **Best-effort**: any error is swallowed and logged at debug level —
  seeding is an optimisation, not a correctness gate. The agent can
  always re-Read explicitly.
* **Return value**: the number of files seeded, for telemetry / tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = ["seed_read_file_state_from_history"]


def seed_read_file_state_from_history(
    messages: Iterable[Any],
    context: Any,
    *,
    workspace_root: Path | None = None,
) -> int:
    """Populate ``context.read_file_fingerprints`` from historical Read calls.

    Walks ``messages`` (any object exposing ``.role`` / ``.content`` or
    a ``dict`` with the same shape), finds ``Read`` ``tool_use`` blocks,
    and marks the referenced files as read on ``context`` so subsequent
    Edit/Write staleness checks and Read dedup behave as if the model
    had just read them in this session.

    Parameters
    ----------
    messages
        The resumed conversation's message iterable. Each message may
        be a dataclass with a ``content`` attribute (list of blocks) or
        a ``dict`` with a ``"content"`` key.
    context
        A :class:`src.tool_system.context.ToolContext` (or compatible)
        exposing ``mark_file_read(path, *, partial=False)`` and
        ``read_file_fingerprints``.
    workspace_root
        Used to resolve relative ``file_path`` arguments. When ``None``,
        only absolute paths are seeded.

    Returns
    -------
    int
        The number of files successfully seeded (silently skips files
        that don't exist or fail to stat).
    """
    if context is None:
        return 0
    mark = getattr(context, "mark_file_read", None)
    if mark is None or not callable(mark):
        return 0

    seeded = 0
    for message in messages:
        for block in _iter_tool_use_blocks(message):
            if str(block.get("name", "")).lower() != "read":
                continue
            path = _resolve_read_path(block.get("input", {}), workspace_root)
            if path is None or not path.exists():
                continue
            partial = _is_partial_read(block.get("input", {}))
            try:
                mark(path, partial=partial)
                seeded += 1
            except Exception:  # nosec
                # mark_file_read stats the file; if it races with a
                # delete or a permission change, treat as a skip.
                logger.debug(
                    "F-125 read-file-state seed: mark_file_read failed for %s",
                    path,
                    exc_info=True,
                )
    if seeded:
        logger.debug(
            "F-125 read-file-state seed: populated %d file fingerprint(s) from resumed history",
            seeded,
        )
    return seeded


def _iter_tool_use_blocks(message: Any) -> Iterable[dict]:
    """Yield ``tool_use`` content blocks from a message as plain dicts.

    Tolerates both dataclass messages (``.content`` is a list of
    block objects / dicts) and dict-shaped messages. Non-tool_use
    blocks are filtered out.
    """
    content = _get(message, "content")
    if content is None:
        return
    if isinstance(content, str):
        return
    if not isinstance(content, (list, tuple)):
        return
    for block in content:
        block_dict = _block_as_dict(block)
        if block_dict is None:
            continue
        if str(block_dict.get("type", "")).lower() != "tool_use":
            continue
        yield block_dict


def _block_as_dict(block: Any) -> dict | None:
    """Coerce a content block (dataclass or dict) to a plain dict."""
    if isinstance(block, dict):
        return block
    # dataclass-style block with attributes
    type_attr = getattr(block, "type", None)
    if type_attr is None:
        return None
    name = getattr(block, "name", "")
    input_ = getattr(block, "input", None)
    return {
        "type": str(type_attr),
        "name": str(name or ""),
        "input": dict(input_) if isinstance(input_, dict) else {},
    }


def _get(obj: Any, key: str) -> Any:
    """Get an attribute or dict key, returning ``None`` if absent."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _resolve_read_path(input_data: dict, workspace_root: Path | None) -> Path | None:
    """Resolve the ``file_path`` argument of a Read tool_use block.

    Accepts both ``file_path`` (canonical) and ``path`` (legacy alias)
    keys. Returns ``None`` when the argument is absent or not a string.
    Relative paths are resolved against ``workspace_root`` when given;
    absolute paths are returned as-is.
    """
    raw = input_data.get("file_path")
    if raw is None:
        raw = input_data.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        p = Path(raw)
    except (ValueError, TypeError):
        return None
    if not p.is_absolute():
        if workspace_root is None:
            return None
        p = workspace_root / p
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return None


def _is_partial_read(input_data: dict) -> bool:
    """Return ``True`` when the historical Read used ``offset`` or ``limit``.

    Partial reads must not be deduped to ``file_unchanged`` because the
    model only saw a slice — re-reading the full file must return real
    content. See ``read.py`` FILE_UNCHANGED_STUB / partial fingerprint
    logic for the rationale.
    """
    offset = input_data.get("offset")
    limit = input_data.get("limit")
    if offset is not None and int(offset or 0) > 0:
        return True
    if limit is not None and int(limit or 0) > 0:
        return True
    return False
