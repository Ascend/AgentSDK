"""ContextCollapseBoundary marker.

A ``ContextCollapseBoundary`` is the explicit, machine-detectable
placeholder injected into a projected message view to mark the
boundary between an archived range and the live messages that follow.
The existing :class:`ContextCollapseStore` injects a synthetic
``UserMessage`` with a ``[Collapsed context]`` text prefix; the
boundary module upgrades that to a structured marker that downstream
consumers (TUI, providers, audit logs) can detect without parsing
strings.

Two helper classes:

* :class:`BoundaryText` — the canonical text representation, with a
  stable prefix that the rest of the system greps for.
* :class:`BoundaryDetector` — a small helper that scans a list of
  messages and returns the indices of all boundary messages, along
  with the committed archive id each boundary corresponds to.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

BOUNDARY_PREFIX = "[CTX-COLLAPSE:"

#: Regex for the canonical boundary line. The trailing ``]`` is the
#: end of the prefix; everything after is the archive id.
_BOUNDARY_RE = re.compile(r"^\[CTX-COLLAPSE:([A-Za-z0-9._\-]{1,64})\]\s*$")


def make_boundary_text(archive_id: str) -> str:
    """Return the canonical boundary text for the given archive id.

    The archive id is validated against the same regex as plan ids to
    keep the boundary line unambiguously parseable.
    """
    if not isinstance(archive_id, str) or not archive_id:
        raise ValueError("archive_id must be a non-empty string")
    if not _BOUNDARY_RE.match(f"{BOUNDARY_PREFIX}{archive_id}]\n"):
        raise ValueError(f"archive_id has invalid characters: {archive_id!r}")
    return f"{BOUNDARY_PREFIX}{archive_id}]"


@dataclass(frozen=True)
class BoundaryHit:
    """A detected boundary in a message list."""

    message_index: int
    archive_id: str


class BoundaryDetector:
    """Scan messages and report boundary hits.

    The detector matches both the canonical ``[CTX-COLLAPSE:abc]``
    line (when injected as a standalone message) and the legacy
    ``[Collapsed context]`` prefix that the existing
    :class:`ContextCollapseStore` produces, so the rest of the system
    can adopt the new format gradually without breaking older audit
    logs.
    """

    LEGACY_PREFIX = "[Collapsed context]"

    def __init__(self, *, treat_legacy_as_boundary: bool = True) -> None:
        self._treat_legacy = treat_legacy_as_boundary
        self._lock = threading.RLock()
        # Auto-incrementing archive id seed for callers that want to
        # mint their own ids (the trigger does this when generating
        # commits).
        self._next_id = 0

    def detect(self, messages: Iterable[Any]) -> list[BoundaryHit]:
        with self._lock:
            out: list[BoundaryHit] = []
            for i, msg in enumerate(messages):
                text = _text(msg).strip()
                if not text:
                    continue
                first_line = text.splitlines()[0] if text else ""
                m = _BOUNDARY_RE.match(first_line)
                if m:
                    out.append(BoundaryHit(message_index=i, archive_id=m.group(1)))
                    continue
                if self._treat_legacy and text.startswith(self.LEGACY_PREFIX):
                    out.append(BoundaryHit(message_index=i, archive_id="legacy"))
            return out

    def mint_archive_id(self) -> str:
        with self._lock:
            self._next_id += 1
            return f"arc{self._next_id:06d}"


def _text(msg: Any) -> str:
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "\n".join(parts)
        return ""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    text = getattr(msg, "text", None)
    if isinstance(text, str):
        return text
    return ""


__all__ = [
    "BOUNDARY_PREFIX",
    "BoundaryDetector",
    "BoundaryHit",
    "make_boundary_text",
]
