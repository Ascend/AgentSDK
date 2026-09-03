"""Stable error fingerprinting.

The fingerprint is the join key used by :mod:`telemetry.aggregator`
to merge per-error occurrences into ``top_error_fingerprints`` and crash
summaries. It is intentionally short (16 hex chars) and stable across runs
that hit the same exception class, location and message shape.
"""

from __future__ import annotations

import hashlib
import re
import traceback
from typing import Final

_FINGERPRINT_LENGTH: Final[int] = 16

# Patterns we strip from exception messages before hashing. The order
# matters: bigger patterns first so e.g. UUIDs are matched before bare
# digit runs.
_STRIP_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
    re.compile(r"\b[0-9a-fA-F]{16,}\b"),
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\b\d+\b"),
    re.compile(r"//[^/\s]+/"),
    re.compile(r"\\[^\\\s]+\\"),
    re.compile(r"/[^/\s]+/"),
)

_REDACTED_PLACEHOLDER: Final[str] = "<num>"


def _normalize_message(message: str) -> str:
    """Strip volatile tokens (UUIDs, hex blobs, numbers, absolute paths)."""
    text = message
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub(_REDACTED_PLACEHOLDER, text)
    return text.strip()


def _pick_project_frame(
    tb: traceback.TracebackException | None,
    project_roots: tuple[str, ...],
) -> tuple[str, str, int] | None:
    """Pick the deepest frame whose filename lives under any project root.

    Returns ``(module, function, line)`` or ``None`` if no candidate frame
    is found. ``tb`` is consumed eagerly so the caller can rely on the
    returned tuple without holding a reference to the original stack.
    """
    if tb is None:
        return None
    last_match: tuple[str, str, int] | None = None
    for frame in tb.stack:
        filename = frame.filename or ""
        if not filename:
            continue
        if project_roots and not any(filename.startswith(root) for root in project_roots):
            continue
        module = frame.name or "<unknown>"
        last_match = (filename, module, int(frame.lineno or 0))
    return last_match


def compute_fingerprint(
    exc: BaseException,
    *,
    project_roots: tuple[str, ...] = (),
) -> str:
    """Return a 16-char hex fingerprint for ``exc``.

    The hash is computed over the join of:

    * the exception class name,
    * the deepest in-project frame (module / function / line), if any,
      otherwise the type's ``__module__`` and the first frame's filename,
    * a normalized version of ``str(exc)`` with volatile tokens stripped.

    The same exception raised from the same line with the same message
    shape always yields the same fingerprint.
    """
    cls_name = type(exc).__name__
    module_hint = type(exc).__module__ or ""

    try:
        tb = traceback.TracebackException.from_exception(exc)
    except Exception:
        tb = None

    project_frame = _pick_project_frame(tb, project_roots)
    if project_frame is not None:
        filename, func, lineno = project_frame
    elif tb is not None and tb.stack:
        first = tb.stack[0]
        filename = first.filename
        func = first.name
        lineno = int(first.lineno or 0)
    else:
        filename, func, lineno = "", "", 0

    message = _normalize_message(str(exc) or "")
    payload = f"{cls_name}|{module_hint}|{filename}|{func}|{lineno}|{message}"
    digest = hashlib.sha1(payload.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]
