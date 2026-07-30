"""Ultraplan slash-keyword detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TRIGGER_KEYWORDS: tuple[str, ...] = ("/ultraplan", "/ultra", "/up")


@dataclass(frozen=True)
class TriggerHit:
    start: int
    end: int
    keyword: str
    abbrev: bool = False


def find_ultraplan_trigger_positions(
    text: str,
    *,
    keywords: tuple[str, ...] = TRIGGER_KEYWORDS,
    inside_quotes: Literal["skip", "include"] = "skip",
    inside_code_fence: bool = False,
) -> list[TriggerHit]:
    """Find unescaped ultraplan trigger keywords in *text*.

    The scanner skips single, double, and backtick quoted spans by default.
    Passing ``inside_code_fence=True`` disables all hits for callers that
    already know the cursor is in a fenced block.
    """

    if not text or inside_code_fence:
        return []
    ordered = tuple(sorted(keywords, key=len, reverse=True))
    hits: list[TriggerHit] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            i += 1
            continue
        if inside_quotes == "skip" and ch in {"'", '"', "`"}:
            quote = ch
            i += 1
            continue
        match = next((kw for kw in ordered if text.startswith(kw, i)), None)
        if match is None:
            i += 1
            continue
        before_ok = i == 0 or text[i - 1].isspace()
        after_index = i + len(match)
        after_ok = after_index == len(text) or text[after_index].isspace()
        if before_ok and after_ok:
            hits.append(
                TriggerHit(
                    start=i,
                    end=after_index,
                    keyword=match,
                    abbrev=match != "/ultraplan",
                )
            )
        i = after_index
    return hits


def replace_ultraplan_keyword(
    text: str,
    old: str,
    new: str,
    *,
    positions: list[TriggerHit] | None = None,
) -> str:
    """Replace detected trigger keywords without touching quoted text."""

    hits = positions if positions is not None else find_ultraplan_trigger_positions(text)
    out = text
    for hit in sorted(hits, key=lambda h: h.start, reverse=True):
        if out[hit.start : hit.end] == old:
            out = out[: hit.start] + new + out[hit.end :]
    return out


def is_ultraplan_command(text: str) -> bool:
    """Return True when *text* begins with an ultraplan trigger after spaces."""

    leading = len(text) - len(text.lstrip())
    hits = find_ultraplan_trigger_positions(text)
    return bool(hits and hits[0].start == leading)
