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

# pylint: disable=relative-beyond-top-level
"""Rule-first feature extractor for SR-5.1.

Mirrors the contract sketched in the feature design. The extractor
takes a release ``body`` (and metadata) and produces a list of
:class:`FeatureRecord` candidates using Markdown heuristics:

* ``## Added / ## New / ## Features`` sections
* ``- [x]`` checkbox completed items (Keep-a-Changelog style)
* ``## Breaking Changes`` sections
* ``## Changed / ## Improved / ## Enhanced`` sections
* Top-level bullet points when no section is detected

The optional LLM fallback is intentionally pluggable: callers can pass
a callable that takes a list of candidates + the body and returns
refined records. We never call an LLM implicitly — that decision lives
in the pipeline runner (and behind the ``use_llm`` config flag).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .models import (
    FeatureRecord,
    FeatureType,
    Release,
    make_feature_id,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------


# Sections we treat as "new features". Order matters: the parser keeps
# the *first* matching heading it finds.
NEW_SECTION_HEADINGS = (
    "added",
    "new",
    "features",
    "new features",
    "what's new",
    "whats new",
)

ENHANCEMENT_SECTION_HEADINGS = (
    "changed",
    "improved",
    "enhanced",
    "updates",
)

BREAKING_SECTION_HEADINGS = (
    "breaking",
    "breaking changes",
    "removed",
)

DEPRECATION_SECTION_HEADINGS = (
    "deprecated",
    "deprecations",
)

BUGFIX_SECTION_HEADINGS = (
    "fixed",
    "bug fixes",
    "bugfixes",
)

ALL_SECTION_HEADINGS = (
    *NEW_SECTION_HEADINGS,
    *ENHANCEMENT_SECTION_HEADINGS,
    *BREAKING_SECTION_HEADINGS,
    *DEPRECATION_SECTION_HEADINGS,
    *BUGFIX_SECTION_HEADINGS,
)


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>[^\n]+?)\s*$", re.MULTILINE)
_CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+(?P<text>.+?)\s*$", re.MULTILINE)
# Negative lookahead so checkboxes (``- [x] foo``) match exactly once
# rather than being captured both as a checkbox and as a bullet (the
# bullet branch would have produced ``"[x] foo"`` as the text, which
# would silently survive the dedup-by-text and double-count the line).
_BULLET_RE = re.compile(r"^\s*[-*]\s+(?!\[)(?P<text>.+?)\s*$", re.MULTILINE)

# Patterns for bullet-level bugfix detection.  Applied only when the section
# heading doesn't give us an explicit classification (e.g. "## What's Changed").
# Word boundaries (\b) on single-word patterns prevent matching "prefix",
# "debug", etc.
_BUGFIX_TEXT_RE = re.compile(
    r"\b(?:bug|fix|fixed|hotfix|crash|regression|backport)\b|"
    r"hot-fix|race[-\s]condition|memory[-\s]leak",
    re.IGNORECASE,
)


def _is_bugfix_by_text(text: str) -> bool:
    """Return True if *text* contains bugfix-indicating keywords."""
    return bool(_BUGFIX_TEXT_RE.search(text))


@dataclass
class _Section:
    """A Markdown section captured by :func:`_split_sections`."""

    heading: str
    body: str


def _split_sections(text: str) -> list[_Section]:
    """Split ``text`` into ``(heading, body)`` pairs.

    A section spans from one heading line until the next heading of the
    same or higher level. Pre-heading preamble (e.g. ``# v1.2.3``) is
    returned as an empty-heading section; the extraction layer skips
    its bullets unless the body has no section headings at all.
    """
    matches = list(_HEADING_RE.finditer(text or ""))
    if not matches:
        return [_Section(heading="", body=text or "")]

    sections: list[_Section] = []
    if matches[0].start() > 0:
        sections.append(_Section(heading="", body=text[: matches[0].start()]))

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group("title").strip().lower()
        body = text[start:end].strip("\n")
        sections.append(_Section(heading=heading, body=body))

    return sections


def _classify_heading(heading: str) -> FeatureType | None:
    """Map a section heading to a :class:`FeatureType` (or None)."""
    h = heading.lower().strip()
    # Drop leading numbering/decorations like "1.2 Added" / "[v2] Changed"
    h = re.sub(r"^[\[\(\d\.\)\]\s]+", "", h)
    if any(token in h for token in BREAKING_SECTION_HEADINGS):
        return FeatureType.BREAKING
    if any(token in h for token in NEW_SECTION_HEADINGS):
        return FeatureType.NEW
    if any(token in h for token in DEPRECATION_SECTION_HEADINGS):
        return FeatureType.DEPRECATION
    if any(token in h for token in BUGFIX_SECTION_HEADINGS):
        return FeatureType.BUGFIX
    if any(token in h for token in ENHANCEMENT_SECTION_HEADINGS):
        return FeatureType.ENHANCEMENT
    return None


def _extract_bullets(section_body: str) -> list[str]:
    """Return ordered, de-duplicated list items inside ``section_body``."""
    items: list[str] = []
    seen: set[str] = set()
    for regex in (_CHECKBOX_RE, _BULLET_RE):
        for match in regex.finditer(section_body or ""):
            text = match.group("text").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(text)
    return items


def _short_title(text: str, *, limit: int = 80) -> str:
    """Produce a stable, short title for ``text``.

    Title-cases the first 80 chars and strips trailing punctuation.
    Used to keep the digest readable even when the upstream bullet is
    multi-sentence.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.rstrip(".!?:;,")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


LLMHook = Callable[[list[FeatureRecord], str], list[FeatureRecord]]


class FeatureExtractor:
    """Rule-first feature extractor.

    The extractor is intentionally stateless and pure — every method
    takes its inputs as arguments so callers can chain it inside
    pipeline tests without monkeypatching globals.
    """

    #: Section names that always indicate breaking changes (matched
    #: verbatim, in addition to the fuzzy :func:`_classify_heading`
    #: logic). Useful for projects that don't follow Keep-a-Changelog.
    BREAKING_SECTION_ALIASES: tuple[str, ...] = (
        "⚠ breaking",
        "⚠️ breaking",
        "migration",
    )

    #: Section names that always indicate deprecations.
    DEPRECATION_SECTION_ALIASES: tuple[str, ...] = ("deprecation notice",)

    def __init__(self, llm_hook: LLMHook | None = None) -> None:
        self._llm_hook = llm_hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, release: Release, source: str) -> list[FeatureRecord]:
        """Extract candidate feature records from ``release``.

        Prefers ``raw_body`` (CHANGELOG text from Layer 1.5) over ``body``
        (GitHub Release body) when available — CHANGELOG sections provide
        richer ``## Added / Changed / Fixed`` markup for pattern extraction.
        """
        source_text = release.raw_body or release.body
        if not source_text:
            return []
        candidates = list(self._extract_by_patterns(source_text))
        records: list[FeatureRecord] = []
        for candidate in candidates:
            records.append(
                FeatureRecord(
                    id=make_feature_id(source, candidate.title, candidate.kind.value),
                    source=source,
                    title=candidate.title,
                    description=candidate.description,
                    feature_type=candidate.kind,
                    released_at=release.published_at,
                    url=release.url,
                    raw_body=candidate.raw,
                )
            )
        if self._llm_hook is not None:
            try:
                records = self._llm_hook(records, source_text)
            except Exception as exc:  # noqa: BLE001
                _log.warning("LLM hook raised (%s); keeping rule-based output", exc)
        return records

    def extract_many(self, releases: Iterable[Release], source: str) -> list[FeatureRecord]:
        """Run :meth:`extract` over multiple releases and merge."""
        out: list[FeatureRecord] = []
        for release in releases:
            out.extend(self.extract(release, source))
        return out

    # ------------------------------------------------------------------
    # Rule-based pattern extraction
    # ------------------------------------------------------------------

    @dataclass
    class _Candidate:
        title: str
        description: str
        kind: FeatureType
        raw: str

    def _extract_by_patterns(self, body: str) -> list[FeatureExtractor._Candidate]:
        sections = _split_sections(body)
        candidates: list[FeatureExtractor._Candidate] = []

        # Track the last explicit heading kind so untagged bullets in
        # unrecognised sections can still be classified.
        last_kind: FeatureType | None = None
        for section in sections:
            section_kind = self._classify_section(section.heading)
            section_explicit = section_kind is not None
            if section_kind is not None:
                last_kind = section_kind
            elif not section.heading:
                # Pre-amble: keep whatever default the caller wants.
                # We default to NEW only when the release body has no
                # sections at all — otherwise leave it untagged.
                if not any(s.heading for s in sections):
                    last_kind = FeatureType.NEW
                else:
                    continue
            else:
                # Unrecognised section (e.g. "Documentation", "Tests"):
                # still emit candidates so the classifier can decide.
                pass

            for bullet in _extract_bullets(section.body):
                title = _short_title(bullet)
                if not title:
                    continue
                kind = last_kind or FeatureType.NEW
                # ── Bullet-level bugfix detection ──
                # When the section heading is unrecognised (e.g. "## What's
                # Changed"), fall back to keyword matching on the bullet
                # text itself.  We deliberately skip sections with an
                # explicit heading classification — a "fix" inside
                # "## Added" is a new capability, not a bugfix.
                if not section_explicit and kind != FeatureType.BUGFIX and _is_bugfix_by_text(title):
                    kind = FeatureType.BUGFIX
                candidates.append(
                    FeatureExtractor._Candidate(
                        title=title,
                        description=bullet,
                        kind=kind,
                        raw=bullet,
                    )
                )
        return candidates

    def _classify_section(self, heading: str) -> FeatureType | None:
        if not heading:
            return None
        h = heading.lower()
        for alias in self.BREAKING_SECTION_ALIASES:
            if alias in h:
                return FeatureType.BREAKING
        for alias in self.DEPRECATION_SECTION_ALIASES:
            if alias in h:
                return FeatureType.DEPRECATION
        return _classify_heading(heading)
