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

"""Chain filter — parentUuid chain walker and byte-level pruner.

This module is the read-side counterpart. The write side lives
in :mod:`extensions.agent.session_persist` (``_inject_parent_uuids``).

Design (mirrors docs/FEATURE_PLAN.md §1.4.6):

    transcript.jsonl
        ↓ (raw bytes)
    walk_chain_before_parse()
        ↓ (filtered bytes — dead branches dropped)
    json.loads per line
        ↓ (message dicts)
    build_conversation_chain()
        ↓ (ordered active-chain messages)
    Session.load() / resume()

Why byte-level pruning:

    Per CCB architecture, ``walkChainBeforeParse`` scans the raw
    transcript for the substring ``"parentUuid":`` to build a uuid
    index **without** parsing each JSON line. This keeps the cost
    of the filter proportional to dead-branch size, not to
    conversation size, on long histories where most rewinds /
    forks are noise.

Two-stage gating:

    * ``DEAD_BRANCH_RATIO`` — skip filtering when < 50% of lines are
      off the active chain. Rationale: small sessions are fast
      enough to parse in full, and the byte scan itself is wasted
      work below this threshold.
    * ``ABS_SIZE_THRESHOLD`` — skip filtering on transcripts
      smaller than 10 KB. Same rationale: tiny transcripts benefit
      more from direct ``json.loads`` than from byte scanning.

Backward compatibility:

    Transcripts without any ``"parentUuid":`` substring (legacy
    sessions without parent links) skip the filter automatically.
    Mixed transcripts created by resuming a legacy session retain
    every legacy message while newer, chain-aware lines can still
    be pruned to their active branch.

Decoupling:

    This module imports no project internals from ``src/`` and no
    SessionStorage / Conversation glue. It operates on ``bytes``
    and ``list[dict]`` exclusively, so it can be unit-tested in
    isolation and reused by Visualizer / Telemetry consumers if
    they decide to honour the active-chain view too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Threshold for skipping the byte-level scan when dead-branch
# density is low. Mirrors CCB's ``SKIP_PRECOMPACT_THRESHOLD`` (0.5).
DEAD_BRANCH_RATIO: float = 0.5

# Threshold for skipping the byte-level scan on tiny transcripts.
# 10 KB ≈ ~50 short messages; below this the scan cost dominates.
ABS_SIZE_THRESHOLD: int = 10 * 1024

# Marker substring used by the byte-level indexer to locate
# ``"parentUuid":`` fields without parsing each line. Kept as a
# module constant so tests can vary it if the schema evolves.
PARENT_UUID_TOKEN: bytes = b'"parentUuid":'


@dataclass
class ChainFilterResult:
    """Outcome of ``walk_chain_before_parse``.

    Attributes:
        raw_bytes: filtered transcript bytes (active-chain lines +
            session_init / session_snapshot). May be empty if the
            input had no usable content.
        active_line_indices: zero-based line offsets (into the
            original ``raw_text`` split by ``\\n``) of lines that
            survived the filter. Useful for debugging / metrics.
        skipped_line_indices: zero-based line offsets of dropped
            dead-branch lines.
        skipped: True when the gate short-circuited and no pruning
            was attempted (small transcript or low dead-branch
            ratio). Callers can distinguish "no pruning needed"
            from "no dead branches found".
        total_lines: total non-blank lines in the input.
        dead_branch_lines: number of lines classified as
            dead-branch by the chain walk.
    """

    raw_bytes: bytes
    active_line_indices: list[int] = field(default_factory=list)
    skipped_line_indices: list[int] = field(default_factory=list)
    skipped: bool = False
    total_lines: int = 0
    dead_branch_lines: int = 0

    @property
    def dead_branch_ratio(self) -> float:
        """Ratio of dead branches in the input (0.0 ~ 1.0)."""
        if self.total_lines <= 0:
            return 0.0
        return self.dead_branch_lines / self.total_lines


@dataclass
class ChainFilterConfig:
    """Tunable thresholds for ``walk_chain_before_parse``.

    ``None`` falls back to the module-level constants so tests and
    callers can override selectively without touching globals.
    """

    dead_branch_ratio: float | None = None
    abs_size_threshold: int | None = None

    def resolved(self) -> tuple[float, int]:
        return (
            self.dead_branch_ratio if self.dead_branch_ratio is not None else DEAD_BRANCH_RATIO,
            self.abs_size_threshold if self.abs_size_threshold is not None else ABS_SIZE_THRESHOLD,
        )


def walk_chain_before_parse(
    raw: bytes,
    *,
    config: ChainFilterConfig | None = None,
) -> ChainFilterResult:
    """Prune dead-branch lines from a transcript byte stream.

    The function is byte-level: it does **not** parse JSON. It scans
    for ``"parentUuid":`` tokens, builds a uuid → line-index map,
    then walks from the leaf (the last message whose uuid is not
    referenced by any other message) back to the root. Lines not
    on the resulting chain are dropped.

    Metadata lines (``session_init`` / ``session_snapshot`` /
    ``cost_block``) are detected by a lightweight ``"type":`` scan
    and **always preserved** regardless of chain membership — they
    carry session identity / cost state, not conversation state.

    Args:
        raw: transcript bytes (UTF-8 JSONL).
        config: optional thresholds override.

    Returns:
        :class:`ChainFilterResult` with filtered bytes and
        observability metadata. ``result.skipped`` is True when
        no pruning was attempted (gate short-circuited).
    """
    if not raw:
        return ChainFilterResult(raw_bytes=b"", skipped=True)

    ratio_threshold, size_threshold = (
        config.resolved() if config is not None else (DEAD_BRANCH_RATIO, ABS_SIZE_THRESHOLD)
    )

    # Gate 1: tiny transcripts — direct parse is cheaper.
    if len(raw) < size_threshold:
        return ChainFilterResult(raw_bytes=raw, skipped=True)

    lines = raw.split(b"\n")
    # Strip trailing empty element from a final newline.
    while lines and not lines[-1]:
        lines.pop()

    total = len(lines)
    if total == 0:
        return ChainFilterResult(raw_bytes=raw, skipped=True)

    # Pre-scan: for each line, decide whether it carries a
    # ``parentUuid`` token, and whether it's a metadata line. The
    # metadata detection is intentionally a fast byte check (looks
    # for ``"type":`` followed by a recognised marker) so we don't
    # pay JSON-parse cost on every line.
    parent_indices: list[int] = []
    legacy_message_indices: set[int] = set()
    metadata_indices: set[int] = set()
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if PARENT_UUID_TOKEN in line:
            parent_indices.append(idx)
        elif b'"uuid"' in line:
            # A resumed transcript has a legacy prefix without
            # parentUuid followed by newly chained messages. Preserve every
            # valid legacy message: there is no topology with which to prune
            # that prefix safely, and a new message may point at its tail.
            entry = _parse_line_safely(line)
            if entry is not None:
                uuid = entry.get("uuid")
                if isinstance(uuid, str) and uuid:
                    legacy_message_indices.add(idx)
        if _is_metadata_line(line):
            metadata_indices.add(idx)

    # Gate 2: legacy transcripts have no ``parentUuid`` at all —
    # we cannot build a chain, so the safest thing is to skip the
    # filter and let the JSON parser handle them.
    if not parent_indices:
        return ChainFilterResult(
            raw_bytes=raw,
            skipped=True,
            total_lines=total,
        )

    # Build uuid index. We need each line's uuid **and** its
    # parentUuid. The byte scan finds candidate lines quickly, but
    # we still need to extract values — using a tiny per-line JSON
    # parse here keeps the implementation portable. Each parse is
    # O(line size) so total cost is O(transcript size), same as a
    # direct full parse — the saving comes from skipping dead
    # branches on the way out, not from skipping the parse itself.
    uuid_to_line: dict[str, int] = {}
    parent_uuids: dict[int, str | None] = {}
    for idx in parent_indices:
        entry = _parse_line_safely(lines[idx])
        if entry is None:
            continue
        uuid = entry.get("uuid")
        parent_uuid = entry.get("parentUuid")
        if not isinstance(uuid, str) or not uuid:
            continue
        uuid_to_line[uuid] = idx
        parent_uuids[idx] = parent_uuid if isinstance(parent_uuid, str) else None

    # The leaf is the message whose uuid is not anyone's parent.
    # In a /rewind scenario the file can carry TWO leaves: the
    # end of the dead branch (e.g. u4) and the end of the new
    # branch (e.g. u5). Both qualify under the "no child"
    # criterion. To pick the right one we use **on-disk line
    # order**, not chain length — the latest leaf in the file
    # is the most recently appended message, which is what the
    # user just sent. This also handles forks: each fork's
    # tail is its own leaf, and the most recently written fork
    # wins.
    referenced_as_parent = set(parent_uuids.values()) - {None}
    leaf_candidates = [idx for uuid, idx in uuid_to_line.items() if uuid not in referenced_as_parent]

    # Walk from the latest leaf back to the root. ``max`` on
    # line indices gives us the right answer both for rewinds
    # (the new branch is appended after the dead branch) and
    # for forks (the most recent fork is written last).
    best_chain: set[int] = set()
    if leaf_candidates:
        leaf_idx = max(leaf_candidates)
        chain: set[int] = set()
        cursor = leaf_idx
        seen: set[int] = set()
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            # ``parent_uuids`` is keyed by line index (int), so
            # checking membership tells us whether this line
            # carries a parentUuid at all. ``uuid_to_line`` is
            # keyed by uuid (str) and would never match an int
            # cursor — that's the trap we just fell into.
            if cursor not in parent_uuids:
                # Parent uuid points to a line we couldn't read —
                # stop walking rather than crash. Defensive against
                # partial writes.
                break
            chain.add(cursor)
            parent_uuid = parent_uuids.get(cursor)
            cursor = uuid_to_line.get(parent_uuid) if parent_uuid else None
        best_chain = chain | legacy_message_indices

    # If we couldn't find any leaf, fall back to "everything is
    # active" — better to over-include than to lose the entire
    # conversation because of a malformed parent pointer.
    if not best_chain:
        return ChainFilterResult(
            raw_bytes=raw,
            skipped=True,
            total_lines=total,
        )

    # Compute gate ratio. We treat metadata lines as NOT part of
    # the "dead-branch" pool — they're always preserved, so
    # counting them as dead would inflate the ratio.
    chainable_lines = total - len(metadata_indices)
    dead_lines = max(0, chainable_lines - len(best_chain))
    dead_ratio = dead_lines / chainable_lines if chainable_lines > 0 else 0.0

    if dead_ratio < ratio_threshold:
        # Below threshold — skip the heavy concat step. The
        # caller can still benefit from the analysis (e.g. for
        # metrics) via the returned metadata.
        return ChainFilterResult(
            raw_bytes=raw,
            skipped=True,
            total_lines=total,
            dead_branch_lines=dead_lines,
        )

    # Rebuild the byte stream: active-chain lines + metadata,
    # in original order. ``bytes`` is rebuilt via ``b"\n".join``
    # to keep the JSONL shape intact.
    active_set = best_chain | metadata_indices
    kept: list[bytes] = []
    skipped: list[int] = []
    for idx, line in enumerate(lines):
        if idx in active_set:
            kept.append(line)
        else:
            skipped.append(idx)
    # Trailing newline matches the convention used by
    # ``SessionStorage.flush()``.
    out = b"\n".join(kept) + (b"\n" if kept else b"")

    return ChainFilterResult(
        raw_bytes=out,
        active_line_indices=sorted(idx for idx in best_chain),
        skipped_line_indices=skipped,
        skipped=False,
        total_lines=total,
        dead_branch_lines=dead_lines,
    )


def build_conversation_chain(
    messages: Iterable[dict[str, Any]],
    leaf_uuid: str | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct an ordered active-chain list from message dicts.

    Walks from the leaf message (the one with no child referencing
    it) back to the root via ``parentUuid``, then reverses the
    result so callers get chronological order.

    Args:
        messages: iterable of message dicts. Each dict must carry
            a ``uuid`` and (optionally) a ``parentUuid``. Metadata
            entries (``type`` other than chat message) are passed
            through unchanged when ``leaf_uuid`` is None — they
            don't belong to the conversation chain.
        leaf_uuid: explicit leaf to start from. When ``None`` the
            leaf is inferred (longest chain among candidates).

    Returns:
        Ordered list of message dicts. Empty when ``messages`` is
        empty. When no chain can be reconstructed (no leaf, no
        parentUuid links), the input is returned in its original
        order as a fallback.
    """
    msg_list = list(messages)
    if not msg_list:
        return []

    # Index by uuid. Skip entries that don't carry a uuid — they
    # are either metadata or malformed lines.
    by_uuid: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str | None] = {}
    for entry in msg_list:
        if not isinstance(entry, dict):
            continue
        uuid = entry.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            continue
        by_uuid[uuid] = entry
        parent_uuid = entry.get("parentUuid")
        parent_of[uuid] = parent_uuid if isinstance(parent_uuid, str) else None

    if not by_uuid:
        return msg_list

    # No chain topology at all — every message has parentUuid=None.
    # Returning the input as-is keeps the helper a no-op for legacy
    # transcripts that don't carry parentUuid fields, instead of
    # collapsing to a single-entry chain.
    if not any(p is not None for p in parent_of.values()) and leaf_uuid is None:
        return msg_list

    # If an explicit leaf was given, use it. Otherwise infer: pick
    # the message whose uuid is not anyone's parent — when there
    # are several (forked chains), the one closest to the end of
    # the input list wins (most-recent branch).
    leaf: str | None = leaf_uuid
    if leaf is None or leaf not in by_uuid:
        referenced = set(parent_of.values()) - {None}
        candidates = [u for u in by_uuid if u not in referenced]
        if not candidates:
            # All messages are referenced — pick the last entry in
            # the input list as the leaf (the conversation tail).
            for entry in reversed(msg_list):
                if isinstance(entry, dict):
                    uuid = entry.get("uuid")
                    if isinstance(uuid, str) and uuid and uuid in by_uuid:
                        leaf = uuid
                        break
        else:
            # Prefer the candidate that appears latest in the
            # original input (mimicking "most recent branch").
            for entry in reversed(msg_list):
                if not isinstance(entry, dict):
                    continue
                uuid = entry.get("uuid")
                if uuid in candidates:
                    leaf = uuid
                    break
            if leaf is None:
                leaf = candidates[0]

    if leaf is None or leaf not in by_uuid:
        return msg_list

    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: str | None = leaf
    while cursor is not None and cursor not in seen:
        seen.add(cursor)
        entry = by_uuid.get(cursor)
        if entry is None:
            break
        chain.append(entry)
        cursor = parent_of.get(cursor)
    chain.reverse()
    return chain


def filter_active_chain_messages(
    entries: Sequence[dict[str, Any]],
    *,
    config: ChainFilterConfig | None = None,
) -> list[dict[str, Any]]:
    """High-level helper: prune dead branches and rebuild the chain.

    Combines :func:`walk_chain_before_parse` and
    :func:`build_conversation_chain` into a single call for
    transcript consumers that already hold parsed dicts (e.g.
    :class:`SessionStorage.read_transcript`). The function
    delegates byte-level gating to ``walk_chain_before_parse`` via
    an in-memory serialisation, so the threshold-based short
    circuit still works.

    Note:
        Serialising and re-parsing here is wasteful if the caller
        has the raw bytes handy. Prefer calling
        :func:`walk_chain_before_parse` directly and parsing the
        output bytes — this helper exists for code paths that
        only have the parsed dicts in hand (e.g. tests).

    Args:
        entries: parsed transcript entries (typically from
            ``SessionStorage.read_transcript()``).
        config: optional thresholds override.

    Returns:
        Ordered active-chain message dicts. Metadata entries
        (``type`` other than chat) are preserved at their
        original positions.
    """
    if not entries:
        return []

    # Re-serialise to drive the byte-level gate. ``ensure_ascii``
    # is False to keep the round-trip identical to the original
    # writer.
    raw = (
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries if isinstance(entry, dict)).encode("utf-8")
        + b"\n"
    )

    result = walk_chain_before_parse(raw, config=config)

    if result.skipped:
        # Below threshold or legacy format — return everything,
        # but still run chain rebuilder so callers get
        # deterministic ordering when parentUuid links exist.
        return build_conversation_chain(entries)

    # Parse the filtered bytes back into dicts and rebuild chain.
    filtered_entries: list[dict[str, Any]] = []
    for line in result.raw_bytes.split(b"\n"):
        if not line.strip():
            continue
        try:
            filtered_entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return build_conversation_chain(filtered_entries)


# --- Private helpers ----------------------------------------------------


def _is_metadata_line(line: bytes) -> bool:
    """Return True if ``line`` looks like a metadata entry.

    Metadata lines carry a ``"type":`` field with a known marker
    (session_init / session_snapshot / cost_block). The check is
    intentionally cheap — a substring scan — because we run it on
    every line of the transcript.
    """
    # Avoid the cost of json.loads here. The known markers are
    # distinct enough that a substring test is safe.
    for marker in (
        b'"type": "session_init"',
        b'"type":"session_init"',
        b'"type": "session_snapshot"',
        b'"type":"session_snapshot"',
        b'"type": "cost_block"',
        b'"type":"cost_block"',
    ):
        if marker in line:
            return True
    return False


def _parse_line_safely(line: bytes) -> dict[str, Any] | None:
    """Parse a single JSONL line, returning None on any failure."""
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(entry, dict):
        return None
    return entry


__all__ = [
    "ABS_SIZE_THRESHOLD",
    "ChainFilterConfig",
    "ChainFilterResult",
    "DEAD_BRANCH_RATIO",
    "PARENT_UUID_TOKEN",
    "build_conversation_chain",
    "filter_active_chain_messages",
    "walk_chain_before_parse",
]
