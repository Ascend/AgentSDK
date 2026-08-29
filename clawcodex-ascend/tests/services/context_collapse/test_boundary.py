#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Boundary detector tests: canonical text + legacy prefix detection."""

from __future__ import annotations

import pytest

from src.services.context_collapse.boundary import (
    BOUNDARY_PREFIX,
    BoundaryDetector,
    BoundaryHit,
    make_boundary_text,
)


# ---------------------------------------------------------------------------
# make_boundary_text
# ---------------------------------------------------------------------------


def test_make_boundary_text_basic() -> None:
    assert make_boundary_text("arc000001") == "[CTX-COLLAPSE:arc000001]"


def test_make_boundary_text_rejects_empty() -> None:
    with pytest.raises(ValueError):
        make_boundary_text("")


def test_make_boundary_text_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        make_boundary_text(123)  # type: ignore[arg-type]


def test_make_boundary_text_accepts_alphanumeric() -> None:
    assert "abc123" in make_boundary_text("abc123")
    assert "XyZ-99" in make_boundary_text("XyZ-99")


def test_make_boundary_text_rejects_invalid_chars() -> None:
    with pytest.raises(ValueError):
        make_boundary_text("has spaces")


def test_make_boundary_text_rejects_overly_long_id() -> None:
    with pytest.raises(ValueError):
        make_boundary_text("a" * 65)


def test_make_boundary_text_prefix_is_stable() -> None:
    assert BOUNDARY_PREFIX == "[CTX-COLLAPSE:"
    # Multiple generations with the same id produce identical text.
    assert make_boundary_text("abc") == make_boundary_text("abc")


# ---------------------------------------------------------------------------
# BoundaryDetector.detect
# ---------------------------------------------------------------------------


def test_detector_empty_input() -> None:
    assert BoundaryDetector().detect([]) == []


def test_detector_finds_canonical_boundary() -> None:
    det = BoundaryDetector()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "[CTX-COLLAPSE:arc000007]"},
        {"role": "assistant", "content": "hi"},
    ]
    hits = det.detect(msgs)
    assert len(hits) == 1
    assert hits[0] == BoundaryHit(message_index=1, archive_id="arc000007")


def test_detector_skips_empty_messages() -> None:
    det = BoundaryDetector()
    msgs = [{"role": "user", "content": ""}]
    assert det.detect(msgs) == []


def test_detector_finds_legacy_prefix_when_enabled() -> None:
    det = BoundaryDetector(treat_legacy_as_boundary=True)
    msgs = [
        {"role": "user", "content": "[Collapsed context]\nsome summary"},
    ]
    hits = det.detect(msgs)
    assert len(hits) == 1
    assert hits[0].message_index == 0
    assert hits[0].archive_id == "legacy"


def test_detector_ignores_legacy_prefix_when_disabled() -> None:
    det = BoundaryDetector(treat_legacy_as_boundary=False)
    msgs = [
        {"role": "user", "content": "[Collapsed context]\nsome summary"},
    ]
    assert det.detect(msgs) == []


def test_detector_handles_object_messages() -> None:
    det = BoundaryDetector()

    class M:
        def __init__(self, text: str) -> None:
            self.content = text

    msgs = [M("[CTX-COLLAPSE:arc123]")]
    hits = det.detect(msgs)
    assert hits == [BoundaryHit(message_index=0, archive_id="arc123")]


def test_detector_handles_structured_content_blocks() -> None:
    det = BoundaryDetector()
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "[CTX-COLLAPSE:arc42]"}],
        }
    ]
    hits = det.detect(msgs)
    assert hits == [BoundaryHit(message_index=0, archive_id="arc42")]


def test_detector_rejects_oversized_archive_id() -> None:
    """The detector regex limits archive ids to 64 chars; longer ids don't match."""
    det = BoundaryDetector()
    msgs = [{"role": "user", "content": f"[CTX-COLLAPSE:{'a' * 100}]"}]
    # Not a boundary hit because the regex's [A-Za-z0-9._\-]{1,64} rejects it.
    # However, the legacy prefix does not match either, so no hit.
    assert det.detect(msgs) == []


def test_detector_finds_multiple_boundaries_in_order() -> None:
    det = BoundaryDetector()
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "[CTX-COLLAPSE:arc1]"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "[CTX-COLLAPSE:arc2]"},
        {"role": "assistant", "content": "tail"},
    ]
    hits = det.detect(msgs)
    assert [h.message_index for h in hits] == [1, 3]
    assert [h.archive_id for h in hits] == ["arc1", "arc2"]


def test_detector_multiline_message_only_inspects_first_line() -> None:
    det = BoundaryDetector()
    msgs = [
        {
            "role": "user",
            "content": "[CTX-COLLAPSE:arc1]\nsome trailing body\n[CTX-COLLAPSE:arc2]",
        }
    ]
    # Only the first line qualifies as a boundary.
    hits = det.detect(msgs)
    assert hits == [BoundaryHit(message_index=0, archive_id="arc1")]


# ---------------------------------------------------------------------------
# BoundaryDetector.mint_archive_id
# ---------------------------------------------------------------------------


def test_mint_archive_id_returns_increasing_sequence() -> None:
    det = BoundaryDetector()
    assert det.mint_archive_id() == "arc000001"
    assert det.mint_archive_id() == "arc000002"
    assert det.mint_archive_id() == "arc000003"


def test_mint_archive_id_format_is_six_digit() -> None:
    det = BoundaryDetector()
    for _ in range(5):
        aid = det.mint_archive_id()
        assert aid.startswith("arc")
        assert len(aid) == len("arc") + 6


def test_mint_archive_id_thread_safe() -> None:
    """Concurrent mints must produce distinct ids."""
    import threading

    det = BoundaryDetector()
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        aid = det.mint_archive_id()
        with lock:
            results.append(aid)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 100
    assert len(set(results)) == 100


# ---------------------------------------------------------------------------
# BoundaryHit (frozen dataclass)
# ---------------------------------------------------------------------------


def test_boundary_hit_is_immutable() -> None:
    hit = BoundaryHit(message_index=0, archive_id="arc1")
    with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
        hit.message_index = 99  # type: ignore[misc]


def test_boundary_hit_equality() -> None:
    a = BoundaryHit(message_index=1, archive_id="arc")
    b = BoundaryHit(message_index=1, archive_id="arc")
    c = BoundaryHit(message_index=2, archive_id="arc")
    assert a == b
    assert a != c
