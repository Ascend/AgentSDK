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

"""Tests for schema v1 → v2 migration helpers."""

from __future__ import annotations

import pytest

from telemetry.events import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    TelemetryEvent,
)
from telemetry.migration import (
    _fingerprint_dict_to_hash,
    migrate_v1_to_v2,
    normalize_event,
)


def test_migrate_v1_to_v2_wraps_fingerprint_string():
    """v1 fingerprint strings are wrapped in the v2 dict form."""
    raw = {
        "type": "error",
        "timestamp": 1718400000.0,
        "session_id": "abc",
        "schema_version": 1,
        "fields": {
            "error_class": "ValueError",
            "fingerprint": "deadbeef01234567",
            "stacktrace": ["ValueError: boom"],
        },
    }

    out = migrate_v1_to_v2(raw)

    assert out["schema_version"] == SCHEMA_VERSION_V2
    fp = out["fields"]["fingerprint"]
    assert isinstance(fp, dict)
    assert fp["hash"] == "deadbeef01234567"
    assert fp["version"] == 1
    assert fp["method"] == "legacy"
    # Unrelated fields preserved
    assert out["fields"]["error_class"] == "ValueError"
    assert out["fields"]["stacktrace"] == ["ValueError: boom"]


def test_migrate_v1_to_v2_preserves_unrelated_fields():
    """Migration only touches ``fields.fingerprint`` — every other
    field is preserved verbatim (including unknown ones).
    """
    raw = {
        "type": "session_start",
        "timestamp": 1.0,
        "session_id": "s1",
        "schema_version": 1,
        "fields": {
            "entrypoint": "cli",
            "platform": "Linux",
            "fingerprint": "abcdef",
            "custom_field": "kept",
        },
    }

    out = migrate_v1_to_v2(raw)

    assert out["fields"]["entrypoint"] == "cli"
    assert out["fields"]["platform"] == "Linux"
    assert out["fields"]["custom_field"] == "kept"
    assert out["fields"]["fingerprint"] == {
        "hash": "abcdef",
        "version": 1,
        "method": "legacy",
    }


def test_migrate_v1_to_v2_no_op_on_already_v2():
    """Feeding a v2 event back through the migrator is idempotent —
    the v2 fingerprint dict is returned untouched.
    """
    raw = {
        "type": "error",
        "timestamp": 1.0,
        "session_id": "s1",
        "schema_version": SCHEMA_VERSION_V2,
        "fields": {
            "fingerprint": {
                "hash": "deadbeef",
                "version": 2,
                "method": "sha1-truncate",
            }
        },
    }

    out = migrate_v1_to_v2(raw)

    assert out is raw  # not a copy
    assert out["fields"]["fingerprint"] == {
        "hash": "deadbeef",
        "version": 2,
        "method": "sha1-truncate",
    }


def test_migrate_v1_to_v2_handles_missing_schema_version_field():
    """Events lacking ``schema_version`` are treated as v1 (read
    default), so the migrator still upgrades them.
    """
    raw = {
        "type": "error",
        "timestamp": 1.0,
        "session_id": "s1",
        "fields": {"fingerprint": "abc1234567890def"},
    }

    out = migrate_v1_to_v2(raw)

    assert out["schema_version"] == SCHEMA_VERSION_V2
    assert out["fields"]["fingerprint"]["hash"] == "abc1234567890def"


def test_normalize_event_v1_passthrough_when_already_v2():
    """``normalize_event`` is a dispatcher: v2+ events are returned
    as-is so the aggregator's read path is idempotent.
    """
    raw = {
        "type": "error",
        "schema_version": SCHEMA_VERSION_V2,
        "fields": {"fingerprint": {"hash": "abc", "version": 2}},
    }

    out = normalize_event(raw)

    assert out is raw
    assert out["schema_version"] == SCHEMA_VERSION_V2


def test_normalize_event_missing_schema_version_defaults_to_v1():
    """Events with no ``schema_version`` field default to v1 and are
    upgraded by the migrator.
    """
    raw = {
        "type": "error",
        "fields": {"fingerprint": "abc"},
    }

    out = normalize_event(raw)

    assert out["schema_version"] == SCHEMA_VERSION_V2
    assert isinstance(out["fields"]["fingerprint"], dict)


def test_fingerprint_dict_to_hash_handles_both_shapes():
    """``_fingerprint_dict_to_hash`` is the single join-key extractor
    used by the aggregator's crash summary — it must accept v1
    strings and v2 dicts alike.
    """
    assert _fingerprint_dict_to_hash("abc1234567890def") == "abc1234567890def"
    assert _fingerprint_dict_to_hash({"hash": "abc1234567890def", "version": 2, "method": "x"}) == "abc1234567890def"
    # Unknown shapes collapse to "" so the bucket is stable
    assert _fingerprint_dict_to_hash(None) == ""
    assert _fingerprint_dict_to_hash(42) == ""
    assert _fingerprint_dict_to_hash({}) == ""


def test_telemetry_event_field_default_is_v2():
    """a freshly-constructed ``TelemetryEvent`` writes v2 by
    default so the recorder keeps producing v2 rows automatically.
    """
    event = TelemetryEvent(type="error")  # type: ignore[arg-type]
    assert event.schema_version == SCHEMA_VERSION_V2
    # Sanity: SCHEMA_VERSION remains 1 for read-default compatibility.
    assert SCHEMA_VERSION == 1


def test_privacy_audit_v1_v2_fingerprint_hash_equivalent():
    """cross-version guarantee: a v1 event and a v2 event with
    the same 16-char hash must end up in the same aggregator bucket
    after ``normalize_event`` and ``_fingerprint_dict_to_hash``.

    This is the single privacy / dedup invariant the migration
    promise rests on — the daily crash summary must not double-count
    legacy events when v2 is rolled out.
    """
    v1_event = {
        "type": "error",
        "schema_version": 1,
        "fields": {"fingerprint": "sharedfingerprint00", "error_class": "E"},
    }
    v2_event = {
        "type": "error",
        "schema_version": SCHEMA_VERSION_V2,
        "fields": {
            "fingerprint": {
                "hash": "sharedfingerprint00",
                "version": 2,
                "method": "sha1-truncate",
            },
            "error_class": "E",
        },
    }

    v1_normalized = normalize_event(v1_event)
    v1_hash = _fingerprint_dict_to_hash(v1_normalized["fields"]["fingerprint"])
    v2_hash = _fingerprint_dict_to_hash(v2_event["fields"]["fingerprint"])

    assert v1_hash == v2_hash == "sharedfingerprint00"
    # And both end up stamped as v2 after normalization
    assert v1_normalized["schema_version"] == SCHEMA_VERSION_V2
    assert v2_event["schema_version"] == SCHEMA_VERSION_V2


@pytest.mark.parametrize(
    "garbage",
    [None, 42, "string", [], {"no_hash_key": "x"}, object()],
)
def test_normalize_event_does_not_raise_on_malformed(garbage):
    """``normalize_event`` must never raise out of its public path;
    the aggregator's read loop relies on that to keep building
    summaries even when a single row is corrupt.
    """
    # Should not raise; result type matches input
    out = normalize_event(garbage)
    assert out is garbage or out == garbage
