#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Template model tests: validation matrix + round-trip."""
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import pytest

from src.services.templates import Template


def _tpl(**overrides) -> Template:
    defaults: dict = {
        "id": "general-purpose",
        "title": "General-purpose agent",
        "description": "A baseline agent for general work.",
        "fields": {"tools": ["Read", "Bash"], "max_turns": 25},
        "metadata": {"owner": "core", "version": 1},
        "source": "built-in",
    }
    defaults.update(overrides)
    return Template(**defaults)


# ---------------------------------------------------------------------------
# Defaults and minimal construction
# ---------------------------------------------------------------------------


def test_minimal_template() -> None:
    t = Template(id="x", title="X")
    assert t.id == "x"
    assert t.title == "X"
    assert t.description is None
    assert dict(t.fields) == {}
    assert dict(t.metadata) == {}
    assert t.source == "user"


def test_full_template_preserves_fields() -> None:
    t = _tpl()
    assert t.id == "general-purpose"
    assert t.title == "General-purpose agent"
    assert t.description is not None
    assert t.fields["max_turns"] == 25
    assert t.metadata["owner"] == "core"
    assert t.source == "built-in"


def test_template_is_frozen() -> None:
    t = Template(id="x", title="X")
    with pytest.raises(Exception):  # FrozenInstanceError is AttributeError subclass
        t.title = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# id validation
# ---------------------------------------------------------------------------


def test_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        Template(id="", title="x")


def test_rejects_non_string_id() -> None:
    with pytest.raises(ValueError):
        Template(id=42, title="x")  # type: ignore[arg-type]


def test_rejects_id_with_spaces() -> None:
    with pytest.raises(ValueError):
        Template(id="has spaces", title="x")


def test_rejects_id_with_path_separators() -> None:
    with pytest.raises(ValueError):
        Template(id="../etc/passwd", title="x")
    with pytest.raises(ValueError):
        Template(id="a/b", title="x")


def test_rejects_overly_long_id() -> None:
    with pytest.raises(ValueError):
        Template(id="a" * 65, title="x")


def test_accepts_alphanumeric_with_dots_and_dashes() -> None:
    t = Template(id="a.b-c_1.2", title="x")
    assert t.id == "a.b-c_1.2"


def test_accepts_id_at_max_length() -> None:
    t = Template(id="a" * 64, title="x")
    assert len(t.id) == 64


# ---------------------------------------------------------------------------
# title validation
# ---------------------------------------------------------------------------


def test_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="")


def test_rejects_non_string_title() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title=123)  # type: ignore[arg-type]


def test_rejects_overly_long_title() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x" * 201)


def test_accepts_title_at_max_length() -> None:
    t = Template(id="x", title="x" * 200)
    assert len(t.title) == 200


# ---------------------------------------------------------------------------
# description validation
# ---------------------------------------------------------------------------


def test_description_defaults_to_none() -> None:
    t = Template(id="x", title="x")
    assert t.description is None


def test_rejects_non_string_description() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", description=42)  # type: ignore[arg-type]


def test_rejects_overly_long_description() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", description="x" * 2001)


# ---------------------------------------------------------------------------
# fields validation
# ---------------------------------------------------------------------------


def test_fields_default_empty_mapping() -> None:
    t = Template(id="x", title="x")
    assert dict(t.fields) == {}


def test_rejects_non_mapping_fields() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", fields=[("a", 1)])  # type: ignore[arg-type]


def test_rejects_field_name_with_spaces() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", fields={"bad name": 1})


def test_rejects_field_name_starting_with_digit() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", fields={"1bad": 1})


def test_accepts_underscore_field_names() -> None:
    t = Template(id="x", title="x", fields={"_internal": True})
    assert "_internal" in t.fields


def test_rejects_too_many_fields() -> None:
    big = {f"f{i}": i for i in range(65)}
    with pytest.raises(ValueError):
        Template(id="x", title="x", fields=big)


# ---------------------------------------------------------------------------
# metadata validation
# ---------------------------------------------------------------------------


def test_metadata_default_empty() -> None:
    t = Template(id="x", title="x")
    assert dict(t.metadata) == {}


def test_rejects_non_mapping_metadata() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", metadata=["bad"])  # type: ignore[arg-type]


def test_rejects_metadata_with_non_string_key() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", metadata={1: "v"})  # type: ignore[dict-item]


def test_rejects_metadata_with_empty_key() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", metadata={"": "v"})


# ---------------------------------------------------------------------------
# source validation
# ---------------------------------------------------------------------------


def test_rejects_empty_source() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", source="")


def test_rejects_non_string_source() -> None:
    with pytest.raises(ValueError):
        Template(id="x", title="x", source=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_round_trip_full_template() -> None:
    t = _tpl()
    d = t.to_dict()
    t2 = Template.from_dict(d)
    assert t2.id == t.id
    assert t2.title == t.title
    assert t2.description == t.description
    assert dict(t2.fields) == dict(t.fields)
    assert dict(t2.metadata) == dict(t.metadata)
    assert t2.source == t.source


def test_round_trip_minimal_template() -> None:
    t = Template(id="x", title="x")
    d = t.to_dict()
    assert d == {"id": "x", "title": "x", "source": "user"}
    t2 = Template.from_dict(d)
    assert t2.id == t.id
    assert t2.title == t.title


def test_to_dict_omits_empty_optional_fields() -> None:
    t = Template(id="x", title="x")
    d = t.to_dict()
    assert "description" not in d
    assert "fields" not in d
    assert "metadata" not in d


def test_to_dict_includes_description_when_set() -> None:
    t = Template(id="x", title="x", description="hello")
    d = t.to_dict()
    assert d["description"] == "hello"


def test_to_dict_includes_fields_when_nonempty() -> None:
    t = Template(id="x", title="x", fields={"k": 1})
    d = t.to_dict()
    assert d["fields"] == {"k": 1}


# ---------------------------------------------------------------------------
# from_dict validation (corrupt payloads)
# ---------------------------------------------------------------------------


def test_from_dict_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        Template.from_dict("not a dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Template.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_dict_rejects_missing_id() -> None:
    with pytest.raises(ValueError):
        Template.from_dict({"title": "x"})


def test_from_dict_rejects_missing_title() -> None:
    with pytest.raises(ValueError):
        Template.from_dict({"id": "x"})


def test_from_dict_defaults_source_to_user() -> None:
    t = Template.from_dict({"id": "x", "title": "x"})
    assert t.source == "user"


def test_from_dict_coerces_none_fields_and_metadata_to_empty() -> None:
    t = Template.from_dict({"id": "x", "title": "x", "fields": None, "metadata": None})
    assert dict(t.fields) == {}
    assert dict(t.metadata) == {}


def test_from_dict_propagates_invalid_field_name() -> None:
    with pytest.raises(ValueError):
        Template.from_dict({"id": "x", "title": "x", "fields": {"1bad": 1}})


def test_from_dict_propagates_invalid_id() -> None:
    with pytest.raises(ValueError):
        Template.from_dict({"id": "has spaces", "title": "x"})
