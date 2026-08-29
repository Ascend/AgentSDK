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

"""Unit tests for :mod:`src.services.templates.schema` (P85-A).

Covers the canonical schema contract:

* Top-level shape — single mapping OR list of mappings (bundle).
* Top-level keys — strict mode rejects unknown keys; lenient mode drops them.
* Per-field validation — ``id`` and ``title`` required; ``fields`` and
  ``metadata`` are free-form mappings.
* File parsing — YAML vs JSON dispatch, corrupt / missing / empty files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.templates import (
    SCHEMA_DESCRIPTION,
    TEMPLATE_FIELD_KEYS,
    TEMPLATE_SCHEMA_VERSION,
    TEMPLATE_TOP_LEVEL_KEYS,
    TemplateCorruptError,
    TemplateValidationError,
    parse_template_file,
    parse_template_file_payload,
    parse_template_payload,
)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


def test_schema_version_is_set() -> None:
    assert TEMPLATE_SCHEMA_VERSION == "1.0"


def test_top_level_keys_are_canonical_set() -> None:
    assert TEMPLATE_TOP_LEVEL_KEYS == frozenset(
        {
            "id",
            "title",
            "description",
            "fields",
            "metadata",
            "source",
            "kind",
            "variables",
            "tags",
            "category",
            "schema_version",
            "min_clawcodex_version",
            "output_path_template",
            "content_template",
            "content_template_ref",
        }
    )


def test_field_keys_match_top_level() -> None:
    """``fields`` and ``metadata`` use the same key set as top-level."""
    assert TEMPLATE_FIELD_KEYS == TEMPLATE_TOP_LEVEL_KEYS


def test_schema_description_is_nonempty() -> None:
    assert SCHEMA_DESCRIPTION
    assert TEMPLATE_SCHEMA_VERSION in SCHEMA_DESCRIPTION


# ---------------------------------------------------------------------------
# parse_template_payload — single mapping only
# ---------------------------------------------------------------------------


def test_parse_payload_minimal() -> None:
    tpl = parse_template_payload({"id": "x", "title": "X"})
    assert tpl.id == "x"
    assert tpl.title == "X"


def test_parse_payload_full() -> None:
    data = {
        "id": "y",
        "title": "Y",
        "description": "A y template",
        "fields": {"tools": ["Read"]},
        "metadata": {"category": "test"},
        "source": "user",
    }
    tpl = parse_template_payload(data)
    assert tpl.description == "A y template"
    assert tpl.fields["tools"] == ["Read"]
    assert tpl.metadata["category"] == "test"
    assert tpl.source == "user"


def test_parse_payload_normalizes_f95_top_level_keys() -> None:
    tpl = parse_template_payload(
        {
            "id": "skill-demo",
            "title": "Skill Demo",
            "kind": "skill",
            "variables": [{"name": "skill_name", "description": "Name"}],
            "output_path_template": ".claude/skills/{{ skill_name }}/SKILL.md",
            "content_template": "# {{ skill_name }}",
        }
    )
    assert tpl.metadata["kind"] == "skill"
    assert tpl.metadata["variables"] == [{"name": "skill_name", "description": "Name"}]
    assert tpl.metadata["output_path_template"] == ".claude/skills/{{ skill_name }}/SKILL.md"
    assert tpl.fields["content_template"] == "# {{ skill_name }}"


def test_parse_payload_rejects_non_mapping() -> None:
    with pytest.raises(TemplateValidationError):
        parse_template_payload("not a mapping")  # type: ignore[arg-type]


def test_parse_payload_rejects_list_at_top_level() -> None:
    """``parse_template_payload`` only handles single mappings —
    bundles go through ``parse_template_file_payload``.
    """
    with pytest.raises(TemplateValidationError):
        parse_template_payload([{"id": "x", "title": "X"}])  # type: ignore[arg-type]


def test_parse_payload_rejects_missing_id() -> None:
    with pytest.raises(TemplateValidationError):
        parse_template_payload({"title": "X"})


def test_parse_payload_rejects_missing_title() -> None:
    with pytest.raises(TemplateValidationError):
        parse_template_payload({"id": "x"})


def test_parse_payload_strict_mode_rejects_unknown_keys() -> None:
    """Strict mode (default) raises on extra top-level keys."""
    with pytest.raises(TemplateValidationError) as excinfo:
        parse_template_payload({"id": "x", "title": "X", "unknown_key": "value"})
    assert "unknown_key" in str(excinfo.value)


def test_parse_payload_lenient_mode_drops_unknown_keys() -> None:
    """Lenient mode silently drops extra keys (forward-compat)."""
    tpl = parse_template_payload(
        {"id": "x", "title": "X", "future_key": "ignored"},
        strict=False,
    )
    assert tpl.id == "x"


# ---------------------------------------------------------------------------
# parse_template_file_payload — single mapping OR bundle (list)
# ---------------------------------------------------------------------------


def test_parse_file_payload_single_mapping() -> None:
    result = parse_template_file_payload({"id": "a", "title": "A"})
    assert len(result) == 1
    assert result[0].id == "a"


def test_parse_file_payload_bundle() -> None:
    bundle = [
        {"id": "a", "title": "A"},
        {"id": "b", "title": "B"},
        {"id": "c", "title": "C"},
    ]
    result = parse_template_file_payload(bundle)
    assert len(result) == 3
    assert [t.id for t in result] == ["a", "b", "c"]


def test_parse_file_payload_empty_bundle_is_zero_templates() -> None:
    """An empty bundle is degenerate but valid — zero templates, no error."""
    result = parse_template_file_payload([])
    assert result == []


def test_parse_file_payload_rejects_scalar() -> None:
    """A scalar (string / int / None) is not a valid file shape."""
    with pytest.raises(TemplateCorruptError):
        parse_template_file_payload("just a string")  # type: ignore[arg-type]


def test_parse_file_payload_bundle_entry_must_be_mapping() -> None:
    with pytest.raises(TemplateCorruptError):
        parse_template_file_payload(
            [
                {"id": "a", "title": "A"},
                "not a mapping",
            ]
        )


def test_parse_file_payload_includes_bundle_index_in_error() -> None:
    """When an entry is invalid, the error message names its index."""
    with pytest.raises(TemplateValidationError) as excinfo:
        parse_template_file_payload(
            [
                {"id": "a", "title": "A"},
                {"id": "b"},  # missing title
            ]
        )
    assert "#1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# parse_template_file — disk I/O
# ---------------------------------------------------------------------------


def test_parse_file_yaml_single(tmp_path: Path) -> None:
    path = tmp_path / "x.yml"
    path.write_text("id: x\ntitle: X\n", encoding="utf-8")
    result = parse_template_file(path)
    assert len(result) == 1
    assert result[0].id == "x"


def test_parse_file_yaml_bundle(tmp_path: Path) -> None:
    path = tmp_path / "agents.yml"
    path.write_text(
        "- id: a\n  title: A\n- id: b\n  title: B\n",
        encoding="utf-8",
    )
    result = parse_template_file(path)
    assert [t.id for t in result] == ["a", "b"]


def test_parse_file_json_single(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text('{"id": "x", "title": "X"}', encoding="utf-8")
    result = parse_template_file(path)
    assert result[0].id == "x"


def test_parse_file_yaml_extension_treated_as_yaml(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("id: x\ntitle: X\n", encoding="utf-8")
    result = parse_template_file(path)
    assert result[0].id == "x"


def test_parse_file_unknown_extension_treated_as_json(tmp_path: Path) -> None:
    """Files without .yml/.yaml/.json extensions are parsed as JSON."""
    path = tmp_path / "x.txt"
    path.write_text('{"id": "x", "title": "X"}', encoding="utf-8")
    result = parse_template_file(path)
    assert result[0].id == "x"


def test_parse_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(TemplateCorruptError):
        parse_template_file(tmp_path / "does-not-exist.yml")


def test_parse_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(TemplateCorruptError):
        parse_template_file(tmp_path)


def test_parse_file_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("not: valid: yaml: [", encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        parse_template_file(path)


def test_parse_file_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        parse_template_file(path)


def test_parse_file_empty_file_is_zero_templates(tmp_path: Path) -> None:
    """An empty YAML file (yaml.safe_load returns None) is zero templates."""
    path = tmp_path / "empty.yml"
    path.write_text("", encoding="utf-8")
    result = parse_template_file(path)
    assert result == []


def test_parse_file_strict_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "future.yml"
    path.write_text("id: x\ntitle: X\nfuture_key: ignored\n", encoding="utf-8")
    with pytest.raises(TemplateValidationError):
        parse_template_file(path, strict=True)


def test_parse_file_lenient_drops_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "future.yml"
    path.write_text("id: x\ntitle: X\nfuture_key: ignored\n", encoding="utf-8")
    result = parse_template_file(path, strict=False)
    assert result[0].id == "x"


def test_parse_file_rejects_non_mapping_non_list_top_level(
    tmp_path: Path,
) -> None:
    """A scalar at the top level is corrupt."""
    path = tmp_path / "scalar.yml"
    path.write_text('"just a string"\n', encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        parse_template_file(path)
