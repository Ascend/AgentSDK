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

"""Template resolver tests: base + override merge semantics."""

from __future__ import annotations

import pytest

from src.services.templates import (
    Template,
    TemplateRegistry,
    TemplateResolutionError,
    TemplateResolver,
)


def _tpl(**overrides) -> Template:
    defaults: dict = {
        "id": "base",
        "title": "Base template",
        "fields": {
            "tools": ["Read", "Bash"],
            "max_turns": 25,
            "model": "claude-sonnet-4-6",
            "permission_mode": "default",
            "nested": {"a": 1, "b": 2},
        },
    }
    defaults.update(overrides)
    return Template(**defaults)


# ---------------------------------------------------------------------------
# No override
# ---------------------------------------------------------------------------


def test_no_override_returns_base_fields() -> None:
    t = _tpl()
    r = TemplateResolver().resolve(t)
    assert r.fields["max_turns"] == 25
    assert r.fields["model"] == "claude-sonnet-4-6"
    assert r.shadow_keys == []
    assert r.template_id == "base"
    assert r.base_template is t


def test_no_override_none_passes() -> None:
    r = TemplateResolver().resolve(_tpl(), None)
    assert r.fields["max_turns"] == 25


def test_empty_override_returns_base_fields() -> None:
    r = TemplateResolver().resolve(_tpl(), {})
    assert r.fields["max_turns"] == 25
    assert r.shadow_keys == []


# ---------------------------------------------------------------------------
# Scalar override
# ---------------------------------------------------------------------------


def test_scalar_override_replaces_value() -> None:
    r = TemplateResolver().resolve(_tpl(), {"max_turns": 5})
    assert r.fields["max_turns"] == 5
    assert r.fields["model"] == "claude-sonnet-4-6"  # unchanged


def test_falsy_scalar_override_honored() -> None:
    """Override with False / 0 / '' must be applied; only absence skips."""
    r = TemplateResolver().resolve(
        _tpl(fields={"flag": True, "count": 1, "name": "x"}),
        {"flag": False, "count": 0, "name": ""},
    )
    assert r.fields["flag"] is False
    assert r.fields["count"] == 0
    assert r.fields["name"] == ""


def test_override_with_none_sets_value_to_none() -> None:
    """Override value of None is honoured literally (does NOT delete the key).

    Only key *absence* signals "no override"; explicit ``None`` is a real
    value the caller chose. This matches the documented "falsy is allowed"
    semantics in the resolver docstring.
    """
    r = TemplateResolver().resolve(
        _tpl(fields={"a": 1, "b": 2}),
        {"b": None},
    )
    assert r.fields == {"a": 1, "b": None}


# ---------------------------------------------------------------------------
# List override (replace, not append)
# ---------------------------------------------------------------------------


def test_list_override_replaces_wholesale() -> None:
    r = TemplateResolver().resolve(_tpl(), {"tools": ["Read"]})
    assert r.fields["tools"] == ["Read"]


def test_list_override_empty_list_replaces() -> None:
    r = TemplateResolver().resolve(_tpl(), {"tools": []})
    assert r.fields["tools"] == []


# ---------------------------------------------------------------------------
# Dict override (deep merge)
# ---------------------------------------------------------------------------


def test_dict_override_deep_merges() -> None:
    r = TemplateResolver().resolve(_tpl(), {"nested": {"b": 99, "c": 3}})
    assert r.fields["nested"] == {"a": 1, "b": 99, "c": 3}


def test_dict_override_replaces_with_scalars() -> None:
    r = TemplateResolver().resolve(_tpl(), {"nested": "stringified"})
    assert r.fields["nested"] == "stringified"


# ---------------------------------------------------------------------------
# Shadow keys
# ---------------------------------------------------------------------------


def test_shadow_keys_lists_undeclared_overrides() -> None:
    r = TemplateResolver().resolve(
        _tpl(fields={"a": 1}),
        {"unknown_field": "x", "another": True},
    )
    assert set(r.shadow_keys) == {"unknown_field", "another"}
    assert r.fields["unknown_field"] == "x"  # still carried through


def test_shadow_keys_sorted() -> None:
    r = TemplateResolver().resolve(_tpl(fields={}), {"z": 1, "a": 2, "m": 3})
    assert r.shadow_keys == ["a", "m", "z"]


def test_no_shadow_when_all_keys_declared() -> None:
    r = TemplateResolver().resolve(_tpl(), {"max_turns": 10})
    assert r.shadow_keys == []


# ---------------------------------------------------------------------------
# resolve_from_registry
# ---------------------------------------------------------------------------


def test_resolve_from_registry() -> None:
    r = TemplateRegistry()
    r.register(_tpl())
    resolver = TemplateResolver()
    out = resolver.resolve_from_registry(r, "base", {"max_turns": 7})
    assert out.fields["max_turns"] == 7
    assert out.template_id == "base"


def test_resolve_from_registry_missing_raises() -> None:
    r = TemplateRegistry()
    with pytest.raises(Exception):
        TemplateResolver().resolve_from_registry(r, "missing")


def test_resolve_from_registry_rejects_non_registry() -> None:
    with pytest.raises(TypeError):
        TemplateResolver().resolve_from_registry("not a registry", "x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_resolve_rejects_non_template() -> None:
    with pytest.raises(TypeError):
        TemplateResolver().resolve({"id": "x", "title": "x"})  # type: ignore[arg-type]


def test_resolve_rejects_non_mapping_override() -> None:
    with pytest.raises(TemplateResolutionError):
        TemplateResolver().resolve(_tpl(), [("a", 1)])  # type: ignore[arg-type]


def test_resolve_rejects_non_string_override_key() -> None:
    with pytest.raises(TemplateResolutionError):
        TemplateResolver().resolve(_tpl(), {1: "v"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# ResolvedTemplate convenience
# ---------------------------------------------------------------------------


def test_resolved_template_get_with_default() -> None:
    r = TemplateResolver().resolve(_tpl())
    assert r.get("max_turns") == 25
    assert r.get("missing") is None
    assert r.get("missing", "fallback") == "fallback"


def test_resolved_template_is_frozen() -> None:
    r = TemplateResolver().resolve(_tpl())
    with pytest.raises(Exception):  # FrozenInstanceError subclass
        r.fields = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration with registry
# ---------------------------------------------------------------------------


def test_resolve_chain_base_then_override_then_override() -> None:
    """Simulate base -> user override -> project override."""
    r = TemplateRegistry()
    r.register(_tpl())
    resolver = TemplateResolver()
    first = resolver.resolve_from_registry(r, "base", {"max_turns": 5})
    second = resolver.resolve(_tpl(fields=first.fields), {"model": "x"})
    assert second.fields["max_turns"] == 5
    assert second.fields["model"] == "x"
