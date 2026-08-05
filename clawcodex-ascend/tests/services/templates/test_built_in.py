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

"""Unit tests for :mod:`src.services.templates.built_in` (P85-E).

Covers the canonical built-in template catalogue: its shape, the
``register_built_in_templates`` helper, and the integration with the
default-registry bootstrap flow. P85-E adds five built-ins
(``general-purpose`` / ``explore`` / ``plan`` / ``fix`` / ``review``)
that every install starts with, and this test suite pins that
contract.
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

from src.services.templates import (
    SOURCE_BUILT_IN,
    Template,
    TemplateAlreadyExistsError,
    TemplateRegistry,
    bootstrap_default_templates,
    get_built_in_templates,
    get_default_template_registry,
    register_built_in_templates,
    reset_default_template_registry,
)
from clawcodex_ext.services.templates.built_in import (
    _READ_ONLY_DISALLOWED,
)


# ---------------------------------------------------------------------------
# Catalogue shape — the 5 canonical templates ship
# ---------------------------------------------------------------------------


def test_catalogue_has_exactly_five_templates() -> None:
    """P85-E ships exactly five built-in templates."""
    catalogue = get_built_in_templates()
    assert len(catalogue) == 5


def test_catalogue_ids_are_canonical_set() -> None:
    """The five ids are exactly the documented ones, in any order."""
    catalogue = get_built_in_templates()
    assert {t.id for t in catalogue} == {
        "general-purpose",
        "explore",
        "plan",
        "fix",
        "review",
    }


def test_catalogue_ids_are_alphabetical() -> None:
    """The catalogue is ordered alphabetically by id for deterministic
    display in ``/template list`` output.
    """
    catalogue = get_built_in_templates()
    ids = [t.id for t in catalogue]
    assert ids == sorted(ids)


def test_catalogue_returns_immutable_tuple() -> None:
    """``get_built_in_templates`` returns a tuple — callers cannot
    accidentally mutate the catalogue. Note: CPython's ``tuple()``
    constructor may return the same object for tuple inputs, but the
    result is still immutable so the safety contract holds.
    """
    a = get_built_in_templates()
    assert isinstance(a, tuple)
    # Tuples are immutable — a runtime mutation attempt raises.
    try:
        a[0] = "mutated"  # type: ignore[index]
    except TypeError:
        return
    raise AssertionError("catalogue tuple is mutable")


def test_catalogue_templates_are_frozen_dataclass_instances() -> None:
    """All built-ins are :class:`Template` dataclass instances (frozen)."""
    for tpl in get_built_in_templates():
        assert isinstance(tpl, Template)


def test_every_built_in_has_source_built_in() -> None:
    """Every built-in is tagged ``source=SOURCE_BUILT_IN`` so the
    ``/template list --source built-in`` filter works.
    """
    for tpl in get_built_in_templates():
        assert tpl.source == SOURCE_BUILT_IN


def test_every_built_in_has_id_title_and_metadata() -> None:
    """Every built-in satisfies the contract: id + title + non-empty metadata."""
    for tpl in get_built_in_templates():
        assert tpl.id
        assert tpl.title
        assert isinstance(tpl.metadata, dict)
        # Every built-in carries a category tag so a future TUI picker
        # can group without parsing prose.
        assert "category" in tpl.metadata


# ---------------------------------------------------------------------------
# Per-template semantics — each one encodes a distinct agent shape
# ---------------------------------------------------------------------------


def test_general_purpose_has_full_tool_access() -> None:
    gp = next(t for t in get_built_in_templates() if t.id == "general-purpose")
    assert gp.fields["tools"] == ["*"]
    assert gp.fields["permission_mode"] == "acceptEdits"


def test_explore_is_read_only() -> None:
    """Explore forbids every mutation-capable tool."""
    explore = next(t for t in get_built_in_templates() if t.id == "explore")
    assert explore.fields["disallowed_tools"] == list(_READ_ONLY_DISALLOWED)
    # Confirms _READ_ONLY_DISALLOWED matches the documented set.
    assert "Edit" in explore.fields["disallowed_tools"]
    assert "Write" in explore.fields["disallowed_tools"]
    assert "Bash" not in explore.fields["disallowed_tools"]
    assert "Read" not in explore.fields["disallowed_tools"]


def test_plan_is_read_only_and_inherits_parent_model() -> None:
    plan = next(t for t in get_built_in_templates() if t.id == "plan")
    assert plan.fields["disallowed_tools"] == list(_READ_ONLY_DISALLOWED)
    assert plan.fields["model"] == "inherit"


def test_fix_has_full_tools_and_bounded_turn_budget() -> None:
    fix = next(t for t in get_built_in_templates() if t.id == "fix")
    assert fix.fields["tools"] == ["*"]
    assert fix.fields["permission_mode"] == "acceptEdits"
    assert isinstance(fix.fields["max_turns"], int)
    assert fix.fields["max_turns"] > 0


def test_review_is_read_only_with_generous_turn_budget() -> None:
    review = next(t for t in get_built_in_templates() if t.id == "review")
    assert review.fields["disallowed_tools"] == list(_READ_ONLY_DISALLOWED)
    assert isinstance(review.fields["max_turns"], int)
    assert review.fields["max_turns"] > 0


# ---------------------------------------------------------------------------
# register_built_in_templates — the helper used by bootstrap
# ---------------------------------------------------------------------------


def test_register_built_in_templates_into_empty_registry() -> None:
    """First call into an empty registry adds all five templates."""
    registry = TemplateRegistry()
    added = register_built_in_templates(registry)
    assert added == 5
    assert len(registry) == 5


def test_register_built_in_templates_is_idempotent() -> None:
    """Re-registering into the same registry adds zero new entries
    (the templates are already present).
    """
    registry = TemplateRegistry()
    register_built_in_templates(registry)
    added = register_built_in_templates(registry)
    assert added == 0
    assert len(registry) == 5


def test_register_built_in_templates_overwrites_existing() -> None:
    """When ``overwrite=True`` (the default), an existing entry is
    replaced with the canonical built-in.
    """
    registry = TemplateRegistry()
    # Plant a non-canonical version first.
    registry.register(
        Template(
            id="plan",
            title="user override",
            fields={"permission_mode": "default"},
            source="user",
        )
    )
    assert registry.get("plan").title == "user override"

    register_built_in_templates(registry, overwrite=True)
    # The pre-existing plan was overwritten with the canonical version.
    assert registry.get("plan").title != "user override"
    # The other 4 built-ins are new additions.
    assert len(registry) == 5


def test_register_built_in_templates_overwrite_false_keeps_existing() -> None:
    """When ``overwrite=False``, the pre-existing entry wins and the
    other 4 (not pre-existing) built-ins are still added.
    """
    registry = TemplateRegistry()
    pre_existing = Template(
        id="plan",
        title="user override",
        fields={"permission_mode": "default"},
        source="user",
    )
    registry.register(pre_existing)

    added = register_built_in_templates(registry, overwrite=False)
    # The other 4 built-ins were added (they weren't in the registry).
    assert added == 4
    assert len(registry) == 5
    # The pre-existing entry is untouched.
    assert registry.get("plan").title == "user override"


def test_register_built_in_templates_only_counts_new_entries() -> None:
    """When 2 of 5 already exist, the count is 3."""
    registry = TemplateRegistry()
    registry.register(Template(id="plan", title="P"))
    registry.register(Template(id="fix", title="F"))
    added = register_built_in_templates(registry)
    assert added == 3
    assert len(registry) == 5


def test_register_built_in_templates_skips_already_exists_silently() -> None:
    """``TemplateAlreadyExistsError`` is caught silently — the
    function never raises even when an id collides.
    """
    registry = TemplateRegistry()
    registry.register(Template(id="explore", title="custom"))
    # Should not raise; the other 4 are added.
    added = register_built_in_templates(registry, overwrite=False)
    assert added == 4
    # Pre-existing entry untouched.
    assert registry.get("explore").title == "custom"


# ---------------------------------------------------------------------------
# Bootstrap integration — built-ins are first, others can shadow them
# ---------------------------------------------------------------------------


def test_bootstrap_registers_built_ins_first(monkeypatch, tmp_path) -> None:
    """With all sources absent, the built-in agent + orchestrator templates are present."""
    reset_default_template_registry()
    n = bootstrap_default_templates(cwd=tmp_path)
    assert n == 9
    registry = get_default_template_registry()
    assert {t.id for t in registry.list_templates(source=SOURCE_BUILT_IN)} == {
        "general-purpose",
        "explore",
        "plan",
        "fix",
        "review",
        "orchestrator-workflow",
        "orchestrator-workflow-local",
        "orchestrator-workflow-yaml",
        "orchestrator-issue-card",
    }


def test_bootstrap_user_template_can_shadow_built_in(monkeypatch, tmp_path) -> None:
    """A user template with the same id as a built-in wins because
    the user source is registered AFTER the built-in.
    """

    from src.services.templates import (
        CLAWCODEX_CONFIG_DIR_ENV,
        TEMPLATES_SUBDIR,
    )

    user_dir = tmp_path / "user"
    tpl_dir = user_dir / TEMPLATES_SUBDIR
    tpl_dir.mkdir(parents=True)
    (tpl_dir / "plan.yml").write_text(
        "id: plan\ntitle: My Custom Plan\nfields:\n  max_turns: 5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    reset_default_template_registry()
    bootstrap_default_templates()

    registry = get_default_template_registry()
    plan = registry.get("plan")
    assert plan.title == "My Custom Plan"
    assert plan.fields["max_turns"] == 5
    assert plan.source == "user"


def test_bootstrap_built_ins_tagged_as_built_in(monkeypatch, tmp_path) -> None:
    """The built-in catalogue entries are tagged with SOURCE_BUILT_IN
    in the default registry after bootstrap.
    """
    reset_default_template_registry()
    bootstrap_default_templates(cwd=tmp_path)
    registry = get_default_template_registry()
    built_ins = registry.list_templates(source=SOURCE_BUILT_IN)
    assert len(built_ins) == 9
    assert {"orchestrator-workflow", "orchestrator-issue-card"}.issubset({template.id for template in built_ins})


def test_bootstrap_returns_total_including_built_ins(monkeypatch, tmp_path) -> None:
    """The return value counts built-ins alongside on-disk sources."""
    from src.services.templates import CLAWCODEX_CONFIG_DIR_ENV, TEMPLATES_SUBDIR

    user_dir = tmp_path / "user"
    tpl_dir = user_dir / TEMPLATES_SUBDIR
    tpl_dir.mkdir(parents=True)
    (tpl_dir / "x.yml").write_text("id: x\ntitle: X\n", encoding="utf-8")
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    reset_default_template_registry()
    n = bootstrap_default_templates()
    # 5 agent built-ins + 4 orchestrator templates + 1 user template.
    assert n == 10
    assert n == len(get_default_template_registry())


def test_bootstrap_idempotent_with_built_ins(monkeypatch, tmp_path) -> None:
    """A second bootstrap call does not double-register the built-ins."""
    reset_default_template_registry()
    first = bootstrap_default_templates(cwd=tmp_path)
    second = bootstrap_default_templates(cwd=tmp_path)
    # The built-in count stays stable across re-runs because of overwrite.
    assert first == 9
    assert second == 9
    assert len(get_default_template_registry()) == 9


# ---------------------------------------------------------------------------
# Built-in dataclass immutability — Template is frozen
# ---------------------------------------------------------------------------


def test_built_in_template_is_frozen() -> None:
    """A built-in :class:`Template` cannot be mutated in place."""
    import dataclasses

    tpl = get_built_in_templates()[0]
    assert dataclasses.is_dataclass(tpl)
    # frozen=True raises FrozenInstanceError on attribute assignment.
    try:
        tpl.title = "mutated"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("built-in Template is not frozen")


# ---------------------------------------------------------------------------
# Pre-existing TemplateAlreadyExistsError import is reachable
# ---------------------------------------------------------------------------


def test_template_already_exists_error_is_exported() -> None:
    """Sanity check: the exception used by the helper is in the public API."""
    assert TemplateAlreadyExistsError is not None
