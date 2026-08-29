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

"""Integration tests for src/services/templates/bootstrap.py.

Covers :func:`bootstrap_default_templates` end-to-end: the function
that populates the default :class:`TemplateRegistry` from standard
discovery paths (user / project / managed) plus the P85-E built-in
catalogue. Uses ``tmp_path`` and ``monkeypatch`` to isolate the
filesystem from the host environment and from other tests.

P85-E/``bootstrap_default_templates`` now registers the 5 built-in
canonical agent templates plus 4 built-in orchestrator templates BEFORE
walking the user / project / managed dirs. The constant
:data:`_BOOTSTRAP_BASE_COUNT` captures that baseline so the count assertions
below can be expressed as ``_BOOTSTRAP_BASE_COUNT + N`` instead of bare
``N`` - keeping the tests honest about the post-P85-E/contract
without sprinkling magic numbers through the file.
"""
# pylint: disable=redefined-outer-name

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.templates import (
    CLAWCODEX_CONFIG_DIR_ENV,
    CLAWCODEX_MANAGED_CONFIG_DIR_ENV,
    PROJECT_CONFIG_DIR,
    SOURCE_MANAGED,
    SOURCE_PROJECT,
    SOURCE_USER,
    TEMPLATES_SUBDIR,
    bootstrap_default_templates,
    get_built_in_templates,
    get_default_template_registry,
    reset_default_template_registry,
)

# Number of canonical templates registered by :func:`bootstrap_default_templates`
# before any on-disk source: 5 agent templates plus 4 orchestrator templates.
_BOOTSTRAP_BASE_COUNT: int = len(get_built_in_templates()) + 4


@pytest.fixture(autouse=True)
def _isolated_default_registry() -> None:
    """Reset the default registry around every test so cross-test
    pollution cannot leak template entries.
    """
    reset_default_template_registry()
    yield
    reset_default_template_registry()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the CLAWCODEX_* env vars so default-resolution tests
    are not affected by ambient values from the host shell.
    """
    monkeypatch.delenv(CLAWCODEX_CONFIG_DIR_ENV, raising=False)
    monkeypatch.delenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, raising=False)


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point Path.home() at a tmp dir so the unconfigured fallback
    resolves to a directory we control.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    return home


def _write_template(path: Path, *, id_: str, title: str, fields: dict | None = None) -> Path:
    """Write a minimal valid YAML template file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields_block = ""
    if fields:
        fields_lines = "\n".join(f"  {k}: {_yaml_repr(v)}" for k, v in fields.items())
        fields_block = f"\nfields:\n{fields_lines}"
    path.write_text(
        f"id: {id_}\ntitle: {title}{fields_block}\n",
        encoding="utf-8",
    )
    return path


def _yaml_repr(value: object) -> str:
    """Minimal YAML serializer for primitives used in tests."""
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_repr(v) for v in value) + "]"
    if isinstance(value, str):
        return value  # trust the caller for safe test strings
    return repr(value)


# ---------------------------------------------------------------------------
# Basic discovery (one source at a time)
# ---------------------------------------------------------------------------


def test_bootstrap_loads_user_dir_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    tpl_dir = user_dir / TEMPLATES_SUBDIR
    _write_template(tpl_dir / "user_tpl.yml", id_="user_tpl", title="From user")
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT + 1
    registry = get_default_template_registry()
    assert "user_tpl" in registry
    assert registry.get("user_tpl").source == SOURCE_USER


def test_bootstrap_loads_project_dir_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proj = tmp_path / "proj"
    tpl_dir = proj / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR
    _write_template(tpl_dir / "proj_tpl.yml", id_="proj_tpl", title="From project")

    n = bootstrap_default_templates(cwd=proj)
    assert n == _BOOTSTRAP_BASE_COUNT + 1
    registry = get_default_template_registry()
    assert "proj_tpl" in registry
    assert registry.get("proj_tpl").source == SOURCE_PROJECT


def test_bootstrap_loads_managed_dir_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mgr_dir = tmp_path / "mgr"
    tpl_dir = mgr_dir / TEMPLATES_SUBDIR
    _write_template(tpl_dir / "mgr_tpl.yml", id_="mgr_tpl", title="From managed")
    monkeypatch.setenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, str(mgr_dir))

    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT + 1
    registry = get_default_template_registry()
    assert "mgr_tpl" in registry
    assert registry.get("mgr_tpl").source == SOURCE_MANAGED


def test_bootstrap_loads_from_all_three_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    proj = tmp_path / "proj"
    mgr_dir = tmp_path / "mgr"
    _write_template(user_dir / TEMPLATES_SUBDIR / "u.yml", id_="u", title="U")
    _write_template(proj / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "p.yml", id_="p", title="P")
    _write_template(mgr_dir / TEMPLATES_SUBDIR / "m.yml", id_="m", title="M")
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))
    monkeypatch.setenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, str(mgr_dir))

    n = bootstrap_default_templates(cwd=proj)
    assert n == _BOOTSTRAP_BASE_COUNT + 3
    registry = get_default_template_registry()
    assert {"u", "p", "m"}.issubset(set(registry.list_ids()))


# ---------------------------------------------------------------------------
# Precedence (highest priority last, last-wins)
# ---------------------------------------------------------------------------


def test_project_overrides_user_same_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    proj = tmp_path / "proj"
    _write_template(
        user_dir / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="From user",
        fields={"tools": ["Read"]},
    )
    _write_template(
        proj / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="From project",
        fields={"tools": ["Write"]},
    )
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    bootstrap_default_templates(cwd=proj)
    tpl = get_default_template_registry().get("shared")
    assert tpl.fields["tools"] == ["Write"]
    assert tpl.source == SOURCE_PROJECT


def test_managed_overrides_project_same_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proj = tmp_path / "proj"
    mgr_dir = tmp_path / "mgr"
    _write_template(
        proj / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="From project",
        fields={"tools": ["Write"]},
    )
    _write_template(
        mgr_dir / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="From managed",
        fields={"tools": ["Bash"]},
    )
    monkeypatch.setenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, str(mgr_dir))

    bootstrap_default_templates(cwd=proj)
    tpl = get_default_template_registry().get("shared")
    assert tpl.fields["tools"] == ["Bash"]
    assert tpl.source == SOURCE_MANAGED


def test_managed_overrides_user_same_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    mgr_dir = tmp_path / "mgr"
    _write_template(
        user_dir / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="From user",
        fields={"tools": ["Read"]},
    )
    _write_template(
        mgr_dir / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="From managed",
        fields={"tools": ["Bash"]},
    )
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))
    monkeypatch.setenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, str(mgr_dir))

    bootstrap_default_templates()
    tpl = get_default_template_registry().get("shared")
    assert tpl.fields["tools"] == ["Bash"]
    assert tpl.source == SOURCE_MANAGED


# ---------------------------------------------------------------------------
# Missing dirs - must NOT raise
# ---------------------------------------------------------------------------


def test_bootstrap_silent_when_user_dir_missing(
    clean_env: None,
    fake_home: Path,
) -> None:
    """No CLAWCODEX_CONFIG_DIR and no ~/.clawcodex/templates/ -> only built-ins."""
    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT
    assert len(get_default_template_registry()) == _BOOTSTRAP_BASE_COUNT


def test_bootstrap_silent_when_project_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """cwd has no .clawcodex/templates at any ancestor -> only built-ins."""
    proj = tmp_path / "proj"
    proj.mkdir()
    n = bootstrap_default_templates(cwd=proj)
    assert n == _BOOTSTRAP_BASE_COUNT


def test_bootstrap_silent_when_managed_dir_missing(
    clean_env: None,
) -> None:
    """No CLAWCODEX_MANAGED_CONFIG_DIR and no /etc/clawcodex/templates/
    -> built-ins still register (no raise even if /etc is not writable).
    """
    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT


def test_bootstrap_silent_when_all_dirs_missing(
    clean_env: None,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """All three sources absent -> only built-ins register, no raise."""
    proj = tmp_path / "proj"
    proj.mkdir()
    n = bootstrap_default_templates(cwd=proj)
    assert n == _BOOTSTRAP_BASE_COUNT
    assert len(get_default_template_registry()) == _BOOTSTRAP_BASE_COUNT


# ---------------------------------------------------------------------------
# Idempotency + source tagging
# ---------------------------------------------------------------------------


def test_bootstrap_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    _write_template(user_dir / TEMPLATES_SUBDIR / "x.yml", id_="x", title="X")
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    first = bootstrap_default_templates()
    second = bootstrap_default_templates()
    assert first == _BOOTSTRAP_BASE_COUNT + 1
    assert second == _BOOTSTRAP_BASE_COUNT + 1  # not double-counted
    assert len(get_default_template_registry()) == _BOOTSTRAP_BASE_COUNT + 1


def test_bootstrap_tags_source_attribute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every discovered template's .source reflects the dir it came from."""
    user_dir = tmp_path / "user"
    proj = tmp_path / "proj"
    mgr_dir = tmp_path / "mgr"
    _write_template(user_dir / TEMPLATES_SUBDIR / "u.yml", id_="u", title="U")
    _write_template(proj / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "p.yml", id_="p", title="P")
    _write_template(mgr_dir / TEMPLATES_SUBDIR / "m.yml", id_="m", title="M")
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))
    monkeypatch.setenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, str(mgr_dir))

    bootstrap_default_templates(cwd=proj)
    registry = get_default_template_registry()
    assert registry.get("u").source == SOURCE_USER
    assert registry.get("p").source == SOURCE_PROJECT
    assert registry.get("m").source == SOURCE_MANAGED


# ---------------------------------------------------------------------------
# Multi-level project walker integration
# ---------------------------------------------------------------------------


def test_bootstrap_walks_multiple_project_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Innermost project dir wins over outermost project dir."""
    outer = tmp_path / "outer"
    middle = outer / "middle"
    inner = middle / "inner"
    inner.mkdir(parents=True)
    _write_template(
        outer / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="outer",
        fields={"tools": ["Read"]},
    )
    _write_template(
        inner / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="inner",
        fields={"tools": ["Bash"]},
    )

    bootstrap_default_templates(cwd=inner)
    tpl = get_default_template_registry().get("shared")
    assert tpl.fields["tools"] == ["Bash"]
    assert tpl.source == SOURCE_PROJECT


def test_bootstrap_does_not_cross_git_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Templates in a parent-of-repo .clawcodex/templates must not leak."""
    repo = tmp_path / "repo"
    nested = repo / "src"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    # Parent-of-repo has its own templates/ that must NOT appear.
    _write_template(
        tmp_path / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "outside.yml",
        id_="outside",
        title="outside the repo",
    )
    # Inside the repo, the nested dir has its own.
    _write_template(
        nested / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR / "inside.yml",
        id_="inside",
        title="inside the repo",
    )

    bootstrap_default_templates(cwd=nested)
    registry = get_default_template_registry()
    assert "inside" in registry
    assert "outside" not in registry


# ---------------------------------------------------------------------------
# overwrite=False (lower-priority keeps when higher-priority absent)
# ---------------------------------------------------------------------------


def test_bootstrap_with_overwrite_false_does_not_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When overwrite=False and a same-id template already exists in
    the default registry, a higher-priority source does NOT replace it.
    """
    registry = get_default_template_registry()
    registry.register(
        _template(id_="shared", title="pre-existing", fields={"tools": ["Read"]}),
    )

    user_dir = tmp_path / "user"
    _write_template(
        user_dir / TEMPLATES_SUBDIR / "shared.yml",
        id_="shared",
        title="from user",
        fields={"tools": ["Bash"]},
    )
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    bootstrap_default_templates(overwrite=False)
    tpl = registry.get("shared")
    # Pre-existing template kept its identity.
    assert tpl.fields["tools"] == ["Read"]
    assert tpl.title == "pre-existing"


def _template(*, id_: str, title: str, fields: dict | None = None):
    """Local helper to build a Template via the public constructor."""
    from src.services.templates import Template

    return Template(id=id_, title=title, fields=fields or {})


# ---------------------------------------------------------------------------
# Direct return-value contract
# ---------------------------------------------------------------------------


def test_bootstrap_returns_total_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The return value reflects total templates in the default registry."""
    user_dir = tmp_path / "user"
    for i in range(3):
        _write_template(
            user_dir / TEMPLATES_SUBDIR / f"t{i}.yml",
            id_=f"t{i}",
            title=f"T{i}",
        )
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT + 3
    assert n == len(get_default_template_registry())


def test_bootstrap_returns_count_includes_pre_existing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pre-existing entries in the default registry count toward the
    returned total - bootstrap does NOT clear first.
    """
    registry = get_default_template_registry()
    registry.register(_template(id_="preset", title="preset"))
    assert len(registry) == 1

    user_dir = tmp_path / "user"
    _write_template(user_dir / TEMPLATES_SUBDIR / "fresh.yml", id_="fresh", title="fresh")
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT + 2
    assert {"preset", "fresh"}.issubset(set(registry.list_ids()))


# ---------------------------------------------------------------------------
# Bundle files (P85-A extension): a single YAML can register multiple templates
# ---------------------------------------------------------------------------


def test_bootstrap_loads_bundle_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    bundle_path = user_dir / TEMPLATES_SUBDIR / "agents.yml"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        "- id: alpha\n  title: Alpha\n- id: beta\n  title: Beta\n- id: gamma\n  title: Gamma\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT + 3
    assert {"alpha", "beta", "gamma"}.issubset(set(get_default_template_registry().list_ids()))


# ---------------------------------------------------------------------------
# Skips corrupt files silently
# ---------------------------------------------------------------------------


def test_bootstrap_skips_corrupt_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    tpl_dir = user_dir / TEMPLATES_SUBDIR
    tpl_dir.mkdir(parents=True)
    (tpl_dir / "bad.yml").write_text("not: valid: yaml: [", encoding="utf-8")
    _write_template(tpl_dir / "good.yml", id_="good", title="Good")
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(user_dir))

    n = bootstrap_default_templates()
    assert n == _BOOTSTRAP_BASE_COUNT + 1
    assert "good" in get_default_template_registry()
    assert "bad" not in get_default_template_registry()
