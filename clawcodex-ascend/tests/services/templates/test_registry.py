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

"""Template registry tests: register / get / list / discover / concurrency."""
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.services.templates import (
    Template,
    TemplateAlreadyExistsError,
    TemplateNotFoundError,
    TemplateRegistry,
)


def _tpl(tid: str = "x", **overrides) -> Template:
    defaults: dict = {"id": tid, "title": f"Template {tid}"}
    defaults.update(overrides)
    return Template(**defaults)


# ---------------------------------------------------------------------------
# Construction + size + membership
# ---------------------------------------------------------------------------


def test_empty_registry() -> None:
    r = TemplateRegistry()
    assert len(r) == 0
    assert list(r) == []
    assert "x" not in r


def test_store_path_property_preserved() -> None:
    r = TemplateRegistry(store_path=Path("/tmp/store.json"))
    assert r.store_path == Path("/tmp/store.json")


def test_search_dir_property_preserved() -> None:
    r = TemplateRegistry(search_dir=Path("/tmp/templates"))
    assert r.search_dir == Path("/tmp/templates")


def test_store_path_accepts_string() -> None:
    r = TemplateRegistry(store_path="/tmp/store.json")
    assert r.store_path == Path("/tmp/store.json")


# ---------------------------------------------------------------------------
# register / get / unregister
# ---------------------------------------------------------------------------


def test_register_and_get() -> None:
    r = TemplateRegistry()
    t = _tpl("a")
    r.register(t)
    assert r.get("a") is t


def test_register_rejects_non_template() -> None:
    r = TemplateRegistry()
    with pytest.raises(TypeError):
        r.register({"id": "x", "title": "x"})  # type: ignore[arg-type]


def test_register_rejects_duplicate_by_default() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a"))
    with pytest.raises(TemplateAlreadyExistsError):
        r.register(_tpl("a"))


def test_register_overwrite_when_flagged() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a", title="first"))
    r.register(_tpl("a", title="second"), overwrite=True)
    assert r.get("a").title == "second"


def test_unregister_removes_template() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a"))
    r.unregister("a")
    assert "a" not in r


def test_unregister_missing_raises() -> None:
    r = TemplateRegistry()
    with pytest.raises(TemplateNotFoundError):
        r.unregister("missing")


def test_get_missing_raises() -> None:
    r = TemplateRegistry()
    with pytest.raises(TemplateNotFoundError):
        r.get("missing")


def test_try_get_returns_none_for_missing() -> None:
    r = TemplateRegistry()
    assert r.try_get("missing") is None


def test_try_get_returns_template_for_present() -> None:
    r = TemplateRegistry()
    t = _tpl("a")
    r.register(t)
    assert r.try_get("a") is t


# ---------------------------------------------------------------------------
# register_many
# ---------------------------------------------------------------------------


def test_register_many_adds_all_when_no_duplicates() -> None:
    r = TemplateRegistry()
    added = r.register_many([_tpl("a"), _tpl("b"), _tpl("c")])
    assert added == 3
    assert len(r) == 3


def test_register_many_skips_duplicates() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a"))
    added = r.register_many([_tpl("a"), _tpl("b")])
    assert added == 1
    assert len(r) == 2


def test_register_many_with_overwrite_succeeds_on_existing() -> None:
    """overwrite=True is a flat "last write wins" — no collision raised."""
    r = TemplateRegistry()
    r.register(_tpl("a", title="first"))
    added = r.register_many([_tpl("a", title="second"), _tpl("b")], overwrite=True)
    assert added == 2
    assert r.get("a").title == "second"
    assert "b" in r


# ---------------------------------------------------------------------------
# list_templates / list_ids
# ---------------------------------------------------------------------------


def test_list_templates_sorted_by_id() -> None:
    r = TemplateRegistry()
    r.register(_tpl("c"))
    r.register(_tpl("a"))
    r.register(_tpl("b"))
    ids = [t.id for t in r.list_templates()]
    assert ids == ["a", "b", "c"]


def test_list_templates_filters_by_source() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a", source="built-in"))
    r.register(_tpl("b", source="user"))
    r.register(_tpl("c", source="built-in"))
    built_in = r.list_templates(source="built-in")
    assert {t.id for t in built_in} == {"a", "c"}


def test_list_ids_returns_string_list() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a"))
    r.register(_tpl("b"))
    assert r.list_ids() == ["a", "b"]


def test_clear_empties_registry() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a"))
    r.register(_tpl("b"))
    r.clear()
    assert len(r) == 0


# ---------------------------------------------------------------------------
# discover (directory scan)
# ---------------------------------------------------------------------------


def test_discover_requires_search_dir() -> None:
    r = TemplateRegistry()
    with pytest.raises(ValueError):
        r.discover()


def test_discover_raises_without_search_dir(tmp_path: Path) -> None:
    r = TemplateRegistry()
    with pytest.raises(ValueError):
        r.discover()


def test_discover_returns_zero_when_dir_missing(tmp_path: Path) -> None:
    r = TemplateRegistry(search_dir=tmp_path / "missing")
    assert r.discover() == 0


def test_discover_loads_yaml_files(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    sub.mkdir()
    (sub / "general.yml").write_text(
        "id: general\ntitle: General\nfields:\n  tools: [Read, Bash]\n",
        encoding="utf-8",
    )
    (sub / "explore.yaml").write_text(
        "id: explore\ntitle: Explore\n",
        encoding="utf-8",
    )
    r = TemplateRegistry(search_dir=sub)
    added = r.discover()
    assert added == 2
    assert set(r.list_ids()) == {"general", "explore"}
    assert r.get("general").fields["tools"] == ["Read", "Bash"]


def test_discover_loads_json_files(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    sub.mkdir()
    (sub / "fix.json").write_text('{"id": "fix", "title": "Fix bug"}', encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    added = r.discover()
    assert added == 1
    assert r.get("fix").title == "Fix bug"


def test_discover_skips_corrupt_files(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    sub.mkdir()
    (sub / "bad.yml").write_text("not: valid: yaml: [", encoding="utf-8")
    (sub / "good.yml").write_text("id: good\ntitle: Good\n", encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    added = r.discover()
    assert added == 1
    assert r.list_ids() == ["good"]


def test_discover_applies_source_override(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    sub.mkdir()
    (sub / "x.yml").write_text("id: x\ntitle: X\n", encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    r.discover(source="project")
    assert r.get("x").source == "project"


def test_discover_skips_already_registered(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    sub.mkdir()
    (sub / "a.yml").write_text("id: a\ntitle: A\n", encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    r.register(_tpl("a"))
    added = r.discover()
    assert added == 0


def test_discover_recursive_finds_nested(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    nested = sub / "deep" / "deeper"
    nested.mkdir(parents=True)
    (nested / "x.yml").write_text("id: x\ntitle: X\n", encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    added = r.discover()
    assert added == 1
    assert r.list_ids() == ["x"]


def test_discover_non_recursive_skips_nested(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    nested = sub / "deep"
    nested.mkdir(parents=True)
    (nested / "x.yml").write_text("id: x\ntitle: X\n", encoding="utf-8")
    (sub / "y.yml").write_text("id: y\ntitle: Y\n", encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    added = r.discover(recursive=False)
    assert added == 1
    assert r.list_ids() == ["y"]


def test_discover_skips_non_template_files(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    sub.mkdir()
    (sub / "readme.md").write_text("# readme", encoding="utf-8")
    (sub / "x.yml").write_text("id: x\ntitle: X\n", encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    added = r.discover()
    assert added == 1


def test_discover_skips_files_missing_id_or_title(tmp_path: Path) -> None:
    sub = tmp_path / "templates"
    sub.mkdir()
    (sub / "no_id.yml").write_text("title: Missing id\n", encoding="utf-8")
    (sub / "no_title.yml").write_text("id: notitle\n", encoding="utf-8")
    (sub / "good.yml").write_text("id: good\ntitle: Good\n", encoding="utf-8")
    r = TemplateRegistry(search_dir=sub)
    added = r.discover()
    assert added == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_register_uniques_preserved() -> None:
    r = TemplateRegistry()
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            r.register(_tpl(f"t{i:03d}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(r) == 50


def test_concurrent_get_returns_consistent_view() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a", title="A"))
    seen: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(50):
            t = r.get("a")
            with lock:
                seen.append(t.title)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(t == "A" for t in seen)
    assert len(seen) == 500


# ---------------------------------------------------------------------------
# Containment + iteration
# ---------------------------------------------------------------------------


def test_contains_returns_true_for_registered() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a"))
    assert "a" in r
    assert "b" not in r


def test_iter_yields_template_ids() -> None:
    r = TemplateRegistry()
    r.register(_tpl("a"))
    r.register(_tpl("b"))
    assert set(iter(r)) == {"a", "b"}
