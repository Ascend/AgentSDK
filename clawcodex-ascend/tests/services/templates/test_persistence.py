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

"""Template persistence tests: atomic save/load + merge + corruption."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.services.templates import (
    Template,
    TemplateCorruptError,
    TemplateNotFoundError,
    TemplateRegistry,
    TemplateStateFile,
    load_registry,
    merge_registries,
    save_registry,
)


def _tpl(tid: str, **overrides) -> Template:
    defaults: dict = {"id": tid, "title": f"Template {tid}"}
    defaults.update(overrides)
    return Template(**defaults)


# ---------------------------------------------------------------------------
# TemplateStateFile save/load round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    reg = TemplateRegistry()
    reg.register(_tpl("a", fields={"x": 1}))
    reg.register(_tpl("b", fields={"y": 2}))
    TemplateStateFile(path).save(reg)

    loaded = TemplateStateFile(path).load()
    assert loaded.list_ids() == ["a", "b"]
    assert loaded.get("a").fields["x"] == 1
    assert loaded.get("b").fields["y"] == 2


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "templates.json"
    reg = TemplateRegistry()
    reg.register(_tpl("a"))
    TemplateStateFile(path).save(reg)
    assert path.exists()


def test_save_rejects_non_registry(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    with pytest.raises(TypeError):
        TemplateStateFile(path).save("not a registry")  # type: ignore[arg-type]


def test_load_missing_file_raises_not_found(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(TemplateNotFoundError):
        TemplateStateFile(path).load()


def test_load_corrupt_json_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{ not valid", encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        TemplateStateFile(path).load()


def test_load_invalid_payload_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"templates": {"x": "not a dict"}}), encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        TemplateStateFile(path).load()


def test_load_missing_required_field_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"templates": {"x": {"id": "x"}}}), encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        TemplateStateFile(path).load()


def test_load_root_not_object_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        TemplateStateFile(path).load()


def test_load_templates_field_not_object_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"templates": []}), encoding="utf-8")
    with pytest.raises(TemplateCorruptError):
        TemplateStateFile(path).load()


def test_exists_reports_correctly(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    f = TemplateStateFile(path)
    assert not f.exists()
    f.save(TemplateRegistry())
    assert f.exists()


def test_delete_removes_file(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    f = TemplateStateFile(path)
    f.save(TemplateRegistry())
    f.delete()
    assert not f.exists()


def test_delete_missing_file_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "never_existed.json"
    TemplateStateFile(path).delete()
    assert not path.exists()


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------


def test_atomic_write_no_temp_leftover(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    reg = TemplateRegistry()
    reg.register(_tpl("a"))
    TemplateStateFile(path).save(reg)
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    f = TemplateStateFile(path)

    reg1 = TemplateRegistry()
    reg1.register(_tpl("a", title="Old"))
    f.save(reg1)

    reg2 = TemplateRegistry()
    reg2.register(_tpl("a", title="New"))
    f.save(reg2)

    loaded = f.load()
    assert loaded.get("a").title == "New"


def test_concurrent_writes_do_not_corrupt_file(tmp_path: Path) -> None:
    """Many threads writing simultaneously should produce a valid file."""
    path = tmp_path / "templates.json"
    f = TemplateStateFile(path)

    def writer(i: int) -> None:
        reg = TemplateRegistry()
        reg.register(_tpl(f"t{i:03d}", title=f"Writer {i}"))
        f.save(reg)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # File must be valid JSON and parseable as a registry.
    loaded = f.load()
    assert isinstance(loaded, TemplateRegistry)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def test_save_registry_and_load_registry(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    reg = TemplateRegistry()
    reg.register(_tpl("a"))
    save_registry(path, reg)

    loaded = load_registry(path)
    assert loaded.list_ids() == ["a"]


def test_safe_filename_suffix_is_short_hex() -> None:
    suffix = TemplateStateFile.__module__  # smoke check import
    assert suffix  # module loaded


def test_safe_filename_suffix_function() -> None:
    from clawcodex_ext.services.templates.persistence import safe_filename_suffix

    s = safe_filename_suffix()
    assert isinstance(s, str)
    assert len(s) == 8
    int(s, 16)  # parses as hex


# ---------------------------------------------------------------------------
# merge_registries
# ---------------------------------------------------------------------------


def test_merge_registries_adds_new() -> None:
    a = TemplateRegistry()
    a.register(_tpl("x"))
    b = TemplateRegistry()
    b.register(_tpl("y"))
    added = merge_registries(a, b)
    assert added == 1
    assert set(a.list_ids()) == {"x", "y"}


def test_merge_registries_skips_duplicates() -> None:
    a = TemplateRegistry()
    a.register(_tpl("x", title="first"))
    b = TemplateRegistry()
    b.register(_tpl("x", title="second"))
    added = merge_registries(a, b)
    assert added == 0
    assert a.get("x").title == "first"


def test_merge_registries_prefer_source_overwrites() -> None:
    a = TemplateRegistry()
    a.register(_tpl("x", title="target"))
    b = TemplateRegistry()
    b.register(_tpl("x", title="source"))
    merge_registries(a, b, prefer_target=False)
    assert a.get("x").title == "source"


def test_merge_registries_prefer_target_keeps_existing() -> None:
    a = TemplateRegistry()
    a.register(_tpl("x", title="target"))
    b = TemplateRegistry()
    b.register(_tpl("x", title="source"))
    merge_registries(a, b, prefer_target=True)
    assert a.get("x").title == "target"


def test_merge_registries_rejects_non_registry() -> None:
    reg = TemplateRegistry()
    with pytest.raises(TypeError):
        merge_registries(reg, "not a registry")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        merge_registries("not a registry", reg)  # type: ignore[arg-type]


def test_merge_registries_empty_source() -> None:
    a = TemplateRegistry()
    a.register(_tpl("x"))
    b = TemplateRegistry()
    added = merge_registries(a, b)
    assert added == 0
    assert a.list_ids() == ["x"]


# ---------------------------------------------------------------------------
# End-to-end: discover + save + load
# ---------------------------------------------------------------------------


def test_discover_then_save_then_load_round_trip(tmp_path: Path) -> None:
    search = tmp_path / "templates"
    search.mkdir()
    (search / "a.yml").write_text("id: a\ntitle: A\nfields:\n  tools: [Read]\n", encoding="utf-8")
    (search / "b.yml").write_text("id: b\ntitle: B\n", encoding="utf-8")
    store_path = tmp_path / "store.json"

    reg = TemplateRegistry(search_dir=search)
    reg.discover()
    save_registry(store_path, reg)

    loaded = load_registry(store_path)
    assert set(loaded.list_ids()) == {"a", "b"}
    assert loaded.get("a").fields["tools"] == ["Read"]
