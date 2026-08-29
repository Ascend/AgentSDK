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

"""PlanStore tests: atomic writes, corruption, concurrency, list/delete."""
# pylint: disable=redefined-builtin, use-implicit-booleaness-not-comparison

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.services.ultraplan import (
    Plan,
    PlanNotFoundError,
    PlanStatus,
    PlanStore,
    SubPlan,
)


def _plan(id: str = "p1") -> Plan:
    return Plan(
        id=id,
        title="My plan",
        goal="Achieve X",
        sub_plans=[SubPlan(id="sp1", title="A", description="d")],
    )


def test_store_creates_data_dir(tmp_path: Path) -> None:
    target = tmp_path / "plans"
    PlanStore(target)
    assert target.exists() and target.is_dir()


def test_store_save_writes_json_file(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    plan = _plan()
    path = store.save(plan)
    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["id"] == "p1"


def test_store_load_round_trip(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    plan = _plan()
    store.save(plan)
    loaded = store.load("p1")
    assert loaded == plan


def test_store_load_missing_raises(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    with pytest.raises(PlanNotFoundError):
        store.load("nope")


def test_store_load_invalid_json_raises_corrupt(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    (tmp_path / "p1.json").write_text("{ this is not json", encoding="utf-8")
    from src.services.ultraplan import PlanCorruptError

    with pytest.raises(PlanCorruptError):
        store.load("p1")


def test_store_load_invalid_structure_raises_corrupt(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    (tmp_path / "p1.json").write_text(
        json.dumps({"id": "p1", "title": "x", "goal": "x", "sub_plans": "not-a-list"}),
        encoding="utf-8",
    )
    from src.services.ultraplan import PlanCorruptError

    with pytest.raises(PlanCorruptError):
        store.load("p1")


def test_store_delete_removes_file(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    store.save(_plan("p1"))
    assert store.exists("p1")
    store.delete("p1")
    assert not store.exists("p1")


def test_store_delete_silent_on_missing(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    store.delete("never-existed")  # must not raise


def test_store_list_plans_returns_saved(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    store.save(_plan("p1"))
    store.save(_plan("p2"))
    plans = store.list_plans()
    ids = [pid for pid, _, _ in plans]
    assert "p1" in ids and "p2" in ids
    assert all(s is PlanStatus.DRAFT for _, s, _ in plans)


def test_store_list_skips_corrupted_files(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    store.save(_plan("p1"))
    (tmp_path / "broken.json").write_text("not json at all", encoding="utf-8")
    plans = store.list_plans()
    ids = [pid for pid, _, _ in plans]
    assert "p1" in ids
    assert "broken" not in ids


def test_store_save_uses_atomic_replace(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    store.save(_plan("p1"))
    # There should be no leftover .tmp files in the directory.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_store_concurrent_writes_do_not_corrupt(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    n = 50
    barrier = threading.Barrier(n)

    def writer(i: int) -> None:
        barrier.wait()
        plan = _plan(f"p{i}")
        store.save(plan)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    plans = store.list_plans()
    assert len(plans) == n
    # No leftover tmp files.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_store_save_replaces_existing(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    p1 = _plan("p1")
    store.save(p1)
    p1.title = "Updated"
    store.save(p1)
    loaded = store.load("p1")
    assert loaded.title == "Updated"


def test_store_transaction_lock_serialization(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    order: list[str] = []

    def worker(name: str) -> None:
        with store.transaction():
            order.append(f"{name}-enter")
            # Yield so other threads can attempt to enter the lock.
            import time

            time.sleep(0.01)
            order.append(f"{name}-exit")

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Interleaving must be enter-exit-enter-exit... no nested enters.
    assert len(order) == 6
    for i in range(0, 6, 2):
        assert order[i].endswith("-enter")
        assert order[i + 1].endswith("-exit")
        assert order[i].split("-")[0] == order[i + 1].split("-")[0]


def test_store_data_dir_is_path_or_string(tmp_path: Path) -> None:
    store1 = PlanStore(tmp_path)
    store2 = PlanStore(str(tmp_path))
    store1.save(_plan("p1"))
    # Both should reference the same underlying directory.
    assert store2.exists("p1")


def test_store_save_then_load_preserves_status(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    plan = _plan()
    plan.status = PlanStatus.ACTIVE
    store.save(plan)
    assert store.load("p1").status is PlanStatus.ACTIVE


def test_store_persists_nested_step_criteria(tmp_path: Path) -> None:
    from src.services.ultraplan import (
        AcceptanceCriteria,
        CheckKind,
        Step,
        StepKind,
    )

    step = Step(
        id="s1",
        title="T",
        description="D",
        kind=StepKind.IMPLEMENT,
        criteria=[
            AcceptanceCriteria(
                id="c1",
                description="file exists",
                kind=CheckKind.FILE_EXISTS,
                target="/tmp/example",
            )
        ],
    )
    sp = SubPlan(id="sp1", title="A", description="d", steps=[step])
    plan = Plan(id="p1", title="My plan", goal="Goal", sub_plans=[sp])
    store = PlanStore(tmp_path)
    store.save(plan)
    loaded = store.load("p1")
    assert loaded.sub_plans[0].steps[0].criteria[0].target == "/tmp/example"


def test_store_rejects_non_plan_save(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    with pytest.raises(TypeError):
        store.save({"id": "p1", "title": "x", "goal": "x"})  # type: ignore[arg-type]


def test_store_rejects_empty_id_load(tmp_path: Path) -> None:
    store = PlanStore(tmp_path)
    with pytest.raises(ValueError):
        store.load("")  # type: ignore[arg-type]


def test_store_handles_oserror_on_read(tmp_path: Path, monkeypatch) -> None:
    store = PlanStore(tmp_path)
    plan = _plan()
    store.save(plan)
    # Force read_text to raise OSError to verify it is mapped to PlanCorruptError.
    from src.services.ultraplan import PlanCorruptError

    def boom(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(Path, "read_text", boom)
    with pytest.raises(PlanCorruptError):
        store.load("p1")
