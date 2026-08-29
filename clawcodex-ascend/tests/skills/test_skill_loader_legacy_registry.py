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

"""Compatibility contracts for the legacy mutable skill registry."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from clawcodex_ext.skills import bundled, loader, mcp_skill_builders
from clawcodex_ext.skills.model import Skill


@pytest.fixture(autouse=True)
def _clear_legacy_registry() -> Iterator[None]:
    loader.clear_skill_registry()
    yield
    loader.clear_skill_registry()


def test_discover_all_skills_does_not_mutate_legacy_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = Skill(name="legacy", description="legacy")
    discovered = Skill(name="discovered", description="discovered")
    loader.clear_skill_registry()
    loader._skill_registry[legacy.name] = legacy

    monkeypatch.delenv("CLAWCODEX_MANAGED_SKILLS_DIR", raising=False)
    monkeypatch.setattr(loader, "get_skill_dir_commands", lambda *_args, **_kwargs: [discovered])
    monkeypatch.setattr(loader, "_legacy_user_skill_dirs", lambda _path: [])
    monkeypatch.setattr(loader, "_legacy_project_skill_dirs", lambda _path: [])
    monkeypatch.setattr(loader, "get_registered_bundled_skills", lambda: [])
    monkeypatch.setattr(loader, "get_dynamic_skills", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mcp_skill_builders, "get_mcp_skill_builders", lambda: {})

    result = loader.discover_all_skills(project_root=tmp_path)

    assert result == [discovered]
    assert loader.get_registered_skill("legacy") is legacy
    assert loader.get_registered_skill("discovered") is None


def test_get_all_skills_replaces_legacy_registry_atomically_for_api_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = Skill(name="old", description="old")
    fresh = Skill(name="fresh", description="fresh")
    update_started = threading.Event()
    allow_update = threading.Event()
    read_started = threading.Event()
    read_finished = threading.Event()

    class BlockingRegistry(dict[str, Skill]):
        def update(self, *args: Any, **kwargs: Any) -> None:
            update_started.set()
            if not allow_update.wait(timeout=2):
                raise TimeoutError("test did not release registry update")
            super().update(*args, **kwargs)

    registry = BlockingRegistry({old.name: old})
    monkeypatch.setattr(loader, "_skill_registry", registry)
    monkeypatch.setattr(bundled, "init_bundled_skills", lambda: True)
    monkeypatch.setattr(loader, "discover_all_skills", lambda **_kwargs: [fresh])

    writer_errors: list[BaseException] = []
    reader_errors: list[BaseException] = []
    read_result: list[Skill | None] = []

    def write_registry() -> None:
        try:
            loader.get_all_skills(project_root=tmp_path)
        except BaseException as exc:  # pragma: no cover - surfaced below
            writer_errors.append(exc)

    def read_registry() -> None:
        try:
            read_started.set()
            read_result.append(loader.get_registered_skill("fresh"))
        except BaseException as exc:  # pragma: no cover - surfaced below
            reader_errors.append(exc)
        finally:
            read_finished.set()

    writer = threading.Thread(target=write_registry)
    reader = threading.Thread(target=read_registry)
    writer.start()
    assert update_started.wait(timeout=2)

    # The writer has cleared the dict and is paused in update while holding
    # the registry lock. API readers must wait instead of observing that gap.
    reader.start()
    assert read_started.wait(timeout=2)
    assert not read_finished.wait(timeout=0.05)

    allow_update.set()
    writer.join(timeout=2)
    reader.join(timeout=2)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_errors == []
    assert reader_errors == []
    assert read_result == [fresh]
    assert dict(registry) == {"fresh": fresh}


def test_discovery_reserves_bundled_names_and_orders_all_disk_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_shared = Skill(
        name="shared",
        description="project",
        loaded_from="project",
    )
    project_reserved = Skill(
        name="reserved",
        description="project shadow",
        loaded_from="project",
    )
    user_shared = Skill(
        name="shared",
        description="legacy user",
        loaded_from="user",
    )
    bundled_reserved = Skill(
        name="reserved",
        description="bundled",
        loaded_from="bundled",
        source="bundled",
    )

    monkeypatch.delenv("CLAWCODEX_MANAGED_SKILLS_DIR", raising=False)
    monkeypatch.setattr(
        loader,
        "get_skill_dir_commands",
        lambda *_args, **_kwargs: [project_shared, project_reserved],
    )
    monkeypatch.setattr(loader, "_legacy_user_skill_dirs", lambda _path: [tmp_path / "user"])
    monkeypatch.setattr(loader, "_legacy_project_skill_dirs", lambda _path: [])

    def load_dirs(_dirs, *, source: str, loaded_from: str, **_kwargs: object):
        del source
        return [user_shared] if loaded_from == "user" else []

    monkeypatch.setattr(loader, "_load_dirs_as", load_dirs)
    monkeypatch.setattr(
        loader,
        "get_registered_bundled_skills",
        lambda: [bundled_reserved],
    )
    monkeypatch.setattr(loader, "get_dynamic_skills", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mcp_skill_builders, "get_mcp_skill_builders", lambda: {})

    discovered = {
        skill.name: skill
        for skill in loader.discover_all_skills(
            project_root=tmp_path,
            user_skills_dir=tmp_path / "user",
        )
    }

    assert discovered["shared"] is user_shared
    assert discovered["reserved"] is bundled_reserved
