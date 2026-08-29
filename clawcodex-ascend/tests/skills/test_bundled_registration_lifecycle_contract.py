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

"""Failure-isolation and resource-root contracts for bundled skills."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

from clawcodex_ext.skills import bundled as bundled_catalog
from clawcodex_ext.skills import bundled_skills
from clawcodex_ext.skills.bundled_skills import (
    BundledSkillDefinition,
    clear_bundled_skills,
    get_bundled_skill_by_name,
    register_bundled_skill,
)


@pytest.fixture(autouse=True)
def _reset_bundled_lifecycle() -> Iterator[None]:
    clear_bundled_skills()
    yield
    clear_bundled_skills()


def test_one_failing_registrar_does_not_block_following_registrars_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    flaky_attempts = 0

    def flaky() -> None:
        nonlocal flaky_attempts
        flaky_attempts += 1
        calls.append("flaky")
        if flaky_attempts == 1:
            raise RuntimeError("transient registrar failure")

    def succeeding(name: str):
        def registrar() -> None:
            calls.append(name)

        return registrar

    monkeypatch.setattr(bundled_catalog, "register_simplify_skill", flaky)
    monkeypatch.setattr(bundled_catalog, "register_debug_skill", succeeding("debug"))
    monkeypatch.setattr(bundled_catalog, "register_loop_skill", succeeding("loop"))
    monkeypatch.setattr(
        bundled_catalog,
        "register_spec_audit_skill",
        succeeding("spec-audit"),
    )
    monkeypatch.setattr(bundled_catalog, "register_stuck_skill", succeeding("stuck"))
    monkeypatch.setattr(
        bundled_catalog,
        "register_verify_content_skill",
        succeeding("verify-content"),
    )
    monkeypatch.setattr(
        bundled_catalog,
        "register_update_config_skill",
        succeeding("update-config"),
    )
    monkeypatch.setattr(
        bundled_catalog,
        "register_orchestrator_skill",
        succeeding("orchestrator"),
    )

    with caplog.at_level(logging.WARNING, logger=bundled_catalog.__name__):
        assert bundled_catalog.init_bundled_skills() is False

    assert calls == [
        "flaky",
        "debug",
        "loop",
        "stuck",
        "verify-content",
        "update-config",
        "spec-audit",
        "orchestrator",
    ]
    assert "transient registrar failure" in caplog.text

    calls.clear()
    assert bundled_catalog.init_bundled_skills() is True
    assert flaky_attempts == 2
    assert calls[0] == "flaky"


@pytest.mark.parametrize(
    ("module_name", "registrar_name", "skill_name"),
    [
        ("simplify", "register_simplify_skill", "simplify"),
        ("debug", "register_debug_skill", "debug"),
        ("loop", "register_loop_skill", "loop"),
        ("orchestrator", "register_orchestrator_skill", "orchestrator"),
        ("spec_audit", "register_spec_audit_skill", "spec-audit"),
        ("stuck", "register_stuck_skill", "stuck"),
        ("verify_content", "register_verify_content_skill", "verify-content"),
        ("update_config", "register_update_config_skill", "update-config"),
    ],
)
def test_each_registrar_forwards_registry_rejection(
    module_name: str,
    registrar_name: str,
    skill_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module(f"clawcodex_ext.skills.bundled.{module_name}")
    received_names: list[str] = []

    def reject(definition: BundledSkillDefinition) -> bool:
        received_names.append(definition.name)
        return False

    monkeypatch.setattr(module, "register_bundled_skill", reject)

    assert getattr(module, registrar_name)() is False
    assert received_names == [skill_name]


def test_rejected_registrar_is_diagnosed_and_retried(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0

    def flaky() -> bool:
        nonlocal attempts
        attempts += 1
        return attempts > 1

    def succeeding() -> bool:
        return True

    flaky.__name__ = "register_simplify_skill"
    monkeypatch.setattr(bundled_catalog, "register_simplify_skill", flaky)
    monkeypatch.setattr(bundled_catalog, "register_debug_skill", succeeding)
    monkeypatch.setattr(bundled_catalog, "register_loop_skill", succeeding)
    monkeypatch.setattr(bundled_catalog, "register_orchestrator_skill", succeeding)
    monkeypatch.setattr(bundled_catalog, "register_spec_audit_skill", succeeding)
    monkeypatch.setattr(bundled_catalog, "register_stuck_skill", succeeding)
    monkeypatch.setattr(bundled_catalog, "register_verify_content_skill", succeeding)
    monkeypatch.setattr(bundled_catalog, "register_update_config_skill", succeeding)

    with caplog.at_level(logging.WARNING, logger=bundled_catalog.__name__):
        assert bundled_catalog.init_bundled_skills() is False

    assert "register_simplify_skill rejected its definition" in caplog.text
    assert bundled_catalog.init_bundled_skills() is True
    assert attempts == 2


def test_invalid_definition_is_diagnosed_and_same_name_can_be_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = f"retry-{uuid4().hex}"

    with caplog.at_level(logging.WARNING, logger=bundled_skills.__name__):
        accepted = register_bundled_skill(
            BundledSkillDefinition(
                name=name,
                description="",
                get_prompt_for_command=lambda _args: "invalid",
            )
        )

    assert accepted is False
    assert get_bundled_skill_by_name(name) is None
    assert f"bundled skill '{name}' rejected" in caplog.text
    assert "description" in caplog.text

    assert register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="valid after retry",
            get_prompt_for_command=lambda args: f"valid: {args}",
        )
    )
    recovered = get_bundled_skill_by_name(name)
    assert recovered is not None
    assert recovered.description == "valid after retry"
    assert recovered.get_prompt("argument") == "valid: argument"


def test_temp_resource_root_contains_product_version_and_process_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = "9.8.7-test"
    nonce = "fixed-process-nonce"
    monkeypatch.setattr(bundled_skills.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(bundled_skills, "_BUNDLED_VERSION", version)
    monkeypatch.setattr(bundled_skills, "_PROCESS_NONCE", nonce)
    monkeypatch.setattr(bundled_skills, "_bundled_skills_root", None)

    root = Path(bundled_skills.get_bundled_skills_root())

    assert root == tmp_path / "clawcodex" / "bundled-skills" / version / nonce
    assert root.is_dir()


def test_bundled_canonical_name_resolves_before_an_earlier_alias() -> None:
    assert register_bundled_skill(
        BundledSkillDefinition(
            name="alias-owner",
            description="owns an alias",
            aliases=["canonical-target"],
            get_prompt_for_command=lambda _args: "alias owner",
        )
    )
    assert register_bundled_skill(
        BundledSkillDefinition(
            name="canonical-target",
            description="owns the canonical name",
            get_prompt_for_command=lambda _args: "canonical owner",
        )
    )

    resolved = get_bundled_skill_by_name("canonical-target")
    assert resolved is not None
    assert resolved.name == "canonical-target"
