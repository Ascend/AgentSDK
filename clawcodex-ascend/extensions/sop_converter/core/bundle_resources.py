#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Declarative resource lifecycle bindings for SOP bundles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def normalize_resource_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


@dataclass(frozen=True)
class ResourceBinding:
    """One explicit create/invoke resource pair from ``resources.yaml``."""

    resource_type: str
    create: str
    invoke: str
    handle_field: str = "id"

    @property
    def normalized_resource_type(self) -> str:
        return normalize_resource_type(self.resource_type)

    def matches_create(self, *references: str) -> bool:
        return self._matches(self.create, references)

    def matches_invoke(self, *references: str) -> bool:
        return self._matches(self.invoke, references)

    @staticmethod
    def _matches(expected: str, references: tuple[str, ...]) -> bool:
        target = normalize_resource_type(expected)
        if not target:
            return False
        return any(normalize_resource_type(reference) == target for reference in references)


def _sidecar_path(bundle_path: Path) -> Path | None:
    candidates = (
        bundle_path / ".clawcodex" / "resources.yaml",
        bundle_path / ".clawcodex" / "resources.yml",
    )
    return next((path for path in candidates if path.is_file()), None)


def load_resource_bindings(bundle_path: str | Path | None) -> list[ResourceBinding]:
    """Load a bundle's resource overrides, returning an empty list if absent."""
    if bundle_path is None:
        return []
    path = _sidecar_path(Path(bundle_path).expanduser().resolve())
    if path is None:
        return []

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, dict) or not isinstance(raw.get("resources", []), list):
        raise ValueError(f"{path}: top-level 'resources' must be a list")

    bindings: list[ResourceBinding] = []
    for index, item in enumerate(raw.get("resources", []), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: resources[{index}] must be an object")
        resource_type = str(item.get("type") or item.get("resource_type") or "").strip()
        create = str(item.get("create") or "").strip()
        invoke = str(item.get("invoke") or "").strip()
        handle_field = str(item.get("handle_field") or "id").strip() or "id"
        if not resource_type or not create or not invoke:
            raise ValueError(f"{path}: resources[{index}] requires type, create, and invoke")
        bindings.append(
            ResourceBinding(
                resource_type=resource_type,
                create=create,
                invoke=invoke,
                handle_field=handle_field,
            )
        )
    return bindings


__all__ = ["ResourceBinding", "load_resource_bindings", "normalize_resource_type"]
