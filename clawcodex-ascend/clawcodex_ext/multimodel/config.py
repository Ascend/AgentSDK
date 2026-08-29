#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Persistent configuration and selection precedence."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

VALID_STRATEGIES = frozenset({"parallel", "voting", "routing", "fallback"})
VALID_AGGREGATORS = frozenset({"passthrough", "first_success", "majority", "rank", "scoring", "fusion"})


class MultiModelConfigError(ValueError):
    """Raised when multi-model configuration is invalid."""


@dataclass(frozen=True)
class SlotConfig:
    name: str
    provider: str
    model: str
    weight: float = 1.0
    timeout_ms: int = 120_000
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "weight": self.weight,
            "timeout_ms": self.timeout_ms,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class RouteConfig:
    """A portable keyword route: any matching term selects ``slot``."""

    pattern: str
    slot: str

    def to_dict(self) -> dict[str, str]:
        return {"pattern": self.pattern, "slot": self.slot}


@dataclass(frozen=True)
class GroupConfig:
    strategy: str
    slots: tuple[SlotConfig, ...]
    aggregator: str | None = None
    max_concurrent: int = 5
    min_votes: int | None = None
    scorer_provider: str = "openai"
    scorer_model: str = "gpt-4o"
    routes: tuple[RouteConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "strategy": self.strategy,
            "max_concurrent": self.max_concurrent,
            "slots": [slot.to_dict() for slot in self.slots],
        }
        if self.aggregator:
            data["aggregator"] = self.aggregator
        if self.min_votes is not None:
            data["min_votes"] = self.min_votes
        if self.aggregator in {"scoring", "rank", "fusion"}:
            data["scorer_provider"] = self.scorer_provider
            data["scorer_model"] = self.scorer_model
        if self.routes:
            data["routes"] = [route.to_dict() for route in self.routes]
        return data


@dataclass(frozen=True)
class MultiModelConfig:
    default_group: str = ""
    groups: dict[str, GroupConfig] = field(default_factory=dict)


def default_config_path() -> Path:
    """Return the user config path, with a test-friendly environment override."""
    root = Path(os.environ.get("CLAWCODEX_CONFIG_DIR", "~/.clawcodex")).expanduser()
    return root / "config.yaml"


def parse_slot(spec: str) -> SlotConfig:
    """Parse ``name:model@provider[,weight=N][,timeout_ms=N]``."""
    if not isinstance(spec, str) or not spec.strip():
        raise MultiModelConfigError("slot must not be empty")
    pieces = [piece.strip() for piece in spec.split(",")]
    identity = pieces[0]
    try:
        name, target = identity.split(":", 1)
        model, provider = target.rsplit("@", 1)
    except ValueError as exc:
        raise MultiModelConfigError("slot must have the form name:model@provider[,weight=N][,timeout_ms=N]") from exc
    if not name or not model or not provider:
        raise MultiModelConfigError("slot name, model, and provider are required")
    values: dict[str, Any] = {"weight": 1.0, "timeout_ms": 120_000, "enabled": True}
    for option in pieces[1:]:
        if "=" not in option:
            raise MultiModelConfigError(f"invalid slot option: {option}")
        key, value = (part.strip() for part in option.split("=", 1))
        if key == "weight":
            try:
                values[key] = float(value)
            except ValueError as exc:
                raise MultiModelConfigError("slot weight must be a number") from exc
        elif key == "timeout_ms":
            try:
                values[key] = int(value)
            except ValueError as exc:
                raise MultiModelConfigError("slot timeout_ms must be an integer") from exc
        else:
            raise MultiModelConfigError(f"unsupported slot option: {key}")
    return _validate_slot(SlotConfig(name=name, provider=provider, model=model, **values))


def _validate_slot(slot: SlotConfig) -> SlotConfig:
    if not slot.name or not slot.provider or not slot.model:
        raise MultiModelConfigError("slot name, provider, and model are required")
    if slot.weight <= 0:
        raise MultiModelConfigError("slot weight must be greater than zero")
    if slot.timeout_ms <= 0:
        raise MultiModelConfigError("slot timeout_ms must be greater than zero")
    return slot


def validate_group(group: GroupConfig) -> GroupConfig:
    if group.strategy not in VALID_STRATEGIES:
        raise MultiModelConfigError(f"unknown strategy: {group.strategy}")
    if not group.slots:
        raise MultiModelConfigError("a model group needs at least one slot")
    names = [slot.name for slot in group.slots]
    if len(names) != len(set(names)):
        raise MultiModelConfigError("slot names must be unique")
    for slot in group.slots:
        _validate_slot(slot)
    names = set(names)
    for route in group.routes:
        if not route.pattern.strip() or route.slot not in names:
            raise MultiModelConfigError("each route needs a non-empty pattern and an existing slot")
    if group.aggregator is not None and group.aggregator not in VALID_AGGREGATORS:
        raise MultiModelConfigError(f"unknown aggregator: {group.aggregator}")
    if group.max_concurrent <= 0:
        raise MultiModelConfigError("max_concurrent must be greater than zero")
    if group.min_votes is not None and group.min_votes <= 0:
        raise MultiModelConfigError("min_votes must be greater than zero")
    return group


def _group_from_mapping(value: Mapping[str, Any]) -> GroupConfig:
    raw_slots = value.get("slots", [])
    if not isinstance(raw_slots, list):
        raise MultiModelConfigError("group slots must be a list")
    slots = tuple(
        _validate_slot(
            SlotConfig(
                name=str(raw["name"]),
                provider=str(raw["provider"]),
                model=str(raw["model"]),
                weight=float(raw.get("weight", 1.0)),
                timeout_ms=int(raw.get("timeout_ms", 120_000)),
                enabled=bool(raw.get("enabled", True)),
            )
        )
        for raw in raw_slots
    )
    raw_routes = value.get("routes", [])
    if not isinstance(raw_routes, list):
        raise MultiModelConfigError("group routes must be a list")
    routes = tuple(
        RouteConfig(str(item.get("pattern", "")), str(item.get("slot", "")))
        for item in raw_routes
        if isinstance(item, Mapping)
    )
    return validate_group(
        GroupConfig(
            strategy=str(value.get("strategy", "parallel")),
            slots=slots,
            aggregator=value.get("aggregator"),
            max_concurrent=int(value.get("max_concurrent", 5)),
            min_votes=(int(value["min_votes"]) if value.get("min_votes") is not None else None),
            scorer_provider=str(value.get("scorer_provider", "openai")),
            scorer_model=str(value.get("scorer_model", "gpt-4o")),
            routes=routes,
        )
    )


def load_config(path: Path | None = None) -> MultiModelConfig:
    path = path or default_config_path()
    if not path.exists():
        return MultiModelConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise MultiModelConfigError(f"cannot read {path}: {exc}") from exc
    section = raw.get("multimodel", {})
    if not isinstance(section, Mapping):
        raise MultiModelConfigError("multimodel must be a mapping")
    groups_raw = section.get("groups", {})
    if not isinstance(groups_raw, Mapping):
        raise MultiModelConfigError("multimodel.groups must be a mapping")
    groups = {str(name): _group_from_mapping(value) for name, value in groups_raw.items() if isinstance(value, Mapping)}
    default_group = str(section.get("default_group", ""))
    if default_group and default_group not in groups:
        raise MultiModelConfigError(f"default_group '{default_group}' does not exist")
    return MultiModelConfig(default_group=default_group, groups=groups)


def save_config(config: MultiModelConfig, path: Path | None = None) -> Path:
    path = path or default_config_path()
    if config.default_group and config.default_group not in config.groups:
        raise MultiModelConfigError(f"default_group '{config.default_group}' does not exist")
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise MultiModelConfigError(f"cannot read {path}: {exc}") from exc
    raw = deepcopy(raw)
    raw["multimodel"] = {
        "default_group": config.default_group,
        "groups": {name: group.to_dict() for name, group in config.groups.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def resolve_active_group(
    *, cli_group: str | None = None, runtime_group: str | None = None, config: MultiModelConfig | None = None
) -> str:
    """Apply the documented precedence: CLI > runtime > config default."""
    return cli_group or runtime_group or (config or load_config()).default_group
