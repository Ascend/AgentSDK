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

"""Data model for the ``tool-dependencies.yaml`` schema.

Pure dataclasses — no IO, no detection logic.  Downstream readers
(task guide, system prompt) accept instances of these classes and
serialize them themselves.

Schema layout:

.. code-block:: yaml

    version: 1
    dependencies:
      - from: agentbuilder-build-agent
        to: invoke-existing-agent
        shared_params: [agent_id]
        hidden_steps:
          - action: persist_agent_catalog
            description: ...
        lifecycle: create → invoke
    intent_groups:
      agent_lifecycle:
        description: "..."
        tools: [...]
        primary_entry: agentbuilder-build-agent
    priority_routes:
      - keywords: ["create agent", ...]
        intent_group: agent_lifecycle
        entry_first: true

The on-disk YAML keys use ``from`` / ``to`` (not ``from_tool`` /
``to_tool``) for ergonomics; the dataclass fields keep the unambiguous
names.
"""

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HiddenStep:
    """A runtime step that occurs between two Agent-visible tools."""

    action: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action, "description": self.description}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HiddenStep":
        return cls(
            action=str(payload.get("action", "")),
            description=str(payload.get("description", "")),
        )


@dataclass
class ToolDependency:
    """A directed edge ``from_tool`` → ``to_tool`` in the tool lifecycle graph."""

    from_tool: str
    to_tool: str
    shared_params: list[str] = field(default_factory=list)
    hidden_steps: list[HiddenStep] = field(default_factory=list)
    lifecycle: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_tool,
            "to": self.to_tool,
            "shared_params": list(self.shared_params),
            "hidden_steps": [s.to_dict() for s in self.hidden_steps],
            "lifecycle": self.lifecycle,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolDependency":
        steps_raw = payload.get("hidden_steps") or []
        steps = [HiddenStep.from_dict(step) for step in steps_raw if isinstance(step, dict)]
        return cls(
            from_tool=str(payload.get("from") or payload.get("from_tool") or ""),
            to_tool=str(payload.get("to") or payload.get("to_tool") or ""),
            shared_params=[str(p) for p in payload.get("shared_params", [])],
            hidden_steps=steps,
            lifecycle=str(payload.get("lifecycle", "")),
        )


@dataclass
class IntentGroup:
    """A set of tools that share a single semantic intent.

    The ``name`` field becomes the YAML map key, so it must be a stable
    identifier (e.g. ``agent_lifecycle``).
    """

    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    primary_entry: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "tools": list(self.tools),
            "primary_entry": self.primary_entry,
        }

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "IntentGroup":
        return cls(
            name=name,
            description=str(payload.get("description", "")),
            tools=[str(t) for t in payload.get("tools", [])],
            primary_entry=(str(payload["primary_entry"]) if payload.get("primary_entry") else None),
        )


@dataclass
class PriorityRoute:
    """A keyword match that biases ToolSearch toward an :class:`IntentGroup`."""

    keywords: list[str] = field(default_factory=list)
    intent_group: str = ""
    entry_first: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "intent_group": self.intent_group,
            "entry_first": self.entry_first,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PriorityRoute":
        return cls(
            keywords=[str(k) for k in payload.get("keywords", [])],
            intent_group=str(payload.get("intent_group", "")),
            entry_first=bool(payload.get("entry_first", True)),
        )


@dataclass
class ToolDependencyGraph:
    """Top-level container persisted to ``tool-dependencies.yaml``."""

    version: int = 1
    dependencies: list[ToolDependency] = field(default_factory=list)
    intent_groups: list[IntentGroup] = field(default_factory=list)
    priority_routes: list[PriorityRoute] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_intent_group(self, name: str) -> IntentGroup | None:
        for group in self.intent_groups:
            if group.name == name:
                return group
        return None

    def dependencies_from(self, tool_name: str) -> list[ToolDependency]:
        return [d for d in self.dependencies if d.from_tool == tool_name]

    def dependencies_to(self, tool_name: str) -> list[ToolDependency]:
        return [d for d in self.dependencies if d.to_tool == tool_name]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "intent_groups": {g.name: g.to_dict() for g in self.intent_groups},
            "priority_routes": [r.to_dict() for r in self.priority_routes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolDependencyGraph":
        deps_raw = payload.get("dependencies") or []
        deps = [ToolDependency.from_dict(d) for d in deps_raw if isinstance(d, dict)]
        groups_raw = payload.get("intent_groups") or {}
        groups: list[IntentGroup] = []
        if isinstance(groups_raw, dict):
            for name, body in groups_raw.items():
                if isinstance(body, dict):
                    groups.append(IntentGroup.from_dict(str(name), body))
        elif isinstance(groups_raw, list):
            for body in groups_raw:
                if isinstance(body, dict) and body.get("name"):
                    groups.append(IntentGroup.from_dict(str(body["name"]), body))
        routes_raw = payload.get("priority_routes") or []
        routes = [PriorityRoute.from_dict(r) for r in routes_raw if isinstance(r, dict)]
        version = payload.get("version", 1)
        try:
            version = int(version)
        except (TypeError, ValueError):
            version = 1
        return cls(
            version=version,
            dependencies=deps,
            intent_groups=groups,
            priority_routes=routes,
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def merge_overrides(self, override: "ToolDependencyGraph | None") -> None:
        """Merge an override graph on top of ``self`` in place.

        Override rules:
        * ``dependencies`` are appended; same ``(from, to)`` pair is replaced.
        * ``intent_groups`` with the same name replace the existing one.
        * ``priority_routes`` are appended.
        Unknown fields are ignored silently.
        """
        if override is None:
            return
        existing_pairs = {(d.from_tool, d.to_tool) for d in self.dependencies}
        for dep in override.dependencies:
            if (dep.from_tool, dep.to_tool) in existing_pairs:
                # Replace
                self.dependencies = [
                    d for d in self.dependencies if (d.from_tool, d.to_tool) != (dep.from_tool, dep.to_tool)
                ]
            self.dependencies.append(dep)
            existing_pairs.add((dep.from_tool, dep.to_tool))
        # Intent groups: name-keyed replacement
        names = {g.name for g in self.intent_groups}
        for group in override.intent_groups:
            if group.name in names:
                self.intent_groups = [g for g in self.intent_groups if g.name != group.name]
            self.intent_groups.append(group)
            names.add(group.name)
        self.priority_routes.extend(override.priority_routes)

    def is_empty(self) -> bool:
        return not (self.dependencies or self.intent_groups or self.priority_routes)

    @classmethod
    def detect_from_components(cls, components: list[Any]) -> "ToolDependencyGraph":
        """Convenience alias for :func:`detect_lifecycle_patterns`."""
        from .detector import detect_lifecycle_patterns

        return detect_lifecycle_patterns(components)

    def __len__(self) -> int:  # pragma: no cover — debugging helper
        return len(self.dependencies) + len(self.intent_groups) + len(self.priority_routes)


__all__ = [
    "HiddenStep",
    "ToolDependency",
    "IntentGroup",
    "PriorityRoute",
    "ToolDependencyGraph",
]
