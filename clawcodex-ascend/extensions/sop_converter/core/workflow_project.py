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

"""Helpers for resolving workflow project prefix from a SOP bundle."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_workflow_project_name(bundle_path: Path) -> str | None:
    """Read ``name:`` from ``workflow.yaml`` (lightweight line parse)."""
    path = bundle_path / "workflow.yaml"
    if not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("name:"):
                continue
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            return value or None
    except OSError:
        return None
    return None


def read_workflow_first_stage_skill_name(bundle_path: Path) -> str | None:
    """Resolve Stage 1 skill name from ``workflow.yaml`` (``phase`` or agent mapping)."""
    path = bundle_path / "workflow.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Cannot parse workflow.yaml for stage-1 skill: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages:
        return None

    first: dict | None = None
    for stage in stages:
        if isinstance(stage, dict) and stage.get("id") == 1:
            first = stage
            break
    if first is None:
        numbered: list[tuple[int, dict]] = []
        for stage in stages:
            if not isinstance(stage, dict) or stage.get("id") is None:
                continue
            try:
                sid = int(stage["id"])
            except (TypeError, ValueError):
                continue
            numbered.append((sid, stage))
        if numbered:
            first = min(numbered, key=lambda item: item[0])[1]

    if first is None:
        return None

    phase = first.get("phase")
    if isinstance(phase, str) and phase.strip():
        kebab = phase.strip().replace("_", "-")
        return kebab if kebab.endswith("-skill") else f"{kebab}-skill"

    agent_cfg = first.get("agent_config")
    agent = agent_cfg.get("agent") if isinstance(agent_cfg, dict) else None
    if isinstance(agent, str) and agent.strip():
        from extensions.sop_converter.sop_prompts import agent_type_to_skill_name

        return agent_type_to_skill_name(
            agent.strip(),
            project_prefix=read_workflow_project_name(bundle_path),
        )
    return None


def read_workflow_stage_for_agent(
    bundle_path: Path,
    agent_type: str,
) -> dict[str, object] | None:
    """Return workflow stage row ``{id, name, phase, output_files}`` for *agent_type*."""
    path = bundle_path / "workflow.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Cannot parse workflow.yaml for agent %s: %s", agent_type, exc)
        return None
    if not isinstance(data, dict):
        return None
    stages = data.get("stages")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        agent_cfg = stage.get("agent_config")
        agent = agent_cfg.get("agent") if isinstance(agent_cfg, dict) else None
        if agent != agent_type:
            continue
        stage_id = stage.get("id")
        try:
            parsed_id = int(stage_id) if stage_id is not None else None
        except (TypeError, ValueError):
            parsed_id = None
        return {
            "id": parsed_id,
            "name": stage.get("name"),
            "phase": stage.get("phase"),
            "output_files": stage.get("output_files") or [],
        }
    return None


def read_workflow_stage_pipeline(bundle_path: Path) -> list[dict[str, object]]:
    """Return ordered workflow rows for overview stage-orchestration prompts."""
    path = bundle_path / "workflow.yaml"
    if not path.is_file():
        return []
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Cannot parse workflow.yaml pipeline: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    stages = data.get("stages")
    if not isinstance(stages, list):
        return []

    rows: list[dict[str, object]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        kind = str(stage.get("kind") or "agent")
        agent_cfg = stage.get("agent_config")
        agent = agent_cfg.get("agent") if isinstance(agent_cfg, dict) else None
        outputs = stage.get("output_files")
        if not isinstance(outputs, list):
            outputs = []
        rows.append(
            {
                "id": stage.get("id"),
                "name": stage.get("name"),
                "kind": kind,
                "agent": agent,
                "phase": stage.get("phase"),
                "output_files": [str(f) for f in outputs if f],
                "depends_on": stage.get("depends_on") or [],
                "gate_mode": stage.get("gate_mode"),
            }
        )

    def _sort_key(row: dict[str, object]) -> tuple[int, str]:
        raw_id = row.get("id")
        try:
            return (int(raw_id), str(row.get("name") or ""))
        except (TypeError, ValueError):
            return (9999, str(row.get("name") or ""))

    return sorted(rows, key=_sort_key)


def is_prefixed_stage_agent(agent_type: str, project_name: str | None) -> bool:
    """True for ``{Project}-topic-init-agent`` style stage agents."""
    if not project_name or not agent_type.endswith("-agent"):
        return False
    prefix = f"{project_name}-"
    return agent_type.startswith(prefix) and agent_type != f"{project_name}-agent"
