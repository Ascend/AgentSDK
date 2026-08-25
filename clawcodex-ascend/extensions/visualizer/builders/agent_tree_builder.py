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

"""Build the flat, parent-linked agent tree consumed by API and UI callers.

Hierarchy inference from mailbox cross-references is intentionally outside the
current data contract; callers receive the parent links already present in the
session data.
"""

from __future__ import annotations

from typing import Any

from ..models.viz_models import (  # pylint: disable=relative-beyond-top-level
    AgentTreeNode,
    SessionVizData,
)


class AgentTreeBuilder:
    """Build agent tree visualization data."""

    def build(self, session: SessionVizData) -> dict[str, Any]:
        """Build tree data for ECharts tree chart."""
        nodes = session.agent_tree
        if not nodes:
            return {"nodes": [], "edges": [], "root": None}

        # Build edges from parent_id references
        edges: list[dict[str, str]] = []
        node_map: dict[str, AgentTreeNode] = {}
        root_id: str | None = None

        for node in nodes:
            node_map[node.agent_id] = node
            if node.parent_id is None:
                root_id = node.agent_id
            else:
                edges.append({"source": node.parent_id, "target": node.agent_id})

        # If no explicit root, pick the first node
        if root_id is None and nodes:
            root_id = nodes[0].agent_id

        return {
            "nodes": [self._node_to_dict(n) for n in nodes],
            "edges": edges,
            "root": root_id,
        }

    def _node_to_dict(self, node: AgentTreeNode) -> dict[str, Any]:
        return {
            "id": node.agent_id,
            "name": node.name,
            "parentId": node.parent_id,
            "children": node.children,
            "depth": node.depth,
            "status": node.status.value,
            "stats": node.stats.model_dump(),
            "metadata": node.metadata,
        }
