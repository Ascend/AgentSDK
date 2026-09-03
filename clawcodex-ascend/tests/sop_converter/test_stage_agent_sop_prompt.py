#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Tests for generic stage-agent default execution prompts."""

from __future__ import annotations

import unittest

from extensions.sop_converter.sop_prompts import (
    append_sop_overview_routing,
    infer_stage_label_from_skill,
    pick_pipeline_execute_tool,
    stage_agent_sop_body,
)
from extensions.sop_converter.workflow_project import (
    read_workflow_stage_for_agent,
    read_workflow_stage_pipeline,
)


class TestStageAgentSopPrompt(unittest.TestCase):
    def test_pick_pipeline_execute_tool(self) -> None:
        tools = [
            "autoresearchclaw-execute-stage",
            "myproject-pipeline-execute-stage",
        ]
        self.assertEqual(
            pick_pipeline_execute_tool(tools),
            "myproject-pipeline-execute-stage",
        )

    def test_infer_stage_label_from_skill(self) -> None:
        self.assertEqual(
            infer_stage_label_from_skill("literature-collect-skill"),
            "LITERATURE_COLLECT",
        )

    def test_stage_agent_sop_body_is_generic(self) -> None:
        body = stage_agent_sop_body(
            agent_type="MyProject-literature-collect-agent",
            skill_name="literature-collect-skill",
            stage_label="LITERATURE_COLLECT",
            pipeline_tool="myproject-pipeline-execute-stage",
            stage_id=4,
            output_files=["candidates.jsonl"],
            contract_dod=">=N candidate papers",
        )
        self.assertIn("默认用户指令", body)
        self.assertIn("run_dir=<绝对路径> 执行本阶段", body)
        self.assertIn("myproject-pipeline-execute-stage", body)
        self.assertNotIn("researchclaw-pipeline-execute-stage", body)
        self.assertIn("禁止 pipeline 主路径失败后静默 fallback", body)
        self.assertIn("stage-04/", body)
        self.assertIn("candidates.jsonl", body)

    def test_read_workflow_stage_for_agent(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflow.yaml").write_text(
                "name: MyProject\n"
                "stages:\n"
                "  - id: 4\n"
                "    name: LITERATURE_COLLECT\n"
                "    phase: literature-collect\n"
                "    agent_config:\n"
                "      agent: MyProject-literature-collect-agent\n"
                "    output_files:\n"
                "      - candidates.jsonl\n",
                encoding="utf-8",
            )
            meta = read_workflow_stage_for_agent(root, "MyProject-literature-collect-agent")
            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertEqual(meta["id"], 4)
            self.assertEqual(meta["name"], "LITERATURE_COLLECT")
            self.assertEqual(meta["output_files"], ["candidates.jsonl"])

    def test_read_workflow_stage_pipeline(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflow.yaml").write_text(
                "name: MyProject\n"
                "stages:\n"
                "  - id: 4\n"
                "    name: LITERATURE_COLLECT\n"
                "    kind: agent\n"
                "    agent_config:\n"
                "      agent: MyProject-literature-collect-agent\n"
                "    output_files:\n"
                "      - candidates.jsonl\n"
                "  - id: 5\n"
                "    name: LITERATURE_SCREEN\n"
                "    kind: gate\n"
                "    gate_mode: manual\n",
                encoding="utf-8",
            )
            rows = read_workflow_stage_pipeline(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["name"], "LITERATURE_COLLECT")
            self.assertEqual(rows[1]["kind"], "gate")

    def test_append_sop_overview_routing_includes_pipeline_block(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflow.yaml").write_text(
                "name: MyProject\n"
                "stages:\n"
                "  - id: 4\n"
                "    name: LITERATURE_COLLECT\n"
                "    kind: agent\n"
                "    agent_config:\n"
                "      agent: MyProject-literature-collect-agent\n",
                encoding="utf-8",
            )
            body = append_sop_overview_routing(
                "# Overview",
                bundle_path=root,
            )
            self.assertIn("流水线 Stage 编排", body)
            self.assertIn("从 Stage 4 做到 Stage 6", body)
            self.assertIn("MyProject-literature-collect-agent", body)
            self.assertIn("SOP 路由", body)


if __name__ == "__main__":
    unittest.main()
