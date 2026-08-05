#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# pylint: disable=no-name-in-module
from __future__ import annotations

import json

from clawcodex_ext.services.ultraplan.audit import AuditLogger
from clawcodex_ext.services.ultraplan.keyword_detector import TriggerHit
from clawcodex_ext.services.ultraplan.templates import TemplateLibrary
from clawcodex_ext.tui.rainbow_highlight import highlight_triggers, should_render_rainbow


def test_builtin_templates_apply_goal() -> None:
    rendered = TemplateLibrary().apply("refactor", "executor.py")
    assert "executor.py" in rendered
    assert "refactor" in rendered.lower()


def test_custom_template_json_is_loaded(tmp_path) -> None:
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "id": "custom",
                "title": "Custom",
                "description": "Custom template",
                "prompt": "Plan this: {goal}",
            }
        ),
        encoding="utf-8",
    )
    assert TemplateLibrary(tmp_path).apply("custom", "ship it") == "Plan this: ship it"


def test_custom_template_yaml_is_loaded(tmp_path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        "\n".join(
            [
                "id: custom_yaml",
                "title: Custom YAML",
                "description: Custom YAML template",
                "prompt: |",
                "  Plan YAML: {goal}",
            ]
        ),
        encoding="utf-8",
    )
    assert TemplateLibrary(tmp_path).apply("custom_yaml", "ship it") == "Plan YAML: ship it"


def test_audit_logger_writes_ndjson(tmp_path) -> None:
    logger = AuditLogger(tmp_path)
    logger.append("p1", "plan.created", {"title": "T"})
    entries = logger.read("p1")
    assert entries[0].event == "plan.created"
    assert entries[0].payload == {"title": "T"}
    assert logger.path_for("p1").read_text(encoding="utf-8").count("\n") == 1


def test_highlight_triggers_preserves_text() -> None:
    text = "/ultraplan foo"
    highlighted = highlight_triggers(text, [TriggerHit(0, 10, "/ultraplan")])
    assert highlighted.plain == text


def test_should_render_rainbow_uses_isatty() -> None:
    class Stream:
        def isatty(self) -> bool:
            return True

    assert should_render_rainbow(Stream())
