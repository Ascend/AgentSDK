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

from __future__ import annotations

from clawcodex_ext.context_system.prompt_assembly import (
    build_full_system_prompt,
    build_full_system_prompt_blocks,
)
from clawcodex_ext.services.proactive import reset_default_controller_for_tests


def test_prompt_assembly_omits_proactive_section_when_inactive() -> None:
    reset_default_controller_for_tests()

    prompt = build_full_system_prompt(cwd=".")

    assert "<proactive-mode" not in prompt


def test_prompt_assembly_injects_proactive_section_when_active() -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test", focus="minimal")

    prompt = build_full_system_prompt(cwd=".")
    blocks = build_full_system_prompt_blocks(cwd=".")
    block_text = "\n".join(str(block.get("text", "")) for block in blocks)

    assert '<proactive-mode phase="active" focus="minimal">' in prompt
    assert '<proactive-mode phase="active" focus="minimal">' in block_text
