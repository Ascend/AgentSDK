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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass(frozen=True)
class Redirect:
    op: Literal[">", ">>", "<", "<<", ">&", ">|", "<&", "&>", "&>>", "<<<"]
    target: str
    fd: int | None = None


@dataclass(frozen=True)
class SimpleCommand:
    argv: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    redirects: list[Redirect] = field(default_factory=list)
    text: str = ""

    @property
    def name(self) -> str | None:
        return self.argv[0] if self.argv else None


@dataclass(frozen=True)
class Pipeline:
    commands: list[SimpleCommand] = field(default_factory=list)


@dataclass(frozen=True)
class CommandList:
    entries: list[CommandListEntry] = field(default_factory=list)


@dataclass(frozen=True)
class CommandListEntry:
    node: ASTNode
    operator: Literal["&&", "||", ";", "&", ""] = ""


@dataclass(frozen=True)
class Subshell:
    body: CommandList = field(default_factory=CommandList)


ASTNode = Union[SimpleCommand, Pipeline, CommandList, Subshell]
