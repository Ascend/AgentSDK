#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

"""CommandRouter — map ``command`` semantic to existing entry points (P5).

Reuses the orchestrator's ``parse_agent_command`` for
``/agent retry|follow-up|unblock`` and recognizes the issue-CLI control
verbs (pause/resume/stop/inject/clarify/review/feedback/retry). It also
normalizes the README-documented IM orchestrator commands (``/issue ...`` and
``/server status``) to the existing orchestrator CLI argv shape. It does NOT
invent new synonyms — same names as the existing surfaces.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

# pylint: disable=no-name-in-module  # services.im_gateway: pending patch migration
from clawcodex_ext.services.im_gateway.models import InboundMessage

_AGENT_CMD_RE = re.compile(r"^\s*/agent\s+(retry|follow-up|unblock)\b[^\n]*", re.IGNORECASE)
_CONTROL_VERB_RE = re.compile(
    r"^\s*/(?P<verb>pause|resume|stop|takeover|inject|detach|clarify|review|feedback)\b[^\n]*",
    re.IGNORECASE,
)

_ORCHESTRATOR_ISSUE_COMMANDS = frozenset(
    {
        "list",
        "show",
        "tail",
        "stop",
        "pause",
        "resume",
        "clarify",
        "inject",
        "feedback",
        "review",
        "retry",
        "workspace",
        "rebase",
    }
)

_ISSUE_ID_OPTION = "--" + "id"


@dataclass
class CommandRoute:
    kind: str  # "agent_intent" | "control_verb" | "orchestrator_cli"
    verb: str  # retry | follow-up | unblock | pause | resume | ...
    issue_hint: str | None = None
    payload: str = ""
    argv: tuple[str, ...] = ()


class CommandRouter:
    def route(self, message: InboundMessage) -> CommandRoute | None:
        text = (message.text or "").strip()
        argv = self._orchestrator_argv(text)
        if argv is not None:
            return CommandRoute(
                kind="orchestrator_cli",
                verb=argv[0],
                issue_hint=self._extract_issue(text, argv),
                payload=text,
                argv=tuple(argv),
            )
        m = _AGENT_CMD_RE.match(text)
        if m:
            verb = m.group(1).lower()
            issue_hint = self._extract_issue(text)
            return CommandRoute(kind="agent_intent", verb=verb, issue_hint=issue_hint, payload=text)
        m = _CONTROL_VERB_RE.match(text)
        if m:
            verb = m.group("verb").lower()
            issue_hint = self._extract_issue(text)
            return CommandRoute(kind="control_verb", verb=verb, issue_hint=issue_hint, payload=text)
        return None

    @staticmethod
    def _orchestrator_argv(text: str) -> list[str] | None:
        tokens = _split_command_tokens(text)
        if not tokens:
            return None
        command = tokens[0].lower()
        if command == "issue" and len(tokens) >= 2:
            issue_subcommand = tokens[1].lower()
            if issue_subcommand in _ORCHESTRATOR_ISSUE_COMMANDS:
                return ["issue", issue_subcommand, *tokens[2:]]
        if command == "server" and len(tokens) >= 2 and tokens[1].lower() == "status":
            return ["server", "status", *tokens[2:]]
        return None

    @staticmethod
    def _extract_issue(text: str, argv: list[str] | None = None) -> str | None:
        if argv:
            for idx, token in enumerate(argv):
                if token == _ISSUE_ID_OPTION and idx + 1 < len(argv):
                    return argv[idx + 1]
                if token.startswith(f"{_ISSUE_ID_OPTION}="):
                    return token.split("=", 1)[1]
        # Loose match for issue ids like AGENTSDK-15, PROJ-128, etc.
        m = re.search(r"\b([A-Z][A-Z0-9_]+-\d+)\b", text)
        return m.group(1) if m else None


def _split_command_tokens(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return []
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    if not tokens:
        return []
    tokens[0] = tokens[0].lstrip("/")
    return tokens


__all__ = ["CommandRoute", "CommandRouter"]
