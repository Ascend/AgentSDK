#!/usr/bin/env python3
# coding=utf-8

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

"""``@agent-<type>`` mention completer for the REPL's prompt.

Completes agent names when the user types ``@agent-`` followed by a partial
name. Mirrors the ``@agent-<type>`` mention syntax defined in
:mod:`clawcodex_ext.command_system.input_processing`.

The class implements ``prompt_toolkit.completion.Completer`` so the same
instance can plug into the idle ``PromptSession`` and the live
``LiveStatus`` input buffer used while the agent is working.

Originally lived in :mod:`src.utils.agent_mention_completer`; moved to the
extension layer per F-48 Phase 2-B (see ``DECOUPLING_PLAN.md`` §3.6). The
``src`` copy is now a thin facade.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

try:
    from prompt_toolkit.completion import Completer, Completion
except ModuleNotFoundError:  # pragma: no cover

    class Completer:  # type: ignore[no-redef]
        pass

    class Completion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


# Match ``@agent-<partial>`` at the end of the text before cursor.
# The ``@`` must be at the start of a token (preceded by whitespace or
# beginning-of-line). Captures the full token (e.g. ``agent-explor``)
# so the replacement can swap it for the completed form.
_AGENT_COMPLETE_RE = re.compile(r"(?:^|(?<=\s))@(agent-[\w:.@\-]*)$")

_DEFAULT_MAX_SUGGESTIONS = 10


class AgentMentionCompleter(Completer):
    """Completer that surfaces agent names after ``@agent-``.

    Construct once per REPL session and pass to the ``PromptSession``
    via ``merge_completers``. The agent list is read lazily from the
    provided callable each time completions are requested, so newly-added
    agents appear without restarting the REPL.

    Args:
        agents_provider:
            A zero-argument callable returning the list of available agent
            definitions (same shape returned by
            ``ClawcodexREPL._available_agents()``). Each item should have
            an ``agent_type`` attribute (or dict key) giving the name that
            follows ``@agent-`` in the mention syntax.

    Keyword Args:
        max_suggestions:
            Maximum number of completion candidates to yield. Default 10.
    """

    def __init__(
        self,
        agents_provider: Callable[[], list[Any]],
        *,
        max_suggestions: int = _DEFAULT_MAX_SUGGESTIONS,
    ) -> None:
        self._agents_provider = agents_provider
        self._max_suggestions = max_suggestions

    def get_completions(self, document, complete_event) -> Iterable[Completion]:
        """Yield agent-name completions when the token starts with ``@agent-``."""
        text = document.text_before_cursor
        match = _AGENT_COMPLETE_RE.search(text)
        if match is None:
            return

        # The ``@`` must be at the start of a token — skip ``foo@agent-``
        at_pos = match.start()
        if at_pos > 0 and not text[at_pos - 1].isspace():
            return

        token = match.group(1)  # e.g. ``"agent-explor"`` or ``"agent-"``
        partial = token[len("agent-") :].lower()  # e.g. ``"explor"`` or ``""``
        replace_len = len(match.group(0))

        try:
            agents = self._agents_provider()
        except Exception:
            return

        if not agents:
            return

        count = 0
        for agent in agents:
            agent_type = _get_agent_type(agent)
            if agent_type is None:
                continue

            if partial in agent_type.lower():
                if count >= self._max_suggestions:
                    break
                count += 1

                name = _get_agent_name(agent)
                yield Completion(
                    text=f"@agent-{agent_type}",
                    start_position=-replace_len,
                    display=agent_type,
                    display_meta=name or None,
                )


def _get_agent_type(agent: Any) -> str | None:
    """Extract the ``agent_type`` from an agent definition."""
    agent_type = getattr(agent, "agent_type", None) or (agent.get("agent_type") if isinstance(agent, dict) else None)
    if isinstance(agent_type, str) and agent_type:
        return agent_type
    return None


def _get_agent_name(agent: Any) -> str:
    """Extract the display ``name`` from an agent definition."""
    name = getattr(agent, "name", None) or (agent.get("name", "") if isinstance(agent, dict) else "")
    return str(name) if name else ""


__all__ = [
    "AgentMentionCompleter",
]
