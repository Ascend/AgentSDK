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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""LKB proof-trace widget for in-transcript denial display.

When a tool call (TaskUpdate, TodoWrite, …) is denied by the Logical
Kanban commit gate, the tool output contains an ``lkb`` or
``logicalKanban`` key with the denial payload.  This widget renders a
structured panel showing the violated rule, proof trace, human-readable
explanation and repair suggestions.

Usage
-----
The widget is mounted by :class:`AssistantToolUseMessage` when it
detects a denial in the tool output::

    denial = extract_lkb_denial(output)
    if denial is not None:
        self.mount(LKBProofWidget(denial))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static


# ── payload type ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LkbDenialPayload:
    """Structured LKB denial extracted from a tool output dictionary."""

    decision: str  # 'denied'
    result: str  # 'fail'
    reason: str  # e.g. 'blocked_task_cannot_enter_doing'
    violated_rule: str | None = None
    proof_trace: tuple[dict[str, Any], ...] = ()
    repair_suggestions: tuple[dict[str, Any], ...] = ()
    fuzzy_ambiguities: tuple[dict[str, Any], ...] = ()
    human_message_zh: str | None = None
    human_message_en: str | None = None


# ── rule descriptions (6 MVP rules from spec Ch 6.3) ───────────────────

_RULE_DESCRIPTIONS: dict[str, str] = {
    "R-001": "Blocked by unmet preconditions",
    "R-002": "Blocked task cannot enter Doing",
    "R-003": "Doing task must be Ready and unblocked",
    "R-004": "State transition permission check",
    "R-005": "Completed tasks must have acceptance proof",
    "R-006": "Conflicting assertions invalidate each other",
}


# ── extraction helper ───────────────────────────────────────────────────


def extract_lkb_denial(output: Any) -> LkbDenialPayload | None:
    """Recursively search *output* for an LKB denial payload.

    The tool output may place the LKB data under a ``lkb`` or
    ``logicalKanban`` key at any nesting depth.  Returns ``None`` when
    no denial is found (the tool was accepted or LKB is disabled).
    """
    payloads: list[dict[str, Any]] = []

    def _walk(d: Any) -> None:
        if not isinstance(d, dict):
            return
        for key in ("lkb", "logicalKanban"):
            sub = d.get(key)
            if isinstance(sub, dict):
                if sub.get("decision") == "denied":
                    payloads.append(sub)
                # Also recurse in case the denial is nested further.
                _walk(sub)
        for v in d.values():
            _walk(v)

    _walk(output)
    if not payloads:
        return None

    # Pick the first denial found (usually there is only one).
    p = payloads[0]
    hm = p.get("humanMessage") or p.get("human_message") or {}
    return LkbDenialPayload(
        decision=str(p.get("decision", "denied")),
        result=str(p.get("result", "fail")),
        reason=str(p.get("reason", "")),
        violated_rule=str(p.get("violatedRule")) if p.get("violatedRule") else None,
        proof_trace=tuple(p.get("proofTrace") or p.get("proof_trace") or []),
        repair_suggestions=tuple(p.get("repairSuggestions") or p.get("repair_suggestions") or []),
        fuzzy_ambiguities=tuple(p.get("legacyTodoAmbiguities") or p.get("fuzzy_ambiguities") or []),
        human_message_zh=(hm.get("zh") if isinstance(hm, dict) else None) or None,
        human_message_en=(hm.get("en") if isinstance(hm, dict) else None) or None,
    )


# ── widget ──────────────────────────────────────────────────────────────


class LKBProofWidget(Static):
    """Renders a structured LKB denial panel in the transcript.

    The panel shows the violated rule, proof trace (derivation chain),
    human-readable semantic explanation, any detected ambiguities, and
    actionable repair suggestions.
    """

    DEFAULT_CSS = """
    LKBProofWidget {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, denial: LkbDenialPayload) -> None:
        self.denial = denial
        super().__init__(self._build_panel(), markup=False)

    # ── panel construction ──────────────────────────────────────────

    def _build_panel(self) -> Panel:
        lines: list[Text] = []

        # 1. Violated rule header
        lines.append(self._rule_header())

        # 2. Proof trace (derivation chain)
        if self.denial.proof_trace:
            lines.append(Text(""))  # blank line separator
            lines.append(Text("Proof trace:", style="bold"))
            for step in self.denial.proof_trace:
                lines.append(self._render_step(step))

        # 3. Human-readable explanation
        if self.denial.human_message_en:
            lines.append(Text(""))
            lines.append(Text("Explanation:", style="bold"))
            lines.append(Text(f"  {self.denial.human_message_en}", style="dim"))
        elif self.denial.human_message_zh:
            lines.append(Text(""))
            lines.append(Text("Explanation:", style="bold"))
            lines.append(Text(f"  {self.denial.human_message_zh}", style="dim"))

        # 4. Detected ambiguities (fuzzy input)
        if self.denial.fuzzy_ambiguities:
            lines.append(Text(""))
            lines.append(Text("Detected ambiguities:", style="bold yellow"))
            for amb in self.denial.fuzzy_ambiguities:
                phrase = amb.get("phrase", amb.get("text", ""))
                sev = amb.get("severity", "?")
                amb_kind = amb.get("kind", "?")
                lines.append(Text(f'  ? "{phrase}"', style="yellow") + Text(f"  ({sev} / {amb_kind})", style="dim"))

        # 5. Repair suggestions
        if self.denial.repair_suggestions:
            lines.append(Text(""))
            lines.append(Text("Repair suggestions:", style="bold"))
            for idx, sug in enumerate(self.denial.repair_suggestions, 1):
                lines.append(self._render_suggestion(idx, sug))

        # 6. Hint for more detail
        if self.denial.reason:
            lines.append(Text(""))
            lines.append(
                Text(
                    f"  (run /lkb explain {self.denial.reason.split()[-1] if self.denial.reason.split() else ''} "
                    f"for full proof)",
                    style="dim",
                )
            )

        body = Text("\n").join(lines)
        return Panel(body, title="❌ LKB validation failed", border_style="red")

    # ── internal helpers ────────────────────────────────────────────

    def _rule_header(self) -> Text:
        rule_id = self.denial.violated_rule or "—"
        desc = _RULE_DESCRIPTIONS.get(rule_id, rule_id)
        out = Text()
        out.append(f"Violated rule: {rule_id}  ", style="bold red")
        out.append(desc, style="red")
        return out

    def _render_step(self, step: dict) -> Text:
        rule = step.get("rule", "?")
        premises: list[str] = step.get("premises", [])
        conclusion: str = step.get("conclusion", "")
        seq = step.get("step", "")

        p_text = " + ".join(str(p) for p in premises[:4])
        if len(premises) > 4:
            p_text += f" + …({len(premises)} premises)"

        out = Text()
        out.append(f"  Step {seq}  ", style="dim")
        out.append(p_text, style="white")
        out.append("\n")
        out.append(f"           ──[{rule}]──→ ", style="bold green")
        out.append(conclusion, style="bold white")
        return out

    def _render_suggestion(self, idx: int, sug: dict) -> Text:
        action: str = sug.get("action", "?")
        target: str = sug.get("target", "")
        message: str = sug.get("message", "")
        out = Text()
        out.append(f"  [{idx}] ", style="dim")
        out.append(action, style="bold cyan")
        if target:
            out.append(f"  {target}", style="yellow")
        if message:
            out.append(f"  {message}", style="dim")
        return out
