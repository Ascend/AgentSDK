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
"""Build agent prompts from Linear issue data.

Port of Symphony's PromptBuilder (Solid template → Jinja2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, StrictUndefined, TemplateError

from clawcodex_ext.agent.agent_definitions import task_v2_guidelines

from .premise_check import build_premise_block, check_issue_premise
from .prompt_context import (
    USER_MESSAGE_MARKER as _USER_MESSAGE_MARKER,
    _build_sequential_workspace_context,
    _expand_agent_mentions_in_prompt,
    _get_git_log_summary,
    _get_operator_hints,
    _get_workspace_diff,
    _resolve_workspace_path,
    _to_jinja_value,
    resolve_python_executable,  # noqa: F401
)
from .rules_learner import RuleEngine
from .workflow_store import get_workflow_store

if TYPE_CHECKING:
    from .tracker import PullRequestFeedback, PullRequestRef

logger = logging.getLogger(__name__)

# Jinja2 environment with strict undefined handling (mirrors Solid's strict_variables)
_jinja_env = Environment(undefined=StrictUndefined)

_DEFAULT_PROMPT = """You are an autonomous software engineering agent.

Issue: {{ issue.identifier }} - {{ issue.title }}
{% if issue.description %}
Description:
{{ issue.description }}
{% endif %}
{% if issue.priority %}
Priority: {{ issue.priority }}
{% endif %}
{% if issue.state %}
State: {{ issue.state }}
{% endif %}

Please analyze the issue, implement the necessary changes, and ensure all tests pass.

## CLI Usage Guidelines
When you need to suggest terminal commands for the user:
- Always use the `clawcodex-dev` CLI entrypoint, NOT `python3 -c` or `PYTHONPATH=`.
- For orchestrator status: `clawcodex-dev orchestrator server status`
- For issue list: `clawcodex-dev orchestrator issue list`
- For issue tail: `clawcodex-dev orchestrator issue tail --id <id>`
- For other commands: use `clawcodex-dev orchestrator --help` or `clawcodex-dev --help`
{% if clarification %}
{{ clarification }}
{% endif %}
"""


# Jinja2 template for clarification guidance injected into the prompt.
# Rendered when an issue is in the clarification flow.
_CLARIFICATION_TEMPLATE = """
---
## Clarification Context

{% if clarification_answer %}
The issue author or operator supplied clarification before this run. Treat the
answer below as part of the issue requirements.

- Question: "{{ pending_question or 'Pre-dispatch clarification' }}"
- Answer{% if answer_source %} ({{ answer_source }}){% endif %}: "{{ clarification_answer }}"
{% else %}
This issue is currently awaiting clarification. When the answer is available,
it will be provided below. If you are unsure about any aspect of the issue,
use the `AskIssueAuthor` tool to request clarification from the issue author
or local operator.

When requesting clarification:
- Be specific: ask exactly what is ambiguous (e.g., "Should this function be sync or async?")
- Provide context: include relevant code snippets or error messages
- Limit to one question at a time to avoid overwhelming responders
{% if pending_question %}
- Current pending question: "{{ pending_question }}"
{% if options %}
- Available options: {{ options|join(', ') }}
{% endif %}
{% endif %}
{% endif %}
---"""

_REVIEW_FEEDBACK_TEMPLATE = """You are an autonomous software engineering agent fixing pull request feedback.

Issue: {{ issue.identifier }} - {{ issue.title }}
Pull request: {% if pull_request.number %}#{{ pull_request.number }}{% else %}unknown{% endif %}{% if pull_request.url %} ({{ pull_request.url }}){% endif %}
Branch: {{ branch_name }}

Current task:
- Fix only the PR review feedback and CI failures listed below.
- Do not expand scope or reimplement unrelated issue requirements.
- Work on the current branch only; do not create a new branch or pull request.
- Prefer the smallest correct change that addresses the feedback.
- If feedback is conflicting or unclear, leave code unchanged for that item and explain what clarification is needed.
- Run relevant tests or record why they cannot be run.
- CLI Usage: when suggesting terminal commands, use `clawcodex-dev` not `python3 -c` or `PYTHONPATH=`.

Feedback:
{% for item in feedback %}
{{ loop.index }}. [{{ item.source }}] {{ item.id }}{% if item.severity %} severity={{ item.severity }}{% endif %}{% if item.status %} status={{ item.status }}{% endif %}
{% if item.file_path %}   File: {{ item.file_path }}{% if item.line %}:{{ item.line }}{% endif %}
{% endif %}{% if item.commit_sha %}   Commit: {{ item.commit_sha }}
{% endif %}{% if item.url %}   URL: {{ item.url }}
{% endif %}{% if item.diff_hunk %}   Diff hunk:
```diff
{{ item.diff_hunk }}
```
{% endif %}   Body:
{{ item.body | indent(3) }}
{% endfor %}
"""


class PromptBuilder:
    """Render agent prompts from issue data + workflow config."""

    @staticmethod
    def render(
        issue: Any,
        attempt: int | None = None,
        clarification_context: str | None = None,
        pending_question: str | None = None,
        options: list[str] | None = None,
        session: Any | None = None,
        python_executable: str | None = None,
        previous_run_ids: list[str] | None = None,
        conflict_files: tuple[str, ...] | list[str] | None = None,
    ) -> str:
        """Build prompt using workflow's WORKFLOW.md body template + issue data.

        Args:
            issue: Issue object with to_dict() method or dict-like
            attempt: Current attempt number (for retry tracking)
            clarification_context: Pre-rendered clarification guidance block
            pending_question: If issue is in clarification flow, the pending question
            options: If in clarification flow, the available options for the question
            previous_run_ids: Run IDs from previous failed attempts; injected as a
                hint so the agent can Read() past transcripts to learn what was tried.
            conflict_files: F-120 — when the agent is in a rebase-resolution
                reentry run, this lists the files that git left in conflict
                state. Injected into the prompt so the agent can read each
                file's conflict markers and resolve them.
        """
        store = get_workflow_store()
        current = store.current()

        if current:
            template_str = current[1]
        else:
            template_str = _DEFAULT_PROMPT

        if not template_str or not template_str.strip():
            template_str = _DEFAULT_PROMPT

        try:
            template = _jinja_env.from_string(template_str)
        except TemplateError as exc:
            logger.error("Template parse error: %s", exc)
            template = _jinja_env.from_string(_DEFAULT_PROMPT)

        issue_dict = issue.to_dict() if hasattr(issue, "to_dict") else issue
        context = {
            "attempt": attempt,
            "issue": _to_jinja_value(issue_dict),
            "clarification": clarification_context,
            "pending_question": pending_question,
            "options": options,
        }

        try:
            rendered = template.render(context).strip()
        except TemplateError as exc:
            logger.error("Template render error: %s", exc)
            # Fallback to default prompt
            fallback = _jinja_env.from_string(_DEFAULT_PROMPT)
            rendered = fallback.render(context).strip()
        if session is not None and getattr(session, "workspace_strategy", None) == "sequential":
            rendered = f"{rendered}\n\n{_build_sequential_workspace_context(session)}"

        ws_path = _resolve_workspace_path(session)

        # Operator hints injection: if the workspace has .operator_hints.md,
        # prepend operator guidance before the issue context so it is the
        # first thing the agent sees on every turn.
        operator_hints = _get_operator_hints(ws_path) if ws_path else None
        if operator_hints:
            rendered = f"---\n## Operator Hints\n\n{operator_hints}\n---\n\n{rendered}"

        # F-40 root-cause fix: inject workspace diff context so the
        # agent sees exactly which files are already modified and can
        # skip re-exploration when code already exists on disk.
        # Only injected when there are uncommitted changes (first turn).
        ws_diff = _get_workspace_diff(ws_path) if ws_path else None
        if ws_diff:
            rendered = (
                "---\n"
                "## Current Workspace Changes\n"
                "\n"
                "The following files have already been modified or created in the\n"
                "workspace but are not yet committed. If these changes match the\n"
                "current issue's requirements, **do not re-implement them**.\n"
                "Skip directly to `git add` + `git commit`.\n"
                "\n"
                f"{ws_diff}\n"
                "---\n"
                "\n"
                f"{rendered}"
            )

        # Premise check (defect R3): when the issue references files that
        # do not exist in the workspace, warn the agent up front and hand
        # it the honest-exit protocol, so "fabricate the missing file" is
        # no longer the path of least resistance.
        if ws_path:
            try:
                missing_paths = check_issue_premise(issue_dict, ws_path)
            except Exception:  # premise checking must never break prompts
                logger.warning("premise check failed", exc_info=True)
                missing_paths = []
            if missing_paths:
                rendered = f"{rendered}\n\n{build_premise_block(missing_paths)}"

        if previous_run_ids:
            sessions_home = Path.home() / ".clawcodex" / "sessions"
            prev_lines = "\n".join(
                f'- `{rid}` — `Read(path="{sessions_home / rid / "transcript.jsonl"}")`' for rid in previous_run_ids
            )
            rendered = (
                "---\n"
                "## Previous Attempts\n"
                "\n"
                "This issue has been attempted before and failed.  You can inspect\n"
                "the full conversation transcript of each previous run to understand\n"
                "what was tried, what went wrong, and what to avoid this time.\n"
                "\n"
                f"{prev_lines}\n"
                "---\n"
                "\n"
                f"{rendered}"
            )

        if python_executable:
            rendered = (
                f"⛔ **约束提醒**：始终用 `{python_executable}` 绝对路径运行 Python，不要调试环境差异。\n\n{rendered}"
            )

        # F-120: inject the list of files git left in conflict state so the
        # agent can read each file's conflict markers and resolve them in
        # place. Only emitted when conflict_files is non-empty.
        if conflict_files:
            file_lines = "\n".join(f"- `{name}`" for name in conflict_files)
            rendered = (
                "---\n"
                "## Conflicting Files (F-120 rebase reentry)\n"
                "\n"
                "The orchestrator's automated rebase left the following files in a\n"
                "conflict state (REBASE_HEAD is set in the workspace). Read each\n"
                "file, resolve the conflict markers (`<<<<<<<`, `=======`,\n"
                "`>>>>>>>`), then continue the rebase and push:\n"
                "\n"
                f"{file_lines}\n"
                "\n"
                "Suggested commands (run from the workspace root):\n"
                "\n"
                "```bash\n"
                "git status              # confirm REBASE_HEAD state\n"
                "# Edit each file above to remove conflict markers.\n"
                "git add <resolved files>\n"
                "git rebase --continue\n"
                "git push --force-with-lease=origin/<branch>:<remote_sha> \\\n"
                "    origin <branch>\n"
                "```\n"
                "---"
                "\n\n"
                f"{rendered}"
            )

        # F-121: rules file reference injection
        rendered = PromptBuilder._inject_rules_reference_from_store(rendered)

        # F-140: inject Task V2 / Logical Kanban guidance so orchestrator-launched
        # agents use the same task-loop discipline as interactive sessions.
        lkb_guidance = task_v2_guidelines()
        if lkb_guidance:
            rendered = f"{rendered}\n\n---\n{lkb_guidance}\n---"

        return rendered

    # F-?? prompt split: marker that separates the constant workflow
    # background (system prompt candidate) from the per-issue data
    # (user message candidate) in workflow.md. Lives in workflow.md
    # between the system section and the issue section. The marker is
    # an HTML comment so it is invisible in Markdown rendering.
    USER_MESSAGE_MARKER = _USER_MESSAGE_MARKER

    # ------------------------------------------------------------------
    # F-121: rules reference injection (shared by all prompt flows)
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_rules_reference(prompt: str, rules_path: str | None) -> str:
        """Append a rules file reference line to *prompt* if *rules_path* is set.

        The reference is deliberately a single line — the agent is expected
        to ``Read()`` the file on demand (F-121 §1.2: "参考示例而非强制约束").
        """
        if not rules_path:
            return prompt
        return (
            f"{prompt}\n\n"
            f"---\n"
            f"\U0001f4d0 **Review conventions**: `{rules_path}`\n"
            f"The file contains illustrative conventions extracted from "
            f"previous PR reviews. Read it with `Read()` when relevant \u2014 "
            f"the rules are **reference examples**, not mandatory requirements.\n"
            f"---"
        )

    @staticmethod
    def render_parts(
        issue: Any,
        attempt: int | None = None,
        clarification_context: str | None = None,
        pending_question: str | None = None,
        options: list[str] | None = None,
        session: Any | None = None,
        python_executable: str | None = None,
        previous_run_ids: list[str] | None = None,
        conflict_files: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[str, str]:
        """Render prompt split into (system, user) by USER_MESSAGE_MARKER.

        The marker lives in workflow.md between the constant background
        / constraint block (system) and the per-issue data block (user).
        The system part is appended to the headless session's effective
        system prompt (alongside CLAUDE.md + git status + style) so the
        daemon sees the same "rich system + short user" structure as
        CCB's interactive session. The user part becomes the per-turn
        user message.

        Falls back to ("", full) when the marker is missing so callers
        that pass an old / un-migrated workflow.md still work — the full
        prompt lands in user and the system append is empty.

        F-89: ``@agent-<type>`` mentions in either half of the prompt are
        expanded into ``agent_mention`` attachments (matching REPL/TUI/
        headless). Unknown agents are stripped with a logged warning —
        orchestrator runs must not abort on a typo in the issue body.
        """
        full = PromptBuilder.render(
            issue,
            attempt=attempt,
            clarification_context=clarification_context,
            pending_question=pending_question,
            options=options,
            session=session,
            python_executable=python_executable,
            previous_run_ids=previous_run_ids,
            conflict_files=conflict_files,
        )
        marker = PromptBuilder.USER_MESSAGE_MARKER
        if marker in full:
            system_part, user_part = full.split(marker, 1)
            system_part, user_part = _expand_agent_mentions_in_prompt(
                system_part.strip(), user_part.strip(), session=session
            )
            return system_part, user_part
        user_part = _expand_agent_mentions_in_prompt("", full.strip(), session=session)[1]
        return "", user_part

    @staticmethod
    def render_rebase(
        *,
        issue: Any,
        branch_name: str,
        base_branch: str,
        conflict_files: tuple[str, ...] | list[str] = (),
        reason: str | None = None,
    ) -> str:
        """F-120: build a prompt for an agent run that resolves a rebase conflict.

        This is used when ``_process_rebase_intent`` left content conflicts
        (has_conflict=True) and the daemon launches a fresh ``agent_rebase``
        run to resolve them. The prompt is intentionally minimal — the agent
        is told exactly which files git marked as conflicting and the
        suggested git commands to finish the rebase + push.
        """
        issue_dict = issue.to_dict() if hasattr(issue, "to_dict") else issue
        title = (issue_dict.get("title") if isinstance(issue_dict, dict) else getattr(issue, "title", "")) or ""
        identifier = (
            issue_dict.get("identifier") if isinstance(issue_dict, dict) else getattr(issue, "identifier", "")
        ) or ""

        files_block = (
            "\n".join(f"- `{name}`" for name in conflict_files)
            if conflict_files
            else "- (no specific files reported — run `git diff --name-only --diff-filter=U` to list them)"
        )

        reason_block = f"\n## Reason\n\n{reason}\n" if reason else ""

        template = (
            "---\n"
            f"# F-120 PR Conflict Resolution — {identifier}\n"
            f"\n**Title:** {title}\n"
            f"**Branch:** `{branch_name}` (base `{base_branch}`)\n"
            f"{reason_block}"
            "\n"
            "## Task\n"
            "\n"
            "The orchestrator's automated `git rebase origin/<base>` left this\n"
            "branch with content conflicts. Your job is to resolve each conflict,\n"
            "continue the rebase, and push the rebased branch with\n"
            "`--force-with-lease` (the default) so the PR becomes mergeable\n"
            "again. **Do NOT close the PR or open a new one.**\n"
            "\n"
            "## Conflicting Files\n"
            "\n"
            f"{files_block}\n"
            "\n"
            "## Procedure\n"
            "\n"
            "1. `git status` — confirm REBASE_HEAD is set.\n"
            "2. For each file above: read the file, remove the\n"
            "   `<<<<<<<`/`=======`/`>>>>>>>` markers, write the merged\n"
            "   content you want kept.\n"
            "3. `git add <file>` for each resolved file.\n"
            "4. `git rebase --continue` (or `--skip` if the upstream commit is\n"
            "   the one to drop — but only when clearly safe).\n"
            "5. `git log --oneline -5` to verify the rebased history.\n"
            "6. Capture the new `HEAD` SHA, then push:\n"
            "\n"
            "   ```bash\n"
            "   REMOTE_SHA=$(git rev-parse origin/<branch>)\n"
            "   git push --force-with-lease=<branch>:$REMOTE_SHA origin <branch>\n"
            "   ```\n"
            "\n"
            "7. Print the final head SHA in your response so the orchestrator\n"
            "   can record it.\n"
            "\n"
            "## Constraints\n"
            "\n"
            "- **Do not** run `git rebase --abort` unless explicitly asked; we\n"
            "  want the rebased history, not the pre-rebase one.\n"
            "- **Do not** use plain `git push --force`; the orchestrator\n"
            "  defaults to `--force-with-lease` to avoid clobbering concurrent\n"
            "  pushes. Only use `--force` if the operator explicitly passed\n"
            "  `--force` to the rebase CLI.\n"
            "- **Do not** open a new PR; the existing PR will pick up the\n"
            "  rebased head automatically once the push lands.\n"
            "---"
        )
        return template

    @staticmethod
    def render_review_feedback(
        *,
        issue: Any,
        pull_request: PullRequestRef,
        branch_name: str,
        feedback: list[PullRequestFeedback],
    ) -> str:
        issue_dict = issue.to_dict() if hasattr(issue, "to_dict") else issue
        context = {
            "issue": _to_jinja_value(issue_dict),
            "pull_request": pull_request,
            "branch_name": branch_name,
            "feedback": feedback,
        }
        try:
            rendered = _jinja_env.from_string(_REVIEW_FEEDBACK_TEMPLATE).render(context).strip()
        except TemplateError as exc:
            logger.error("Review feedback template render error: %s", exc)
            return _DEFAULT_PROMPT

        # F-121: inject rules reference
        rendered = PromptBuilder._inject_rules_reference_from_store(rendered)
        return rendered

    @staticmethod
    def _inject_rules_reference_from_store(prompt: str) -> str:
        """Resolve rules path from WorkflowStore and inject reference."""
        store = get_workflow_store()
        current = store.current()
        if not current:
            return prompt
        config = current[0]
        workflow_path = getattr(config, "source_path", None) or getattr(config, "_source_path", None)
        rules_path = RuleEngine.get_rules_path(config, workflow_path)
        return PromptBuilder._inject_rules_reference(prompt, rules_path)

    @staticmethod
    def render_feedback_summary(
        *,
        attempt: int,
        processed: list[PullRequestFeedback],
        skipped: list[dict],
    ) -> str:
        """Render a post-followup summary for the PR.

        Args:
            attempt: Follow-up attempt number.
            processed: Feedback items that were auto-handled.
            skipped: Dicts with keys ``feedback`` (PullRequestFeedback)
                and ``reason`` (str) for items needing human attention.
        """
        lines = [
            "## ClawCodex PR Review Follow-up Summary",
            "",
            f"**Follow-up attempt**: #{attempt}",
            f"**Processed**: {len(processed)} item(s)",
        ]
        if processed:
            lines += ["", "### Auto-handled"]
            for item in processed:
                loc = ""
                if item.file_path:
                    loc = f" (`{item.file_path}"
                    if item.line:
                        loc += f":{item.line}"
                    loc += "`)"
                body_preview = (item.body or "")[:80]
                if len(item.body or "") > 80:
                    body_preview += "..."
                lines.append(f"- [{item.source}] {item.id}{loc}: {body_preview}")
        if skipped:
            lines += ["", "### Needs human attention"]
            for entry in skipped:
                fb = entry["feedback"]
                reason = entry["reason"]
                loc = ""
                if fb.file_path:
                    loc = f" (`{fb.file_path}"
                    if fb.line:
                        loc += f":{fb.line}"
                    loc += "`)"
                lines.append(f"- [{fb.source}] {fb.id}{loc}: {reason}")
        return "\n".join(lines)

    @staticmethod
    def build_continuation_prompt(
        turn_number: int,
        max_turns: int,
        issue_context: str | None = None,
        session: Any | None = None,
        python_executable: str | None = None,
    ) -> str:
        """Build continuation prompt for subsequent turns.

        F-54 root-cause fix: inject a summary of recent git commits
        so the LLM can see what has already been done in previous
        turns and avoid re-exploring from scratch.
        """
        context_block = f"\n\nCurrent issue context:\n{issue_context}\n" if issue_context else ""
        urgency = (
            f"\n- ⚠️  You have only {max_turns - turn_number + 1} turn(s) remaining. "
            f"Prioritize code implementation over reading more files. "
            f"Use Write/Edit to make concrete changes NOW."
            if turn_number >= max_turns // 2
            else ""
        )

        # F-54 root-cause fix: inject recent git log so the LLM knows
        # what was already done in previous turns.
        git_log_summary = _get_git_log_summary(session)

        # Operator hints injection for continuation turns.
        ws_path = _resolve_workspace_path(session)
        operator_hints = _get_operator_hints(ws_path) if ws_path else None
        hints_block = f"---\n## Operator Hints\n\n{operator_hints}\n---\n\n" if operator_hints else ""

        python_constraint = (
            f"⛔ **约束提醒**：始终用 `{python_executable}` 绝对路径，不要调试环境差异。\n" if python_executable else ""
        )

        prompt = (
            f"{hints_block}"
            f"Continuation guidance:\n\n"
            f"{python_constraint}"
            f"⛔ `pytest` 禁止使用管道 `| tail -40`/`| head -50`，用 `--tb=short -q` 替代。\n"
            f"⛔ 建议终端命令时用 `clawcodex-dev` CLI，不要用 `python3 -c` 或 `PYTHONPATH=`。\n"
            f"- This is continuation turn #{turn_number} of {max_turns}.{context_block}{urgency}\n"
            f"- Resume from the current workspace state and continue implementing.\n"
            f"- Use available tools (Bash, Write, Edit, Grep, Glob, etc.) to make changes.\n"
            f"- Focus on completing the issue requirements. Do NOT re-read files you have already explored.\n"
            f"- Your FIRST action should be a Write or Edit to implement the feature.\n"
            f"{git_log_summary}"
        )
        # F-121: inject rules reference on continuation turns
        prompt = PromptBuilder._inject_rules_reference_from_store(prompt)
        return prompt

    @staticmethod
    def build_clarification_context(
        pending_question: str | None = None,
        options: list[str] | None = None,
        clarification_answer: str | None = None,
        answer_source: str | None = None,
    ) -> str:
        """Build a clarification guidance block for the system prompt.

        This text is injected into the agent's prompt when an issue is in
        the clarification flow, guiding the agent to use AskIssueAuthor
        correctly and informing it about any pending question.

        Args:
            pending_question: The pending clarification question, if any
            options: Available options (for multiple-choice questions)

        Returns:
            A formatted clarification guidance block, or empty string if
            clarification is not active
        """
        if not pending_question and not clarification_answer:
            return ""

        template_str = _CLARIFICATION_TEMPLATE.strip()
        try:
            template = _jinja_env.from_string(template_str)
        except TemplateError as exc:
            logger.error("Clarification template parse error: %s", exc)
            return ""

        context = {
            "pending_question": pending_question,
            "options": options or [],
            "clarification_answer": clarification_answer,
            "answer_source": answer_source,
        }
        try:
            return template.render(context).strip()
        except TemplateError as exc:
            logger.error("Clarification template render error: %s", exc)
            return ""
