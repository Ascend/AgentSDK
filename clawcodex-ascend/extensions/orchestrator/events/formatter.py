#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Format :class:`OrchestratorEvent` → IM-friendly text.

Only actionable state is pushed; token/tool-call detail stays in
LiveView/logs/reports. Each event type has a concise template; unknown
types fall back to ``<event_type>: <message>``.
"""

from __future__ import annotations

from .types import EventLevel, OrchestratorEvent

# event_type -> emoji prefix (actionable signal)
_PREFIX = {
    EventLevel.SUCCESS: "✅",
    EventLevel.WARN: "⚠️",
    EventLevel.ERROR: "❌",
    EventLevel.INFO: "ℹ️",
}


def format_event(event: OrchestratorEvent) -> str:
    prefix = _PREFIX.get(event.level, "ℹ️")
    issue = event.issue_id or "?"
    msg = event.message.strip() if event.message else event.event_type
    handler = _TEMPLATES.get(event.event_type)
    if handler is not None:
        return handler(event, prefix, issue)
    return f"{prefix} {issue}: {msg}"


def _issue_line(event: OrchestratorEvent, prefix: str, issue: str) -> str:
    msg = event.message.strip() if event.message else event.event_type
    pr = event.payload.get("pr")
    tail = f" · PR {pr}" if pr else ""
    return f"{prefix} {issue}: {msg}{tail}"


# -- payload helpers --------------------------------------------------------


def _pr(event: OrchestratorEvent) -> str:
    pr = event.payload.get("pr")
    return f" · PR {pr}" if pr else ""


def _title(event: OrchestratorEvent) -> str:
    """Issue title line, e.g. '「ISSUE-Fix login bug」'."""
    title = event.payload.get("title")
    return f"「ISSUE-{title}」" if title else ""


def _branch(event: OrchestratorEvent) -> str:
    """Branch line, e.g. ' · branch clawcodex/AGENTSDK-15'."""
    branch = event.payload.get("branch")
    return f" · branch {branch}" if branch else ""


def _repo(event: OrchestratorEvent) -> str:
    """Repo line, e.g. ' · repo owner/repo'."""
    repo = event.payload.get("repo")
    return f" · repo {repo}" if repo else ""


def _commit(event: OrchestratorEvent) -> str:
    """Commit line, e.g. ' · commit abc1234'."""
    commit = event.payload.get("commit")
    if not commit:
        return ""
    return f" · commit {commit[:7]}" if len(commit) > 7 else f" · commit {commit}"


def _verification(event: OrchestratorEvent) -> str:
    """Verification status line, e.g. ' · verification passed'."""
    ver = event.payload.get("verification")
    return f" · 验证 {ver}" if ver else ""


def _attempts(event: OrchestratorEvent) -> str:
    """Attempts line, e.g. ' · attempt 2'."""
    attempts = event.payload.get("attempts")
    return f" · 第 {attempts} 次尝试" if attempts else ""


def _turns(event: OrchestratorEvent) -> str:
    """Turn count line, e.g. ' · 42 turns'."""
    turns = event.payload.get("turns")
    return f" · {turns} 轮" if turns else ""


def _url(event: OrchestratorEvent) -> str:
    """Issue URL line, e.g. ' · https://gitcode.com/owner/repo/issues/15'."""
    url = event.payload.get("url")
    return f" · {url}" if url else ""


_TEMPLATES = {
    "orchestrator.started": lambda e, p, i: f"{p} clawcodex-orchestrator: IM notifications enabled",
    "orchestrator.im_registered": lambda e, p, i: f"{p} clawcodex-orchestrator: IM gateway registered",
    "orchestrator.im_reconnected": lambda e, p, i: f"{p} clawcodex-orchestrator: IM gateway reconnected",
    "issue.detected": lambda e, p, i: f"{p} {i}: 新增 ISSUE (new ISSUE) {_title(e)}{_repo(e)}{_url(e)}",
    "issue.started": lambda e, p, i: f"{p} {i}: 任务已启动 (task started) {_title(e)}{_branch(e)}{_repo(e)}",
    "issue.completed": lambda e, p, i: (
        f"{p} {i}: 任务完成 (task completed) {_title(e)}{_branch(e)}{_verification(e)}{_commit(e)}{_pr(e)}"
    ),
    "issue.failed": lambda e, p, i: (
        f"{p} {i}: 任务失败 (task failed) — {e.message}{_title(e)}{_branch(e)}{_attempts(e)}{_turns(e)}{_pr(e)}"
    ),
    "issue.verification_failed": lambda e,
    p,
    i: f"{p} {i}: 验证失败 (verification failed) — {e.message}{_branch(e)}{_pr(e)}",
    "issue.cancelled": lambda e, p, i: f"{p} {i}: 已取消 (cancelled){_branch(e)}",
    "issue.rate_limit_paused": lambda e, p, i: f"{p} {i}: 因限流暂停，稍后重试 (rate-limited, will retry)",
    "pr.opened": lambda e, p, i: f"{p} {i}: PR 已开启 (PR opened){_branch(e)}{_commit(e)}{_pr(e)}",
    "pr.pending_review_gate": lambda e, p, i: f"{p} {i}: PR 待审批 (PR pending review){_branch(e)}{_pr(e)}",
    "pr.updated": lambda e, p, i: f"{p} {i}: PR 已更新 (PR updated){_branch(e)}{_commit(e)}{_pr(e)}",
    "git.push_failed": lambda e, p, i: f"{p} {i}: 推送失败 (push failed) — {e.message}{_branch(e)}",
    "post_commit_failed": lambda e,
    p,
    i: f"{p} {i}: 已提交但后续步骤失败，需人工介入 (committed, post steps failed){_branch(e)}{_commit(e)}{_pr(e)}",
    "agent.rate_limit_circuit_open": lambda e, p, i: f"{p} {i}: 触发限流熔断，已暂停 (rate limit circuit open, paused)",
    "agent.stagnation": lambda e, p, i: f"{p} {i}: agent 长时间无进展 (agent stalled){_turns(e)}",
    "agent.loop_detected": lambda e, p, i: f"{p} {i}: 检测到 agent 循环 (agent loop detected){_turns(e)}",
    "agent.max_turns_exceeded": lambda e, p, i: f"{p} {i}: 达到最大轮次 (max turns exceeded){_turns(e)}",
    "verification.failed": lambda e,
    p,
    i: f"{p} {i}: 验证失败 (verification failed) — {e.message}{_branch(e)}{_commit(e)}",
    "verification.timeout": lambda e, p, i: f"{p} {i}: 验证超时 (verification timeout){_branch(e)}",
    "clarification.notify_emitted": lambda e, p, i: f"{p} {i}: 需要澄清 (clarification needed) — {e.message}",
    "clarification.escalated_to_author": lambda e, p, i: f"{p} {i}: 澄清升级给作者 (escalated to author)",
    "clarification.exhausted": lambda e, p, i: f"{p} {i}: 澄清次数耗尽 (clarification exhausted)",
    "review_feedback.pending_manual": lambda e, p, i: f"{p} {i}: PR review 待人工处理 (review pending manual){_pr(e)}",
    "control.pause": lambda e, p, i: f"{p} {i}: paused",
    "control.resume": lambda e, p, i: f"{p} {i}: resumed",
    "control.stop": lambda e, p, i: f"{p} {i}: 已停止 (stopped)",
    "control.takeover": lambda e, p, i: f"{p} {i}: 运维接管 (taken over)",
    "intent.retry": lambda e, p, i: f"{p} {i}: 重试已触发 (retry triggered){_attempts(e)}",
    "intent.followup": lambda e, p, i: f"{p} {i}: follow-up 已入队 (follow-up queued)",
    "orchestrator.loop_error": lambda e, p, i: f"{p} {i}: 守护循环错误 (loop error) — {e.message}",
}


__all__ = ["format_event"]
