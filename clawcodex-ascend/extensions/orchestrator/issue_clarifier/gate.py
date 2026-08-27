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
"""Orchestrator dispatch gate that connects issue clarification to the existing resolver."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from .models import ClarifyResult
from .service import IssueClarifierService, format_clarification_request

if TYPE_CHECKING:
    from ..clarification import ClarificationResolver
    from ..issue import Issue
    from ..issue_registry import IssueRegistry
    from ..tracker import TrackerAdapter

logger = logging.getLogger(__name__)


class IssueClarificationGate:
    """Return whether an issue may be dispatched in the current poll."""

    def __init__(
        self,
        *,
        service: IssueClarifierService,
        resolver: "ClarificationResolver",
        registry: "IssueRegistry",
        config: Any,
        tracker: "TrackerAdapter | None" = None,  # ★ P2: 远端标签推送
        workspace_focus_callback: Callable[["Issue"], list[dict]] | None = None,  # ★ P2
    ) -> None:
        self.service = service
        self.resolver = resolver
        self.registry = registry
        self.config = config
        self._tracker = tracker
        self._workspace_focus_callback = workspace_focus_callback
        self._analyses_this_poll = 0

    def begin_poll(self) -> None:
        """Reset the bounded first-analysis budget for a poll cycle."""
        self._analyses_this_poll = 0

    async def should_dispatch(self, issue: "Issue") -> bool:
        issue_id = str(issue.id or "")
        if not issue_id or not self.config.enabled:
            return True
        record = self.registry.get(issue_id)
        if record is None:
            return True

        replies = list(record.clarification_replies)
        current_fingerprint = self.service.fingerprint(issue, prior_replies=replies)
        status = str(record.clarification_status or "")

        if status == "manual_resolved":
            return self._finish_without_author_wait(issue_id, should_dispatch=True)

        if status == "awaiting_author":
            resolved = self.resolver.get_answer(issue_id)
            if resolved is None:
                if record.clarifier_fingerprint == current_fingerprint:
                    return False
                # The issue text changed while waiting. Retire the stale
                # question and analyze the new text immediately. Keep the
                # cumulative round count so repeated edits cannot bypass
                # the configured clarification limit.
                self.resolver.clear(issue_id)
                self.registry.update_clarification(issue_id, open_questions=[])
                self._remove_remote_label(issue_id)
            else:
                resolved_item = self.resolver.get_item(issue_id)
                comment_cursor = (
                    getattr(resolved_item, "last_checked_comment_id", None) if resolved_item is not None else None
                )
                answer = str(resolved.answer or "").strip()
                if answer and answer not in replies:
                    replies.append(answer)
                current_fingerprint = self.service.fingerprint(issue, prior_replies=replies)
                self.resolver.clear(issue_id)
                if comment_cursor:
                    self.registry.update_clarification(
                        issue_id,
                        clarifier_comment_cursor=comment_cursor,
                    )

        if record.clarifier_fingerprint == current_fingerprint and status in {
            "clear",
            "resolved",
            "observation",
        }:
            return self._finish_without_author_wait(issue_id, should_dispatch=True)
        if record.clarifier_fingerprint == current_fingerprint and status == "manual_required":
            return self._finish_without_author_wait(issue_id, should_dispatch=False)

        max_analyses = max(1, int(getattr(self.config, "max_analyses_per_poll", 4)))
        if self._analyses_this_poll >= max_analyses:
            logger.info(
                "Deferring issue clarity analysis for issue %s: per-poll budget %d exhausted",
                issue_id,
                max_analyses,
            )
            return False
        self._analyses_this_poll += 1

        # ★ P2: Follow-up workspace focus 富化
        workspace_focuses = self._workspace_focus_for_followup(issue)

        result = await asyncio.to_thread(
            self.service.analyze,
            issue,
            prior_replies=replies,
            workspace_focuses=workspace_focuses,
        )
        return await self._apply_result(issue, result, replies)

    def _workspace_focus_for_followup(self, issue: "Issue") -> list[dict] | None:
        """在 follow-up 模式下计算 workspace focus 作为澄清上下文富化。

        仅在以下条件全部满足时返回非空列表：
        - workspace_focus_enabled=True
        - workspace_focus_callback 已设置
        - 该 issue 有 follow-up 分支（通过 callback 判断）
        """
        ws_enabled = getattr(self.config, "workspace_focus_enabled", False)
        if not ws_enabled or self._workspace_focus_callback is None:
            return None
        try:
            return self._workspace_focus_callback(issue)
        except Exception as exc:
            logger.warning("Workspace focus callback failed for issue %s: %s", issue.id, exc)
            return None

    async def _apply_result(
        self,
        issue: "Issue",
        result: ClarifyResult,
        replies: list[str],
    ) -> bool:
        issue_id = str(issue.id or "")
        record = self.registry.get(issue_id)
        if record is None:
            return True

        if result.is_clear:
            status = "resolved" if replies else "clear"
            self.registry.mark_clarification_resolved(
                issue_id,
                fingerprint=result.fingerprint,
                answer=(replies[-1] if replies else None),
                source=("author" if replies else None),
                status=status,
                replies=replies,
            )
            # ★ P2: 澄清解决后清除远端标签
            return self._finish_without_author_wait(issue_id, should_dispatch=True)

        questions = result.questions
        if not questions:
            self.registry.mark_clarification_manual_required(
                issue_id,
                questions=["The clarity analyzer could not produce an actionable question."],
                fingerprint=result.fingerprint,
            )
            return self._finish_without_author_wait(issue_id, should_dispatch=False)
        if not self.config.block_on_unclear:
            self.registry.update_clarification(
                issue_id,
                clarification_status="observation",
                open_questions=questions,
                clarifier_fingerprint=result.fingerprint,
                clarification_replies=replies,
            )
            return self._finish_without_author_wait(issue_id, should_dispatch=True)

        if record.clarification_round >= self.config.max_rounds:
            self.registry.mark_clarification_manual_required(
                issue_id,
                questions=questions,
                fingerprint=result.fingerprint,
            )
            logger.warning(
                "Issue %s still unclear after %d rounds; manual input required",
                issue_id,
                record.clarification_round,
            )
            return self._finish_without_author_wait(issue_id, should_dispatch=False)

        if self.config.author_first and not issue.author_login:
            self.registry.mark_clarification_manual_required(
                issue_id,
                questions=questions,
                fingerprint=result.fingerprint,
            )
            logger.warning(
                "Issue %s requires clarification but its author identity is unavailable",
                issue_id,
            )
            return self._finish_without_author_wait(issue_id, should_dispatch=False)

        prior_item = self.resolver.get_item(issue_id)
        since_comment_id = getattr(record, "clarifier_comment_cursor", None)
        if prior_item is not None and getattr(prior_item, "last_checked_comment_id", None):
            since_comment_id = prior_item.last_checked_comment_id
        self.resolver.clear(issue_id)
        question, options = format_clarification_request(result)
        await self.resolver.request_clarification(
            issue_id=issue_id,
            issue_identifier=str(issue.identifier or issue_id),
            question=question,
            context=str(issue.description or "")[:2000],
            options=options,
            start_with_author=bool(self.config.author_first),
            since_comment_id=since_comment_id,
            author_login=issue.author_login,
        )
        self.registry.update_clarification(
            issue_id,
            clarification_status="awaiting_author",
            author_login=issue.author_login,
            clarification_replies=replies,
        )
        self.registry.mark_clarification_blocked(
            issue_id,
            questions=questions,
            fingerprint=result.fingerprint,
            round_number=record.clarification_round + 1,
        )
        # ★ P2: 阻断时推送远端标签
        self._add_remote_label(issue_id)
        return False

    # --- 运营增强 2: 远端标签推送/清除 ---

    def _finish_without_author_wait(self, issue_id: str, *, should_dispatch: bool) -> bool:
        """Leave the author-waiting state and remove its remote marker."""
        self._remove_remote_label(issue_id)
        return should_dispatch

    def _add_remote_label(self, issue_id: str) -> None:
        """推送远端等待澄清标签到 tracker。"""
        label = getattr(self.config, "remote_label", "")
        if not label or self._tracker is None:
            return
        add = getattr(self._tracker, "add_label", None)
        if callable(add):
            try:
                add(issue_id, label)  # pylint: disable=not-callable
            except Exception as exc:
                logger.warning("Failed to add label %s to issue %s: %s", label, issue_id, exc)

    def _remove_remote_label(self, issue_id: str) -> None:
        """清除远端等待澄清标签。"""
        label = getattr(self.config, "remote_label", "")
        if not label or self._tracker is None:
            return
        remove = getattr(self._tracker, "remove_label", None)
        if callable(remove):
            try:
                remove(issue_id, label)  # pylint: disable=not-callable
            except Exception as exc:
                logger.warning("Failed to remove label %s from issue %s: %s", label, issue_id, exc)


__all__ = ["IssueClarificationGate"]
