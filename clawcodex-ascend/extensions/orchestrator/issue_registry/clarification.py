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

"""Clarification field mutations for the three-channel flow (split from issue_registry.py)."""

from __future__ import annotations

from .models import IssueRecord


class ClarificationMixin:
    """Clarification-related record mutations.

    The host class provides ``_records`` and ``_save`` (from StorageMixin).
    """

    def update_clarification(
        self,
        issue_id: str,
        *,
        clarification_status: str | None = None,
        question: str | None = None,
        author_login: str | None = None,
        local_answer: str | None = None,
        local_answer_source: str | None = None,
        first_response_source: str | None = None,
        open_questions: list[str] | None = None,
        clarification_round: int | None = None,
        clarifier_fingerprint: str | None = None,
        clarification_replies: list[str] | None = None,
        clarifier_comment_cursor: str | None = None,
    ) -> IssueRecord | None:
        """Update clarification-related fields on an issue record.

        Only non-``None`` kwargs are applied; ``question`` is appended
        to the audit trail ``question_history``.

        Returns:
            The updated record, or ``None`` if the issue is unknown.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        if clarification_status is not None:
            record.clarification_status = clarification_status
        if question is not None:
            record.question_history.append(question)
        if author_login is not None:
            record.author_login = author_login
        if local_answer is not None:
            record.local_answer = local_answer
        if local_answer_source is not None:
            record.local_answer_source = local_answer_source
        if first_response_source is not None:
            record.first_response_source = first_response_source
        if open_questions is not None:
            record.open_questions = list(open_questions)
        if clarification_round is not None:
            record.clarification_round = max(0, int(clarification_round))
        if clarifier_fingerprint is not None:
            record.clarifier_fingerprint = clarifier_fingerprint
        if clarification_replies is not None:
            record.clarification_replies = list(clarification_replies)
        if clarifier_comment_cursor is not None:
            record.clarifier_comment_cursor = clarifier_comment_cursor
        record.touch()
        self._save()
        return record

    def mark_clarification_blocked(
        self,
        issue_id: str,
        *,
        questions: list[str],
        fingerprint: str,
        round_number: int,
    ) -> IssueRecord | None:
        """Record a clarification wait without abandoning the issue.

        Sets ``clarification_status`` to ``awaiting_author``, stores the
        open questions / fingerprint / round number, and appends new
        questions to the audit trail.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.clarification_status = "awaiting_author"
        record.open_questions = list(questions)
        record.clarification_round = max(1, int(round_number))
        record.clarifier_fingerprint = fingerprint
        for question in questions:
            if question not in record.question_history:
                record.question_history.append(question)
        record.touch()
        self._save()
        return record

    def mark_clarification_resolved(
        self,
        issue_id: str,
        *,
        fingerprint: str,
        answer: str | None = None,
        source: str | None = None,
        status: str = "resolved",
        replies: list[str] | None = None,
    ) -> IssueRecord | None:
        """Mark the issue's clarification as resolved.

        Clears the open questions, records the answer and its source
        (also stored as the first response source when given), and sets
        the resolution status.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.clarification_status = status
        record.open_questions = []
        record.clarifier_fingerprint = fingerprint
        if answer is not None:
            record.local_answer = answer
        if source is not None:
            record.local_answer_source = source
            record.first_response_source = source
        if replies is not None:
            record.clarification_replies = list(replies)
        record.touch()
        self._save()
        return record

    def mark_clarification_manual_required(
        self,
        issue_id: str,
        *,
        questions: list[str],
        fingerprint: str,
    ) -> IssueRecord | None:
        """Mark the issue as requiring manual clarification.

        Stores the open questions and fingerprint with
        ``clarification_status`` set to ``manual_required``.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.clarification_status = "manual_required"
        record.open_questions = list(questions)
        record.clarifier_fingerprint = fingerprint
        record.touch()
        self._save()
        return record

    def add_stale_answer(self, issue_id: str, stale_answer: str) -> IssueRecord | None:
        """Record a stale (rejected) answer for later notification.

        Appends the answer to ``stale_answers`` for surface to the
        operator (e.g. by the IM/CLI feedback flow).
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.stale_answers.append(stale_answer)
        record.touch()
        self._save()
        return record
