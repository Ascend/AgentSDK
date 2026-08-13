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

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from extensions.orchestrator.clarification import (
    ClarificationConfig,
    ClarificationResolver,
)
from extensions.orchestrator.clarification_queue import (
    ClarificationQueue,
    ClarificationStatus,
)
from extensions.orchestrator.tracker import Comment, CommentHistoryCapability


def _make_resolver(
    tmp: Path,
    escalation: str = "skip",
) -> tuple[ClarificationResolver, ClarificationQueue, MagicMock]:
    queue = ClarificationQueue(queue_path=tmp / "queue.json")
    tracker = MagicMock()
    tracker.capabilities = [CommentHistoryCapability]
    tracker.fetch_new_comments_since = AsyncMock(return_value=[])
    tracker.create_clarification_comment = AsyncMock(return_value=Comment(id="c1", body="", author_login=None))
    config = ClarificationConfig(
        timeout_local_seconds=60,
        timeout_author_seconds=120,
        escalation=escalation,
    )
    resolver = ClarificationResolver(
        clarification_queue=queue,
        tracker=tracker,
        config=config,
    )
    return resolver, queue, tracker


class TestCheckForAnswerHonorsPreDeadlineReply(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_pre_deadline_author_reply_wins_after_expiry(self) -> None:
        resolver, queue, tracker = _make_resolver(Path(self.tmp.name))
        now = time.time()
        queue.enqueue(
            issue_id="i1",
            issue_identifier="x#1",
            question="which?",
            options=["a", "b"],
            timeout_seconds=60,
        )
        queue.mark_awaiting_author("i1", timeout_seconds=60)
        item = queue.get("i1")
        self.assertIsNotNone(item)
        item.expires_at = now - 10  # deadline already passed
        item.author_login = "author1"
        tracker.fetch_new_comments_since = AsyncMock(
            return_value=[
                Comment(
                    id="100",
                    body="option a",
                    author_login="author1",
                    created_at="2026-08-10T09:00:00+08:00",
                )
            ]
        )

        asyncio.run(resolver._check_for_answer(item))

        resolved = queue.get_resolved("i1")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.answer, "option a")
        self.assertEqual(resolved.status, ClarificationStatus.RESOLVED_AUTHOR)
        self.assertNotEqual(resolved.status, ClarificationStatus.EXHAUSTED)

    def test_expired_without_answer_triggers_escalation(self) -> None:
        resolver, queue, tracker = _make_resolver(Path(self.tmp.name))
        queue.enqueue(
            issue_id="i1",
            issue_identifier="x#1",
            question="which?",
            timeout_seconds=60,
        )
        queue.mark_awaiting_author("i1", timeout_seconds=60)
        item = queue.get("i1")
        self.assertIsNotNone(item)
        item.expires_at = time.time() - 10
        item.author_login = "author1"

        asyncio.run(resolver._check_for_answer(item))

        self.assertEqual(queue.get("i1").status, ClarificationStatus.EXHAUSTED)


class TestMissingAuthorIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_start_with_author_without_login_falls_back_to_local(self) -> None:
        resolver, queue, tracker = _make_resolver(Path(self.tmp.name))
        result = asyncio.run(
            resolver.request_clarification(
                issue_id="i1",
                issue_identifier="x#1",
                question="which?",
                start_with_author=True,
                author_login=None,
            )
        )
        self.assertEqual(result.status, ClarificationStatus.AWAITING_LOCAL)
        item = queue.get("i1")
        self.assertIsNotNone(item)
        self.assertEqual(item.status, ClarificationStatus.AWAITING_LOCAL)
        tracker.create_clarification_comment.assert_not_awaited()

    def test_local_timeout_without_login_applies_escalation_directly(self) -> None:
        resolver, queue, tracker = _make_resolver(Path(self.tmp.name))
        queue.enqueue(
            issue_id="i1",
            issue_identifier="x#1",
            question="which?",
            timeout_seconds=60,
        )
        queue.mark_awaiting_local("i1")
        item = queue.get("i1")
        self.assertIsNotNone(item)
        item.expires_at = time.time() - 10
        item.author_login = None

        asyncio.run(resolver._check_for_answer(item))

        tracker.create_clarification_comment.assert_not_awaited()
        self.assertEqual(queue.get("i1").status, ClarificationStatus.EXHAUSTED)
