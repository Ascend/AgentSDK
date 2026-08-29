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

import unittest

from extensions.orchestrator.repo_tracker.client import (
    _PLATFORMS,
    RepositoryIssueClient,
)
from extensions.orchestrator.repo_tracker.normalizers import _build_issue_comment_url
from extensions.orchestrator.tracker import PullRequestFeedback


class TestFeedbackUrlBackfill(unittest.TestCase):
    """GitCode issue-comments omit html_url -> reconstruct the comment permalink."""

    def test_build_issue_comment_url_gitcode_uses_tid_anchor(self) -> None:
        url = _build_issue_comment_url(_PLATFORMS["gitcode"], "Gideon_Zhao", "perf-reference-ascend", "4", "179164353")
        self.assertEqual(
            url,
            "https://gitcode.com/Gideon_Zhao/perf-reference-ascend/issues/4#tid-179164353",
        )

    def test_build_issue_comment_url_gitee_uses_tid_anchor(self) -> None:
        url = _build_issue_comment_url(_PLATFORMS["gitee"], "acme", "widget", "12", "55")
        self.assertEqual(url, "https://gitee.com/acme/widget/issues/12#tid-55")

    def test_build_issue_comment_url_github_uses_issuecomment_anchor(self) -> None:
        url = _build_issue_comment_url(_PLATFORMS["github"], "acme", "widget", "12", "55")
        self.assertEqual(url, "https://github.com/acme/widget/issues/12#issuecomment-55")

    def test_build_issue_comment_url_returns_none_when_missing_parts(self) -> None:
        self.assertIsNone(_build_issue_comment_url(_PLATFORMS["gitcode"], "", "repo", "4", "1"))
        self.assertIsNone(_build_issue_comment_url(_PLATFORMS["gitcode"], "o", "r", "4", ""))

    def test_backfill_conversation_url_when_html_url_missing(self) -> None:
        client = RepositoryIssueClient(
            owner="Gideon_Zhao",
            repo="perf-reference-ascend",
            platform="gitcode",
            api_key="t",
        )
        # GitCode payload: no html_url, only the comment id.
        item = PullRequestFeedback(
            id="conversation:179164353",
            source="conversation",
            body="please fix",
            url=None,
        )
        result = client._backfill_feedback_url(item, pr_number="4")
        self.assertEqual(
            result.url,
            "https://gitcode.com/Gideon_Zhao/perf-reference-ascend/issues/4#tid-179164353",
        )

    def test_backfill_keeps_existing_html_url(self) -> None:
        client = RepositoryIssueClient(owner="acme", repo="widget", platform="github", api_key="t")
        existing = "https://github.com/acme/widget/issues/12#issuecomment-99"
        item = PullRequestFeedback(id="conversation:99", source="conversation", body="x", url=existing)
        # Should not overwrite a real html_url from the API.
        self.assertEqual(client._backfill_feedback_url(item, pr_number="12").url, existing)

    def test_backfill_inline_review_url(self) -> None:
        client = RepositoryIssueClient(
            owner="Gideon_Zhao", repo="perf-reference-ascend", platform="gitcode", api_key="t"
        )
        item = PullRequestFeedback(id="inline_review:202", source="inline_review", body="nit", url=None)
        result = client._backfill_feedback_url(item, pr_number="3")
        self.assertEqual(
            result.url,
            "https://gitcode.com/Gideon_Zhao/perf-reference-ascend/issues/3#tid-202",
        )

    def test_backfill_skips_review_summary_and_ci(self) -> None:
        client = RepositoryIssueClient(owner="acme", repo="widget", platform="gitcode", api_key="t")
        for source in ("review_summary", "ci"):
            item = PullRequestFeedback(
                id=f"{source}:5",
                source=source,
                body="x",
                url=None,  # type: ignore[arg-type]
            )
            # review_summary / ci anchors don't map to #tid-; leave url empty.
            self.assertIsNone(client._backfill_feedback_url(item, pr_number="3").url)
