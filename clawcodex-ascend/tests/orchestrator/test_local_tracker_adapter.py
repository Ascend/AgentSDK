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
from pathlib import Path
from tempfile import TemporaryDirectory

from extensions.orchestrator.local_tracker.adapter import LocalTrackerAdapter
from extensions.orchestrator.tracker import PullRequestRef


def _write_issue(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestLocalTrackerAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_markdown_issues_are_filtered_and_sorted(self) -> None:
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp)
            _write_issue(
                issues_path / "ready.md",
                """---
id: LOCAL-002
identifier: LOCAL-002
state: ready
priority: 2
labels:
  - orchestrator
---
# Ready issue

Do this second.
""",
            )
            _write_issue(
                issues_path / "open.md",
                """---
id: LOCAL-001
identifier: LOCAL-001
state: open
priority: 1
---
# Open issue

Do this first.
""",
            )
            _write_issue(
                issues_path / "done.md",
                """---
id: LOCAL-003
identifier: LOCAL-003
state: completed
priority: 0
---
# Done issue
""",
            )

            adapter = LocalTrackerAdapter(issues_path)
            issues = await adapter.fetch_candidate_issues()

        self.assertEqual([issue.id for issue in issues], ["LOCAL-001", "LOCAL-002"])
        self.assertEqual(issues[0].title, "Open issue")
        self.assertEqual(issues[0].description, "Do this first.")
        self.assertEqual(issues[0].branch_name, "local/local-001-open-issue")
        self.assertEqual(issues[0].depends_on, [])
        self.assertEqual(issues[1].labels, ["orchestrator"])

    async def test_markdown_issue_parses_depends_on_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp)
            _write_issue(
                issues_path / "dependent.md",
                """---
id: LOCAL-002
identifier: LOCAL-002
state: open
depends_on:
  - LOCAL-001
  - LOCAL-003
---
# Dependent issue
""",
            )

            adapter = LocalTrackerAdapter(issues_path)
            issues = await adapter.fetch_candidate_issues()

        self.assertEqual(issues[0].depends_on, ["LOCAL-001", "LOCAL-003"])

    async def test_fetch_issue_states_rereads_document(self) -> None:
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp)
            issue_path = issues_path / "issue.md"
            _write_issue(
                issue_path,
                """---
id: LOCAL-001
state: open
---
# Test issue
""",
            )
            adapter = LocalTrackerAdapter(issues_path)

            first = await adapter.fetch_issue_states_by_ids(["LOCAL-001"])
            _write_issue(
                issue_path,
                """---
id: LOCAL-001
state: ready
---
# Test issue
""",
            )
            second = await adapter.fetch_issue_states_by_ids(["LOCAL-001"])

        self.assertEqual(first["LOCAL-001"].state, "open")
        self.assertEqual(second["LOCAL-001"].state, "ready")

    async def test_update_issue_state_preserves_body(self) -> None:
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp)
            issue_path = issues_path / "issue.md"
            _write_issue(
                issue_path,
                """---
id: LOCAL-001
state: open
---
# Keep me

Body remains.
""",
            )
            adapter = LocalTrackerAdapter(issues_path)

            await adapter.update_issue_state("LOCAL-001", "completed")

            updated = issue_path.read_text(encoding="utf-8")

        self.assertIn("state: completed", updated)
        self.assertIn("# Keep me\n\nBody remains.", updated)
        self.assertIn("updated_at:", updated)

    async def test_comments_round_trip_through_ndjson(self) -> None:
        with TemporaryDirectory() as tmp:
            adapter = LocalTrackerAdapter(Path(tmp))

            await adapter.create_comment("LOCAL-001", "sync complete")
            clarification = await adapter.create_clarification_comment(
                "LOCAL-001",
                "Need details",
                mentions=["alice"],
            )
            comments = await adapter.fetch_issue_comments("LOCAL-001")
            new_comments = await adapter.fetch_new_comments_since(
                "LOCAL-001",
                comments[0].id,
            )

        self.assertIsNotNone(clarification)
        self.assertEqual([comment.author_login for comment in comments], ["clawcodex", "clawcodex"])
        self.assertEqual(comments[0].body, "sync complete")
        self.assertEqual(comments[1].body, "@alice\n\nNeed details")
        self.assertEqual(new_comments, [comments[1]])

    async def test_update_comment_rewrites_matching_ndjson_record(self) -> None:
        with TemporaryDirectory() as tmp:
            adapter = LocalTrackerAdapter(Path(tmp))
            first = await adapter.create_comment("LOCAL-001", "in progress")
            second = await adapter.create_comment("LOCAL-001", "unchanged")
            assert first is not None
            assert second is not None

            updated = await adapter.update_comment("LOCAL-001", first.id or "", "sync complete")
            comments = await adapter.fetch_issue_comments("LOCAL-001")
            tmp_files = list(Path(tmp).glob("*.tmp"))

        assert updated is not None
        self.assertEqual(updated.id, first.id)
        self.assertEqual(updated.body, "sync complete")
        self.assertEqual(comments[0].body, "sync complete")
        self.assertEqual(comments[1].body, "unchanged")
        self.assertEqual(tmp_files, [])

    async def test_comment_files_include_hash_to_avoid_sanitized_name_collision(self) -> None:
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp)
            adapter = LocalTrackerAdapter(issues_path)

            await adapter.create_comment("LOCAL/001", "first")
            await adapter.create_comment("LOCAL:001", "second")

            comment_files = sorted(issues_path.glob("*.comments.ndjson"))

        self.assertEqual(len(comment_files), 2)

    async def test_adapter_state_lists_are_returned_as_copies(self) -> None:
        adapter = LocalTrackerAdapter("/tmp/issues")

        active_states = adapter.active_states
        active_states.append("mutated")

        self.assertEqual(adapter.active_states, ["open", "ready"])

    async def test_find_pull_request_skips_matching_document_without_pr_url(self) -> None:
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp)
            _write_issue(
                issues_path / "without-pr.md",
                """---
id: LOCAL-001
state: open
branch_name: local/branch
base_branch: main
---
# Missing PR URL
""",
            )
            _write_issue(
                issues_path / "with-pr.md",
                """---
id: LOCAL-002
state: open
branch_name: local/branch
base_branch: main
pr_number: '43'
pr_url: https://example.invalid/pr/43
pr_title: Complete PR
---
# Complete PR
""",
            )
            adapter = LocalTrackerAdapter(issues_path)

            pr = await adapter.find_pull_request(
                head_branch="local/branch",
                base_branch="main",
            )

        self.assertEqual(
            pr,
            PullRequestRef(
                number="43",
                url="https://example.invalid/pr/43",
                title="Complete PR",
            ),
        )

    async def test_find_pull_request_uses_local_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp)
            _write_issue(
                issues_path / "issue.md",
                """---
id: LOCAL-001
state: open
branch_name: local/branch
base_branch: main
pr_number: '42'
pr_url: https://example.invalid/pr/42
pr_title: Local PR
---
# Test issue
""",
            )
            adapter = LocalTrackerAdapter(issues_path)

            pr = await adapter.find_pull_request(
                head_branch="local/branch",
                base_branch="main",
            )

        self.assertEqual(
            pr,
            PullRequestRef(
                number="42",
                url="https://example.invalid/pr/42",
                title="Local PR",
            ),
        )
