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

"""Review-feedback tracking mutations (split from issue_registry.py)."""

from __future__ import annotations

import time
from collections.abc import Mapping

from .models import IssueRecord


class FeedbackMixin:
    """Review-feedback bookkeeping on issue records.

    The host class provides ``_records`` and ``_save`` (from StorageMixin).
    """

    def mark_feedback_pending(
        self,
        issue_id: str,
        feedback_ids: list[str],
        *,
        cursor: str | None = None,
        feedback_urls: Mapping[str, str] | None = None,
    ) -> IssueRecord | None:
        """Record newly discovered feedback ids as pending.

        Skips already-processed ids, deduplicates against the pending
        set, stores canonical URLs when provided, and starts the
        staleness clock when the pending set was previously empty.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        now = time.time()
        if not record.pending_feedback_ids:
            record.pending_feedback_since = now
        seen = set(record.pending_feedback_ids)
        processed = set(record.processed_feedback_ids)
        for feedback_id in feedback_ids:
            if feedback_id in processed:
                continue
            if feedback_id not in seen:
                record.pending_feedback_ids.append(feedback_id)
                seen.add(feedback_id)
                # Record the first-seen time per id so staleness can be
                # judged individually (see ``clear_stale_pending``).
                record.pending_feedback_since_map[feedback_id] = now
            # A feedback item may first be discovered without ``html_url``
            # and receive a reconstructed URL on a later poll. Update the
            # lookup even when the pending ID already exists.
            url = feedback_urls.get(feedback_id) if feedback_urls else None
            if url:
                record.pending_feedback_urls[feedback_id] = url
        if cursor is not None:
            record.feedback_cursor = cursor
        record.last_feedback_checked_at = time.time()
        record.touch()
        self._save()
        return record

    def mark_feedback_processed(
        self,
        issue_id: str,
        feedback_ids: list[str],
        *,
        commit_sha: str | None = None,
        cursor: str | None = None,
    ) -> IssueRecord | None:
        """Move feedback ids from pending to processed.

        Removes their URL lookups, clears the staleness clock once
        nothing remains pending, and records the follow-up commit sha /
        cursor when provided.
        """
        record = self._records.get(issue_id)
        if record is None:
            return None
        processed = set(record.processed_feedback_ids)
        for feedback_id in feedback_ids:
            if feedback_id not in processed:
                record.processed_feedback_ids.append(feedback_id)
                processed.add(feedback_id)
        record.pending_feedback_ids = [
            feedback_id for feedback_id in record.pending_feedback_ids if feedback_id not in processed
        ]
        for feedback_id in feedback_ids:
            record.pending_feedback_urls.pop(feedback_id, None)
            record.pending_feedback_since_map.pop(feedback_id, None)
        if not record.pending_feedback_ids:
            record.pending_feedback_since = None
            record.pending_feedback_since_map.clear()
        if commit_sha is not None:
            record.last_followup_commit_sha = commit_sha
        if cursor is not None:
            record.feedback_cursor = cursor
        record.last_feedback_checked_at = time.time()
        record.touch()
        self._save()
        return record

    def increment_followup_attempt(self, issue_id: str) -> IssueRecord | None:
        """Increment the follow-up attempt counter for an issue."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.followup_attempt_count += 1
        record.touch()
        self._save()
        return record

    def clear_stale_pending(self, issue_id: str, timeout_seconds: int = 600) -> int:
        """Drop pending feedback older than ``timeout_seconds``.

        Age is judged per id via ``pending_feedback_since_map``, falling
        back to the legacy ``pending_feedback_since`` clock; ids with no
        recorded time are kept.

        Returns:
            The number of dropped ids, or ``0`` when nothing was stale.
        """
        record = self._records.get(issue_id)
        if record is None or not record.pending_feedback_ids:
            return 0
        now = time.time()
        stale_ids = []
        for feedback_id in record.pending_feedback_ids:
            first_seen = record.pending_feedback_since_map.get(feedback_id)
            if first_seen is None:
                # Back-compat: the per-id map may be absent on records
                # loaded from registry.json written before that field.
                first_seen = record.pending_feedback_since
            if first_seen is not None and now - first_seen >= timeout_seconds:
                stale_ids.append(feedback_id)
        if not stale_ids:
            return 0
        stale_set = set(stale_ids)
        record.pending_feedback_ids = [
            feedback_id for feedback_id in record.pending_feedback_ids if feedback_id not in stale_set
        ]
        for feedback_id in stale_ids:
            record.pending_feedback_urls.pop(feedback_id, None)
            record.pending_feedback_since_map.pop(feedback_id, None)
        if not record.pending_feedback_ids:
            record.pending_feedback_since = None
            record.pending_feedback_since_map.clear()
        record.touch()
        self._save()
        return len(stale_ids)

    def mark_feedback_checked(self, issue_id: str) -> IssueRecord | None:
        """Stamp ``last_feedback_checked_at`` with the current time."""
        record = self._records.get(issue_id)
        if record is None:
            return None
        record.last_feedback_checked_at = time.time()
        record.touch()
        self._save()
        return record
