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

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from extensions.skills_ext.registry_ext import SkillRegistryExt

from .config import SkillSearchConfig
from .document import extract_search_document

if TYPE_CHECKING:
    from clawcodex_ext.skills.model import Skill

    from .searcher import SkillSearcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SkillIndexWatcher
# ---------------------------------------------------------------------------


class SkillIndexWatcher:
    """Watches ``SkillRegistryExt`` for changes and updates the index incrementally.

    Responsibilities:
    - Registers as a callback on ``SkillRegistryExt.on_skill_registered``
    - On skill registration: extracts document → upserts into memory index
    - Cooldown save: batches multiple consecutive registrations into a single
      disk write (controlled by ``config.save_cooldown_seconds``)
    - Thread safety: ``threading.Lock`` protects concurrent index mutations
    - Lifecycle: ``start()`` / ``stop()`` control whether listening is active

    When ``config.enabled`` is ``False``, ``start()`` is a no-op (zero overhead).

    Usage::

        searcher = SkillSearcher(registry, config=config)
        await searcher.ensure_index()
        watcher = SkillIndexWatcher(searcher, registry, config=config)
        watcher.start()
    """

    def __init__(
        self,
        searcher: SkillSearcher,
        registry: SkillRegistryExt,
        *,
        config: SkillSearchConfig,
    ) -> None:
        self._searcher = searcher
        self._registry = registry
        self._config = config
        self._lock = threading.Lock()
        self._last_save_time: float = 0.0
        self._active = False

    # ------------------------------------------------------------------
    # Public API: start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening for registry changes.

        If ``config.enabled`` is ``False``, this is a silent no-op.
        If already active, repeated calls are no-ops.
        """
        if not self._config.enabled:
            return
        if self._active:
            return
        self._registry.on_skill_registered(self._on_skill_registered)
        self._active = True
        logger.info("SkillIndexWatcher started")

    def stop(self) -> None:
        """Stop listening for registry changes.

        Unregisters the callback.  If not active, this is a no-op.

        Note: ``stop()`` does **not** trigger a save — any modifications
        that have not yet been persisted will be lost.
        """
        if not self._active:
            return
        self._registry.off_skill_registered(self._on_skill_registered)
        self._active = False
        logger.info("SkillIndexWatcher stopped")

    # ------------------------------------------------------------------
    # Internal: registry callback
    # ------------------------------------------------------------------

    def _on_skill_registered(self, skill: Skill) -> None:
        """Callback invoked when a skill is registered in the registry.

        Extracts a ``SkillSearchDocument`` from the skill, upserts it
        into the in-memory index, rebuilds the name→doc_id mapping, and
        schedules a cooldown-batched save to disk.

        Hidden skills are filtered out by ``extract_search_document``.
        If the index is not yet loaded, the callback silently skips.
        """
        with self._lock:
            index = self._searcher._index
            if index is None:
                return

            try:
                doc = extract_search_document(skill)
            except Exception:
                logger.warning(
                    "Failed to extract search document for skill %r",
                    getattr(skill, "name", "?"),
                    exc_info=True,
                )
                return

            if doc is None:
                return

            index.upsert(doc)
            self._searcher._rebuild_name_to_doc_ids()
            self._schedule_save()

    # ------------------------------------------------------------------
    # Internal: cooldown save
    # ------------------------------------------------------------------

    def _schedule_save(self) -> None:
        """Save the index to disk if the cooldown period has elapsed.

        Uses ``time.monotonic()`` so the cooldown is unaffected by
        system clock adjustments.  The cooldown duration is controlled
        by ``config.save_cooldown_seconds`` (default 5s).
        """
        now = time.monotonic()
        if now - self._last_save_time < self._config.save_cooldown_seconds:
            return

        index = self._searcher._index
        if index is None:
            return

        try:
            index.save(self._config.index_path)
            self._last_save_time = now
            logger.debug(
                "Skill index saved (%d docs, %d terms)",
                index.total_docs,
                len(index.inverted_index),
            )
        except OSError:
            logger.warning(
                "Failed to save skill index to %s",
                self._config.index_path,
            )
