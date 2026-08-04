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

"""P92-D: SkillSearcher — high-level search API built on TfIdfSkillIndex.

Wraps index lifecycle (load/build/save), post-filtering, pin management,
and inspection behind a single ``SkillSearcher`` entry point.

Architecture
------------
::

    SkillSearcher
      ├─ TfIdfSkillIndex (P92-C)  ← scoring + persistence
      ├─ SkillRegistryExt         ← skill source
      ├─ Tokenizer (P92-B)        ← query + document tokenization
      └─ pinned.json              ← persisted pin list
"""

from __future__ import annotations


from extensions.skills_ext.registry_ext import SkillRegistryExt  # noqa: E402

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SkillSearchConfig
from .document import SkillSearchDocument, SourceType, extract_batch
from .exceptions import IndexCorruptError, SearchDisabledError
from .index import IndexStats, SkillSearchResult, TfIdfSkillIndex
from .tokenizer import create_default_tokenizer

if TYPE_CHECKING:
    from .tokenizer import Tokenizer
    from .watcher import SkillIndexWatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# InspectResult / FieldInspect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldInspect:
    """Per-field token breakdown for a single skill document."""

    token_count: int
    """Number of tokens in this field."""

    token_sample: list[str]
    """First 20 token samples from this field."""


@dataclass(frozen=True)
class InspectResult:
    """Diagnostic view of a skill document in the index."""

    name: str
    """Skill name."""

    source: SourceType
    """Skill source type."""

    token_count: int
    """Total raw token count across all fields."""

    fields: dict[str, FieldInspect]
    """Per-field token breakdown (name, title, description, body, tags)."""


# ---------------------------------------------------------------------------
# SkillSearcher
# ---------------------------------------------------------------------------


class SkillSearcher:
    """High-level skill search API.

    Responsibilities:
    - Index lifecycle: load from disk or build from registry
    - Search with post-filtering (tags, source) and pinned prioritization
    - Pin management with persistence
    - Inspection and statistics

    Usage::

        searcher = SkillSearcher(registry, config=config)
        await searcher.ensure_index()
        results = await searcher.search("browser automation")
    """

    def __init__(
        self,
        registry: SkillRegistryExt,  # SkillRegistryExt
        *,
        config: SkillSearchConfig,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._registry = registry
        self._config = config
        self._tokenizer = tokenizer or create_default_tokenizer()
        self._index: TfIdfSkillIndex | None = None
        self._pinned_names: list[str] = []
        self._name_to_doc_ids: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Public API: ensure_index / refresh
    # ------------------------------------------------------------------

    async def ensure_index(self) -> None:
        """Ensure the index is loaded and ready for search.

        On first call, attempts to load the index from disk.  If the
        persisted index is missing or corrupt, it is rebuilt from the
        registry and saved back to disk.

        Subsequent calls are no-ops if the index is already loaded.

        Raises:
            SearchDisabledError: If ``config.enabled`` is ``False``.
        """
        if not self._config.enabled:
            raise SearchDisabledError("Skill search is disabled (SKILL_SEARCH_TFIDF feature flag is off)")

        if self._index is not None:
            return

        index_path = self._config.index_path.expanduser()
        try:
            self._index = TfIdfSkillIndex.load(index_path, self._tokenizer, self._config)
            self._load_pinned()
            self._rebuild_name_to_doc_ids()
            logger.info(
                "Loaded skill search index from %s (%d docs)",
                index_path,
                self._index.total_docs,
            )
        except (IndexCorruptError, FileNotFoundError):
            logger.info(
                "Index not found or corrupt, building from registry (%s)",
                index_path,
            )
            await self.refresh()

    async def refresh(self) -> None:
        """Force-rebuild the index from the registry and persist to disk.

        If ``config.enabled`` is ``False``, this is a silent no-op.
        """
        if not self._config.enabled:
            return

        documents = self._collect_documents()
        index = TfIdfSkillIndex(tokenizer=self._tokenizer, config=self._config)
        index.build(documents)
        self._index = index
        self._rebuild_name_to_doc_ids()

        index_path = self._config.index_path.expanduser()
        self._load_pinned()
        index.save(index_path)
        logger.info(
            "Skill search index rebuilt: %d docs, %d terms → %s",
            index.total_docs,
            len(index.inverted_index),
            index_path,
        )

    # ------------------------------------------------------------------
    # Public API: search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        tags: list[str] | None = None,
        source: SourceType | None = None,
    ) -> list[SkillSearchResult]:
        """Search for skills matching *query*.

        If ``ensure_index()`` has not been called yet, it is invoked
        implicitly.

        Args:
            query: Natural language query string.
            top_k: Max results to return (defaults to ``config.top_k``).
            tags: If provided, only return results whose skill tags
                include **any** of the given tags.
            source: If provided, only return results whose source type
                matches.

        Returns:
            Ranked list of ``SkillSearchResult``, highest score first.
            Only results with score ≥ ``config.min_score`` pass the
            threshold.  Pinned skills always appear before unpinned.
        """
        await self.ensure_index()
        assert self._index is not None

        top_k = top_k if top_k is not None else self._config.top_k
        pinned_doc_ids = self._resolve_pinned_doc_ids()

        results = self._index.search(query, top_k=None, pinned_doc_ids=pinned_doc_ids)

        if tags:
            results = self._filter_by_tags(results, tags)
        if source:
            results = self._filter_by_source(results, source)

        return results[:top_k]

    # ------------------------------------------------------------------
    # Public API: pin management
    # ------------------------------------------------------------------

    def pin(self, name: str) -> None:
        """Pin a skill by name so it always ranks above unpinned skills.

        If the skill is already pinned, this is a no-op.
        """
        if name not in self._pinned_names:
            self._pinned_names.append(name)
            self._save_pinned()

    def unpin(self, name: str) -> None:
        """Remove a skill from the pinned list.

        If the skill is not pinned, this is a no-op.
        """
        if name in self._pinned_names:
            self._pinned_names.remove(name)
            self._save_pinned()

    def get_pinned(self) -> list[str]:
        """Return the current pinned skill names in pin order."""
        return list(self._pinned_names)

    # ------------------------------------------------------------------
    # Public API: inspect / stats
    # ------------------------------------------------------------------

    def inspect(self, name: str) -> InspectResult | None:
        """Return a diagnostic view of a skill's token representation.

        Returns ``None`` if the index is not loaded or the skill is not
        found.

        Args:
            name: Skill name to inspect.
        """
        if self._index is None:
            return None

        doc = self._find_doc_by_name(name)
        if doc is None:
            return None

        field_texts = doc.field_text()
        fields: dict[str, FieldInspect] = {}
        total_tokens = 0

        for field_name in ("name", "title", "description", "body", "tags"):
            text = field_texts.get(field_name, "")
            tokens = self._tokenizer.tokenize(text)
            fields[field_name] = FieldInspect(
                token_count=len(tokens),
                token_sample=tokens[:20],
            )
            total_tokens += len(tokens)

        return InspectResult(
            name=doc.name,
            source=doc.source,
            token_count=total_tokens,
            fields=fields,
        )

    def stats(self) -> IndexStats | None:
        """Return current index statistics, or ``None`` if not loaded."""
        if self._index is None:
            return None
        return self._index.total_stats()

    # ------------------------------------------------------------------
    # Public API: watcher factory
    # ------------------------------------------------------------------

    def create_watcher(self) -> SkillIndexWatcher:
        """Create a :class:`SkillIndexWatcher` bound to this searcher.

        The returned watcher is **not** started automatically — call
        ``watcher.start()`` to begin listening for registry changes.

        Typical usage::

            searcher = SkillSearcher(registry, config=config)
            await searcher.ensure_index()
            watcher = searcher.create_watcher()
            watcher.start()
        """
        from .watcher import SkillIndexWatcher

        return SkillIndexWatcher(self, self._registry, config=self._config)

    # ------------------------------------------------------------------
    # Internal: index building
    # ------------------------------------------------------------------

    def _collect_documents(self) -> list[SkillSearchDocument]:
        """Collect SkillSearchDocument from the registry."""
        skills = self._registry.get_all_skills(force_refresh=True)
        return extract_batch(skills)

    def _rebuild_name_to_doc_ids(self) -> None:
        """Rebuild the name → doc_ids mapping from the index."""
        self._name_to_doc_ids.clear()
        if self._index is None:
            return
        for doc_id, doc in self._index.doc_store.items():
            self._name_to_doc_ids.setdefault(doc.name, set()).add(doc_id)

    def _resolve_pinned_doc_ids(self) -> set[str]:
        """Resolve pinned skill names to doc IDs."""
        result: set[str] = set()
        for name in self._pinned_names:
            doc_ids = self._name_to_doc_ids.get(name)
            if doc_ids:
                result.update(doc_ids)
        return result

    def _find_doc_by_name(self, name: str) -> SkillSearchDocument | None:
        """Find a document by name in the index."""
        if self._index is None:
            return None
        doc_ids = self._name_to_doc_ids.get(name)
        if not doc_ids:
            return None
        return self._index.doc_store.get(next(iter(doc_ids)))

    # ------------------------------------------------------------------
    # Internal: post-filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_tags(
        results: list[SkillSearchResult],
        tags: list[str],
    ) -> list[SkillSearchResult]:
        tag_set = {t.lower() for t in tags}
        return [r for r in results if tag_set & {t.lower() for t in r.document.tags}]

    @staticmethod
    def _filter_by_source(
        results: list[SkillSearchResult],
        source: SourceType,
    ) -> list[SkillSearchResult]:
        return [r for r in results if r.document.source == source]

    # ------------------------------------------------------------------
    # Internal: pin persistence
    # ------------------------------------------------------------------

    @property
    def _pinned_path(self) -> Path:
        return self._config.index_path.expanduser().parent / "pinned.json"

    def _load_pinned(self) -> None:
        path = self._pinned_path
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._pinned_names = data
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load pinned skills from %s", path)
            self._pinned_names = []

    def _save_pinned(self) -> None:
        path = self._pinned_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(
                json.dumps(self._pinned_names, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except OSError:
            logger.warning("Failed to save pinned skills to %s", path)
