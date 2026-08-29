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

import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SkillSearchConfig
from .document import SkillSearchDocument
from .exceptions import EmptyQueryError, IndexCorruptError

if TYPE_CHECKING:
    from .tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Persistence format version. Increment when serialization format changes.
INDEX_FORMAT_VERSION = 1

# Per-field boost values: name/title matches are more indicative than
# body matches, so they get higher weight.
FIELD_BOOSTS: dict[str, float] = {
    "name": 3.0,
    "title": 3.0,
    "description": 2.0,
    "body": 1.0,
    "tags": 2.5,
}

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexStats:
    """Statistics about the current index state."""

    total_docs: int
    """Number of documents in the index."""

    total_terms: int
    """Number of unique terms in the inverted index."""

    total_inverted_entries: int
    """Total (doc_id, freq) entries across all terms."""

    approximate_bytes: int
    """Approximate in-memory size in bytes."""


# ---------------------------------------------------------------------------
# Scored hit container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredHit:
    """Internal scored hit during search."""

    doc_id: str
    score: float
    matched_terms: frozenset[str]
    term_contributions: dict[str, float]


# ---------------------------------------------------------------------------
# TfIdfSkillIndex
# ---------------------------------------------------------------------------


@dataclass
class TfIdfSkillIndex:
    """In-memory TF-IDF inverted index for skill search.

    Stores documents in ``doc_store``, maintains an inverted index
    mapping terms to (doc_id, frequency) pairs, and precomputes IDF
    values for fast scoring.

    Supports:
    - Full build from a document list
    - Incremental upsert/remove for live updates
    - Atomic save/load to JSON
    - Field-level boosted ranking
    - Length-normalized TF + smoothed IDF
    """

    tokenizer: Tokenizer
    """Tokenizer used to tokenize documents and queries."""

    config: SkillSearchConfig
    """Search configuration (top_k, min_score, etc.)."""

    doc_store: dict[str, SkillSearchDocument] = field(default_factory=dict)
    """Document storage: doc_id → original document."""

    token_counts: dict[str, int] = field(default_factory=dict)
    """Total token count per document (for length normalization)."""

    inverted_index: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    """Inverted index: term → list of (doc_id, term frequency)."""

    doc_freq: dict[str, int] = field(default_factory=dict)
    """Document frequency: term → how many documents contain this term."""

    idf: dict[str, float] = field(default_factory=dict)
    """Precomputed inverse document frequency: term → idf."""

    total_docs: int = 0
    """Total number of documents in the index."""

    # -----------------------------------------------------------------------
    # Public API: Build / update
    # -----------------------------------------------------------------------

    def build(self, documents: Iterable[SkillSearchDocument]) -> None:
        """Full rebuild: clear everything and index all documents.

        Args:
            documents: Documents to index.
        """
        self.doc_store.clear()
        self.token_counts.clear()
        self.inverted_index.clear()
        self.doc_freq.clear()
        self.idf.clear()
        self.total_docs = 0

        for doc in documents:
            self.upsert(doc)

    def upsert(self, doc: SkillSearchDocument) -> None:
        """Incremental upsert: insert or update one document.

        If the document already exists, it is first fully removed from
        the index (including doc_freq and idf updates) before being
        re-indexed with the new content.

        Args:
            doc: Document to insert or update.
        """
        if doc.id in self.doc_store:
            self.remove(doc.id)

        term_freq, raw_token_count = self._tokenize_and_count(doc)
        if not term_freq:
            return

        self.doc_store[doc.id] = doc
        self.token_counts[doc.id] = raw_token_count
        self.total_docs += 1

        for term, freq in term_freq.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            self.inverted_index[term].append((doc.id, freq))
            self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self._recompute_all_idf()

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index.

        After removal, IDF is recomputed for all terms that were in the
        document. If the document is not found, this is a no-op.

        Args:
            doc_id: Document ID to remove.
        """
        if doc_id not in self.doc_store:
            return

        doc = self.doc_store[doc_id]
        term_freq, _ = self._tokenize_and_count(doc)

        self._remove_from_index(doc_id)

        for term in term_freq:
            self.doc_freq[term] -= 1
            if self.doc_freq[term] <= 0:
                if term in self.doc_freq:
                    del self.doc_freq[term]
                if term in self.inverted_index:
                    del self.inverted_index[term]
                if term in self.idf:
                    del self.idf[term]

        del self.doc_store[doc_id]
        del self.token_counts[doc_id]
        self.total_docs -= 1

        if term_freq:
            self._recompute_all_idf()

    # -----------------------------------------------------------------------
    # Public API: Search
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        pinned_doc_ids: set[str] | None = None,
    ) -> list[SkillSearchResult]:
        """Search for documents matching query, return ranked results.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (defaults to config.top_k).
            pinned_doc_ids: Pinned documents always rank above unpinned
                documents, but are still scored among themselves.

        Returns:
            Ranked list of SkillSearchResult, highest score first.
            Only results with score ≥ config.min_score are returned.

        Raises:
            EmptyQueryError: If query is empty after tokenization.
        """
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            raise EmptyQueryError("Empty search query after tokenization")

        top_k = top_k if top_k is not None else self.config.top_k

        scores: dict[str, float] = {}
        matched_terms: dict[str, set[str]] = {}
        term_contributions: dict[str, dict[str, float]] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf_val = self.idf[token]
            idf_sq = idf_val * idf_val

            for doc_id, tf in self.inverted_index[token]:
                total_tokens = self.token_counts[doc_id]
                tf_normalized = tf / math.sqrt(total_tokens)
                doc = self.doc_store[doc_id]
                doc_weight = doc.weight
                contribution = tf_normalized * idf_sq * doc_weight

                scores[doc_id] = scores.get(doc_id, 0.0) + contribution
                if doc_id not in matched_terms:
                    matched_terms[doc_id] = set()
                matched_terms[doc_id].add(token)
                if doc_id not in term_contributions:
                    term_contributions[doc_id] = {}
                term_contributions[doc_id][token] = contribution

        if not scores:
            return []

        scored = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        pinned: list[SkillSearchResult] = []
        unpinned: list[SkillSearchResult] = []
        min_score = self.config.min_score
        pin_set = pinned_doc_ids or set()

        for doc_id, score in scored:
            if score < min_score:
                continue
            doc = self.doc_store[doc_id]
            mts = tuple(sorted(matched_terms[doc_id]))
            result = SkillSearchResult(
                document=doc,
                score=score,
                matched_terms=mts,
                term_contributions=term_contributions[doc_id],
                reason=SkillSearchResult._build_reason(mts),
            )
            if doc_id in pin_set:
                pinned.append(result)
            else:
                unpinned.append(result)

        all_results = pinned + unpinned
        return all_results[:top_k]

    # -----------------------------------------------------------------------
    # Public API: Stats / Persistence
    # -----------------------------------------------------------------------

    def total_stats(self) -> IndexStats:
        """Return current statistics about the index."""
        total_entries = sum(len(entries) for entries in self.inverted_index.values())
        approx_bytes = sum(len(doc_id) for doc_id in self.doc_store) + sum(
            len(term) * len(entries) for term, entries in self.inverted_index.items()
        )
        return IndexStats(
            total_docs=self.total_docs,
            total_terms=len(self.inverted_index),
            total_inverted_entries=total_entries,
            approximate_bytes=approx_bytes,
        )

    def save(self, path: Path) -> Path:
        """Save index to JSON file atomically.

        Writes first to a temporary ``.tmp`` file, then renames it to the
        target path. This ensures the target file is always either the
        old version or a complete new version — no corrupt partial
        write on disk.

        Args:
            path: Target path to save to.

        Returns:
            The target path where the index was saved.
        """
        resolved = path.expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        temp_path = resolved.with_suffix(".json.tmp")

        doc_store_serialized: dict[str, dict] = {}
        for doc_id, doc in self.doc_store.items():
            doc_store_serialized[doc_id] = {
                "id": doc.id,
                "name": doc.name,
                "title": doc.title,
                "description": doc.description,
                "body": doc.body,
                "source": doc.source,
                "tags": list(doc.tags),
                "updated_at": doc.updated_at,
                "weight": doc.weight,
            }

        inverted_serialized: dict[str, list[list]] = {}
        for term, entries in self.inverted_index.items():
            inverted_serialized[term] = [list(entry) for entry in entries]

        data = {
            "version": INDEX_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_docs": self.total_docs,
            "doc_store": doc_store_serialized,
            "token_counts": self.token_counts,
            "inverted_index": inverted_serialized,
            "doc_freq": self.doc_freq,
            "idf": self.idf,
        }

        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(temp_path, resolved)
        return resolved

    @classmethod
    def load(
        cls,
        path: Path,
        tokenizer: Tokenizer,
        config: SkillSearchConfig,
    ) -> TfIdfSkillIndex:
        """Load index from JSON file.

        Args:
            path: Path to the saved index file.
            tokenizer: Tokenizer to use (tokenizer state not persisted).
            config: Search configuration.

        Returns:
            Loaded TfIdfSkillIndex.

        Raises:
            IndexCorruptError: If the file cannot be parsed, or the
                version is incompatible.
        """
        resolved = path.expanduser()
        if not resolved.exists():
            raise IndexCorruptError(f"Index file not found: {resolved}")

        try:
            with resolved.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise IndexCorruptError(f"Invalid JSON in index: {e}") from e

        version = data.get("version")
        if version != INDEX_FORMAT_VERSION:
            raise IndexCorruptError(f"Incompatible index version: got {version}, expected {INDEX_FORMAT_VERSION}")

        doc_store: dict[str, SkillSearchDocument] = {}
        doc_store_data = data.get("doc_store", {})
        for doc_id, doc_data in doc_store_data.items():
            doc = SkillSearchDocument(
                id=doc_data.get("id", doc_id),
                name=doc_data.get("name", ""),
                title=doc_data.get("title", ""),
                description=doc_data.get("description", ""),
                body=doc_data.get("body", ""),
                source=doc_data.get("source", "local"),
                tags=tuple(doc_data.get("tags", [])),
                updated_at=doc_data.get("updated_at"),
                weight=doc_data.get("weight", 1.0),
            )
            doc_store[doc.id] = doc

        index = cls(
            tokenizer=tokenizer,
            config=config,
            doc_store=doc_store,
            token_counts=data.get("token_counts", {}),
            inverted_index={
                term: [(doc_id, freq) for doc_id, freq in entries]  # pylint: disable=unnecessary-comprehension
                for term, entries in data.get("inverted_index", {}).items()
            },
            doc_freq=data.get("doc_freq", {}),
            idf=data.get("idf", {}),
            total_docs=data.get("total_docs", 0),
        )

        return index

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _tokenize_and_count(self, doc: SkillSearchDocument) -> tuple[dict[str, int], int]:
        """Tokenize document by fields, count term frequencies with boosts.

        The field boosts are baked into the term frequencies at index
        time (e.g. a term in ``name`` gets 3.0x frequency vs 1.0x in
        ``body``).  The raw (unboosted) token count is returned
        separately for length normalization::

            score = weight * Σ (boosted_tf / sqrt(raw_token_count)) * idf²

        Args:
            doc: Document to tokenize.

        Returns:
            (Mapping from term to boosted frequency, raw token count)
        """
        term_freq: dict[str, float] = {}
        field_texts = doc.field_text()
        raw_token_count = 0

        for field_name, text in field_texts.items():
            if not text:
                continue
            boost = FIELD_BOOSTS.get(field_name, 1.0)
            tokens = self.tokenizer.tokenize(text)
            raw_token_count += len(tokens)
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0.0) + boost

        freq_int: dict[str, int] = {}
        for term, f in term_freq.items():
            freq_int[term] = round(f) if f.is_integer() else math.ceil(f)

        return freq_int, raw_token_count

    def _remove_from_index(self, doc_id: str) -> None:
        """Remove all references to doc_id from the inverted index.

        Only modifies ``inverted_index``; the caller is responsible for
        updating ``doc_freq``, ``idf``, and cleaning up empty terms.
        """
        for term, entries in self.inverted_index.items():
            new_entries = [entry for entry in entries if entry[0] != doc_id]
            if len(new_entries) != len(entries):
                self.inverted_index[term] = new_entries

    def _recompute_all_idf(self) -> None:
        """Recompute IDF for all terms in the index after index change."""
        for term in self.doc_freq:  # pylint: disable=consider-using-dict-items
            df = self.doc_freq[term]
            self.idf[term] = self._compute_idf(df)

    def _compute_idf(self, df: int) -> float:
        """Compute smoothed IDF value.

        Formula::

            idf(t) = log( (N + 1) / (df + 1) ) + 1

        - +1 on numerator and denominator for smoothing
        - +1 at the end ensures idf is always positive
        - At search time we square idf to give rare terms more weight
        """
        if self.total_docs == 0:
            return 1.0
        return math.log((self.total_docs + 1) / (df + 1)) + 1


# ---------------------------------------------------------------------------
# SkillSearchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillSearchResult:
    """Result of a skill search query.

    Contains the matched document, the composite score, the list of
    matched terms, per-term contribution breakdown, and a human-readable
    reason string.
    """

    document: SkillSearchDocument
    """The matched skill document."""

    score: float
    """Final composite score (higher = more relevant)."""

    matched_terms: tuple[str, ...]
    """Terms from the query that matched this document."""

    term_contributions: dict[str, float]
    """Per-term contribution to the final score, for inspection."""

    reason: str
    """Human-readable explanation, e.g. matched "browser", "playwright"."""

    @staticmethod
    def _build_reason(matched_terms: tuple[str, ...]) -> str:
        if not matched_terms:
            return ""
        return "matched " + ", ".join(f'"{t}"' for t in sorted(matched_terms))
