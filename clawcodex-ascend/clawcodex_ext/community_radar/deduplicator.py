# pylint: disable=relative-beyond-top-level
"""Cross-project feature deduplicator for SR-5.1.

Mirrors the contract sketched in the feature design:

> Cross-project dedup: semantic similarity over title + description
> (TF-IDF + cosine); features appearing in several projects merge into
> a single FeatureRecord whose related_projects lists every source.

The implementation ships two strategies:

1. **TF-IDF cosine** (default, pure-stdlib) — tokenise ``title +
   description`` after lowercasing + stop-word stripping, build a
   sparse TF vector per record, then cosine-similarity. Vectors are
   **pre-computed once** for all records to avoid O(N²) rebuilds.

2. **Exact-match fallback** — when only one record is present or TF-IDF
   would produce a degenerate zero-vector (e.g. all stop words), fall
   back to case-folded title prefix matching so common short titles
   like "Add context compression" are not silently dropped.

``scikit-learn`` is **not** required: the built-in TF-IDF scorer is
intentionally dependency-free so the radar runs on the same minimal
environment as the rest of ClawCodex.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .models import FeatureRecord

_log = logging.getLogger(__name__)


# Minimal English stop-word list; enough to deflate common tokens that
# otherwise dominate TF-IDF ("the", "and", "for", "to", …). Avoiding
# NLTK keeps the radar free of an extra dependency.
_STOP_WORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "while",
        "for",
        "from",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "without",
        "within",
        "about",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "shall",
        "will",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "there",
        "here",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "you",
        "your",
        "i",
        "me",
        "my",
        "our",
        "we",
        "us",
        "he",
        "she",
        "his",
        "her",
        "not",
        "no",
        "nor",
        "so",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "into",
        "onto",
        "via",
        "per",
        "also",
        "just",
        "very",
        "more",
        "most",
        "much",
        "many",
        "some",
        "any",
        "all",
        "each",
        "every",
        "both",
        "few",
        "other",
        "another",
        "such",
        "only",
        "own",
        "same",
        "than",
        "too",
    ]
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")

# Threshold (0..1) used by the cheap prefix-overlap fallback merge.
PREFIX_OVERLAP_THRESHOLD: float = 0.6


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (no external deps)
# ---------------------------------------------------------------------------


@dataclass
class _Vector:
    """A sparse term-frequency vector."""

    counts: Counter[str]

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.counts.values()))


def _build_tfidf(
    docs: list[list[str]],
) -> tuple[list[_Vector], dict[str, float]]:
    """Compute TF-IDF vectors and the IDF map for ``docs``."""
    if not docs:
        return [], {}

    df: Counter[str] = Counter()
    for tokens in docs:
        for term in set(tokens):
            df[term] += 1
    n = len(docs)
    idf = {
        term: math.log((1 + n) / (1 + freq)) + 1.0  # smoothed IDF
        for term, freq in df.items()
    }

    vectors: list[_Vector] = []
    for tokens in docs:
        if not tokens:
            vectors.append(_Vector(counts=Counter()))
            continue
        tf = Counter(tokens)
        # Length-normalise TF so long descriptions don't dominate.
        length = max(len(tokens), 1)
        weighted = {term: (count / length) * idf.get(term, 1.0) for term, count in tf.items()}
        vectors.append(_Vector(counts=Counter(weighted)))
    return vectors, idf


def _cosine(a: _Vector, b: _Vector) -> float:
    if not a.counts or not b.counts:
        return 0.0
    # Iterate over the smaller vector for speed.
    small, big = (a.counts, b.counts) if len(a.counts) < len(b.counts) else (b.counts, a.counts)
    dot = 0.0
    for term, weight in small.items():
        other = big.get(term)
        if other:
            dot += weight * other
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return dot / denom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


SimilarityFn = Callable[[FeatureRecord, FeatureRecord], float]


def default_scorer() -> SimilarityFn:
    """Return a fresh scorer using TF-IDF cosine on title + description.

    .. note::
       This scorer rebuilds TF-IDF vectors **per pair**. For batch
       deduplication prefer :meth:`FeatureDeduplicator.deduplicate`
       which pre-computes vectors once.
    """

    def _scorer(left: FeatureRecord, right: FeatureRecord) -> float:
        text_left = f"{left.title or ''} {left.description or ''}".strip().lower()
        text_right = f"{right.title or ''} {right.description or ''}".strip().lower()
        tokens_left = _tokenise(text_left)
        tokens_right = _tokenise(text_right)
        if not tokens_left or not tokens_right:
            return 0.0
        vectors, _ = _build_tfidf([tokens_left, tokens_right])
        if len(vectors) != 2:
            return 0.0
        return _cosine(vectors[0], vectors[1])

    return _scorer


def _prefix_overlap(left: FeatureRecord, right: FeatureRecord) -> float:
    """Cheap fallback: normalised shared-prefix length over the title."""
    a = (left.title or "").lower().strip()
    b = (right.title or "").lower().strip()
    if not a or not b:
        return 0.0
    limit = min(len(a), len(b), 32)
    shared = 0
    for i in range(limit):
        if a[i] == b[i]:
            shared += 1
        else:
            break
    if shared < 6:
        return 0.0
    return shared / max(len(a), len(b), 1)


class FeatureDeduplicator:
    """Cluster :class:`FeatureRecord` objects by similarity.

    Records that score ``>= threshold`` are merged into a single record;
    the canonical record absorbs the losers' ``source`` into
    ``related_projects`` so the digest can show "Aider + Claude Code
    both ship this".

    Vectors are pre-computed once in :meth:`deduplicate` so that the
    pairwise cosine comparisons are O(N²) on cheap vector operations
    rather than O(N²) on expensive TF-IDF rebuilds.
    """

    def __init__(self, threshold: float = 0.55) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deduplicate(self, records: Iterable[FeatureRecord]) -> list[FeatureRecord]:
        """Return a deduplicated list of records.

        Pre-computes TF-IDF vectors for all records in a single pass so
        pairwise cosine comparisons are O(number-of-terms) rather than
        O(tokenisation + IDF-build + vector-build) per pair.
        """
        rec_list = list(records)
        if len(rec_list) <= 1:
            return rec_list

        # -- Pre-compute tokens and TF-IDF vectors -----------------------
        texts: list[str] = []
        for r in rec_list:
            texts.append(f"{r.title or ''} {r.description or ''}".strip().lower())
        token_lists = [_tokenise(t) for t in texts]
        vectors, _ = _build_tfidf(token_lists)
        # Map record identity → original index for O(1) vector lookup.
        _idx = {id(r): i for i, r in enumerate(rec_list)}

        def _fast_cosine(record_a: FeatureRecord, record_b: FeatureRecord) -> float:
            i = _idx.get(id(record_a), -1)
            j = _idx.get(id(record_b), -1)
            if i < 0 or j < 0:
                return 0.0
            vi = vectors[i]
            vj = vectors[j]
            if not vi.counts or not vj.counts:
                return 0.0
            return _cosine(vi, vj)

        # -- Cluster ----------------------------------------------------
        canonical: list[FeatureRecord] = []
        for record in rec_list:
            merged = False
            for existing in canonical:
                # Pre-computed TF-IDF cosine
                if _fast_cosine(record, existing) >= self.threshold:
                    self._absorb(existing, record)
                    merged = True
                    break
                # Cheap prefix fallback
                if _prefix_overlap(record, existing) >= PREFIX_OVERLAP_THRESHOLD:
                    self._absorb(existing, record)
                    merged = True
                    break
            if not merged:
                canonical.append(record)
        return canonical

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _absorb(canonical: FeatureRecord, other: FeatureRecord) -> None:
        if other.source and other.source != canonical.source and other.source not in canonical.related_projects:
            canonical.related_projects.append(other.source)
        for tag in other.tags:
            if tag not in canonical.tags:
                canonical.tags.append(tag)
        if not canonical.released_at and other.released_at:
            canonical.released_at = other.released_at
