# pylint: disable=relative-beyond-top-level
"""Multi-dimension feature scorer for SR-5.1.

Implements the multi-dimension scoring model:

| dimension           | default weight | inputs                                      |
|---------------------|---------------:|---------------------------------------------|
| popularity          | 15%           | release spread + tag presence               |
| maturity            | 20%           | is_prerelease + body length                 |
| adaptation_cost     | 25%           | category → estimated effort                 |
| strategic_value     | 25%           | roadmap keyword overlap + related_projects |
| architecture_fit    | 15%           | whether the category lives in clawcodex_ext |

The scorer is pure: every input is on the record itself (plus the
:class:`~clawcodex_ext.community_radar.config.RadarConfig` carrying the
weights + roadmap keyword list). There is no I/O; the orchestration
layer decides when to call :meth:`score` or :meth:`score_many`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from .config import RadarConfig
from .models import FeatureCategory, FeatureRecord, FeatureScore, FeatureType

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default adaptation-cost / architecture-fit heuristics. The numbers are
# deliberately conservative — they only affect ordering inside the
# digest, not absolute gating.
# ---------------------------------------------------------------------------


# Lower number ⇒ cheaper to absorb into ClawCodex (higher score).
_ADAPTATION_COST: dict[FeatureCategory, float] = {
    FeatureCategory.CLI: 20.0,  # well-bounded CLI additions
    FeatureCategory.TUI_REPL: 25.0,
    FeatureCategory.TOOL_SYSTEM: 30.0,
    FeatureCategory.MEMORY: 35.0,
    FeatureCategory.OBSERVABILITY: 35.0,
    FeatureCategory.AGENT_LOOP: 45.0,
    FeatureCategory.PROVIDER: 45.0,
    FeatureCategory.MCP: 50.0,
    FeatureCategory.PERMISSION: 40.0,
    FeatureCategory.MULTI_AGENT: 50.0,
    FeatureCategory.ORCHESTRATOR: 55.0,
    FeatureCategory.INFRA: 80.0,
    FeatureCategory.EMBODIED_AI: 55.0,
    FeatureCategory.SPATIAL_INTELLIGENCE: 50.0,
    FeatureCategory.CODE_AGENT: 70.0,  # defensive: never assigned, but safe
    FeatureCategory.UNKNOWN: 60.0,
}


# Architecture fit: 100 if a category already lives fully under
# ``clawcodex_ext/*``; lower if the change has to touch the
# ``src/upstream`` shape.
_ARCHITECTURE_FIT: dict[FeatureCategory, float] = {
    FeatureCategory.CLI: 95.0,
    FeatureCategory.TUI_REPL: 95.0,
    FeatureCategory.TOOL_SYSTEM: 90.0,
    FeatureCategory.MEMORY: 90.0,
    FeatureCategory.OBSERVABILITY: 90.0,
    FeatureCategory.MCP: 85.0,
    FeatureCategory.MULTI_AGENT: 85.0,
    FeatureCategory.ORCHESTRATOR: 90.0,
    FeatureCategory.PERMISSION: 80.0,
    FeatureCategory.PROVIDER: 70.0,
    FeatureCategory.AGENT_LOOP: 60.0,
    FeatureCategory.INFRA: 50.0,
    FeatureCategory.EMBODIED_AI: 70.0,
    FeatureCategory.SPATIAL_INTELLIGENCE: 75.0,
    FeatureCategory.CODE_AGENT: 80.0,  # defensive: never assigned, but safe
    FeatureCategory.UNKNOWN: 50.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def _tokenise(text: str) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _popularity(record: FeatureRecord) -> float:
    """Heuristic popularity score in [0, 100].

    Inputs (no network — community signals are intentionally cheap):

    * Each ``related_projects`` entry adds a bonus (a feature showing up
      in two projects is more interesting than one only seen once).
    * Title length inversely normalised (shorter titles tend to be the
      features GitHub release-note curators pick).
    * Tags inherited from the classifier add a small bump per tag.
    """
    score = 50.0
    score += min(len(record.related_projects), 5) * 8  # +8 per cross-project ref
    title_len = len(record.title or "")
    if 4 <= title_len <= 80:
        score += 5
    elif title_len > 120:
        score -= 5
    score += min(len(record.tags), 8) * 2  # tag density proxy
    return float(max(0, min(100, score)))


def _maturity(record: FeatureRecord) -> float:
    """Maturity score in [0, 100].

    Without GitHub metadata the extractor relies on signals already on
    the record (release body presence, length, prerelease hint coming
    from the source ``raw_body``).
    """
    score = 40.0  # neutral default for a typical release-note entry
    body = record.raw_body or record.description or ""
    if len(body) >= 80:
        score += 20  # elaborated change log entries usually ship with tests
    if len(body) >= 200:
        score += 10
    if record.feature_type == FeatureType.BREAKING:
        score -= 15  # breaking changes are usually less mature
    if record.feature_type == FeatureType.DEPRECATION:
        score -= 10
    if record.feature_type == FeatureType.NEW:
        score += 5
    if record.released_at:
        score += 5
    return float(max(0, min(100, score)))


def _adaptation_cost(record: FeatureRecord) -> float:
    """Adaptation cost inverted to a [0, 100] score.

    Cost 0  → score 100
    Cost 100 → score 0
    """
    cost = _ADAPTATION_COST.get(record.category, 60.0)
    if record.feature_type == FeatureType.BREAKING:
        cost += 20
    if record.feature_type == FeatureType.DEPRECATION:
        cost += 10
    if record.related_projects:
        # When 2+ projects ship the same thing, the implementation has
        # already converged — adaptation becomes cheaper.
        cost = max(10.0, cost - 10 * len(record.related_projects))
    cost = max(0.0, min(100.0, cost))
    return float(round(100.0 - cost, 2))


def _strategic_value(record: FeatureRecord, keywords: Iterable[str]) -> float:
    """Score how well ``record`` aligns with the ClawCodex roadmap."""
    keyword_set = {k.lower() for k in keywords}
    if not keyword_set:
        return 25.0
    haystack = _tokenise(f"{record.title} {record.description} {' '.join(record.tags)}")
    if not haystack:
        return 25.0
    overlap = keyword_set & haystack
    if not overlap:
        base = 25.0
        # FeatureCategory as a coarse alignment signal — categories
        # closer to the project's core roadmap still score above
        # random infrastructure noise.
        if record.category in {
            FeatureCategory.AGENT_LOOP,
            FeatureCategory.MEMORY,
            FeatureCategory.MCP,
            FeatureCategory.PERMISSION,
            FeatureCategory.ORCHESTRATOR,
        }:
            base += 10
        return float(base)
    return float(max(0, min(100, 35 + 12 * len(overlap))))


def _architecture_fit(record: FeatureRecord) -> float:
    """Score whether absorbing ``record`` stays inside clawcodex_ext/*."""
    base = _ARCHITECTURE_FIT.get(record.category, 50.0)
    if record.feature_type == FeatureType.BREAKING:
        base -= 10
    return float(max(0, min(100, base)))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class FeatureScorer:
    """Compute weighted :class:`FeatureScore` for one or more records."""

    def __init__(self, config: RadarConfig | None = None) -> None:
        self.config = config or RadarConfig()

    def score(self, record: FeatureRecord) -> FeatureScore:
        weights = self.config.normalized_weights()
        dims = {
            "popularity": _popularity(record),
            "maturity": _maturity(record),
            "adaptation_cost": _adaptation_cost(record),
            "strategic_value": _strategic_value(record, self.config.roadmap_keywords),
            "architecture_fit": _architecture_fit(record),
        }
        # pylint: disable=consider-using-dict-items
        overall = sum(dims[name] * weights.get(name, 0.0) for name in dims)
        return FeatureScore(
            record_id=record.id,
            overall=round(overall, 2),
            popularity=round(dims["popularity"], 2),
            maturity=round(dims["maturity"], 2),
            adaptation_cost=round(dims["adaptation_cost"], 2),
            strategic_value=round(dims["strategic_value"], 2),
            architecture_fit=round(dims["architecture_fit"], 2),
        )

    def score_many(self, records: Iterable[FeatureRecord]) -> list[FeatureScore]:
        return [self.score(r) for r in records]
