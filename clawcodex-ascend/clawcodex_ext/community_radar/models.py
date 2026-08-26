"""Data models for SR-5.1 Community Feature Radar.

Mirrors the dataclasses declared in the feature design so the
rest of the pipeline (registry → fetcher → extractor → classifier →
deduplicator → scorer → reporter) can speak a single shared vocabulary.

Naming + field shapes are deliberately kept Pythonic (snake_case) rather
than mirroring the camelCase JSON shape used in GitHub APIs; the layer
boundary between API responses and internal records is the Fetcher.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FeatureCategory(str, Enum):
    # Parent-level category (never assigned to a feature — pure aggregation)
    CODE_AGENT = "code_agent"
    # Leaf categories under CODE_AGENT
    AGENT_LOOP = "agent_loop"
    TOOL_SYSTEM = "tool_system"
    PROVIDER = "provider"
    PERMISSION = "permission"
    MEMORY = "memory"
    MCP = "mcp"
    MULTI_AGENT = "multi_agent"
    ORCHESTRATOR = "orchestrator"
    TUI_REPL = "tui_repl"
    CLI = "cli"
    OBSERVABILITY = "observability"
    INFRA = "infra"
    # Both leaf and parent (contains SPATIAL_INTELLIGENCE)
    EMBODIED_AI = "embodied_ai"
    # Leaf under EMBODIED_AI
    SPATIAL_INTELLIGENCE = "spatial_intelligence"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# FeatureCategory hierarchy helpers (path-based, supports N levels)
# ---------------------------------------------------------------------------

# Category → path tuple from root to self.
# Root-only nodes (e.g. CODE_AGENT) are single-element paths.
# Leaf nodes under a root are two-element paths (root, self).
# Add deeper paths (root, mid, leaf) for 3+ levels — nothing else changes.
_CATEGORY_PATH: dict[FeatureCategory, tuple[FeatureCategory, ...]] = {
    FeatureCategory.CODE_AGENT: (FeatureCategory.CODE_AGENT,),
    FeatureCategory.AGENT_LOOP: (FeatureCategory.CODE_AGENT, FeatureCategory.AGENT_LOOP),
    FeatureCategory.TOOL_SYSTEM: (FeatureCategory.CODE_AGENT, FeatureCategory.TOOL_SYSTEM),
    FeatureCategory.PROVIDER: (FeatureCategory.CODE_AGENT, FeatureCategory.PROVIDER),
    FeatureCategory.PERMISSION: (FeatureCategory.CODE_AGENT, FeatureCategory.PERMISSION),
    FeatureCategory.MEMORY: (FeatureCategory.CODE_AGENT, FeatureCategory.MEMORY),
    FeatureCategory.MCP: (FeatureCategory.CODE_AGENT, FeatureCategory.MCP),
    FeatureCategory.MULTI_AGENT: (FeatureCategory.CODE_AGENT, FeatureCategory.MULTI_AGENT),
    FeatureCategory.ORCHESTRATOR: (FeatureCategory.CODE_AGENT, FeatureCategory.ORCHESTRATOR),
    FeatureCategory.TUI_REPL: (FeatureCategory.CODE_AGENT, FeatureCategory.TUI_REPL),
    FeatureCategory.CLI: (FeatureCategory.CODE_AGENT, FeatureCategory.CLI),
    FeatureCategory.OBSERVABILITY: (FeatureCategory.CODE_AGENT, FeatureCategory.OBSERVABILITY),
    FeatureCategory.INFRA: (FeatureCategory.CODE_AGENT, FeatureCategory.INFRA),
    FeatureCategory.EMBODIED_AI: (FeatureCategory.EMBODIED_AI,),
    FeatureCategory.SPATIAL_INTELLIGENCE: (
        FeatureCategory.EMBODIED_AI,
        FeatureCategory.SPATIAL_INTELLIGENCE,
    ),
    FeatureCategory.UNKNOWN: (FeatureCategory.UNKNOWN,),
}


def get_path(cat: FeatureCategory) -> tuple[FeatureCategory, ...]:
    """Return the full path from root to *cat*.

    >>> get_path(FeatureCategory.AGENT_LOOP)
    (CODE_AGENT, AGENT_LOOP)
    >>> get_path(FeatureCategory.EMBODIED_AI)
    (EMBODIED_AI,)
    """
    return _CATEGORY_PATH.get(cat, (cat,))


def get_root(cat: FeatureCategory) -> FeatureCategory:
    """Return the root (first element) of *cat*'s path.

    >>> get_root(FeatureCategory.TOOL_SYSTEM).value
    'code_agent'
    >>> get_root(FeatureCategory.EMBODIED_AI).value
    'embodied_ai'
    """
    return get_path(cat)[0]


def get_level(cat: FeatureCategory) -> int:
    """Return 0 for root-only nodes, 1 for direct children, 2+ for deeper.

    >>> get_level(FeatureCategory.CODE_AGENT)
    0
    >>> get_level(FeatureCategory.AGENT_LOOP)
    1
    """
    return len(get_path(cat)) - 1


def get_subtree(root: FeatureCategory) -> list[FeatureCategory]:
    """Return all categories whose path starts with *root*, excluding *root*.

    >>> get_subtree(FeatureCategory.CODE_AGENT)
    [AGENT_LOOP, TOOL_SYSTEM, ..., INFRA]
    >>> get_subtree(FeatureCategory.EMBODIED_AI)
    [SPATIAL_INTELLIGENCE]
    """
    result: list[FeatureCategory] = []
    for cat, path in _CATEGORY_PATH.items():
        if len(path) > 1 and path[0] == root:
            result.append(cat)
    return result


def is_leaf(cat: FeatureCategory) -> bool:
    """True when *cat* can be assigned to a feature.

    CODE_AGENT is the only non-leaf — it exists purely for aggregation.
    """
    return cat != FeatureCategory.CODE_AGENT


class FeatureType(str, Enum):
    NEW = "new"
    ENHANCEMENT = "enhancement"
    BREAKING = "breaking"
    DEPRECATION = "deprecation"
    BUGFIX = "bugfix"


class SourceDomain(str, Enum):
    """Known domain for a WatchSource, used to penalise cross-domain
    classification (e.g. a software-engineering project whose features
    happen to mention ``robot`` is NOT an embodied-AI feature).
    """

    CODE_AGENT = "code_agent"
    EMBODIED_AI = "embodied_ai"
    SPATIAL_INTELLIGENCE = "spatial_intelligence"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Source registration
# ---------------------------------------------------------------------------


@dataclass
class WatchSource:
    """A single upstream project being tracked by the radar.

    Matches the dataclass sketched in the feature design. Only fields
    with defaults can be omitted from YAML; ``name`` and ``repo`` are
    required.
    """

    name: str
    repo: str
    domain: str = SourceDomain.GENERAL.value
    track_releases: bool = True
    track_commits: bool = False
    track_prs: bool = False
    track_issues: bool = False
    release_tag_filter: str | None = None
    changelog_path: str | None = None
    notes: str | None = None
    # Tags used by the strategic-value scorer.
    # Optional project-side metadata that helps the keyword match decide
    # whether a feature aligns with a ClawCodex roadmap item.
    roadmap_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchSource":
        if not isinstance(data, dict):
            raise ValueError("WatchSource.from_dict expects a dict")
        name = data.get("name")
        repo = data.get("repo")
        if not isinstance(name, str) or not name:
            raise ValueError("WatchSource requires a non-empty 'name'")
        if not isinstance(repo, str) or "/" not in repo:
            raise ValueError(f"WatchSource[{name}] requires 'repo' in 'owner/name' form")
        keywords = data.get("roadmap_keywords") or data.get("roadmapKeywords") or []
        if not isinstance(keywords, list):
            keywords = []
        domain_raw = data.get("domain")
        try:
            domain = SourceDomain(domain_raw).value if domain_raw else SourceDomain.GENERAL.value
        except ValueError:
            domain = SourceDomain.GENERAL.value
        return cls(
            name=name,
            repo=repo,
            domain=domain,
            track_releases=bool(data.get("track_releases", True)),
            track_commits=bool(data.get("track_commits", False)),
            track_prs=bool(data.get("track_prs", False)),
            track_issues=bool(data.get("track_issues", False)),
            release_tag_filter=data.get("release_tag_filter") or data.get("releaseTagFilter"),
            changelog_path=data.get("changelog_path") or data.get("changelogPath"),
            notes=data.get("notes"),
            roadmap_keywords=[str(k) for k in keywords],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo": self.repo,
            "domain": self.domain,
            "track_releases": self.track_releases,
            "track_commits": self.track_commits,
            "track_prs": self.track_prs,
            "track_issues": self.track_issues,
            "release_tag_filter": self.release_tag_filter,
            "changelog_path": self.changelog_path,
            "notes": self.notes,
            "roadmap_keywords": list(self.roadmap_keywords),
        }


# ---------------------------------------------------------------------------
# Raw upstream objects (post-Fetcher)
# ---------------------------------------------------------------------------


@dataclass
class Release:
    tag: str
    name: str
    body: str
    published_at: str | None  # ISO 8601
    url: str
    is_prerelease: bool = False
    raw_body: str = ""  # CHANGELOG raw body (filled by Layer 1.5), preferred by extractor

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "name": self.name,
            "body": self.body,
            "published_at": self.published_at,
            "url": self.url,
            "is_prerelease": self.is_prerelease,
            "raw_body": self.raw_body,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Release":
        return cls(
            tag=str(data.get("tag", "")),
            name=str(data.get("name", "")),
            body=str(data.get("body", "") or ""),
            published_at=data.get("published_at"),
            url=str(data.get("url", "")),
            is_prerelease=bool(data.get("is_prerelease", False)),
            raw_body=str(data.get("raw_body", "") or ""),
        )


@dataclass
class Commit:
    sha: str
    message: str
    author: str | None
    committed_at: str | None
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha": self.sha,
            "message": self.message,
            "author": self.author,
            "committed_at": self.committed_at,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Commit":
        return cls(
            sha=str(data.get("sha", "")),
            message=str(data.get("message") or ""),
            author=data.get("author"),
            committed_at=data.get("committed_at"),
            url=str(data.get("url") or ""),
        )


@dataclass
class PullRequest:
    number: int
    title: str
    state: str
    merged_at: str | None
    url: str
    body: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "merged_at": self.merged_at,
            "url": self.url,
            "body": self.body,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PullRequest":
        return cls(
            number=int(data.get("number") or 0),
            title=str(data.get("title") or ""),
            state=str(data.get("state") or ""),
            merged_at=data.get("merged_at"),
            url=str(data.get("url") or ""),
            body=str(data.get("body") or ""),
        )


@dataclass
class FetchResult:
    """Bundles everything a Fetcher produced for one WatchSource.

    ``errors`` is a free-form list of ``str`` descriptions (network
    failures, parse errors, rate-limit warnings). The pipeline never
    raises out of a fetch — callers render errors into the digest so
    users see why a source contributed 0 records.
    """

    source: str
    releases: list[Release] = field(default_factory=list)
    commits: list[Commit] = field(default_factory=list)
    prs: list[PullRequest] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    detected_domain: str | None = None  # auto-detected project domain

    @property
    def is_empty(self) -> bool:
        return not (self.releases or self.commits or self.prs)


# ---------------------------------------------------------------------------
# Pipeline records
# ---------------------------------------------------------------------------


def make_feature_id(source: str, title: str, kind: str) -> str:
    """Stable id for a FeatureRecord before dedup.

    Uses the first 12 hex chars of SHA-1(source|title|kind). Long enough
    to be unique across the ~10 default sources, short enough to remain
    readable in log lines.
    """
    h = hashlib.sha1(usedforsecurity=False)
    h.update(source.encode("utf-8"))
    h.update(b"|")
    h.update((title or "").lower().encode("utf-8"))
    h.update(b"|")
    h.update((kind or "").encode("utf-8"))
    return h.hexdigest()[:12]


@dataclass
class FeatureRecord:
    """A single community-feature candidate produced by the pipeline.

    Mirrors the dataclass in the feature design. Fields are kept
    loose (``category`` defaults to UNKNOWN) so the rule-based
    extractor can construct records before the classifier has had a
    chance to label them.
    """

    id: str
    source: str
    title: str
    description: str
    category: FeatureCategory = FeatureCategory.UNKNOWN
    feature_type: FeatureType = FeatureType.NEW
    released_at: str | None = None
    url: str = ""
    related_projects: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    raw_body: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "feature_type": self.feature_type.value,
            "released_at": self.released_at,
            "url": self.url,
            "related_projects": list(self.related_projects),
            "tags": list(self.tags),
            "raw_body": self.raw_body,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureRecord":
        try:
            category = FeatureCategory(data.get("category") or "unknown")
        except ValueError:
            category = FeatureCategory.UNKNOWN
        try:
            ftype = FeatureType(data.get("feature_type") or "new")
        except ValueError:
            ftype = FeatureType.NEW
        related = data.get("related_projects") or []
        tags = data.get("tags") or []
        return cls(
            id=str(
                data.get("id")
                or make_feature_id(
                    data.get("source", ""),
                    data.get("title", ""),
                    str(data.get("feature_type", "new")),
                )
            ),
            source=str(data.get("source", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "") or ""),
            category=category,
            feature_type=ftype,
            released_at=data.get("released_at"),
            url=str(data.get("url", "") or ""),
            related_projects=[str(p) for p in related],
            tags=[str(t) for t in tags],
            raw_body=data.get("raw_body"),
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class FeatureScore:
    record_id: str
    overall: float
    popularity: float
    maturity: float
    adaptation_cost: float
    strategic_value: float
    architecture_fit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "overall": self.overall,
            "popularity": self.popularity,
            "maturity": self.maturity,
            "adaptation_cost": self.adaptation_cost,
            "strategic_value": self.strategic_value,
            "architecture_fit": self.architecture_fit,
            "dimensions": {
                "popularity": self.popularity,
                "maturity": self.maturity,
                "adaptation_cost": self.adaptation_cost,
                "strategic_value": self.strategic_value,
                "architecture_fit": self.architecture_fit,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureScore:
        if not isinstance(data, dict):
            raise TypeError("FeatureScore.from_dict expects a dict")
        dims = data.get("dimensions")
        if not isinstance(dims, dict):
            dims = {}
        return cls(
            record_id=str(data.get("record_id", "")),
            overall=float(data.get("overall", 0.0)),
            popularity=float(data.get("popularity", dims.get("popularity", 0.0))),
            maturity=float(data.get("maturity", dims.get("maturity", 0.0))),
            adaptation_cost=float(data.get("adaptation_cost", dims.get("adaptation_cost", 0.0))),
            strategic_value=float(data.get("strategic_value", dims.get("strategic_value", 0.0))),
            architecture_fit=float(data.get("architecture_fit", dims.get("architecture_fit", 0.0))),
        )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class DigestStats:
    total_versions: int = 0
    total_features: int = 0
    filtered_count: int = 0  # features excluded by exclude_feature_types
    major_count: int = 0  # features promoted to Highlights section
    minor_count: int = 0  # features in Detail Table only
    by_category: dict[str, int] = field(default_factory=dict)
    by_root_category: dict[str, int] = field(default_factory=dict)
    top_projects: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_versions": self.total_versions,
            "total_features": self.total_features,
            "filtered_count": self.filtered_count,
            "major_count": self.major_count,
            "minor_count": self.minor_count,
            "by_category": dict(self.by_category),
            "by_root_category": dict(self.by_root_category),
            "top_projects": [list(item) for item in self.top_projects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DigestStats:
        if not isinstance(data, dict):
            raise TypeError("DigestStats.from_dict expects a dict")
        by_category = data.get("by_category")
        by_root = data.get("by_root_category")
        return cls(
            total_versions=int(data.get("total_versions", 0)),
            total_features=int(data.get("total_features", 0)),
            filtered_count=int(data.get("filtered_count", 0)),
            major_count=int(data.get("major_count", 0)),
            minor_count=int(data.get("minor_count", 0)),
            by_category={str(k): int(v) for k, v in dict(by_category or {}).items()},
            by_root_category={str(k): int(v) for k, v in dict(by_root or {}).items()},
            top_projects=[(str(item[0]), int(item[1])) for item in data.get("top_projects", [])],
        )


@dataclass
class ScoredFeature:
    """A feature bundled with its FeatureScore for easy rendering."""

    record: FeatureRecord
    score: FeatureScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": self.score.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoredFeature:
        if not isinstance(data, dict):
            raise TypeError("ScoredFeature.from_dict expects a dict")
        return cls(
            record=FeatureRecord.from_dict(data.get("record", {})),
            score=FeatureScore.from_dict(data.get("score", {})),
        )


@dataclass
class CommunityDigest:
    period: str  # "weekly" | "monthly"
    generated_at: str
    summary: str
    period_start: str = ""  # ISO-8601, the "since" cutoff for this scan
    new_features: list[FeatureRecord] = field(default_factory=list)
    trending: list[ScoredFeature] = field(default_factory=list)
    # Phase 4 / SR-5.3: major-feature highlights for the prose-summary block.
    highlights: list[ScoredFeature] = field(default_factory=list)
    # LLM importance data: feature_id → {"level": "MAJOR"|"MINOR", "highlight": "intro text"}
    llm_importance: dict[str, dict[str, str]] = field(default_factory=dict)
    breaking_changes: list[FeatureRecord] = field(default_factory=list)
    stats: DigestStats = field(default_factory=DigestStats)
    # When the run was manual (CLI), include raw sources so users can
    # audit why a feature appeared.
    sources_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "generated_at": self.generated_at,
            "period_start": self.period_start,
            "summary": self.summary,
            "new_features": [r.to_dict() for r in self.new_features],
            "trending": [s.to_dict() for s in self.trending],
            "highlights": [s.to_dict() for s in self.highlights],
            "llm_importance": dict(self.llm_importance),
            "breaking_changes": [r.to_dict() for r in self.breaking_changes],
            "stats": self.stats.to_dict(),
            "sources_used": list(self.sources_used),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunityDigest:
        if not isinstance(data, dict):
            raise TypeError("CommunityDigest.from_dict expects a dict")
        return cls(
            period=str(data.get("period", "")),
            generated_at=str(data.get("generated_at", "")),
            period_start=str(data.get("period_start", "")),
            summary=str(data.get("summary", "")),
            new_features=[FeatureRecord.from_dict(r) for r in data.get("new_features", [])],
            trending=[ScoredFeature.from_dict(s) for s in data.get("trending", [])],
            highlights=[ScoredFeature.from_dict(s) for s in data.get("highlights", [])],
            llm_importance=dict(data.get("llm_importance") or {}),
            breaking_changes=[FeatureRecord.from_dict(r) for r in data.get("breaking_changes", [])],
            stats=DigestStats.from_dict(data.get("stats") or {}),
            sources_used=[str(s) for s in data.get("sources_used", [])],
            errors=[str(e) for e in data.get("errors", [])],
        )


@dataclass
class HistoryComparison:
    """Delta between two consecutive CommunityDigest runs.

    Built by :func:`reporter.compare_digests` and rendered as a
    "changes comparison" section in the Markdown digest.
    """

    previous_period: str
    previous_generated_at: str
    previous_stem: str
    new_count: int
    disappeared_count: int
    score_changed: list[dict[str, Any]] = field(default_factory=list)
    new_features: list[FeatureRecord] = field(default_factory=list)
    disappeared_features: list[FeatureRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_period": self.previous_period,
            "previous_generated_at": self.previous_generated_at,
            "previous_stem": self.previous_stem,
            "new_count": self.new_count,
            "disappeared_count": self.disappeared_count,
            "score_changed": [dict(item) for item in self.score_changed],
            "new_features": [r.to_dict() for r in self.new_features],
            "disappeared_features": [r.to_dict() for r in self.disappeared_features],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryComparison:
        if not isinstance(data, dict):
            raise TypeError("HistoryComparison.from_dict expects a dict")
        return cls(
            previous_period=str(data.get("previous_period", "")),
            previous_generated_at=str(data.get("previous_generated_at", "")),
            previous_stem=str(data.get("previous_stem", "")),
            new_count=int(data.get("new_count", 0)),
            disappeared_count=int(data.get("disappeared_count", 0)),
            score_changed=[dict(item) for item in data.get("score_changed", [])],
            new_features=[FeatureRecord.from_dict(r) for r in data.get("new_features", [])],
            disappeared_features=[FeatureRecord.from_dict(r) for r in data.get("disappeared_features", [])],
        )


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp used for digest.generated_at."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
