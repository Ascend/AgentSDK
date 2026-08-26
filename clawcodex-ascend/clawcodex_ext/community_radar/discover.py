# pylint: disable=relative-beyond-top-level
"""``source discover`` — search GitHub repos and add them to the Radar monitoring sources.

Flow: build the GitHub Search API query → fetch candidates → verify domain via
detect_repo_domain → dedup → sort by stars → write to SourceRegistry.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .fetcher import detect_repo_domain
from .models import SourceDomain, WatchSource

if TYPE_CHECKING:
    from .fetcher import Fetcher
    from .registry import SourceRegistry

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

MAX_COUNT_PER_DISCOVER: int = 10
MAX_SOURCES_TOTAL: int = 30

# GitHub Search API returns at most 100 results per page.  We fetch 30
# candidates so there is enough headroom after dedup + domain filtering.
_CANDIDATE_PER_PAGE: int = 30

# GitHub Search API rejects queries longer than 256 characters.  We keep
# each per-domain keyword set small enough that the final query (including
# filters like ``stars:>=100 language:python``) stays under this limit.
# The full keyword sets in _QUICK_MATCH_KW (below) cover the wider spectrum
# for local topic/description matching without API cost.

# ── Per-domain search keywords (6-7 each, kept small for API limits) ──

# GitHub Search API limits operators (AND/OR/NOT) to 5 per query.
# With N keywords joined by OR, we get (N-1) OR operators → max N = 6.
# We use 5 keywords per domain (4 ORs) so we stay safely under the limit.
# These 5 must be the most discriminating high-signal keywords per domain.
_SEARCH_KEYWORDS: dict[str, list[str]] = {
    "code_agent": [
        "coding agent",
        "LLM agent",
        "software engineering agent",
        "SWE agent",
        "AI coding assistant",
    ],
    "embodied_ai": [
        "robot learning",
        "imitation learning",
        "robot manipulation",
        "embodied AI",
        "humanoid robot",
    ],
    "spatial_intelligence": [
        "3D reconstruction",
        "gaussian splatting",
        "NeRF",
        "neural rendering",
        "3D generation",
    ],
}


# ---------------------------------------------------------------------------
# Domain keywords for fast matching (no API call)
# ---------------------------------------------------------------------------


_EMBODIED_AI_QUICK_KW = [
    "robot",
    "robotics",
    "manipulation",
    "locomotion",
    "grasping",
    "embodied",
    "vla",
    "reinforcement-learning",
    "imitation-learning",
    "humanoid",
    "legged-robot",
    "mobile-manipulation",
    "teleoperation",
    "sim-to-real",
    "robot-learning",
    "mobile robot",
    "robot simulation",
    "autonomous navigation",
    "dexterous manipulation",
]

_SPATIAL_QUICK_KW = [
    "nerf",
    "neural-radiance",
    "3d-reconstruction",
    "3d-vision",
    "gaussian-splatting",
    "point-cloud",
    "slam",
    "lidar",
    "novel-view-synthesis",
    "volumetric",
    "mesh",
    "voxel",
    "spatial-intelligence",
    "radiance-field",
    "depth-estimation",
    "multi-view-stereo",
    "structure-from-motion",
    "3d-generation",
    "text-to-3d",
    "image-to-3d",
    "3d-scene-understanding",
    "point-cloud-processing",
]

_CODE_AGENT_QUICK_KW = [
    "ai-coding",
    "coding-agent",
    "llm-agent",
    "ai-assistant",
    "ai-programming",
    "agent-framework",
    "multi-agent",
    "code-generation",
    "ai-developer-tool",
    "autonomous-coding",
    "software-engineering-agent",
]


def _format_stars(n: int) -> str:
    """Format star count for human display (e.g. 28500 → 28.5k)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


# ---------------------------------------------------------------------------
# DiscoverCandidate
# ---------------------------------------------------------------------------


@dataclass
class DiscoverCandidate:
    """A repo returned by GitHub Search before domain verification."""

    repo: str  # "owner/name"
    stars: int
    description: str
    topics: list[str]
    language: str | None = None
    domain: str | None = None  # filled after detect_repo_domain()


# ---------------------------------------------------------------------------
# Discover result
# ---------------------------------------------------------------------------


@dataclass
class DiscoverResult:
    """Outcome of a discover run."""

    added: list[WatchSource] = field(default_factory=list)
    added_stars: dict[str, int] = field(default_factory=dict)  # repo → star count
    skipped_duplicate: int = 0
    skipped_domain_mismatch: int = 0
    count_ceiling_warning: bool = False
    total_limit_warning: bool = False
    not_enough_warning: bool = False
    requested_count: int = 0
    search_total: int = 0  # total results returned by GitHub Search


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


def build_search_query(
    domain: str,
    min_stars: int = 100,
    lang: str | None = None,
) -> str:
    """Build a GitHub Search API ``q`` parameter value for a single domain.

    The returned string is guaranteed to be ≤ 256 characters so it passes
    GitHub's validation.

    Args:
        domain: Which domain to build keywords for (required, one domain).
        min_stars: Minimum star count (0 disables the filter).
        lang: Optional programming language filter (e.g. ``"python"``).
    """
    keywords = _SEARCH_KEYWORDS.get(domain, [])
    if not keywords:
        keywords = ["AI"]  # fallback — should never happen

    # Build OR-clause from keywords, each double-quoted
    if len(keywords) == 1:
        q_parts = [f'"{keywords[0]}"']
    else:
        q_parts = [" OR ".join(f'"{kw}"' for kw in keywords)]

    # Scope to name, description, topics
    q_parts.append("in:name,description,topics")

    # Quality filters
    q_parts.append("fork:false")
    q_parts.append("archived:false")

    if min_stars > 0:
        q_parts.append(f"stars:>={min_stars}")

    if lang:
        q_parts.append(f"language:{lang}")

    result = " ".join(q_parts)

    # Safety check — should never trigger with current keyword sets,
    # but guards against future keyword additions breaking the limit.
    if len(result) > 256:
        _log.warning(
            "search query length %d exceeds 256-char GitHub limit; truncating keywords",
            len(result),
        )
        # Rebuild with fewer keywords to stay under the limit
        return _build_trimmed_query(keywords, min_stars, lang)

    return result


def _build_trimmed_query(
    keywords: list[str],
    min_stars: int,
    lang: str | None,
) -> str:
    """Build a query that stays under 256 chars by dropping keywords."""
    suffix = " in:name,description,topics fork:false archived:false"
    if min_stars > 0:
        suffix += f" stars:>={min_stars}"
    if lang:
        suffix += f" language:{lang}"

    best = ""
    for n in range(len(keywords), 0, -1):
        if len(keywords[:n]) == 1:
            cand = f'"{keywords[0]}"{suffix}'
        else:
            cand = " OR ".join(f'"{kw}"' for kw in keywords[:n]) + suffix
        if len(cand) <= 256:
            best = cand
            break

    return best or '"AI" in:name,description,topics fork:false archived:false'


def quick_domain_match(candidate: DiscoverCandidate) -> str | None:
    """Try to determine a candidate's domain from topics + description alone.

    Returns one of ``"embodied_ai"``, ``"spatial_intelligence"``,
    ``"code_agent"``, or ``None`` when uncertain (caller should fall back
    to :func:`detect_repo_domain`).  Priority is embodied_ai, then
    spatial_intelligence, then code_agent; the first keyword match wins.
    """
    combined = (
        " " + " ".join(t.lower() for t in (candidate.topics or [])) + " " + (candidate.description or "").lower() + " "
    )

    for kw in _EMBODIED_AI_QUICK_KW:
        if f" {kw} " in combined:
            return "embodied_ai"

    for kw in _SPATIAL_QUICK_KW:
        if f" {kw} " in combined:
            return "spatial_intelligence"

    for kw in _CODE_AGENT_QUICK_KW:
        if kw in combined:
            return "code_agent"

    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def discover_sources(
    fetcher: Fetcher,
    registry: SourceRegistry,
    *,
    domain: str | None = None,
    min_stars: int = 100,
    count: int = 5,
    lang: str | None = None,
) -> DiscoverResult:
    """Search GitHub and add qualifying repos to *registry*.

    When *domain* is ``None``, runs three parallel searches (one per known
    domain) to avoid hitting GitHub's 256-char query-length limit on a
    single mega-query.  Results are merged and deduplicated.

    Returns a :class:`DiscoverResult` summarising what happened so the CLI
    handler can print a user-friendly summary.
    """
    result = DiscoverResult(requested_count=count)

    if count < 1:
        raise ValueError(f"count must be >= 1 (got {count})")
    if count > MAX_COUNT_PER_DISCOVER:
        result.count_ceiling_warning = True
        count = MAX_COUNT_PER_DISCOVER
        _log.info("--count clamped to %d", MAX_COUNT_PER_DISCOVER)

    existing_count = len(registry)
    available_slots = MAX_SOURCES_TOTAL - existing_count
    if available_slots <= 0:
        result.total_limit_warning = True
        _log.warning("source registry already at limit (%d)", MAX_SOURCES_TOTAL)
        return result

    if count > available_slots:
        result.total_limit_warning = True
        count = available_slots
        _log.info("count reduced to %d (only %d slots remain)", count, available_slots)

    if domain is not None:
        # Single-domain: one search query
        domains_to_search = [domain]
    else:
        # All domains: parallel searches to stay within 256-char limit
        domains_to_search = list(_SEARCH_KEYWORDS.keys())

    raw_results = _search_parallel(fetcher, domains_to_search, min_stars, lang)
    result.search_total = len(raw_results)

    if not raw_results:
        result.not_enough_warning = True
        _log.info("GitHub Search returned no results")
        return result

    existing_repos = {s.repo.lower() for s in registry.list()}
    candidates: list[DiscoverCandidate] = []
    seen_repos: set[str] = set()

    for item in raw_results:
        full_name = (item.get("full_name") or "").strip()
        if not full_name:
            continue
        key = full_name.lower()
        if key in seen_repos:
            continue
        seen_repos.add(key)

        if key in existing_repos:
            result.skipped_duplicate += 1
            continue

        candidates.append(
            DiscoverCandidate(
                repo=full_name,
                stars=item.get("stargazers_count", 0),
                description=item.get("description") or "",
                topics=item.get("topics") or [],
                language=item.get("language"),
            )
        )

    _log.info("candidates after dedup: %d (skipped %d duplicates)", len(candidates), result.skipped_duplicate)

    _resolve_domains_parallel(fetcher, candidates)

    if domain is not None:
        kept: list[DiscoverCandidate] = []
        for c in candidates:
            if c.domain == domain:
                kept.append(c)
            else:
                result.skipped_domain_mismatch += 1
        candidates = kept
    else:
        # Assign "general" to anything that didn't match a known domain
        for c in candidates:
            if c.domain is None:
                c.domain = SourceDomain.GENERAL.value

    _log.info("candidates after domain filter: %d (mismatch: %d)", len(candidates), result.skipped_domain_mismatch)

    candidates.sort(key=lambda c: c.stars, reverse=True)

    if len(candidates) < count:
        result.not_enough_warning = True

    selected = candidates[:count]

    for c in selected:
        name = _repo_to_name(c.repo)
        if name in registry:
            name = c.repo.replace("/", "-")
        source = WatchSource(
            name=name,
            repo=c.repo,
            domain=c.domain or SourceDomain.GENERAL.value,
            track_releases=True,
            track_commits=False,
            track_prs=False,
            track_issues=False,
        )
        registry.add(source)
        result.added.append(source)
        result.added_stars[c.repo] = c.stars

    if result.added:
        registry.save()
        _log.info("added %d sources to registry", len(result.added))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _repo_to_name(repo: str) -> str:
    """Derive a short source name from ``owner/name``."""
    return repo.split("/", 1)[1] if "/" in repo else repo


def _search_parallel(
    fetcher: Fetcher,
    domains: list[str],
    min_stars: int,
    lang: str | None,
    max_workers: int = 6,
) -> list[dict]:
    """Run one search per domain, optionally in parallel.

    When only one domain is given the call is sequential (no thread overhead).
    When multiple domains are given, searches run in parallel via
    :class:`~concurrent.futures.ThreadPoolExecutor` and results are merged
    with deduplication by ``full_name``.
    """
    if len(domains) == 1:
        q = build_search_query(domains[0], min_stars=min_stars, lang=lang)
        _log.info("search query [%s] (%d chars): %.120s...", domains[0], len(q), q)
        return fetcher.search_repositories(q, per_page=_CANDIDATE_PER_PAGE)

    # Parallel search across domains
    def _search_one(d: str) -> list[dict]:
        q = build_search_query(d, min_stars=min_stars, lang=lang)
        _log.info("search query [%s] (%d chars): %.120s...", d, len(q), q)
        return fetcher.search_repositories(q, per_page=_CANDIDATE_PER_PAGE)

    merged: list[dict] = []
    seen: set[str] = set()

    with ThreadPoolExecutor(max_workers=min(max_workers, len(domains))) as ex:
        futures = {ex.submit(_search_one, d): d for d in domains}
        for fut in as_completed(futures):
            for item in fut.result():
                full_name = (item.get("full_name") or "").lower()
                if full_name and full_name not in seen:
                    seen.add(full_name)
                    merged.append(item)

    return merged


def _resolve_domains_parallel(
    fetcher: Fetcher,
    candidates: list[DiscoverCandidate],
    max_workers: int = 6,
) -> None:
    """Resolve domain for every candidate, trying quick-match first.

    Quick-match (topics + description) avoids an API call per candidate.
    Fallback: parallel calls to ``detect_repo_domain()`` for the uncertain
    ones, reusing the fetcher's cache dir and token.
    """
    uncertain: list[int] = []  # indices into candidates

    for i, c in enumerate(candidates):
        qd = quick_domain_match(c)
        if qd is not None:
            c.domain = qd
        else:
            uncertain.append(i)

    if not uncertain:
        return

    _log.info(
        "quick-match resolved %d/%d; running API detection for %d",
        len(candidates) - len(uncertain),
        len(candidates),
        len(uncertain),
    )

    def _detect_one(idx: int) -> tuple[int, str | None]:
        c = candidates[idx]
        d = detect_repo_domain(
            c.repo,
            fetcher.cache_dir,
            github_token=fetcher.github_token,
        )
        return (idx, d)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(uncertain))) as ex:
        futures = {ex.submit(_detect_one, idx): idx for idx in uncertain}
        for fut in as_completed(futures):
            try:
                idx, d = fut.result()
                candidates[idx].domain = d
            except Exception as exc:  # noqa: BLE001
                _log.warning("domain resolution failed for %s: %s", candidates[futures[fut]].repo, exc)
