# pylint: disable=relative-beyond-top-level
"""Keyword-based feature classifier for SR-5.1.

Implements the keyword-based feature classifier module.
Classification is rule-first: every :class:`FeatureCategory` has a list
of keyword signals that score ``title + description`` after lowercasing.

Design notes:
* Multiple categories can match. We pick the category with the highest
  weighted score; ties fall back to ``FeatureCategory.AGENT_LOOP`` (the
  most strategic area for ClawCodex) so the digest skews towards
  impactful features instead of infrastructure noise.
* The taxonomy is intentionally shallow — only the 12 top-level
  categories from the design are emitted. Sub-categories
  (``agent_loop/prompt_engineering`` etc.) are recorded as ``tags``
  so callers can drill down without expanding the enum.
* The classifier never mutates the input list; it returns a fresh list
  of records with ``category`` and ``tags`` filled in.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .config import DEFAULT_ROADMAP_KEYWORDS
from .models import FeatureCategory, FeatureRecord, SourceDomain

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword taxonomy. Weights are positive; the classifier adds the weight
# of every match to produce a category score. Ties resolve via the
# ``_TIEBREAK`` order defined below.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CategoryRule:
    category: FeatureCategory
    keywords: tuple[tuple[str, float], ...]
    # Sub-tags to add when this category wins. Kept as a tuple of
    # (keyword, sub_tag) so the digest can show why a record landed in
    # "agent_loop/context_management" instead of just "agent_loop".
    sub_tags: tuple[tuple[str, str], ...] = ()
    # Ambiguous keywords that only score when at least one domain-anchor
    # keyword also appears in the text.  Format:
    #   (keyword, weight, (anchor_keyword, ...))
    # When no anchor matches the weight is skipped entirely so generic
    # terms like "rendering" or "policy" don't cause false positives
    # across unrelated domains.
    conditional_keywords: tuple[tuple[str, float, tuple[str, ...]], ...] = ()
    # Sub-tags gated by the same anchor condition.
    conditional_sub_tags: tuple[tuple[str, str, tuple[str, ...]], ...] = ()


_RULES: tuple[_CategoryRule, ...] = (
    _CategoryRule(
        category=FeatureCategory.AGENT_LOOP,
        keywords=(
            ("agent loop", 2.0),
            ("agent", 1.0),
            ("planner", 1.5),
            ("planning", 1.2),
            ("self-correct", 1.6),
            ("self-correction", 1.6),
            ("reflection", 1.2),
            ("prompt", 0.8),
            ("context compression", 1.5),
            ("context management", 1.4),
            ("system prompt", 0.9),
        ),
        sub_tags=(
            ("planner", "planning"),
            ("self-correct", "self_correction"),
            ("self-correction", "self_correction"),
            ("reflection", "self_correction"),
            ("context compression", "context_management"),
            ("context management", "context_management"),
            ("prompt", "prompt_engineering"),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.TOOL_SYSTEM,
        keywords=(
            ("tool", 1.0),
            ("mcp", 1.2),
            ("function call", 1.3),
            ("tool call", 1.3),
            ("tool registry", 1.4),
            ("tool definition", 1.2),
            ("slash command", 1.4),
        ),
        sub_tags=(
            ("mcp", "mcp_extension"),
            ("slash command", "new_tool"),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.PROVIDER,
        keywords=(
            ("provider", 1.5),
            ("anthropic", 1.0),
            ("openai", 1.0),
            ("litellm", 1.4),
            ("model", 0.6),
            ("claude", 0.7),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.PERMISSION,
        keywords=(
            ("permission", 1.6),
            ("allow", 0.8),
            ("deny", 0.8),
            ("sandbox", 1.2),
            ("approval", 1.3),
            ("yolo", 1.0),
            ("security", 0.9),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.MEMORY,
        keywords=(
            ("memory", 1.5),
            ("remember", 1.2),
            ("compact", 1.4),
            ("context collapse", 1.6),
            ("dreaming", 1.6),
            ("forget", 0.9),
            ("session history", 1.0),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.MCP,
        keywords=(
            ("mcp", 1.6),
            ("model context protocol", 1.8),
            ("mcp server", 1.7),
        ),
        sub_tags=(("server", "mcp_server"), ("client", "mcp_client")),
    ),
    _CategoryRule(
        category=FeatureCategory.MULTI_AGENT,
        keywords=(
            ("multi-agent", 1.6),
            ("multi agent", 1.6),
            ("agent2agent", 1.7),
            ("a2a", 1.4),
            ("team", 0.9),
            ("subagent", 1.2),
            ("buddy", 1.1),
            ("crew", 1.0),
        ),
        sub_tags=(
            ("a2a", "a2a_protocol"),
            ("agent2agent", "a2a_protocol"),
            ("team", "team_management"),
            ("subagent", "task_delegation"),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.ORCHESTRATOR,
        keywords=(
            ("orchestrator", 1.6),
            ("workflow", 1.2),
            ("cron", 1.0),
            ("scheduler", 1.3),
            ("kairos", 1.4),
            ("ultraplan", 1.6),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.TUI_REPL,
        keywords=(
            ("tui", 1.5),
            ("repl", 1.3),
            ("prompt-toolkit", 1.4),
            ("textual", 1.4),
            ("ui", 0.6),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.CLI,
        keywords=(
            ("cli", 1.3),
            ("subcommand", 1.4),
            ("command line", 1.2),
            ("flag", 0.6),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.OBSERVABILITY,
        keywords=(
            ("telemetry", 1.6),
            ("observability", 1.6),
            ("trace", 1.0),
            ("metric", 0.9),
            ("langfuse", 1.5),
            ("analytics", 1.1),
            ("log", 0.5),
            ("event", 0.6),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.INFRA,
        keywords=(
            ("ci", 1.0),
            ("build", 0.6),
            ("deploy", 0.9),
            ("docker", 1.1),
            ("release", 0.5),
            ("packaging", 1.0),
            ("dependency", 0.7),
            ("performance", 0.8),
            ("bench", 0.8),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.EMBODIED_AI,
        keywords=(
            ("embodied", 1.6),
            ("robot", 1.2),
            ("robotics", 1.3),
            ("locomotion", 1.3),
            ("imitation learning", 1.5),
            ("teleoperation", 1.4),
            ("sim-to-real", 1.5),
            ("sim2real", 1.5),
            ("grasping", 1.3),
            ("dexterous", 1.3),
            ("end-effector", 1.2),
            ("actuator", 1.1),
            ("kinematic", 1.2),
            ("reinforcement learning", 1.3),
            ("vla", 1.4),
            ("vision language action", 1.5),
            ("robot learning", 1.4),
            ("foundation model for robotics", 1.6),
            ("generalist robot", 1.5),
            ("mobile manipulation", 1.4),
            ("humanoid", 1.2),
            ("legged", 1.2),
        ),
        sub_tags=(
            ("vla", "vla_model"),
            ("vision language action", "vla_model"),
            ("imitation learning", "imitation_learning"),
            ("sim-to-real", "sim2real_transfer"),
            ("sim2real", "sim2real_transfer"),
            ("foundation model for robotics", "foundation_model"),
            ("generalist robot", "generalist_policy"),
        ),
        # Ambiguous keywords that require a robotics/embodied anchor to score.
        conditional_keywords=(
            # "policy" is robot_policy only near embodied terms; otherwise
            # it's usage-policy / permissions-policy in software projects.
            (
                "policy",
                0.9,
                (
                    "robot",
                    "robotics",
                    "vla",
                    "manipulation",
                    "grasping",
                    "trajectory",
                    "locomotion",
                    "end-effector",
                    "actuator",
                ),
            ),
            # "navigation" in the context of robotics, SLAM, or habitat.
            (
                "navigation",
                1.3,
                (
                    "robot",
                    "robotics",
                    "embodied",
                    "habitat",
                    "slam",
                    "lidar",
                    "locomotion",
                    "manipulation",
                ),
            ),
            # "imitation" alone is too broad; anchor it to robotics terms.
            (
                "imitation",
                1.2,
                (
                    "robot",
                    "robotics",
                    "learning",
                    "policy",
                    "manipulation",
                    "trajectory",
                    "grasping",
                    "locomotion",
                ),
            ),
            # "trajectory" means physical motion path in robotics but
            # conversation/action trace in software agents.
            (
                "trajectory",
                1.1,
                (
                    "robot",
                    "robotics",
                    "manipulation",
                    "kinematic",
                    "actuator",
                    "grasping",
                    "locomotion",
                    "end-effector",
                ),
            ),
            # "manipulation" is robotic manipulation when near embodied
            # terms; otherwise it's data/string/node manipulation.
            (
                "manipulation",
                1.4,
                (
                    "robot",
                    "robotics",
                    "grasping",
                    "dexterous",
                    "end-effector",
                    "actuator",
                    "locomotion",
                    "embodied",
                    "teleoperation",
                ),
            ),
        ),
        conditional_sub_tags=(
            (
                "policy",
                "robot_policy",
                (
                    "robot",
                    "robotics",
                    "vla",
                    "manipulation",
                    "grasping",
                    "trajectory",
                    "locomotion",
                ),
            ),
            (
                "imitation",
                "imitation_learning",
                ("robot", "robotics", "learning", "policy", "manipulation"),
            ),
        ),
    ),
    _CategoryRule(
        category=FeatureCategory.SPATIAL_INTELLIGENCE,
        keywords=(
            ("point cloud", 1.4),
            ("nerf", 1.5),
            ("gaussian splatting", 1.6),
            ("neural radiance", 1.6),
            ("novel view", 1.3),
            ("slam", 1.5),
            ("lidar", 1.3),
            ("voxel", 1.2),
            ("mesh", 1.0),
            ("volumetric", 1.2),
            ("radiance field", 1.6),
            ("occupancy", 1.2),
            ("physical simulation", 1.5),
        ),
        sub_tags=(
            ("nerf", "nerf"),
            ("gaussian splatting", "gaussian_splatting"),
            ("neural radiance", "nerf"),
            ("slam", "slam"),
            ("point cloud", "point_cloud"),
        ),
        # Ambiguous keywords that require a 3D/spatial anchor to score.
        # Without anchors generic terms like "rendering" catch UI rendering
        # in software projects, "scene" catches narrative scenes, etc.
        conditional_keywords=(
            (
                "rendering",
                1.1,
                (
                    "nerf",
                    "gaussian",
                    "splatting",
                    "point cloud",
                    "3d",
                    "mesh",
                    "depth",
                    "volumetric",
                    "radiance",
                    "scene",
                    "reconstruction",
                ),
            ),
            (
                "scene",
                0.9,
                (
                    "nerf",
                    "gaussian",
                    "3d",
                    "reconstruction",
                    "rendering",
                    "point cloud",
                    "volumetric",
                ),
            ),
            (
                "reconstruction",
                1.2,
                ("nerf", "gaussian", "3d", "point cloud", "mesh", "slam", "volumetric", "depth"),
            ),
            (
                "depth",
                1.0,
                (
                    "nerf",
                    "gaussian",
                    "point cloud",
                    "3d",
                    "lidar",
                    "rendering",
                    "volumetric",
                    "scene",
                ),
            ),
            ("segmentation", 1.0, ("point cloud", "3d", "lidar", "voxel", "mesh", "scene")),
            (
                "3d",
                1.0,
                (
                    "nerf",
                    "gaussian",
                    "point cloud",
                    "rendering",
                    "slam",
                    "mesh",
                    "voxel",
                    "reconstruction",
                    "scene",
                    "depth",
                ),
            ),
            (
                "spatial",
                1.5,
                (
                    "nerf",
                    "gaussian",
                    "3d",
                    "point cloud",
                    "slam",
                    "reconstruction",
                    "rendering",
                    "mesh",
                ),
            ),
            ("geometric", 1.1, ("nerf", "gaussian", "3d", "point cloud", "mesh", "reconstruction")),
            (
                "world model",
                1.4,
                ("nerf", "gaussian", "3d", "spatial", "point cloud", "scene", "rendering"),
            ),
        ),
        conditional_sub_tags=(
            (
                "world model",
                "world_model",
                ("nerf", "gaussian", "3d", "spatial", "point cloud", "scene"),
            ),
        ),
    ),
)


# Categories that imply a source lives in the embodied / spatial domain.
# Used by _check_domain to reject cross-domain keyword matches.
_EMBODIED_CATS: frozenset[FeatureCategory] = frozenset({FeatureCategory.EMBODIED_AI})
_SPATIAL_CATS: frozenset[FeatureCategory] = frozenset({FeatureCategory.SPATIAL_INTELLIGENCE})


# Categories the tie-breaker prefers (highest priority first).
_TIEBREAK: tuple[FeatureCategory, ...] = (
    FeatureCategory.AGENT_LOOP,
    FeatureCategory.TOOL_SYSTEM,
    FeatureCategory.MEMORY,
    FeatureCategory.PERMISSION,
    FeatureCategory.MCP,
    FeatureCategory.MULTI_AGENT,
    FeatureCategory.ORCHESTRATOR,
    FeatureCategory.EMBODIED_AI,
    FeatureCategory.SPATIAL_INTELLIGENCE,
    FeatureCategory.OBSERVABILITY,
    FeatureCategory.PROVIDER,
    FeatureCategory.TUI_REPL,
    FeatureCategory.CLI,
    FeatureCategory.INFRA,
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


LLMClassifierHook = Callable[[FeatureRecord], FeatureCategory]


class FeatureClassifier:
    """Keyword-based feature classifier with an optional LLM hook.

    ``roadmap_keywords`` lets callers tweak the strategic-value scoring
    downstream (the scorer reads the same list).
    """

    def __init__(
        self,
        rules: Iterable[_CategoryRule] | None = None,
        llm_hook: LLMClassifierHook | None = None,
        roadmap_keywords: Iterable[str] | None = None,
        source_domain_map: dict[str, str] | None = None,
    ) -> None:
        self._rules: tuple[_CategoryRule, ...] = tuple(rules) if rules is not None else _RULES
        self._llm_hook = llm_hook
        self.roadmap_keywords: tuple[str, ...] = tuple(
            k.lower() for k in (roadmap_keywords or DEFAULT_ROADMAP_KEYWORDS)
        )
        # source_name → SourceDomain value, used by _check_domain to
        # prevent cross-domain misclassification (e.g. a software-eng
        # project tagged as EMBODIED_AI just because it mentions "robot").
        self._source_domain_map: dict[str, str] = dict(source_domain_map) if source_domain_map else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, record: FeatureRecord) -> FeatureRecord:
        """Return ``record`` with ``category`` + ``tags`` filled in.

        The original record is mutated (it is a dataclass, so this is
        the same as returning a copy from the caller's POV) and also
        returned so the method can be used inline.
        """
        text = self._join_text(record)
        scores: dict[FeatureCategory, float] = {}
        sub_tags: dict[FeatureCategory, list[str]] = {}

        for rule in self._rules:
            score = 0.0
            matched_sub_tags: list[str] = []
            for keyword, weight in rule.keywords:
                if self._keyword_matches(keyword, text):
                    score += weight
            for keyword, sub_tag in rule.sub_tags:
                if self._keyword_matches(keyword, text):
                    matched_sub_tags.append(sub_tag)
            for keyword, weight, anchors in rule.conditional_keywords:
                if self._keyword_matches(keyword, text) and any(self._keyword_matches(a, text) for a in anchors):
                    score += weight
            for keyword, sub_tag, anchors in rule.conditional_sub_tags:
                if self._keyword_matches(keyword, text) and any(self._keyword_matches(a, text) for a in anchors):
                    matched_sub_tags.append(sub_tag)
            if score > 0:
                scores[rule.category] = scores.get(rule.category, 0.0) + score
                if matched_sub_tags:
                    sub_tags.setdefault(rule.category, []).extend(matched_sub_tags)

        category = self._pick_category(scores)
        # Source-domain fallback: when no keyword matches, use the
        # source's known domain so features from embodied / spatial
        # projects don't fall through to UNKNOWN.
        if category == FeatureCategory.UNKNOWN:
            domain = self._source_domain_map.get(record.source, SourceDomain.GENERAL.value)
            if domain in {SourceDomain.EMBODIED_AI.value, SourceDomain.SPATIAL_INTELLIGENCE.value}:
                try:
                    category = FeatureCategory(domain)
                except ValueError:
                    pass
        category = self._check_domain(category, record.source)
        record.category = category

        # Combine rule sub_tags with roadmap keyword tags so the
        # ``tags`` field carries enough context for filtering.
        new_tags = list(record.tags)
        for tag in sub_tags.get(category, ()):
            if tag not in new_tags:
                new_tags.append(tag)
        for keyword in self.roadmap_keywords:
            if keyword and keyword in text and keyword not in new_tags:
                new_tags.append(keyword)
        record.tags = new_tags

        if category == FeatureCategory.UNKNOWN and self._llm_hook is not None:
            try:
                refined = self._llm_hook(record)
            except Exception as exc:  # noqa: BLE001
                _log.warning("LLM classifier hook raised (%s); keeping UNKNOWN", exc)
                refined = category
            else:
                record.category = self._check_domain(refined, record.source)

        return record

    def classify_many(self, records: Iterable[FeatureRecord]) -> list[FeatureRecord]:
        return [self.classify(r) for r in records]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _join_text(record: FeatureRecord) -> str:
        parts = [record.title or "", record.description or ""]
        return "\n".join(parts).lower()

    @staticmethod
    def _keyword_matches(keyword: str, haystack: str) -> bool:
        """Match ``keyword`` against ``haystack`` using word boundaries.

        Multi-word keywords use simple substring search because word
        boundaries don't behave well across spaces. Single-word keywords
        use ``\\b`` to avoid matching ``agent`` inside ``agentic``.
        """
        kw = keyword.lower().strip()
        if not kw:
            return False
        if " " in kw or "-" in kw or "/" in kw:
            return kw in haystack
        return bool(re.search(rf"\b{re.escape(kw)}\b", haystack))

    def _pick_category(self, scores: dict[FeatureCategory, float]) -> FeatureCategory:
        if not scores:
            return FeatureCategory.UNKNOWN
        max_score = max(scores.values())
        # All tied candidates ranked by tiebreak order; the first
        # surviving entry is the winner.
        tied = [cat for cat, score in scores.items() if score == max_score]
        for cat in _TIEBREAK:
            if cat in tied:
                return cat
        return min(tied, key=lambda cat: cat.value)

    def _check_domain(self, category: FeatureCategory, source: str) -> FeatureCategory:
        """Reject ``category`` when it contradicts the source's known domain.

        A ``code_agent`` source mentioning "robot" in a gallery
        example is NOT an embodied-AI feature; a ``spatial_intelligence``
        source mentioning "policy" is NOT a robot-policy feature.
        """
        domain = self._source_domain_map.get(source, SourceDomain.GENERAL.value)
        if domain == SourceDomain.CODE_AGENT.value:
            if category in _EMBODIED_CATS or category in _SPATIAL_CATS:
                return FeatureCategory.UNKNOWN
        elif domain == SourceDomain.EMBODIED_AI.value:
            if category in _SPATIAL_CATS:
                return FeatureCategory.UNKNOWN
        elif domain == SourceDomain.SPATIAL_INTELLIGENCE.value and category in _EMBODIED_CATS:
            return FeatureCategory.UNKNOWN
        return category
