#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Runtime configuration for SR-5.1 Community Feature Radar.

Mirrors the radar config block of the original design:

    community_radar:
      enabled: false
      cron_schedule: "0 8 * * 1"
      max_features_per_report: 20
      output_dir: ".reports/community-radar"
      notify: false

Since Phase 3, ``enabled`` and ``notify`` default to ``True`` so a
fresh install immediately gets a working Cron durable task; users who
want to opt out set ``CLAWCODEX_RADAR_ENABLED=0`` or override
``community_radar.enabled: false`` in their workflow.md. All values are
overridable via CLI/env.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


DEFAULT_OUTPUT_DIR = ".reports/community-radar"
DEFAULT_CRON_SCHEDULE = "0 8 * * 1"  # Mondays 08:00 UTC
DEFAULT_MAX_FEATURES = 30
DEFAULT_LANGUAGE = "zh"
# Feature types excluded from reports (case-insensitive, matched against
# FeatureType.value). BUGFIX and DEPRECATION are excluded by default because
# they don't represent new capabilities.
DEFAULT_EXCLUDE_FEATURE_TYPES: list[str] = ["bugfix", "deprecation"]
# Categories that qualify for the "Highlights" summary section.
# Only features in these categories + scoring above highlight_min_score are
# promoted to the prose-summary block.
DEFAULT_HIGHLIGHT_CATEGORIES: list[str] = [
    "agent_loop",
    "tool_system",
    "multi_agent",
    "orchestrator",
    "provider",
    "memory",
    "mcp",
    "permission",
    "observability",
]
DEFAULT_HIGHLIGHT_MIN_SCORE = 55.0
# Issue sync defaults
DEFAULT_SYNC_ISSUES = False
DEFAULT_SYNC_ISSUES_MAX_PER_SCAN = 2
DEFAULT_SYNC_ISSUES_LABELS: list[str] = ["community-radar"]
DEFAULT_SYNC_ISSUES_CLOSED_ISSUE_MODE = "ask"  # "ask" | "skip" | "retry"
DEFAULT_WEIGHTS: dict[str, float] = {
    "popularity": 0.15,
    "maturity": 0.20,
    "adaptation_cost": 0.25,
    "strategic_value": 0.25,
    "architecture_fit": 0.15,
}
# Tokens that suggest a feature aligns with the ClawCodex roadmap
# (roadmap headings). The scorer matches tokens against
# the free-form ``title + description`` text after lowercasing.
DEFAULT_ROADMAP_KEYWORDS: list[str] = [
    "agent",
    "tool",
    "permission",
    "memory",
    "compact",
    "mcp",
    "sandbox",
    "cron",
    "telemetry",
    "session",
    "multi-agent",
    "orchestrator",
    "provider",
    "context",
    # SR-5.1 embodied / spatial intelligence domain keywords
    "embodied",
    "robot",
    "robotics",
    "manipulation",
    "navigation",
    "policy",
    "vla",
    "simulation",
    "spatial",
    "3d",
    "nerf",
    "gaussian",
    "slam",
    "rendering",
    "imitation",
]


@dataclass
class RadarConfig:
    """Runtime knobs for the radar.

    ``weights`` is keyed by :class:`FeatureScore` dimension name. They
    are normalised to sum to 1.0 by :meth:`normalized_weights` so the
    scorer never has to worry about caller-supplied totals.

    Phase 3 changes:

    * ``enabled`` defaults to ``True`` so a fresh install immediately
      gets a working Cron durable task. Users who want to opt out set
      the env var ``CLAWCODEX_RADAR_ENABLED=0`` or override
      ``community_radar.enabled: false`` in their workflow.md.
    * ``notify`` defaults to ``True``; the actual delivery only happens
      when at least one channel is configured (see
      :mod:`clawcodex_ext.community_radar.notifier`).
    """

    enabled: bool = True
    cron_schedule: str = DEFAULT_CRON_SCHEDULE
    max_features_per_report: int = DEFAULT_MAX_FEATURES
    output_dir: str = DEFAULT_OUTPUT_DIR
    notify: bool = True
    use_llm: bool = False
    cache_dir: str = ".cache/community-radar"
    # i18n & report filtering (Phase 4 / SR-5.3)
    language: str = DEFAULT_LANGUAGE  # "zh" | "en"
    exclude_feature_types: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_FEATURE_TYPES))
    highlight_categories: list[str] = field(default_factory=lambda: list(DEFAULT_HIGHLIGHT_CATEGORIES))
    highlight_min_score: float = DEFAULT_HIGHLIGHT_MIN_SCORE
    # ``weights`` + ``roadmap_keywords`` are deliberately not part of
    # the public workflow.md schema; tests and advanced users can
    # override them via env vars (CLAWCODEX_RADAR_WEIGHT_POPULARITY=0.20)
    # or via the Python API.
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    roadmap_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_ROADMAP_KEYWORDS))
    # LLM classification (MAJOR/MINOR + highlights) is always enabled.
    # Translation (title_zh / desc_zh) is controlled by ``language``:
    # when "zh" the LLM prompt includes translation; when "en" it does not.
    #
    # --- GitCode / GitHub / Gitee issue sync ---
    sync_issues: bool = DEFAULT_SYNC_ISSUES
    target_repo: str = ""  # "owner/repo" or "platform.com/owner/repo"
    api_token: str = ""  # platform API token (prefer env vars)
    sync_issues_max_per_scan: int = DEFAULT_SYNC_ISSUES_MAX_PER_SCAN
    sync_issues_labels: list[str] = field(default_factory=lambda: list(DEFAULT_SYNC_ISSUES_LABELS))
    sync_issues_closed_issue_mode: str = DEFAULT_SYNC_ISSUES_CLOSED_ISSUE_MODE

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def normalized_weights(self) -> dict[str, float]:
        """Return ``weights`` rescaled to sum to 1.0.

        Falls back to :data:`DEFAULT_WEIGHTS` when the user supplied a
        dict whose entries are all zero / non-numeric, so the scorer is
        never asked to divide by zero.
        """
        cleaned: dict[str, float] = {}
        for key, default_val in DEFAULT_WEIGHTS.items():
            raw = self.weights.get(key, default_val)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = default_val
            cleaned[key] = max(0.0, value)
        total = sum(cleaned.values())
        if total <= 0:
            return dict(DEFAULT_WEIGHTS)
        return {k: v / total for k, v in cleaned.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cron_schedule": self.cron_schedule,
            "max_features_per_report": self.max_features_per_report,
            "output_dir": self.output_dir,
            "notify": self.notify,
            "cache_dir": self.cache_dir,
            "language": self.language,
            "exclude_feature_types": list(self.exclude_feature_types),
            "highlight_categories": list(self.highlight_categories),
            "highlight_min_score": self.highlight_min_score,
            "weights": dict(self.weights),
            "roadmap_keywords": list(self.roadmap_keywords),
            "use_llm": self.use_llm,
            # Issue sync
            "sync_issues": self.sync_issues,
            "target_repo": self.target_repo,
            # api_token deliberately omitted: credentials must not be
            # persisted via to_dict (logs / cache files).
            "sync_issues_max_per_scan": self.sync_issues_max_per_scan,
            "sync_issues_labels": list(self.sync_issues_labels),
            "sync_issues_closed_issue_mode": self.sync_issues_closed_issue_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RadarConfig:
        if not isinstance(data, dict):
            return cls()
        weights_raw = data.get("weights") or {}
        weights = dict(DEFAULT_WEIGHTS)
        if isinstance(weights_raw, dict):
            for key, value in weights_raw.items():
                if key in DEFAULT_WEIGHTS:
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(parsed):
                        weights[key] = parsed
        keywords_raw = data.get("roadmap_keywords") or data.get("roadmapKeywords") or []
        if isinstance(keywords_raw, list) and keywords_raw:
            keywords = [str(k) for k in keywords_raw]
        else:
            keywords = list(DEFAULT_ROADMAP_KEYWORDS)

        # Language
        lang = str(data.get("language") or DEFAULT_LANGUAGE)
        if lang not in ("zh", "en"):
            lang = DEFAULT_LANGUAGE

        # Exclude feature types
        exclude_raw = data.get("exclude_feature_types")
        if exclude_raw is None:
            exclude_raw = data.get("excludeFeatureTypes")
        if isinstance(exclude_raw, list):
            exclude_types = [str(t).lower() for t in exclude_raw]
        else:
            exclude_types = list(DEFAULT_EXCLUDE_FEATURE_TYPES)

        # Highlight categories
        hl_cats_raw = data.get("highlight_categories")
        if hl_cats_raw is None:
            hl_cats_raw = data.get("highlightCategories")
        if isinstance(hl_cats_raw, list):
            hl_cats = [str(c).lower() for c in hl_cats_raw]
        else:
            hl_cats = list(DEFAULT_HIGHLIGHT_CATEGORIES)

        # Highlight min score
        hl_raw = data.get("highlight_min_score")
        if hl_raw is None:
            hl_raw = data.get("highlightMinScore")
        if hl_raw is None:
            hl_raw = DEFAULT_HIGHLIGHT_MIN_SCORE
        try:
            hl_score = float(hl_raw)
        except (TypeError, ValueError):
            hl_score = DEFAULT_HIGHLIGHT_MIN_SCORE

        # Issue sync
        sync_issues = bool(data.get("sync_issues", DEFAULT_SYNC_ISSUES))
        target_repo = str(data.get("target_repo") or data.get("targetRepo") or "")
        api_token = str(data.get("api_token") or data.get("apiToken") or "")
        sync_max_raw = data.get("sync_issues_max_per_scan") or data.get("syncIssuesMaxPerScan")
        try:
            sync_max = int(sync_max_raw) if sync_max_raw is not None else DEFAULT_SYNC_ISSUES_MAX_PER_SCAN
        except (TypeError, ValueError):
            sync_max = DEFAULT_SYNC_ISSUES_MAX_PER_SCAN
        sync_labels_raw = data.get("sync_issues_labels")
        if sync_labels_raw is None:
            sync_labels_raw = data.get("syncIssuesLabels")
        if isinstance(sync_labels_raw, list):
            sync_labels = [str(item) for item in sync_labels_raw]
        else:
            sync_labels = list(DEFAULT_SYNC_ISSUES_LABELS)
        # Closed issue mode
        closed_mode_raw = str(
            data.get("sync_issues_closed_issue_mode")
            or data.get("syncIssuesClosedIssueMode")
            or DEFAULT_SYNC_ISSUES_CLOSED_ISSUE_MODE
        )
        if closed_mode_raw not in ("ask", "skip", "retry"):
            closed_mode_raw = DEFAULT_SYNC_ISSUES_CLOSED_ISSUE_MODE

        # Max features per report
        max_features_raw = data.get("max_features_per_report")
        if max_features_raw is None:
            max_features_raw = data.get("maxFeaturesPerReport")
        if max_features_raw is None:
            max_features_raw = DEFAULT_MAX_FEATURES
        try:
            max_features = int(max_features_raw)
        except (TypeError, ValueError):
            max_features = DEFAULT_MAX_FEATURES

        return cls(
            enabled=bool(data.get("enabled", True)),
            cron_schedule=str(data.get("cron_schedule") or DEFAULT_CRON_SCHEDULE),
            max_features_per_report=max_features,
            output_dir=str(data.get("output_dir") or data.get("outputDir") or DEFAULT_OUTPUT_DIR),
            notify=bool(data.get("notify", True)),
            cache_dir=str(data.get("cache_dir") or data.get("cacheDir") or ".cache/community-radar"),
            language=lang,
            use_llm=bool(data.get("use_llm", False)),
            exclude_feature_types=exclude_types,
            highlight_categories=hl_cats,
            highlight_min_score=hl_score,
            weights=weights,
            roadmap_keywords=keywords,
            # Issue sync
            sync_issues=sync_issues,
            target_repo=target_repo,
            api_token=api_token,
            sync_issues_max_per_scan=sync_max,
            sync_issues_labels=sync_labels,
            sync_issues_closed_issue_mode=closed_mode_raw,
        )


# ---------------------------------------------------------------------------
# Environment overrides (handy in CI / Cron)
# ---------------------------------------------------------------------------


def apply_env_overrides(config: RadarConfig) -> RadarConfig:
    """Mutate-and-return ``config`` based on ``CLAWCODEX_RADAR_*`` env vars.

    Supported keys:

    * ``CLAWCODEX_RADAR_ENABLED=1``
    * ``CLAWCODEX_RADAR_CRON="0 8 * * 1"``
    * ``CLAWCODEX_RADAR_MAX=25``
    * ``CLAWCODEX_RADAR_OUTPUT=/abs/path/.reports``
    * ``CLAWCODEX_RADAR_NOTIFY=1``
    * ``CLAWCODEX_RADAR_USE_LLM=1``
    * ``CLAWCODEX_RADAR_CACHE_DIR=/var/cache/...``
    * ``CLAWCODEX_RADAR_WEIGHT_POPULARITY=0.20`` (per dimension)
    * ``CLAWCODEX_RADAR_LANGUAGE=en``
    * ``CLAWCODEX_RADAR_EXCLUDE_TYPES=bugfix,deprecation``
    * ``CLAWCODEX_RADAR_HIGHLIGHT_CATEGORIES=agent_loop,tool_system``
    * ``CLAWCODEX_RADAR_HIGHLIGHT_MIN_SCORE=60``

    Unknown keys are silently ignored so a stale env file does not
    crash the radar.
    """

    def _bool(name: str) -> bool | None:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return None
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    if (v := _bool("CLAWCODEX_RADAR_ENABLED")) is not None:
        config.enabled = v
    if (v := _bool("CLAWCODEX_RADAR_NOTIFY")) is not None:
        config.notify = v
    if (v := _bool("CLAWCODEX_RADAR_USE_LLM")) is not None:
        config.use_llm = v

    if raw := os.environ.get("CLAWCODEX_RADAR_LANGUAGE"):
        lang = raw.strip().lower()
        if lang in ("zh", "en"):
            config.language = lang
    if raw := os.environ.get("CLAWCODEX_RADAR_EXCLUDE_TYPES"):
        config.exclude_feature_types = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if raw := os.environ.get("CLAWCODEX_RADAR_HIGHLIGHT_CATEGORIES"):
        config.highlight_categories = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if raw := os.environ.get("CLAWCODEX_RADAR_HIGHLIGHT_MIN_SCORE"):
        try:
            config.highlight_min_score = float(raw)
        except ValueError:
            _log.warning("invalid CLAWCODEX_RADAR_HIGHLIGHT_MIN_SCORE=%r; ignored", raw)

    if raw := os.environ.get("CLAWCODEX_RADAR_CRON"):
        config.cron_schedule = raw.strip()
    if raw := os.environ.get("CLAWCODEX_RADAR_OUTPUT"):
        config.output_dir = raw.strip()
    if raw := os.environ.get("CLAWCODEX_RADAR_CACHE_DIR"):
        config.cache_dir = raw.strip()
    if raw := os.environ.get("CLAWCODEX_RADAR_MAX"):
        try:
            config.max_features_per_report = int(raw)
        except ValueError:
            _log.warning("invalid CLAWCODEX_RADAR_MAX=%r; ignored", raw)

    for dim in DEFAULT_WEIGHTS:
        env_key = f"CLAWCODEX_RADAR_WEIGHT_{dim.upper()}"
        if raw := os.environ.get(env_key):
            try:
                config.weights[dim] = float(raw)
            except ValueError:
                _log.warning("invalid %s=%r; ignored", env_key, raw)

    # ── Issue sync overrides ──
    if (v := _bool("CLAWCODEX_RADAR_SYNC_ISSUES")) is not None:
        config.sync_issues = v
    if raw := os.environ.get("CLAWCODEX_RADAR_SYNC_MAX"):
        try:
            config.sync_issues_max_per_scan = int(raw)
        except ValueError:
            _log.warning("invalid CLAWCODEX_RADAR_SYNC_MAX=%r; ignored", raw)
    if raw := os.environ.get("CLAWCODEX_RADAR_SYNC_LABELS"):
        config.sync_issues_labels = [item.strip() for item in raw.split(",") if item.strip()]
    if raw := os.environ.get("CLAWCODEX_RADAR_CLOSED_ISSUE_MODE"):
        mode = raw.strip().lower()
        if mode in ("ask", "skip", "retry"):
            config.sync_issues_closed_issue_mode = mode

    return config


# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    """``~/.clawcodex/community-radar/config.yaml`` (env-overridable)."""
    base = os.environ.get("CLAWCODEX_HOME")
    root = Path(base) if base else Path.home() / ".clawcodex"
    return root / "community-radar" / "config.yaml"
