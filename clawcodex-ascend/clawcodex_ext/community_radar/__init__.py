"""SR-5.1 Community Feature Radar.

Public surface for the community radar package. Most consumers should
import from the submodules (e.g. ``from clawcodex_ext.community_radar
import run_community_scan``); this file re-exports the high-level
helpers so external modules can grab everything in one import.

This package implements SR-5.1 entirely inside ``clawcodex_ext/*`` so
it never touches ``src/*`` and stays clear of the upstream-sync audit.
The Cron integration is provided by :func:`run_community_scan`, which
the durable task scheduler calls as ``run_community_scan``.
"""

from __future__ import annotations

from .config import (
    DEFAULT_CRON_SCHEDULE,
    DEFAULT_MAX_FEATURES,
    DEFAULT_OUTPUT_DIR,
    RadarConfig,
    apply_env_overrides,
    default_config_path,
)
from .discover import (
    MAX_COUNT_PER_DISCOVER,
    MAX_SOURCES_TOTAL,
    DiscoverCandidate,
    DiscoverResult,
    build_search_query,
    discover_sources,
    quick_domain_match,
)
from .cron_integration import (
    DEFAULT_CRON_PROMPT,
    DEFAULT_CRON_TASK_ID,
    CronTaskSummary,
    ensure_cron_installed,
    get_cron_task_status,
    install_cron_task,
    load_registry_safely,
    uninstall_cron_task,
)
from .i18n import STRINGS, Language, build_template_labels, get_text
from .llm_classifier import (
    LLMConfig,
    build_classifier_hook,
    build_extractor_hook,
    build_summarizer_hook,
    llm_generated_marker,
)
from .models import (
    CommunityDigest,
    DigestStats,
    FeatureCategory,
    FeatureRecord,
    FeatureScore,
    FeatureType,
    FetchResult,
    PullRequest,
    Release,
    ScoredFeature,
    WatchSource,
)
from .notifier import (
    NOTIFY_CONFIG_RELATIVE_PATH,
    DigestNotifier,
    NotifyConfig,
    build_digest_message,
)
from .pipeline import CommunityRadarPipeline, ScanResult, run_community_scan
from .issue_platforms import (
    IssueClient,
    IssuePlatform,
    ResolvedTarget,
    resolve_target,
)
from .issue_sync import (
    IssueSyncCache,
    IssueSyncResult,
    list_candidates_interactive,
    sync_features_to_issues,
    sync_single_feature,
)
from .registry import (
    DEFAULT_SOURCES,
    PHASE1_SOURCES,
    PHASE2_SOURCES,
    SourceRegistry,
    default_registry_path,
)
from .reporter import (
    CommunityReporter,
    DigestWriteResult,
    copy_to_persistent,
    render_proposals,
)

__all__ = [
    # config
    "RadarConfig",
    "apply_env_overrides",
    "default_config_path",
    "DEFAULT_CRON_SCHEDULE",
    "DEFAULT_MAX_FEATURES",
    "DEFAULT_OUTPUT_DIR",
    # models
    "CommunityDigest",
    "DigestStats",
    "FeatureCategory",
    "FeatureRecord",
    "FeatureScore",
    "FeatureType",
    "FetchResult",
    "PullRequest",
    "Release",
    "ScoredFeature",
    "WatchSource",
    # pipeline
    "CommunityRadarPipeline",
    "ScanResult",
    "run_community_scan",
    # registry
    "DEFAULT_SOURCES",
    "PHASE1_SOURCES",
    "PHASE2_SOURCES",
    "SourceRegistry",
    "default_registry_path",
    # cron integration
    "DEFAULT_CRON_PROMPT",
    "DEFAULT_CRON_TASK_ID",
    "CronTaskSummary",
    "ensure_cron_installed",
    "install_cron_task",
    "uninstall_cron_task",
    "get_cron_task_status",
    "load_registry_safely",
    # LLM hooks (Phase 2)
    "LLMConfig",
    "build_classifier_hook",
    "build_extractor_hook",
    "build_summarizer_hook",
    "llm_generated_marker",
    # i18n (Phase 4 / SR-5.3)
    "STRINGS",
    "Language",
    "get_text",
    "build_template_labels",
    # Notifier (Phase 4)
    "NOTIFY_CONFIG_RELATIVE_PATH",
    "DigestNotifier",
    "NotifyConfig",
    "build_digest_message",
    # Reporter extras (Phase 4)
    "CommunityReporter",
    "DigestWriteResult",
    "copy_to_persistent",
    "render_proposals",
    # discover
    "MAX_COUNT_PER_DISCOVER",
    "MAX_SOURCES_TOTAL",
    "DiscoverCandidate",
    "DiscoverResult",
    "build_search_query",
    "discover_sources",
    "quick_domain_match",
    # issue platforms & sync
    "IssueClient",
    "IssuePlatform",
    "ResolvedTarget",
    "resolve_target",
    "IssueSyncCache",
    "IssueSyncResult",
    "sync_features_to_issues",
    "list_candidates_interactive",
    "sync_single_feature",
]
