#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Tracker kind registry & adapter factory.

Split out of ``tracker.py``: static metadata about supported tracker
backends, config validation, and the ``create_tracker_adapter`` factory.
``tracker.py`` re-exports these symbols for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from .tracker import TrackerAdapter


# Supported tracker kinds — adapters accept these in ``tracker.kind``.
SUPPORTED_TRACKERS = frozenset({"linear", "github", "gitee", "gitcode", "local"})


@dataclass(frozen=True)
class TrackerKindInfo:
    """Static metadata used by config validation and adapter creation."""

    kind: str
    label: str
    default_endpoint: str
    default_clone_base_url: str | None
    api_key_env_vars: tuple[str, ...]
    owner_env_vars: tuple[str, ...] = ()
    repo_env_vars: tuple[str, ...] = ()
    assignee_env_vars: tuple[str, ...] = ()
    requires_project_slug: bool = False
    requires_repository: bool = False


class TrackerConfigError(ValueError):
    """Raised when tracker configuration is invalid."""


_TRACKER_KIND_INFO: dict[str, TrackerKindInfo] = {
    "linear": TrackerKindInfo(
        kind="linear",
        label="Linear",
        default_endpoint="https://api.linear.app/graphql",
        default_clone_base_url=None,
        api_key_env_vars=("LINEAR_API_KEY",),
        assignee_env_vars=("LINEAR_ASSIGNEE",),
        requires_project_slug=True,
    ),
    "github": TrackerKindInfo(
        kind="github",
        label="GitHub",
        default_endpoint="https://api.github.com",
        default_clone_base_url="https://github.com",
        api_key_env_vars=("GITHUB_TOKEN", "GITHUB_API_KEY"),
        owner_env_vars=("GITHUB_OWNER", "TRACKER_OWNER"),
        repo_env_vars=("GITHUB_REPO", "TRACKER_REPO"),
        assignee_env_vars=("GITHUB_ASSIGNEE", "TRACKER_ASSIGNEE"),
        requires_repository=True,
    ),
    "gitee": TrackerKindInfo(
        kind="gitee",
        label="Gitee",
        default_endpoint="https://gitee.com/api/v5",
        default_clone_base_url="https://gitee.com",
        api_key_env_vars=("GITEE_TOKEN", "GITEE_API_KEY"),
        owner_env_vars=("GITEE_OWNER", "TRACKER_OWNER"),
        repo_env_vars=("GITEE_REPO", "TRACKER_REPO"),
        assignee_env_vars=("GITEE_ASSIGNEE", "TRACKER_ASSIGNEE"),
        requires_repository=True,
    ),
    "gitcode": TrackerKindInfo(
        kind="gitcode",
        label="GitCode",
        default_endpoint="https://api.gitcode.com/api/v5",
        default_clone_base_url="https://gitcode.com",
        api_key_env_vars=("GITCODE_TOKEN", "GITCODE_API_KEY"),
        owner_env_vars=("GITCODE_OWNER", "TRACKER_OWNER"),
        repo_env_vars=("GITCODE_REPO", "TRACKER_REPO"),
        assignee_env_vars=("GITCODE_ASSIGNEE", "TRACKER_ASSIGNEE"),
        requires_repository=True,
    ),
    "local": TrackerKindInfo(
        kind="local",
        label="Local",
        default_endpoint="",
        default_clone_base_url=None,
        api_key_env_vars=(),
    ),
}


def normalize_tracker_kind(kind: str | None) -> str:
    """Normalize a user-provided tracker kind (lowercase, stripped)."""
    normalized = (kind or "linear").strip().lower()
    if normalized not in SUPPORTED_TRACKERS:
        raise TrackerConfigError(
            f"Unsupported tracker kind: {kind!r}. Supported values: {', '.join(sorted(SUPPORTED_TRACKERS))}"
        )
    return normalized


def tracker_kind_info(kind: str) -> TrackerKindInfo:
    """Return the static ``TrackerKindInfo`` metadata for a tracker kind."""
    normalized = normalize_tracker_kind(kind)
    try:
        return _TRACKER_KIND_INFO[normalized]
    except KeyError as exc:
        raise TrackerConfigError(
            f"Unsupported tracker kind: {kind!r}. Supported values: {', '.join(sorted(SUPPORTED_TRACKERS))}"
        ) from exc


def default_active_states_for_kind(kind: str) -> list[str]:
    """Return per-tracker default active states when config omits them."""
    normalized = normalize_tracker_kind(kind)
    if normalized == "linear":
        return ["Todo", "In Progress"]
    if normalized == "local":
        return ["open", "ready"]
    if normalized == "gitcode":
        return ["opened"]
    return ["open"]


def default_terminal_states_for_kind(kind: str) -> list[str]:
    """Return per-tracker default terminal states when config omits them."""
    normalized = normalize_tracker_kind(kind)
    if normalized == "linear":
        return ["Closed", "Cancelled", "Canceled", "Duplicate", "Done"]
    if normalized == "local":
        return ["completed", "closed", "cancelled", "failed", "abandoned"]
    return ["closed"]


def validate_tracker_config(config: Any) -> None:
    """Validate tracker configuration before adapter creation."""
    info = tracker_kind_info(getattr(config, "kind", None))
    if info.kind == "local":
        if not getattr(config, "issues_path", None):
            raise TrackerConfigError("Local issues path not configured. Set tracker.issues_path in WORKFLOW.md")
        return
    if not getattr(config, "api_key", None):
        env_hint = " or ".join(info.api_key_env_vars)
        raise TrackerConfigError(
            f"{info.label} API key not configured. Set {env_hint} or tracker.api_key in WORKFLOW.md"
        )
    if info.requires_project_slug and not getattr(config, "project_slug", None):
        raise TrackerConfigError(f"{info.label} project slug not configured. Set tracker.project_slug in WORKFLOW.md")
    if info.requires_repository:
        owner = getattr(config, "owner", None)
        repo = getattr(config, "repo", None)
        if not owner or not repo:
            raise TrackerConfigError(
                f"{info.label} repository not configured. Set tracker.owner and tracker.repo in WORKFLOW.md"
            )


def create_tracker_adapter(
    config: Any,
    *,
    http_client: "httpx.AsyncClient | None" = None,
) -> "TrackerAdapter":
    """Create a tracker adapter from workflow tracker config.

    Dispatches to ``LinearAdapter`` / ``LocalTrackerAdapter`` /
    ``RepositoryTrackerAdapter`` based on the normalized kind.

    Args:
        config: the tracker config (e.g. the ``tracker`` section of the
            workflow document).
        http_client: optional pre-configured HTTP client reused by
            repository-backed adapters.

    Returns:
        A concrete ``TrackerAdapter`` instance.
    """
    kind = normalize_tracker_kind(getattr(config, "kind", None))
    validate_tracker_config(config)
    if kind == "linear":
        from .linear.adapter import LinearAdapter

        return LinearAdapter(
            api_key=getattr(config, "api_key", "") or "",
            project_slug=getattr(config, "project_slug", None),
            endpoint=getattr(config, "endpoint", None) or tracker_kind_info("linear").default_endpoint,
            active_states=list(getattr(config, "active_states", []) or []),
            assignee=getattr(config, "assignee", None),
            title_prefixes=list(getattr(config, "title_prefixes", []) or []),
            title_prefix_match=getattr(config, "title_prefix_match", "any"),
        )
    if kind == "local":
        from .local_tracker.adapter import LocalTrackerAdapter

        return LocalTrackerAdapter(
            issues_path=getattr(config, "issues_path", None) or "",
            active_states=list(getattr(config, "active_states", []) or []),
            terminal_states=list(getattr(config, "terminal_states", []) or []),
            title_prefixes=list(getattr(config, "title_prefixes", []) or []),
            title_prefix_match=getattr(config, "title_prefix_match", "any"),
        )

    from .repo_tracker.adapter import RepositoryTrackerAdapter

    return RepositoryTrackerAdapter(
        platform=kind,
        owner=getattr(config, "owner", None) or "",
        repo=getattr(config, "repo", None) or "",
        api_key=getattr(config, "api_key", None),
        endpoint=getattr(config, "endpoint", None) or tracker_kind_info(kind).default_endpoint,
        active_states=list(getattr(config, "active_states", []) or []),
        terminal_states=list(getattr(config, "terminal_states", []) or []),
        assignee=getattr(config, "assignee", None),
        http_client=http_client,
        skip_labels=list(getattr(config, "skip_labels", []) or []),
        require_any_labels=list(getattr(config, "require_any_labels", []) or []),
        title_prefixes=list(getattr(config, "title_prefixes", []) or []),
        title_prefix_match=getattr(config, "title_prefix_match", "any"),
    )


def repository_clone_url_for_tracker(config: Any) -> str | None:
    """Resolve the git clone URL for a repository-backed tracker.

    An explicit ``config.clone_url`` wins; otherwise the URL is built
    from the kind's default clone base plus ``owner``/``repo``.  Returns
    ``None`` for non-repository trackers or when owner/repo are missing.
    """
    clone_url = getattr(config, "clone_url", None)
    if clone_url:
        return clone_url

    info = tracker_kind_info(getattr(config, "kind", None))
    if not info.requires_repository or not info.default_clone_base_url:
        return None

    owner = getattr(config, "owner", None)
    repo = getattr(config, "repo", None)
    if not owner or not repo:
        return None
    return f"{info.default_clone_base_url}/{owner}/{repo}.git"
