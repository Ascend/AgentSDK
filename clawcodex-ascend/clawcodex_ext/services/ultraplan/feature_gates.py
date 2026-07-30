"""Feature-gate helpers for F-87 ultraplan surfaces."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from clawcodex_ext.feature_gate import FeatureFlag, get_registry  # pylint: disable=no-name-in-module

ULTRAPLAN_LLM_PLANNER = "ULTRAPLAN_LLM_PLANNER"
ULTRAPLAN_REMOTE = "ULTRAPLAN_REMOTE"
ULTRAPLAN_RAINBOW = "ULTRAPLAN_RAINBOW"


def register_ultraplan_feature_flags() -> None:
    registry = get_registry()
    flags = (
        FeatureFlag(
            name=ULTRAPLAN_LLM_PLANNER,
            default=True,
            description="Enable LLM-backed /ultraplan plan generation.",
        ),
        FeatureFlag(
            name=ULTRAPLAN_REMOTE,
            default=False,
            description="Enable /ultraplan remote CCR execution.",
        ),
        FeatureFlag(
            name=ULTRAPLAN_RAINBOW,
            default=True,
            description="Enable /ultraplan trigger highlighting in prompt input.",
        ),
    )
    for flag in flags:
        if flag.name not in registry.list_features():
            registry.register(flag)


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def is_ultraplan_llm_enabled() -> bool:
    register_ultraplan_feature_flags()
    legacy = _env_bool("ULTRAPLAN_LLM_PLANNER")
    if legacy is not None:
        return legacy
    return get_registry().is_enabled(ULTRAPLAN_LLM_PLANNER)


def is_ultraplan_remote_enabled() -> bool:
    register_ultraplan_feature_flags()
    legacy = _env_bool("ULTRAPLAN_REMOTE")
    if legacy is not None:
        return legacy
    return get_registry().is_enabled(ULTRAPLAN_REMOTE)


def is_ultraplan_rainbow_enabled() -> bool:
    register_ultraplan_feature_flags()
    legacy = _env_bool("ULTRAPLAN_RAINBOW")
    if legacy is not None:
        return legacy
    return get_registry().is_enabled(ULTRAPLAN_RAINBOW)


def is_ccr_endpoint_allowed(endpoint: str) -> bool:
    allowlist = os.environ.get("CCR_ALLOWLIST", "").strip()
    if not allowlist:
        return True
    parsed = urlparse(endpoint)
    host = parsed.netloc or parsed.path
    allowed = {item.strip() for item in allowlist.split(",") if item.strip()}
    return endpoint in allowed or host in allowed


register_ultraplan_feature_flags()
