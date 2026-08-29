#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Feature Gate — runtime feature toggle system.

Exports the singleton :class:`FeatureRegistry` and convenience helpers.

Usage::

    from clawcodex_ext.feature_gate import get_registry, FeatureFlag

    reg = get_registry()
    reg.register(FeatureFlag("my_feature", default=True))
    assert reg.is_enabled("my_feature")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ConfigStore
from .decorators import (
    feature_gated,
    feature_gated_class,
    guarded_call,
    guarded_is_enabled,
)
from .registry import FeatureRegistry
from .types import FeatureFlag

if TYPE_CHECKING:
    from .cli import run_feature_command

# ---------------------------------------------------------------------------
# Singleton — defined BEFORE .cli import so that cli.py can import get_registry
# ---------------------------------------------------------------------------

_instance: FeatureRegistry | None = None


def get_registry() -> FeatureRegistry:
    """Return the process-global :class:`FeatureRegistry` singleton.

    Lazily creates the instance on first call.
    """
    global _instance
    if _instance is None:
        _instance = FeatureRegistry()
    return _instance


def reset_registry() -> FeatureRegistry:
    """Reset the singleton and return a fresh instance.

    Intended for testing only.
    """
    global _instance
    _instance = FeatureRegistry()
    return _instance


# Lazy import of CLI handler to avoid circular imports at module scope.
# The CLI handler imports get_registry(), which is now defined above.
def __getattr__(name: str):
    if name == "run_feature_command":
        from .cli import run_feature_command as _rfc

        globals()["run_feature_command"] = _rfc
        return _rfc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Default feature registration
# ---------------------------------------------------------------------------

# Core feature flags that ship with ClawCodex.
# These mirror the CCB FEATURE_* compile-time macros at runtime.
_DEFAULT_FLAGS: list[FeatureFlag] = [
    # --- Agent / orchestrator features ---
    FeatureFlag(
        name="AGENTIC_MODE",
        default=False,
        description="Enable agentic multi-step planning mode (plugin dependent)",
        deps=[],
    ),
    FeatureFlag(
        name="ORCHESTRATOR_LOOP",
        default=False,
        description="Enable the autonomous orchestrator agent loop",
        deps=["AGENTIC_MODE"],
    ),
    FeatureFlag(
        name="HOOK_PRE_LLM",
        default=True,
        description="Enable pre-LLM agent loop hooks (plugin prerequisite)",
    ),
    FeatureFlag(
        name="HOOK_POST_LLM",
        default=True,
        description="Enable post-LLM agent loop hooks",
    ),
    # --- Plugin system ---
    FeatureFlag(
        name="PLUGIN_SYSTEM",
        default=False,
        description="Enable the plugin loading system",
    ),
    # --- Tool gap bridging ---
    FeatureFlag(
        name="TOOL_GAP_BRIDGE",
        default=False,
        description="Enable automatic tool-gap bridging for missing capabilities",
        deps=["PLUGIN_SYSTEM"],
    ),
    # --- Multi-API provider ---
    FeatureFlag(
        name="MULTI_API_PROVIDER",
        default=False,
        description="Enable multi-API provider routing",
    ),
    FeatureFlag(
        name="MULTIMODEL",
        default=False,
        description="Enable multi-model parallel dispatch and model groups",
    ),
    # --- Sandbox ---
    FeatureFlag(
        name="SANDBOX_EXECUTION",
        default=False,
        description="Enable sandboxed command execution",
    ),
    # --- Voice mode ---
    FeatureFlag(
        name="VOICE_MODE",
        default=False,
        description="Enable voice input/output mode",
    ),
    # --- Budget mode ---
    FeatureFlag(
        name="BUDGET_MODE",
        default=False,
        description="Enable cost-aware budget-limited agent runs",
    ),
    # --- Goal mode ---
    FeatureFlag(
        name="goals",
        default=True,
        description="Enable upstream-compatible /goal mode",
    ),
    # --- ACP protocol ---
    FeatureFlag(
        name="ACP_PROTOCOL",
        default=False,
        description="Enable ACP (Agent Communication Protocol) for inter-agent messaging",
    ),
    # --- Native modules ---
    FeatureFlag(
        name="NATIVE_MODULES",
        default=False,
        description="Enable loading of compiled native extension modules",
    ),
    # --- Remote control ---
    FeatureFlag(
        name="REMOTE_CONTROL",
        default=False,
        description="Enable remote CLI control via TCP/WebSocket",
    ),
    # --- Daemon subsystem ---
    # Defaults to False; enabling it registers the ``clawcodex-dev daemon``
    # CLI surface and allows supervisor + worker processes.
    FeatureFlag(
        name="DAEMON",
        default=False,
        description=("Enable the long-running daemon supervisor that owns task_server / cron workers"),
    ),
    FeatureFlag(
        name="BRIDGE_MODE",
        default=False,
        description=("Enable the multi-session bridge subsystem"),
    ),
    # --- CI/CD ---
    FeatureFlag(
        name="CICD_MODE",
        default=False,
        description="Enable CI/CD-optimized mode (batch processing, no TTY)",
    ),
    FeatureFlag(
        name="ULTRAPLAN_LLM_PLANNER",
        default=True,
        description="Enable LLM-backed /ultraplan plan generation",
    ),
    FeatureFlag(
        name="ULTRAPLAN_REMOTE",
        default=False,
        description="Enable /ultraplan remote CCR execution",
    ),
    FeatureFlag(
        name="ULTRAPLAN_RAINBOW",
        default=True,
        description="Enable /ultraplan trigger highlighting in prompt input",
    ),
    FeatureFlag(
        name="KAIROS",
        default=False,
        description="Enable Kairos tick scheduling primitives",
    ),
    FeatureFlag(
        name="PROACTIVE",
        default=False,
        description="Enable proactive tick-driven autonomous mode",
    ),
    FeatureFlag(
        name="MONITOR_TOOL",
        default=False,
        description="Enable the Monitor tool and /monitor slash command",
    ),
    FeatureFlag(
        name="LKB_PLAN_GRAPH",
        default=False,
        description=(
            "Master switch for Logical Kanban (LKB): enables the Task V2 "
            "tool surface (including headless sessions) backed by the "
            "project-level persistent Plan Graph (Store authority, "
            "board.json, atomic Claim, revision vectors)."
        ),
    ),
    # --- Skill search ---
    FeatureFlag(
        name="SKILL_SEARCH_TFIDF",
        default=False,
        description="Enable TF-IDF based skill search for smart skill ranking",
    ),
]


def register_defaults(registry: FeatureRegistry | None = None) -> None:
    """Register all built-in default feature flags.

    Safe to call multiple times — duplicate registrations are no-ops.
    """
    reg = registry or get_registry()
    for flag in _DEFAULT_FLAGS:
        if flag.name not in reg._features:
            reg.register(flag)


# ---------------------------------------------------------------------------
# Eager bootstrap: register defaults so the singleton is ready on first use.
# ---------------------------------------------------------------------------

register_defaults()


# Re-export everything at package level for convenience.
__all__ = [
    "ConfigStore",
    "FeatureFlag",
    "FeatureRegistry",
    "feature_gated",
    "feature_gated_class",
    "guarded_call",
    "guarded_is_enabled",
    "get_registry",
    "register_defaults",
    "reset_registry",
    "run_feature_command",
]
