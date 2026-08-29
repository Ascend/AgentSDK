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

"""Permissions package — core + downstream extensions.

Re-exports the migrated :mod:`src.permissions` symbols alongside the
downstream extensions (:class:`ClassificationCache`, danger detector,
``install_permission_extensions``) that lived here before.
"""

from __future__ import annotations

# --- Migrated from src/permissions/__init__.py ---
from .bash_security import (
    CROSS_PLATFORM_CODE_EXEC,
    DANGEROUS_BASH_PATTERNS,
    is_dangerous_bash_permission,
)
from .check import (
    auto_mode_classify,
    check_rule_based_permissions,
    create_permission_request_message,
    has_permissions_to_use_tool,
    has_permissions_to_use_tool_inner,
)

# --- Downstream extensions (was the original content of this __init__) ---
from .classifier import (
    ClassificationCache,
    LLMClassificationResult,
    auto_mode_classify_with_llm,
    get_cache,
    llm_classify_tool_call,
)
from .cycle import (
    can_cycle_to_auto,
    cycle_permission_mode,
    get_auto_mode_availability_reason,
    get_next_permission_mode,
)
from .danger_detector import detect_dangerous_tool_call
from .filesystem import (
    DANGEROUS_DIRECTORIES,
    DANGEROUS_FILES,
    check_path_safety_for_auto_edit,
    check_read_permission_for_path,
    check_write_permission_for_path,
    normalize_case_for_comparison,
)
from .handler import handle_permission_ask
from .loader import apply_rules_to_context, settings_to_rules
from .modes import (
    is_default_mode,
    is_external_permission_mode,
    permission_mode_from_string,
    permission_mode_short_title,
    permission_mode_symbol,
    permission_mode_title,
    to_external_permission_mode,
)
from .rule_parser import (
    escape_rule_content,
    permission_rule_value_from_string,
    permission_rule_value_to_string,
    unescape_rule_content,
)
from .rules import (
    filter_denied_agents,
    get_allow_rules,
    get_ask_rule_for_tool,
    get_ask_rules,
    get_deny_rule_for_tool,
    get_deny_rules,
    get_rule_by_contents_for_tool,
    tool_always_allowed_rule,
)
from .types import (
    EXTERNAL_PERMISSION_MODES,
    PERMISSION_MODES,
    PERMISSION_RULE_SOURCES,
    AdditionalWorkingDirectory,
    AsyncAgentDecisionReason,
    ClassifierDecisionReason,
    ExternalPermissionMode,
    HookDecisionReason,
    ModeDecisionReason,
    OtherDecisionReason,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionAskHandler,
    PermissionAskReply,
    PermissionAskRequest,
    PermissionBehavior,
    PermissionDecision,
    PermissionDecisionReason,
    PermissionDenyDecision,
    PermissionMode,
    PermissionPassthroughResult,
    PermissionPromptToolDecisionReason,
    PermissionResult,
    PermissionRule,
    PermissionRuleSource,
    PermissionRuleValue,
    PermissionUpdate,
    PermissionUpdateAddDirectories,
    PermissionUpdateAddRules,
    PermissionUpdateDestination,
    PermissionUpdateRemoveDirectories,
    PermissionUpdateRemoveRules,
    PermissionUpdateReplaceRules,
    PermissionUpdateSetMode,
    RuleDecisionReason,
    SafetyCheckDecisionReason,
    SandboxOverrideDecisionReason,
    SubcommandResultsDecisionReason,
    ToolPermissionContext,
    ToolPermissionRulesBySource,
    WorkingDirDecisionReason,
)
from .updates import (
    PERSISTABLE_DESTINATIONS,
    apply_permission_update,
    apply_permission_updates,
    create_read_rule_suggestion,
    extract_rules,
    has_rules,
    persist_permission_update,
    persist_permission_updates,
    supports_persistence,
)


def install_permission_extensions() -> None:
    """Register downstream permission cycle steps and auto mode classifier.

    Idempotent — safe to call more than once.
    """
    from .cycle import register_cycle_step

    register_cycle_step("bypassPermissions", "dontAsk", after="bypassPermissions")

    try:
        from clawcodex_ext.agent.auto_mode_runner import install_llm_auto_mode_classifier

        success = install_llm_auto_mode_classifier(use_llm_for_uncertain=True)
        if not success:
            import logging

            log = logging.getLogger(__name__)
            log.warning("Auto mode LLM classifier installation returned False")
    except Exception as e:
        import logging
        import traceback

        log = logging.getLogger(__name__)
        log.warning("Auto mode LLM classifier not installed: %s", e)
        log.warning("Traceback: %s", traceback.format_exc())


__all__ = [
    # Constants
    "CROSS_PLATFORM_CODE_EXEC",
    "DANGEROUS_BASH_PATTERNS",
    "DANGEROUS_DIRECTORIES",
    "DANGEROUS_FILES",
    "EXTERNAL_PERMISSION_MODES",
    "PERMISSION_MODES",
    "PERMISSION_RULE_SOURCES",
    "PERSISTABLE_DESTINATIONS",
    # Types
    "AdditionalWorkingDirectory",
    "AsyncAgentDecisionReason",
    "ClassifierDecisionReason",
    "ExternalPermissionMode",
    "HookDecisionReason",
    "ModeDecisionReason",
    "OtherDecisionReason",
    "PermissionAllowDecision",
    "PermissionAskDecision",
    "PermissionAskHandler",
    "PermissionAskReply",
    "PermissionAskRequest",
    "PermissionBehavior",
    "PermissionDecision",
    "PermissionDecisionReason",
    "PermissionDenyDecision",
    "PermissionMode",
    "PermissionPassthroughResult",
    "PermissionPromptToolDecisionReason",
    "PermissionResult",
    "PermissionRule",
    "PermissionRuleSource",
    "PermissionRuleValue",
    "PermissionUpdate",
    "PermissionUpdateAddDirectories",
    "PermissionUpdateAddRules",
    "PermissionUpdateDestination",
    "PermissionUpdateRemoveDirectories",
    "PermissionUpdateRemoveRules",
    "PermissionUpdateReplaceRules",
    "PermissionUpdateSetMode",
    "RuleDecisionReason",
    "SafetyCheckDecisionReason",
    "SandboxOverrideDecisionReason",
    "SubcommandResultsDecisionReason",
    "ToolPermissionContext",
    "ToolPermissionRulesBySource",
    "WorkingDirDecisionReason",
    # Functions
    "apply_permission_update",
    "apply_permission_updates",
    "apply_rules_to_context",
    "auto_mode_classify",
    "check_path_safety_for_auto_edit",
    "check_read_permission_for_path",
    "check_rule_based_permissions",
    "check_write_permission_for_path",
    "create_permission_request_message",
    "create_read_rule_suggestion",
    "cycle_permission_mode",
    "escape_rule_content",
    "extract_rules",
    "filter_denied_agents",
    "get_allow_rules",
    "get_ask_rule_for_tool",
    "get_ask_rules",
    "get_deny_rule_for_tool",
    "get_deny_rules",
    "get_next_permission_mode",
    "get_rule_by_contents_for_tool",
    "handle_permission_ask",
    "has_permissions_to_use_tool",
    "has_permissions_to_use_tool_inner",
    "has_rules",
    "is_dangerous_bash_permission",
    "is_default_mode",
    "is_external_permission_mode",
    "normalize_case_for_comparison",
    "permission_mode_from_string",
    "permission_mode_short_title",
    "permission_mode_symbol",
    "permission_mode_title",
    "permission_rule_value_from_string",
    "permission_rule_value_to_string",
    "persist_permission_update",
    "persist_permission_updates",
    "settings_to_rules",
    "supports_persistence",
    "to_external_permission_mode",
    "tool_always_allowed_rule",
    "unescape_rule_content",
    # Downstream extensions
    "install_permission_extensions",
    "ClassificationCache",
    "LLMClassificationResult",
    "auto_mode_classify_with_llm",
    "get_cache",
    "llm_classify_tool_call",
    "detect_dangerous_tool_call",
    "can_cycle_to_auto",
    "get_auto_mode_availability_reason",
]
