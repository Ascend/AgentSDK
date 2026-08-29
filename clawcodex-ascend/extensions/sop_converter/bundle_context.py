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

# pylint: disable=undefined-loop-variable
# Backward-compatibility stub — re-exports from runtime/bundle_context.py
from extensions.sop_converter.runtime.bundle_context import (
    logger,
    SOP_CONVERTER_SPEC_SOURCES,
    SOP_BUNDLE_SPEC_SOURCES,
    is_sop_converter_spec_source,
    BundleContext,
    set_active_bundle,
    get_active_bundle,
    build_bundle_context,
    apply_sdk_source_working_directory,
    collect_tool_names_from_skills,
    collect_tool_names_from_bundle_specs,
    is_pos_converter_tool,
    filter_tools_for_bundle,
    load_bundle_persisted_tools,
    ensure_bundle_tools_registered,
    prune_registry_to_bundle,
    activate_bundle_isolation,
    load_bundle_macro_routes,
)

__all__ = [
    "logger",
    "SOP_CONVERTER_SPEC_SOURCES",
    "SOP_BUNDLE_SPEC_SOURCES",
    "is_sop_converter_spec_source",
    "BundleContext",
    "set_active_bundle",
    "get_active_bundle",
    "build_bundle_context",
    "apply_sdk_source_working_directory",
    "collect_tool_names_from_skills",
    "collect_tool_names_from_bundle_specs",
    "is_pos_converter_tool",
    "filter_tools_for_bundle",
    "load_bundle_persisted_tools",
    "ensure_bundle_tools_registered",
    "prune_registry_to_bundle",
    "activate_bundle_isolation",
    "load_bundle_macro_routes",
]

from extensions.sop_converter.runtime import bundle_context as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value, _impl
