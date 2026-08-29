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
# Backward-compatibility stub — re-exports from core/bundle_manifest.py
from extensions.sop_converter.core.bundle_manifest import (
    logger,
    BUNDLE_MANIFEST_NAME,
    BundleManifest,
    manifest_path_for_bundle,
    write_bundle_manifest,
    read_bundle_manifest,
    resolve_sdk_source_dir,
)

__all__ = [
    "logger",
    "BUNDLE_MANIFEST_NAME",
    "BundleManifest",
    "manifest_path_for_bundle",
    "write_bundle_manifest",
    "read_bundle_manifest",
    "resolve_sdk_source_dir",
]

from extensions.sop_converter.core import bundle_manifest as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value, _impl
