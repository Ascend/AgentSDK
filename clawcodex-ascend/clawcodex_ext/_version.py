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

"""Dynamic calendar-version with release-tag freeze.

When the environment variable ``RELEASE_TAG`` is set (e.g. ``v2026.6.24``),
the version is frozen to that tag (with the leading ``v`` stripped).
This ensures a published wheel's version matches its git tag exactly,
even when rebuilt later from the same commit.

Usage
-----
    from clawcodex_ext._version import __version__

Runtime behaviour
    - Dev: date-based CalVer ``YYYY.M.D`` (e.g. ``2026.6.24``)
    - Release (``$RELEASE_TAG=v2026.6.24``): tag-based ``2026.6.24``

Package metadata (``pyproject.toml``) reads this attribute via
``tool.setuptools.dynamic.version.attr`` so that ``pip install`` and
``python -m build`` always embed the correct version.
"""

import os
from datetime import date


def _version() -> str:
    """Return the effective version string.

    Priority
    --------
    1. ``$RELEASE_TAG`` env var (e.g. ``v2026.6.24`` -> ``2026.6.24``)
       -- used by CI workflows for tagged releases.
    2. Today's date as CalVer ``YYYY.M.D`` -- used for development builds.
    """
    release_tag = os.environ.get("RELEASE_TAG", "")
    if release_tag:
        return release_tag.removeprefix("v")
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


__version__ = _version()
__version_info__ = tuple(int(x) for x in __version__.split("."))
