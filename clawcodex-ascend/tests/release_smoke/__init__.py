#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Pre-publish wheel sanity checks.

NOT in CI — run locally before pushing a release tag:

    python -m build
    python -m pytest tests/release_smoke/ -v

The CI release-preflight workflow already runs ``twine check`` (which
validates metadata schema) and a post-install CLI smoke. This directory
adds belt-and-suspenders assertions for the wheel artifact itself:
entry_points presence, ``Requires-Python`` matches pyproject, and
``RELEASE_TAG`` correctly freezes ``__version__`` in the installed
wheel.

Excluded from default pytest collection by NOT being listed in
``pyproject.toml [tool.pytest.ini_options] testpaths`` (default is
``["tests"]`` but pytest skips paths without a configured marker).
The intended invocation is the explicit ``pytest tests/release_smoke/``
shown above.
"""
