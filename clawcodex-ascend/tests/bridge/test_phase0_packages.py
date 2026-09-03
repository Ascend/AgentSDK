#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Phase 0 smoke tests for the four CCR packages and their backwards-compat surface."""

from __future__ import annotations


def test_subsystem_packages_preserve_legacy_metadata_for_porting_workspace() -> None:
    """WI-0.4 must preserve ARCHIVE_NAME etc. for tests/test_porting_workspace.py:73-79."""
    from src import bridge, remote, server, upstreamproxy

    for pkg in (bridge, remote, server, upstreamproxy):
        assert pkg.MODULE_COUNT > 0, f"{pkg.__name__}.MODULE_COUNT is 0/missing"
        assert pkg.ARCHIVE_NAME, f"{pkg.__name__}.ARCHIVE_NAME is empty"
        assert pkg.SAMPLE_FILES, f"{pkg.__name__}.SAMPLE_FILES is empty"
        assert pkg.PORTING_NOTE, f"{pkg.__name__}.PORTING_NOTE is empty"


def test_legacy_remote_runtime_emits_deprecation_warning() -> None:
    """WI-0.3 + ch01 round-2 P3: importing scripts.audit.remote_runtime
    (formerly src.remote_runtime) fires a DeprecationWarning.
    """
    import importlib
    from pathlib import Path
    import sys
    import warnings

    import pytest

    if not Path("scripts/audit/remote_runtime.py").is_file():
        pytest.skip("legacy audit scripts are outside the runtime migration scope")

    sys.modules.pop("scripts.audit.remote_runtime", None)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        importlib.import_module("scripts.audit.remote_runtime")

    assert any(
        issubclass(w.category, DeprecationWarning) and "scripts.audit.remote_runtime is a placeholder" in str(w.message)
        for w in captured
    ), "expected DeprecationWarning from scripts.audit.remote_runtime import"
