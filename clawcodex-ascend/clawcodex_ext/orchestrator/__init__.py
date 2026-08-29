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

"""Downstream extensions for the orchestrator layer.

These patches live here per the project's decoupling mandate: behaviour that
modifies ``extensions/orchestrator/`` runtime behaviour must be implemented
in ``clawcodex_ext/`` via monkey-patch, registry, or hook — never by editing
``extensions/`` source files.

Currently houses:

- :func:`install_stale_registry_patch` — keeps the daemon's in-memory
  ``IssueRegistry`` in sync with the on-disk JSON so that operator actions
  (e.g. ``clawcodex-dev orchestrator issue retry``) written via a separate
  CLI process become visible to the running daemon without a restart.
"""

from __future__ import annotations

from ._patch_stale_registry import install_stale_registry_patch

__all__ = ["install_stale_registry_patch"]
