#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------
"""Isolate ClawCodex home for sop_converter tests.

A corrupt/stale ``~/.clawcodex/sop-resources/**/catalog.json`` on the host
must not poison cross-layer catalog resolution during unit tests.

Env var names are inlined (not imported) so this conftest can load even when
optional sibling packages under ``extensions/`` are not yet present.
"""

from __future__ import annotations

import pytest

HOME_ROOT_ENV = "CLAWCODEX_HOME"
HOME_ONLY_ENV = "CLAWCODEX_CATALOG_HOME_ONLY"


@pytest.fixture(autouse=True)
def _isolate_clawcodex_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path_factory.mktemp("clawcodex-home")
    monkeypatch.setenv(HOME_ROOT_ENV, str(home))
    monkeypatch.delenv(HOME_ONLY_ENV, raising=False)
    yield home
