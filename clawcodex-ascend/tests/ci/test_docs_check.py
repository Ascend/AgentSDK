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

from __future__ import annotations

import importlib


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend("scripts/ci")
    return importlib.import_module("docs_check")


def test_candidate_paths_skip_raw_snapshot_docs(tmp_path, monkeypatch):
    docs_check = _load_module(monkeypatch)
    monkeypatch.setattr(docs_check, "ROOT", tmp_path)

    raw_doc = tmp_path / "docs" / "i18n.raw" / "README_ZH.md"
    raw_doc.parent.mkdir(parents=True)
    raw_doc.write_text("[missing](../missing.md)\n", encoding="utf-8")
    curated_doc = tmp_path / "docs" / "README.md"
    curated_doc.write_text("# Docs\n", encoding="utf-8")

    paths = docs_check._candidate_paths(
        ["docs/i18n.raw/README_ZH.md", "docs/README.md"],
        all_docs=False,
    )

    assert paths == [curated_doc]
