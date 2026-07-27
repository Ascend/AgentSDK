#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

        http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest


def _ensure_aura_src_on_path():
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "agents").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)


def test_iter_tasks_reports_file_and_json_errors():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.coverage import iter_tasks

    tmpdir = Path.cwd() / f"webwalker-coverage-test-{os.getpid()}"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        missing = tmpdir / "missing.jsonl"
        with pytest.raises(ValueError, match="Failed to open task jsonl"):
            list(iter_tasks(str(missing)))

        invalid = tmpdir / "invalid.jsonl"
        invalid.write_text("{bad json\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"Invalid JSON in .*invalid\.jsonl:1"):
            list(iter_tasks(str(invalid)))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_simulate_golden_walk_checks_final_page_strict_usable():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.coverage import simulate_golden_walk
    from agents.webwalker_agent.cache.page_cache_store import normalize_cache_url

    root = "https://example.com"
    final = "https://example.com/final"
    records = {
        normalize_cache_url(root): {
            "status": "ok",
            "buttons_json": json.dumps([{"text": "Next", "url": final}]),
        },
        normalize_cache_url(final): {
            "status": "failed",
            "buttons_json": "[]",
        },
    }

    class Store:
        def get_record(self, url):
            return records.get(normalize_cache_url(url))

        def get(self, url):
            rec = self.get_record(url)
            if rec is not None and rec.get("status") == "ok":
                return "", "markdown"
            return None

    ok, failure = simulate_golden_walk(Store(), root, ["Next"])

    assert ok is False
    assert failure == {
        "page_url": normalize_cache_url(final),
        "page_status": "failed",
        "target_button": "",
        "fail_reason": "final_page_not_strict_ok",
    }
