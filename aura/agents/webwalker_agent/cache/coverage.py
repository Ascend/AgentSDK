# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
#
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
#
"""Shared helpers for golden-path simulation against the page cache."""

from __future__ import annotations

import json
from typing import Any

from agents.webwalker_agent.cache.page_cache_store import normalize_cache_url
from agents.webwalker_agent.golden_path_utils import normalize_button_label


def iter_tasks(path: str):
    try:
        handle = open(path, encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to open task jsonl {path}: {exc}") from exc

    with handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def source_urls(task: dict[str, Any]) -> list[str]:
    info = task.get("info") if isinstance(task.get("info"), dict) else {}
    raw = info.get("source_website") or task.get("source_website")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(u).strip() for u in raw if str(u or "").strip()]


def page_status(store, url: str) -> str:
    rec = store.get_record(normalize_cache_url(url))
    if rec is None:
        return "missing"
    return str(rec.get("status") or "unknown")


def strict_usable(store, url: str) -> bool:
    """True if strict training can fetch (status ok and get() would return)."""
    key = normalize_cache_url(url)
    if not key:
        return False
    return store.get(key) is not None


def buttons_for_url(store, url: str) -> list[dict[str, str]]:
    rec = store.get_record(normalize_cache_url(url)) or {}
    try:
        raw = json.loads(rec.get("buttons_json") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("url"):
            out.append({"text": str(item.get("text") or ""), "url": str(item.get("url"))})
    return out


def find_button_url(buttons: list[dict[str, str]], target: str) -> str:
    want = normalize_button_label(target)
    for item in buttons:
        if normalize_button_label(item.get("text", "")) == want:
            return str(item.get("url") or "")
    return ""


def _walk_failure(
    page_url: str, page_status_value: str, target_button: str, fail_reason: str
) -> tuple[bool, dict[str, str]]:
    return False, {
        "page_url": page_url,
        "page_status": page_status_value,
        "target_button": target_button,
        "fail_reason": fail_reason,
    }


def simulate_golden_walk(
    store, root_url: str, click_path: list[str]
) -> tuple[bool, dict[str, str] | None]:
    """Walk golden clicks using cached buttons. Returns (ok, first_failure_info)."""
    current = normalize_cache_url(root_url) or root_url
    for target in click_path:
        status = page_status(store, current)
        if not strict_usable(store, current):
            return _walk_failure(current, status, target, "page_not_strict_ok")
        buttons = buttons_for_url(store, current)
        if not buttons:
            return _walk_failure(current, status, target, "no_buttons_on_cached_page")
        next_raw = find_button_url(buttons, target)
        if not next_raw:
            return _walk_failure(current, status, target, "golden_button_not_on_page")
        current = normalize_cache_url(next_raw) or next_raw
    status = page_status(store, current)
    if not strict_usable(store, current):
        return _walk_failure(current, status, "", "final_page_not_strict_ok")
    return True, None


def task_label(task: dict[str, Any], index: int) -> str:
    q = str(task.get("question") or "")[:100].replace("\n", " ")
    return f"#{index} {q}"
