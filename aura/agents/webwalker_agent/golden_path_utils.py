#!/usr/bin/env python3
# coding=utf-8
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

from __future__ import annotations

import json
from typing import Any


def normalize_button_label(value: str) -> str:
    return str(value or "").replace("<button>", "").replace("</button>", "").strip()


def normalize_button_sequence(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [normalize_button_label(value) for value in values if normalize_button_label(value)]


def strip_leading_root_placeholders(seq: list[str]) -> list[str]:
    """Remove leading ``root`` nodes from a normalized click path.

    List-form paths ``["root", "a", "b"]`` use this after normalization.
    Arrow-string paths are parsed by :func:`parse_golden_click_path`, which
    applies the same stripping after removing the initial virtual segment.
    """
    out = list(seq)
    while out and normalize_button_label(out[0]).lower() == "root":
        out.pop(0)
    return out


def parse_golden_click_path(path: Any) -> list[str]:
    """Parse one golden path into click targets only.

    Prefer list-form paths, e.g. ``["root", "a->b", "c"]``, because button
    text may contain ``->``. JSON-encoded lists are accepted for the same
    reason. Legacy arrow-string paths are still supported with ``" -> "`` as
    the delimiter, e.g. ``root -> a -> b``; button labels containing the
    delimiter should be provided in list/JSON-list form.

    Some tasks duplicate the placeholder (``root -> root -> a``). We drop the
    first segment (virtual page/root) and then strip any remaining leading
    ``root`` labels so the first click is always a real button target, matching
    :func:`strip_leading_root_placeholders` used for list-form paths.
    """
    if isinstance(path, list):
        nodes = normalize_button_sequence(path)
        if len(nodes) <= 1:
            return []
        return strip_leading_root_placeholders(nodes[1:])

    if not isinstance(path, str):
        return []

    stripped = path.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parse_golden_click_path(parsed)

    if " -> " not in path:
        return []

    nodes = [normalize_button_label(node) for node in path.split(" -> ") if normalize_button_label(node)]
    if len(nodes) <= 1:
        return []
    clicks = nodes[1:]
    return strip_leading_root_placeholders(clicks)


def parse_golden_click_paths(paths: Any) -> list[list[str]]:
    if not isinstance(paths, list):
        return []

    parsed_paths: list[list[str]] = []
    for path in paths:
        click_path = parse_golden_click_path(path)
        if click_path:
            parsed_paths.append(click_path)
    return parsed_paths


def extract_golden_click_paths_from_task(task: dict | None) -> list[list[str]]:
    if not isinstance(task, dict):
        return []

    task_info = task.get("info")
    raw_paths = []
    if isinstance(task_info, dict) and isinstance(task_info.get("golden_path"), list):
        raw_paths = task_info.get("golden_path") or []
    elif isinstance(task.get("golden_path"), list):
        raw_paths = task.get("golden_path") or []

    if raw_paths and isinstance(raw_paths[0], list):
        # Already parsed click-path format: align with ``root -> ...`` by stripping
        # leading root placeholders (see strip_leading_root_placeholders).
        parsed: list[list[str]] = []
        for path in raw_paths:
            if not isinstance(path, list):
                continue
            stripped = strip_leading_root_placeholders(normalize_button_sequence(path))
            if stripped:
                parsed.append(stripped)
        return parsed

    return parse_golden_click_paths(raw_paths)


def get_golden_target_candidates(golden_click_paths: list[list[str]], click_index: int) -> list[str]:
    if not isinstance(click_index, int) or click_index < 0:
        return []

    candidates: list[str] = []
    for path in golden_click_paths or []:
        if 0 <= click_index < len(path):
            candidate = normalize_button_label(path[click_index])
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def get_target_golden_node(golden_click_paths: list[list[str]], click_index: int) -> str:
    candidates = get_golden_target_candidates(golden_click_paths, click_index)
    return candidates[0] if candidates else ""


def get_target_golden_node_from_progress(
    golden_click_paths: list[list[str]],
    progress_indices: list[int] | None,
) -> str:
    if progress_indices is None:
        progress_indices = []

    candidates: list[str] = []
    for path_idx, path in enumerate(golden_click_paths or []):
        progress = progress_indices[path_idx] if path_idx < len(progress_indices) else 0
        if 0 <= progress < len(path):
            candidate = normalize_button_label(path[progress])
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates[0] if candidates else ""


def match_golden_progress(
    golden_click_paths: list[list[str]],
    clicked_button: str,
    progress_indices: list[int] | None,
) -> dict[str, Any]:
    """Match a click against the remaining suffix of any golden path.

    Unlike strict step-index matching, this allows shortcuts such as clicking
    ``b`` or ``c`` directly in a path ``a -> b -> c``. The returned
    ``advance_to`` is the next expected index for that path.
    """
    clicked_button = normalize_button_label(clicked_button)
    progress_indices = progress_indices or []
    empty = {
        "matched": False,
        "path_index": None,
        "node_index": None,
        "advance_to": None,
        "node": get_target_golden_node_from_progress(golden_click_paths, progress_indices),
        "is_current": False,
        "is_future": False,
        "is_final": False,
    }
    if not clicked_button:
        return empty

    matches: list[dict[str, Any]] = []
    for path_idx, path in enumerate(golden_click_paths or []):
        progress = progress_indices[path_idx] if path_idx < len(progress_indices) else 0
        progress = max(0, min(progress, len(path)))
        for node_idx in range(progress, len(path)):
            node = normalize_button_label(path[node_idx])
            if clicked_button != node:
                continue
            matches.append(
                {
                    "matched": True,
                    "path_index": path_idx,
                    "node_index": node_idx,
                    "advance_to": node_idx + 1,
                    "node": node,
                    "is_current": node_idx == progress,
                    "is_future": node_idx > progress,
                    "is_final": node_idx == len(path) - 1,
                }
            )

    if not matches:
        return empty

    # Prefer the least-skipping match, then the shortest resulting path.
    matches.sort(
        key=lambda item: (
            int(item["node_index"]) - (
                progress_indices[int(item["path_index"])]
                if int(item["path_index"]) < len(progress_indices)
                else 0
            ),
            int(item["node_index"]),
        )
    )
    return matches[0]


def match_golden_click(
    golden_click_paths: list[list[str]],
    clicked_button: str,
    click_index: int,
) -> tuple[bool, str]:
    clicked_button = normalize_button_label(clicked_button)
    if not clicked_button:
        return False, get_target_golden_node(golden_click_paths, click_index)

    for target in get_golden_target_candidates(golden_click_paths, click_index):
        if clicked_button == target:
            return True, target
    return False, get_target_golden_node(golden_click_paths, click_index)


def is_golden_path_end_reached(
    golden_click_paths: list[list[str]],
    clicked_button: str,
    click_index: int,
) -> bool:
    clicked_button = normalize_button_label(clicked_button)
    if not clicked_button or not isinstance(click_index, int) or click_index < 0:
        return False

    for path in golden_click_paths or []:
        if click_index == len(path) - 1 and 0 <= click_index < len(path):
            if clicked_button == normalize_button_label(path[click_index]):
                return True
    return False


def get_chain_click_horizon(golden_click_paths: list[list[str]]) -> int | None:
    lengths = [len(path) for path in golden_click_paths or [] if len(path) >= 1]
    return max(lengths) if lengths else None


def is_golden_click_path_prefix(
    golden_click_paths: list[list[str]],
    clicked_buttons: list[str],
) -> bool:
    clicked_buttons = normalize_button_sequence(clicked_buttons)
    if not clicked_buttons:
        return False

    for path in golden_click_paths or []:
        if len(clicked_buttons) <= len(path) and clicked_buttons == path[: len(clicked_buttons)]:
            return True
    return False


def is_exact_golden_click_path(
    golden_click_paths: list[list[str]],
    clicked_buttons: list[str],
) -> bool:
    clicked_buttons = normalize_button_sequence(clicked_buttons)
    if not clicked_buttons:
        return False

    for path in golden_click_paths or []:
        if clicked_buttons == normalize_button_sequence(path):
            return True
    return False
