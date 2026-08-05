#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

import threading

from src.services.computer_use import DryRunRecorder, InputAction


def test_recorder_starts_empty() -> None:
    recorder = DryRunRecorder()
    assert len(recorder) == 0
    assert recorder.action_count == 0
    assert recorder.actions() == []
    assert recorder.screenshots() == []


def test_recorder_captures_actions() -> None:
    recorder = DryRunRecorder()
    recorder.record_action("click", button="left", x=10, y=20)
    recorder.record_action("type_text", text="hello")

    actions = recorder.actions()
    assert len(actions) == 2
    assert actions[0].kind == "click"
    assert actions[0].args == {"button": "left", "x": 10, "y": 20}
    assert actions[1].kind == "type_text"
    assert actions[1].args == {"text": "hello"}


def test_recorder_filter_by_kind() -> None:
    recorder = DryRunRecorder()
    recorder.record_action("click", x=1, y=2)
    recorder.record_action("move_mouse", x=3, y=4)
    recorder.record_action("click", x=5, y=6)

    clicks = list(recorder.filter("click"))
    assert len(clicks) == 2
    assert all(isinstance(a, InputAction) for a in clicks)


def test_recorder_clear() -> None:
    recorder = DryRunRecorder()
    recorder.record_action("click", x=1, y=2)
    recorder.clear()
    assert recorder.action_count == 0
    assert recorder.screenshots() == []


def test_recorder_screenshots_are_stored() -> None:
    recorder = DryRunRecorder()
    recorder.record_screenshot("fullscreen", b"\x89PNG", extra="meta")
    shots = recorder.screenshots()
    assert len(shots) == 1
    kind, payload, meta = shots[0]
    assert kind == "fullscreen"
    assert payload == b"\x89PNG"
    assert meta == {"extra": "meta"}


def test_recorder_is_thread_safe() -> None:
    recorder = DryRunRecorder()

    def worker(i: int) -> None:
        for _ in range(50):
            recorder.record_action("click", x=i, y=i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert recorder.action_count == 8 * 50
