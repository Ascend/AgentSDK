#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

from clawcodex_ext.intent_forecast.learning import (
    build_feedback_features,
    feedback_feature_similarity,
    feedback_weight,
    read_recent_feedback,
    record_feedback,
)
from clawcodex_ext.intent_forecast.context import ForecastContext
from clawcodex_ext.intent_forecast.messages import ForecastSuggestion


def test_feedback_roundtrip(tmp_path) -> None:
    suggestion = ForecastSuggestion(id="s1", title="Do it", prompt="do it", confidence=0.7)
    record_feedback(
        "accepted_completed",
        suggestion=suggestion,
        cwd=tmp_path,
        fingerprint="fp",
        features={"suggestion_kind": "continue_impl"},
        base_dir=tmp_path,
    )
    rows = read_recent_feedback(base_dir=tmp_path)
    assert rows[-1]["event"] == "accepted_completed"
    assert rows[-1]["suggestion_id"] == "s1"
    assert rows[-1]["features"]["suggestion_kind"] == "continue_impl"


def test_feedback_features_and_similarity() -> None:
    suggestion = ForecastSuggestion(id="s1", title="Run focused tests", prompt="pytest tests/intent_forecast")
    context = ForecastContext(
        cwd="repo",
        workspace={
            "git_status": " M file.py",
            "changed_files": ["clawcodex_ext/intent_forecast/service.py"],
        },
        task_state={"blocked_reason": ""},
        intent_stage="test",
        response_language="English",
    )

    features = build_feedback_features(suggestion=suggestion, context=context, trigger="auto")

    assert features["suggestion_kind"] == "run_tests"
    assert features["stage"] == "test"
    assert features["changed_file_globs"] == ["clawcodex_ext/intent_forecast/*"]
    assert feedback_feature_similarity(features, dict(features)) == 1.0


def test_feedback_weight_uses_feature_similarity_without_title_match(tmp_path) -> None:
    accepted = ForecastSuggestion(id="s1", title="Run focused tests", prompt="pytest tests/intent_forecast")
    features = {
        "stage": "test",
        "suggestion_kind": "run_tests",
        "changed_file_globs": ["clawcodex_ext/intent_forecast/*"],
        "has_dirty_worktree": True,
        "had_recent_failure": False,
        "language": "English",
        "trigger": "auto",
    }
    record_feedback(
        "accepted_completed",
        suggestion=accepted,
        cwd=tmp_path,
        fingerprint="old",
        features=features,
        base_dir=tmp_path,
    )

    weight = feedback_weight(
        "Verify with pytest",
        cwd=tmp_path,
        fingerprint="new",
        features=dict(features),
        base_dir=tmp_path,
    )

    assert weight > 0
