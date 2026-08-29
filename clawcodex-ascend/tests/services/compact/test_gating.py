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

"""Unit tests for the lazy compression pipeline gate.

Tests are intentionally small and deterministic; they cover the four
decision branches (forced_source, forced_transition, forced_prev_error,
threshold) plus the env-var override path. The pipeline-side integration
is exercised separately by ``test_compression_gate_integration.py``.
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import os
import unittest

from clawcodex_ext.services.compact.gating import (
    DEFAULT_COMPRESSION_GATE_SKIP_RATIO,
    ENV_COMPRESSION_GATE_SKIP_RATIO,
    resolve_skip_ratio_from_env,
    should_run_compression_pipeline,
)


class TestShouldRunCompressionPipelineThreshold(unittest.TestCase):
    """Default-mode threshold behaviour."""

    def test_below_threshold_returns_false(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=100_000,
            context_window=200_000,
            skip_ratio=0.6,
        )
        self.assertFalse(should_run)
        self.assertEqual(reason, "below_threshold")

    def test_above_threshold_returns_true(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=130_000,
            context_window=200_000,
            skip_ratio=0.6,
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "above_threshold")

    def test_default_skip_ratio_matches_module_constant(self) -> None:
        # 60% of 200_000 = 120_000 threshold
        should_run, _ = should_run_compression_pipeline(
            est_input_tokens=119_999,
            context_window=200_000,
        )
        self.assertFalse(should_run)
        should_run, _ = should_run_compression_pipeline(
            est_input_tokens=120_000,
            context_window=200_000,
        )
        self.assertTrue(should_run)
        self.assertEqual(DEFAULT_COMPRESSION_GATE_SKIP_RATIO, 0.6)

    def test_zero_tokens_returns_below_threshold(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=0,
            context_window=200_000,
        )
        self.assertFalse(should_run)
        self.assertEqual(reason, "below_threshold")

    def test_negative_tokens_clamped_to_zero(self) -> None:
        should_run, _ = should_run_compression_pipeline(
            est_input_tokens=-10,
            context_window=200_000,
        )
        self.assertFalse(should_run)


class TestShouldRunCompressionPipelineForced(unittest.TestCase):
    """Forced-run overrides take precedence over the threshold."""

    def test_query_source_compact_forces_run(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=10,
            context_window=200_000,
            query_source="compact",
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "forced_source")

    def test_query_source_session_memory_forces_run(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=10,
            context_window=200_000,
            query_source="session_memory",
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "forced_source")

    def test_transition_reactive_compact_retry_forces_run(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=10,
            context_window=200_000,
            transition_reason="reactive_compact_retry",
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "forced_transition")

    def test_transition_collapse_drain_retry_forces_run(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=10,
            context_window=200_000,
            transition_reason="collapse_drain_retry",
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "forced_transition")

    def test_previous_pipeline_errored_forces_run(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=10,
            context_window=200_000,
            previous_pipeline_errored=True,
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "forced_prev_error")

    def test_prev_error_wins_over_query_source(self) -> None:
        # Documented precedence: prev_error is checked first; if the
        # previous pipeline errored, we always retry, regardless of
        # source.
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=10,
            context_window=200_000,
            query_source="compact",
            previous_pipeline_errored=True,
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "forced_prev_error")


class TestShouldRunCompressionPipelineDefensive(unittest.TestCase):
    """Defensive branches that disable the gate."""

    def test_zero_skip_ratio_never_skips(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=0,
            context_window=200_000,
            skip_ratio=0.0,
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "above_threshold")

    def test_negative_skip_ratio_treated_as_never_skip(self) -> None:
        should_run, _ = should_run_compression_pipeline(
            est_input_tokens=0,
            context_window=200_000,
            skip_ratio=-0.5,
        )
        self.assertTrue(should_run)

    def test_zero_context_window_never_skips(self) -> None:
        should_run, reason = should_run_compression_pipeline(
            est_input_tokens=0,
            context_window=0,
        )
        self.assertTrue(should_run)
        self.assertEqual(reason, "above_threshold")

    def test_negative_context_window_never_skips(self) -> None:
        should_run, _ = should_run_compression_pipeline(
            est_input_tokens=0,
            context_window=-1,
        )
        self.assertTrue(should_run)


class TestResolveSkipRatioFromEnv(unittest.TestCase):
    """Env-var override and fallback semantics."""

    def setUp(self) -> None:
        self._saved = os.environ.pop(ENV_COMPRESSION_GATE_SKIP_RATIO, None)

    def tearDown(self) -> None:
        os.environ.pop(ENV_COMPRESSION_GATE_SKIP_RATIO, None)
        if self._saved is not None:
            os.environ[ENV_COMPRESSION_GATE_SKIP_RATIO] = self._saved

    def test_default_when_unset_and_no_configured(self) -> None:
        self.assertEqual(
            resolve_skip_ratio_from_env(None),
            DEFAULT_COMPRESSION_GATE_SKIP_RATIO,
        )

    def test_configured_used_when_env_unset(self) -> None:
        self.assertEqual(resolve_skip_ratio_from_env(0.4), 0.4)

    def test_env_wins_over_configured(self) -> None:
        os.environ[ENV_COMPRESSION_GATE_SKIP_RATIO] = "0.25"
        self.assertEqual(resolve_skip_ratio_from_env(0.4), 0.25)

    def test_invalid_env_falls_back_to_configured(self) -> None:
        os.environ[ENV_COMPRESSION_GATE_SKIP_RATIO] = "not-a-float"
        self.assertEqual(resolve_skip_ratio_from_env(0.4), 0.4)

    def test_out_of_range_env_falls_back_to_configured(self) -> None:
        # Values outside (0, 1) are rejected to avoid silently
        # disabling the gate.
        os.environ[ENV_COMPRESSION_GATE_SKIP_RATIO] = "0"
        self.assertEqual(resolve_skip_ratio_from_env(0.4), 0.4)
        os.environ[ENV_COMPRESSION_GATE_SKIP_RATIO] = "1.5"
        self.assertEqual(resolve_skip_ratio_from_env(0.4), 0.4)
        os.environ[ENV_COMPRESSION_GATE_SKIP_RATIO] = "-0.1"
        self.assertEqual(resolve_skip_ratio_from_env(0.4), 0.4)

    def test_invalid_configured_falls_back_to_default(self) -> None:
        # Non-numeric string can't be converted via float(); falls back.
        self.assertEqual(
            resolve_skip_ratio_from_env("not-a-number"),  # type: ignore[arg-type]
            DEFAULT_COMPRESSION_GATE_SKIP_RATIO,
        )


if __name__ == "__main__":
    unittest.main()
