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

from __future__ import annotations

import json
import unittest

from extensions.orchestrator.intent import (
    DEFAULT_INTENT_LABELS,
    Command,
    Intent,
    command_to_intent,
    intent_from_label_set,
    merge_intents,
    merge_intents_with_cli,
    parse_agent_command,
)


class TestIntentEnum(unittest.TestCase):
    def test_intent_values(self) -> None:
        self.assertEqual(Intent.NONE.value, "none")
        self.assertEqual(Intent.RETRY.value, "retry")
        self.assertEqual(Intent.FOLLOWUP.value, "followup")
        self.assertEqual(Intent.BLOCKED.value, "blocked")
        self.assertEqual(Intent.REBASE.value, "rebase")

    def test_command_rebase_value(self) -> None:
        self.assertEqual(Command.REBASE.value, "rebase")

    def test_intent_is_str_enum(self) -> None:
        # Must be JSON-serializable as its string value (registry.json).
        for intent in Intent:
            self.assertEqual(json.dumps(intent.value), json.dumps(intent.value))

    def test_intent_lookup_by_value(self) -> None:
        self.assertIs(Intent("retry"), Intent.RETRY)
        self.assertIs(Intent("none"), Intent.NONE)
        self.assertIs(Intent("rebase"), Intent.REBASE)
        self.assertIs(Command("rebase"), Command.REBASE)


class TestIntentFromLabelSet(unittest.TestCase):
    def test_empty_labels_returns_none(self) -> None:
        self.assertIs(intent_from_label_set([]), Intent.NONE)
        self.assertIs(intent_from_label_set(None), Intent.NONE)

    def test_default_retry_label(self) -> None:
        self.assertIs(
            intent_from_label_set(["agent:retry"]),
            Intent.RETRY,
        )

    def test_default_followup_label(self) -> None:
        self.assertIs(
            intent_from_label_set(["agent:follow-up"]),
            Intent.FOLLOWUP,
        )

    def test_default_blocked_label(self) -> None:
        self.assertIs(
            intent_from_label_set(["agent:blocked"]),
            Intent.BLOCKED,
        )

    def test_unknown_label_returns_none(self) -> None:
        self.assertIs(
            intent_from_label_set(["bug", "enhancement"]),
            Intent.NONE,
        )

    def test_blocked_wins_over_retry(self) -> None:
        # BLOCKED is a permanent skip; must take precedence.
        self.assertIs(
            intent_from_label_set(["agent:retry", "agent:blocked"]),
            Intent.BLOCKED,
        )

    def test_blocked_wins_over_followup(self) -> None:
        self.assertIs(
            intent_from_label_set(["agent:follow-up", "agent:blocked"]),
            Intent.BLOCKED,
        )

    def test_followup_wins_over_retry(self) -> None:
        # FOLLOWUP is more conservative (preserves PR); wins over RETRY.
        self.assertIs(
            intent_from_label_set(["agent:retry", "agent:follow-up"]),
            Intent.FOLLOWUP,
        )

    def test_case_insensitive(self) -> None:
        # Labels are normalized lowercase in _extract_labels, but the
        # helper itself is also case-insensitive for robustness.
        self.assertIs(
            intent_from_label_set(["Agent:Retry"]),
            Intent.RETRY,
        )
        self.assertIs(
            intent_from_label_set(["AGENT:BLOCKED"]),
            Intent.BLOCKED,
        )

    def test_custom_intent_labels(self) -> None:
        custom = {
            "retry": "ops:rerun",
            "followup": "ops:more",
            "blocked": "ops:no",
        }
        self.assertIs(
            intent_from_label_set(["ops:rerun"], custom),
            Intent.RETRY,
        )
        self.assertIs(
            intent_from_label_set(["ops:more"], custom),
            Intent.FOLLOWUP,
        )
        self.assertIs(
            intent_from_label_set(["ops:no"], custom),
            Intent.BLOCKED,
        )
        # Default labels should NOT match when custom mapping is given.
        self.assertIs(
            intent_from_label_set(["agent:retry"], custom),
            Intent.NONE,
        )

    def test_default_intent_labels_constant(self) -> None:
        self.assertEqual(
            DEFAULT_INTENT_LABELS,
            {
                "retry": "agent:retry",
                "followup": "agent:follow-up",
                "blocked": "agent:blocked",
                # Rebase label convention.
                "rebase": "agent:rebase",
            },
        )

    def test_rebase_label_default(self) -> None:
        self.assertIs(
            intent_from_label_set(["agent:rebase"]),
            Intent.REBASE,
        )

    def test_rebase_alongside_retry(self) -> None:
        # REBASE outranks RETRY (force-push is more aggressive than a commit reset).
        self.assertIs(
            intent_from_label_set(["agent:rebase", "agent:retry"]),
            Intent.REBASE,
        )

    def test_rebase_alongside_followup(self) -> None:
        self.assertIs(
            intent_from_label_set(["agent:rebase", "agent:follow-up"]),
            Intent.REBASE,
        )

    def test_blocked_wins_over_rebase(self) -> None:
        # BLOCKED remains the sticky top-priority intent.
        self.assertIs(
            intent_from_label_set(["agent:rebase", "agent:blocked"]),
            Intent.BLOCKED,
        )

    def test_custom_rebase_label(self) -> None:
        labels = {**DEFAULT_INTENT_LABELS, "rebase": "ops:rebase"}
        self.assertIs(
            intent_from_label_set(["ops:rebase"], labels),
            Intent.REBASE,
        )


class TestParseAgentCommand(unittest.TestCase):
    def test_rebase_command(self) -> None:
        self.assertEqual(parse_agent_command("/agent rebase"), Command.REBASE)

    def test_rebase_command_with_whitespace(self) -> None:
        self.assertEqual(parse_agent_command("/agent   rebase"), Command.REBASE)

    def test_rebase_unknown_command_returns_none(self) -> None:
        # Defensive: unknown command tokens are NOT silently mapped to
        # Command.REBASE.
        self.assertIsNone(parse_agent_command("/agent random"))

    def test_existing_commands_unaffected(self) -> None:
        self.assertEqual(parse_agent_command("/agent retry"), Command.RETRY)
        self.assertEqual(parse_agent_command("/agent follow-up"), Command.FOLLOWUP)
        self.assertEqual(parse_agent_command("/agent unblock"), Command.UNBLOCK)

    def test_command_inside_fenced_code_block_ignored(self) -> None:
        # A /agent line inside a fenced code block is example text,
        # not a real command.
        self.assertIsNone(parse_agent_command("```\n/agent retry\n```"))
        self.assertIsNone(parse_agent_command("```text\n/agent rebase\n```"))

    def test_command_after_closed_fence_still_parsed(self) -> None:
        self.assertEqual(
            parse_agent_command("```\n/agent retry\n```\n/agent rebase"),
            Command.REBASE,
        )

    def test_command_in_blockquote_or_indented_block_ignored(self) -> None:
        self.assertIsNone(parse_agent_command("> /agent retry"))
        self.assertIsNone(parse_agent_command("    /agent retry"))


class TestCommandToIntent(unittest.TestCase):
    def test_rebase_command_maps_to_rebase_intent(self) -> None:
        self.assertIs(command_to_intent(Command.REBASE), Intent.REBASE)

    def test_existing_commands_unaffected(self) -> None:
        self.assertIs(command_to_intent(Command.RETRY), Intent.RETRY)
        self.assertIs(command_to_intent(Command.FOLLOWUP), Intent.FOLLOWUP)
        self.assertIs(command_to_intent(Command.UNBLOCK), Intent.NONE)


class TestMergeIntents(unittest.TestCase):
    def test_rebase_beats_retry(self) -> None:
        self.assertIs(
            merge_intents(Intent.RETRY, Intent.REBASE),
            Intent.REBASE,
        )

    def test_rebase_beats_followup(self) -> None:
        self.assertIs(
            merge_intents(Intent.FOLLOWUP, Intent.REBASE),
            Intent.REBASE,
        )

    def test_blocked_beats_rebase(self) -> None:
        self.assertIs(
            merge_intents(Intent.REBASE, Intent.BLOCKED),
            Intent.BLOCKED,
        )

    def test_rebase_beats_none(self) -> None:
        self.assertIs(
            merge_intents(Intent.NONE, Intent.REBASE),
            Intent.REBASE,
        )


class TestMergeIntentsWithCli(unittest.TestCase):
    def test_cli_rebase_wins(self) -> None:
        self.assertIs(
            merge_intents_with_cli(
                label_intent=Intent.RETRY,
                command_intent=Intent.NONE,
                cli_intent=Intent.REBASE,
            ),
            Intent.REBASE,
        )

    def test_cli_blocked_beats_rebase(self) -> None:
        self.assertIs(
            merge_intents_with_cli(
                label_intent=Intent.REBASE,
                command_intent=Intent.REBASE,
                cli_intent=Intent.BLOCKED,
            ),
            Intent.BLOCKED,
        )

    def test_label_rebase_beats_comment_retry(self) -> None:
        # REBASE > RETRY so the label REBASE wins.
        self.assertIs(
            merge_intents_with_cli(
                label_intent=Intent.REBASE,
                command_intent=Intent.RETRY,
                cli_intent=Intent.NONE,
            ),
            Intent.REBASE,
        )


# ---------------------------------------------------------------------------
# Adapter-level overrides
# ---------------------------------------------------------------------------
