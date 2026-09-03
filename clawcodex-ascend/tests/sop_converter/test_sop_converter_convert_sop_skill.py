#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Unit tests for :mod:`extensions.sop_converter.convert_sop_skill`.

Covers the SOP→Agent conversion Skill:

* :func:`convert_sop_to_agent` — full pipeline from SDK spec to a
  serialised result dict (agent_type, skills, tools, persist_status,
  warnings). Error paths (no parsed methods, no grouped skills).
* :func:`_generate_agent_name` — slug generation from requirements,
  including fallback and length cap.
* :func:`get_prompt_for_command` — argument parsing via ``::`` separator.
* :func:`_format_result` — human-readable output for both success and
  error cases.
* :data:`_DEFAULT_RULES` — sanity checks for the built-in mapping rules.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from extensions.sop_converter.convert_sop_skill import (
    _DEFAULT_RULES,
    _SKILL_PROMPT,
    _format_result,
    _generate_agent_name,
    convert_sop_to_agent,
    get_prompt_for_command,
)
from extensions.sop_converter.core.templates import MappingRule


# ---------------------------------------------------------------------------
# _DEFAULT_RULES sanity
# ---------------------------------------------------------------------------


class TestDefaultRules(unittest.TestCase):
    def test_default_rules_is_list_of_mapping_rules(self) -> None:
        self.assertIsInstance(_DEFAULT_RULES, list)
        for r in _DEFAULT_RULES:
            self.assertIsInstance(r, MappingRule)

    def test_default_rules_cover_common_patterns(self) -> None:
        method_names = {r.method_pattern for r in _DEFAULT_RULES}
        # Some common SDK method patterns we expect to see.
        for expected in [
            "docker_build",
            "docker_push",
            "k8s_apply",
            "slack_send",
            "s3_upload",
            "train_model",
        ]:
            self.assertIn(expected, method_names)


# ---------------------------------------------------------------------------
# _generate_agent_name
# ---------------------------------------------------------------------------


class TestGenerateAgentName(unittest.TestCase):
    def test_simple_requirements(self) -> None:
        self.assertEqual(_generate_agent_name("CI/CD pipeline"), "ci-cd-pipeline")

    def test_special_chars_collapsed(self) -> None:
        # Non-alphanumeric runs become single hyphens, then trim edges.
        self.assertEqual(
            _generate_agent_name("  hello!!  world??  "),
            "hello-world",
        )

    def test_empty_falls_back(self) -> None:
        self.assertEqual(_generate_agent_name(""), "converted-agent")

    def test_whitespace_only_falls_back(self) -> None:
        self.assertEqual(_generate_agent_name("   "), "converted-agent")

    def test_special_chars_only_falls_back(self) -> None:
        # After regex stripping, all that's left is empty → fallback.
        self.assertEqual(_generate_agent_name("!!!"), "converted-agent")

    def test_long_name_truncated(self) -> None:
        long_req = "a" * 100
        # Result is capped at 40 chars.
        self.assertEqual(len(_generate_agent_name(long_req)), 40)

    def test_already_short_preserved(self) -> None:
        self.assertEqual(_generate_agent_name("deploy"), "deploy")


# ---------------------------------------------------------------------------
# convert_sop_to_agent — happy path
# ---------------------------------------------------------------------------


class TestConvertPosToAgent(unittest.TestCase):
    def test_simple_list_spec_succeeds(self) -> None:
        # A spec with methods that match the default rules.
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_tag, docker_push, k8s_apply, health_check",
                requirements="CI/CD pipeline",
                agent_name="cicd-agent",
            )
        self.assertEqual(result["status"], "converted")
        self.assertEqual(result["agent_type"], "cicd-agent")
        # Skills were created.
        self.assertGreater(len(result["skills"]), 0)
        # Tools list is populated.
        self.assertIsInstance(result["tools"], list)
        # Skill files list is populated (paths).
        self.assertIsInstance(result["skill_files"], list)
        # Persistence succeeded.
        self.assertEqual(result["persist_status"], "saved")

    def test_auto_generated_agent_name(self) -> None:
        # No agent_name → derive from requirements.
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="My Pipeline",
            )
        self.assertEqual(result["status"], "converted")
        # "My Pipeline" → "my-pipeline" via _generate_agent_name.
        self.assertEqual(result["agent_type"], "my-pipeline")

    def test_custom_agent_description_used(self) -> None:
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="x",
                agent_name="my-agent",
                agent_description="Custom description here",
            )
        self.assertEqual(result["agent_description"], "Custom description here")

    def test_default_description_from_requirements(self) -> None:
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="data processing",
                agent_name="dp-agent",
            )
        # "data processing" is interpolated into the description.
        self.assertIn("data processing", result["agent_description"])

    def test_no_requirements_description(self) -> None:
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="",
                agent_name="a",
            )
        # Without requirements, the default description is generic.
        self.assertIn("SDK", result["agent_description"])

    def test_persist_failure_captured(self) -> None:
        # Force the persist call to fail — the conversion still succeeds.
        from extensions.sop_converter import agent_builder

        with patch.object(
            agent_builder,
            "persist_converted_agent",
            side_effect=OSError("disk full"),
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="x",
                agent_name="a",
            )
        self.assertEqual(result["status"], "converted")
        self.assertTrue(result["persist_status"].startswith("save_failed"))

    def test_model_passthrough(self) -> None:
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="x",
                agent_name="a",
                model="opus-4.7",
            )
        self.assertEqual(result["model"], "opus-4.7")

    def test_no_model_returns_default(self) -> None:
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="x",
                agent_name="a",
            )
        # No model set → either None or "default" string.
        self.assertIn(result["model"], (None, "default"))

    def test_warnings_propagate(self) -> None:
        # The result dict always has a "warnings" key, possibly empty.
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="docker_build, docker_push",
                requirements="x",
                agent_name="a",
            )
        self.assertIn("warnings", result)
        self.assertIsInstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# convert_sop_to_agent — error paths
# ---------------------------------------------------------------------------


class TestConvertPosToAgentErrors(unittest.TestCase):
    def test_no_methods_parsed_returns_error(self) -> None:
        # Empty / comment-only spec → no methods → error.
        result = convert_sop_to_agent(
            sdk_spec="# only comment lines\n# more comments",
            requirements="x",
            agent_name="a",
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("No SDK methods", result["error"])

    def test_unmatched_methods_go_to_default_skill(self) -> None:
        # The skill_grouper puts methods that don't match any rule into
        # a default "sdk_utility" skill — so the conversion still
        # succeeds.
        custom_rules: list[MappingRule] = [
            MappingRule("specific_method", "specific", "group", "Specific"),
        ]
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = convert_sop_to_agent(
                sdk_spec="unrelated_method_one, unrelated_method_two",
                requirements="x",
                agent_name="a",
                mapping_rules=custom_rules,
            )
        self.assertEqual(result["status"], "converted")
        # The unmatched methods ended up in the sdk_utility skill.
        skill_names = [s["name"] for s in result["skills"]]
        self.assertIn("sdk_utility", skill_names)


# ---------------------------------------------------------------------------
# get_prompt_for_command
# ---------------------------------------------------------------------------


class TestGetPromptForCommand(unittest.TestCase):
    def test_empty_args_returns_default_prompt(self) -> None:
        result = get_prompt_for_command("")
        # The default prompt explains the skill.
        self.assertIn("Convert", result)

    def test_default_prompt_constant_is_full_prompt(self) -> None:
        # The default prompt module constant is a multi-line string with
        # the skill description.
        self.assertIn("Convert", _SKILL_PROMPT)
        self.assertIn("SOP", _SKILL_PROMPT)

    def test_simple_args_invokes_conversion(self) -> None:
        # With just a spec, the conversion runs end-to-end.
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = get_prompt_for_command("docker_build, docker_push")
        # Result is the formatted conversion output.
        self.assertIn("Converted SOP", result)

    def test_args_with_requirements(self) -> None:
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = get_prompt_for_command(
                "docker_build :: CI/CD pipeline",
            )
        self.assertIn("Converted SOP", result)
        # The agent type reflects the slug of "CI/CD pipeline".
        self.assertIn("ci-cd-pipeline", result)

    def test_args_with_all_three_fields(self) -> None:
        with patch(
            "extensions.sop_converter.agent_builder.persist_converted_agent",
        ):
            result = get_prompt_for_command(
                "docker_build :: x :: my-agent",
            )
        self.assertIn("my-agent", result)

    def test_error_result_format(self) -> None:
        # An empty spec triggers an error → formatted output reflects that.
        result = get_prompt_for_command("# only comment")
        self.assertIn("Conversion failed", result)


# ---------------------------------------------------------------------------
# _format_result
# ---------------------------------------------------------------------------


class TestFormatResult(unittest.TestCase):
    def test_error_format(self) -> None:
        out = _format_result({"status": "error", "error": "boom"})
        self.assertIn("Conversion failed", out)
        self.assertIn("boom", out)

    def test_error_format_default_message(self) -> None:
        out = _format_result({"status": "error"})
        self.assertIn("unknown error", out)

    def test_success_format_includes_skills_and_tools(self) -> None:
        result = {
            "status": "converted",
            "agent_type": "test-agent",
            "agent_description": "A test agent",
            "model": "opus-4.7",
            "skills": [
                {
                    "name": "skill1",
                    "description": "First skill",
                    "tools": ["a", "b"],
                },
            ],
            "tools": ["a", "b", "c"],
            "skill_files": ["/path/to/skill1.md"],
            "persist_status": "saved",
            "warnings": [],
        }
        out = _format_result(result)
        self.assertIn("test-agent", out)
        self.assertIn("A test agent", out)
        self.assertIn("skill1", out)
        self.assertIn("First skill", out)
        self.assertIn("Tools: a, b", out)
        self.assertIn("Persistence: saved", out)

    def test_warnings_shown_when_present(self) -> None:
        result = {
            "status": "converted",
            "agent_type": "t",
            "agent_description": "d",
            "model": "m",
            "skills": [],
            "tools": [],
            "skill_files": [],
            "persist_status": "saved",
            "warnings": ["warning1", "warning2"],
        }
        out = _format_result(result)
        self.assertIn("Warnings:", out)
        self.assertIn("warning1", out)
        self.assertIn("warning2", out)

    def test_no_warnings_section_when_empty(self) -> None:
        result = {
            "status": "converted",
            "agent_type": "t",
            "agent_description": "d",
            "model": "m",
            "skills": [],
            "tools": [],
            "skill_files": [],
            "persist_status": "saved",
            "warnings": [],
        }
        out = _format_result(result)
        # No "Warnings:" header when list is empty.
        self.assertNotIn("Warnings:", out)


if __name__ == "__main__":
    unittest.main()
