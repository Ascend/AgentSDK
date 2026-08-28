# ruff: noqa
# pylint: skip-file
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

"""Tests for the SOP Converter skill grouper grouping strategies."""

from __future__ import annotations

from extensions.sop_converter.source_parser import (
    SourceComponent,
    SourceOperation,
    ParamSpec,
)
from extensions.sop_converter.skill_grouper import (
    GroupStrategy,
    SkillGrouper,
    group_source_components,
    MappingRule,
    MatchType,
)


# =========================================================================
# SkillGrouper tests
# =========================================================================


class TestGroupStrategy:
    def test_enum_values(self) -> None:
        assert GroupStrategy.KEYWORD_MATCH.value == "keyword_match"
        assert GroupStrategy.COMPONENT_GROUP.value == "component_group"
        assert GroupStrategy.IO_RELATION.value == "io_relation"
        assert GroupStrategy.LLM_SEMANTIC.value == "llm_semantic"

    def test_strategy_dispatch_keyword(self) -> None:
        """KEYWORD_MATCH strategy uses _keyword_match_group()."""
        from extensions.sop_converter.sdk_parser import SdkMethod

        grouper = SkillGrouper(
            [SdkMethod(name="docker_build", description="Build image")],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=[MappingRule("docker_", "docker_ops", "build_image", "Docker build", MatchType.PREFIX)],
        )
        skills = grouper.group()
        assert len(skills) > 0
        assert any(s.name == "build_image" for s in skills)

    def test_component_group_strategy(self) -> None:
        """COMPONENT_GROUP strategy groups operations by component."""
        ops = [
            SourceOperation(name="encode", description="Encode video"),
            SourceOperation(name="decode", description="Decode video"),
        ]
        comp = SourceComponent(
            name="VideoCodec",
            file_path="codec.py",
            description="Video codec operations",
            operations=ops,
        )
        result = group_source_components([comp], strategy=GroupStrategy.COMPONENT_GROUP)
        assert len(result.skills) == 1
        assert result.skills[0].name == "VideoCodec"
        assert "VideoCodec.encode" in result.skills[0].allowed_tools
        assert "VideoCodec.decode" in result.skills[0].allowed_tools

    def test_io_relation_strategy(self) -> None:
        """IO_RELATION strategy groups operations sharing anchor types."""
        ops_a = [
            SourceOperation(
                name="read_file",
                description="Read a file",
                parameters=[ParamSpec(name="path", type_hint="str")],
            ),
        ]
        ops_b = [
            SourceOperation(
                name="write_file",
                description="Write a file",
                parameters=[ParamSpec(name="path", type_hint="str")],
            ),
        ]
        comp_a = SourceComponent(name="Reader", file_path="r.py", description="Reader", operations=ops_a)
        comp_b = SourceComponent(name="Writer", file_path="w.py", description="Writer", operations=ops_b)

        result = group_source_components([comp_a, comp_b], strategy=GroupStrategy.IO_RELATION)
        # read_file and write_file share "str" anchor type → same group
        assert len(result.skills) >= 1
        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert "Reader.read_file" in all_tools
        assert "Writer.write_file" in all_tools
        assert any("str" in s.name for s in result.skills), (
            f"Expected type anchor in name, got: {[s.name for s in result.skills]}"
        )

    def test_io_relation_naming_with_types(self) -> None:
        """IO_RELATION group names include dominant type anchors."""
        ops_a = [
            SourceOperation(
                name="method_a",
                description="A",
                parameters=[
                    ParamSpec(name="x", type_hint="int"),
                    ParamSpec(name="y", type_hint="str"),
                ],
            )
        ]
        ops_b = [
            SourceOperation(
                name="method_b",
                description="B",
                parameters=[ParamSpec(name="x", type_hint="bool")],
            )
        ]
        comp_a = SourceComponent(name="CompA", file_path="a.py", description="A", operations=ops_a)
        comp_b = SourceComponent(name="CompB", file_path="b.py", description="B", operations=ops_b)

        result = group_source_components([comp_a, comp_b], strategy=GroupStrategy.IO_RELATION)
        # Two distinct anchor types → two groups (or merged into one if max_groups=1)
        assert len(result.skills) >= 1
        names = [s.name for s in result.skills]
        assert any("io_group" in n for n in names)

    def test_io_relation_with_no_params(self) -> None:
        """Operations with no parameters grouped into utility bucket."""
        ops = [SourceOperation(name="no_args", description="No args func")]
        comp = SourceComponent(name="Util", file_path="u.py", description="Utility", operations=ops)

        result = group_source_components([comp], strategy=GroupStrategy.IO_RELATION)
        assert len(result.skills) == 1
        assert result.skills[0].name == "io_group_utility"

    def test_io_relation_max_groups_merge(self) -> None:
        """max_io_groups forces merging of groups beyond the limit."""
        ops = []
        for i, type_name in enumerate(["str", "int", "bool", "float", "Path", "dict", "list", "bytes", "tuple", "set"]):
            ops.append(
                SourceOperation(
                    name=f"method_{type_name}",
                    description=f"Method using {type_name}",
                    parameters=[ParamSpec(name="x", type_hint=type_name)],
                )
            )

        comp = SourceComponent(
            name="ManyTypes",
            file_path="m.py",
            description="Many types",
            operations=ops,
        )

        result = group_source_components(
            [comp],
            strategy=GroupStrategy.IO_RELATION,
            max_io_groups=5,
        )
        # 10 distinct types merged down to ≤ 5 groups
        assert len(result.skills) <= 5
        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert len(all_tools) == 10

    def test_io_relation_shared_type_merges(self) -> None:
        """Operations sharing a common type are grouped together."""
        ops = [
            SourceOperation(
                name="op_a",
                description="A",
                parameters=[
                    ParamSpec(name="x", type_hint="str"),
                    ParamSpec(name="y", type_hint="int"),
                ],
            ),
            SourceOperation(
                name="op_b",
                description="B",
                parameters=[
                    ParamSpec(name="x", type_hint="str"),
                    ParamSpec(name="y", type_hint="bool"),
                ],
            ),
            SourceOperation(
                name="op_c",
                description="C",
                parameters=[
                    ParamSpec(name="x", type_hint="str"),
                    ParamSpec(name="y", type_hint="float"),
                ],
            ),
        ]
        comp = SourceComponent(name="Shared", file_path="s.py", description="Shared type str", operations=ops)

        result = group_source_components([comp], strategy=GroupStrategy.IO_RELATION, max_io_groups=2)
        # All 3 ops share "str" anchor → should be in 1-2 groups
        assert len(result.skills) <= 2
        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert "Shared.op_a" in all_tools
        assert "Shared.op_b" in all_tools
        assert "Shared.op_c" in all_tools

    def test_io_relation_rebalancing_no_megagroup(self) -> None:
        """Rebalancing redirects multi-type ops to diverse secondary anchors."""
        ops = []
        for i in range(50):
            ops.append(
                SourceOperation(
                    name=f"op_str_path_{i}",
                    description="Str+Path op",
                    parameters=[
                        ParamSpec(name="x", type_hint="str"),
                        ParamSpec(name="y", type_hint="Path"),
                    ],
                )
            )
        for i in range(50):
            ops.append(
                SourceOperation(
                    name=f"op_str_dict_{i}",
                    description="Str+dict op",
                    parameters=[
                        ParamSpec(name="x", type_hint="str"),
                        ParamSpec(name="y", type_hint="dict"),
                    ],
                )
            )
        for i in range(10):
            ops.append(
                SourceOperation(
                    name=f"op_int_{i}",
                    description="Int op",
                    parameters=[ParamSpec(name="x", type_hint="int")],
                )
            )
        for i in range(5):
            ops.append(
                SourceOperation(
                    name=f"op_bool_{i}",
                    description="Bool op",
                    parameters=[ParamSpec(name="x", type_hint="bool")],
                )
            )

        comp = SourceComponent(
            name="BigSDK",
            file_path="b.py",
            description="Big SDK with diverse secondary types",
            operations=ops,
        )

        result = group_source_components(
            [comp],
            strategy=GroupStrategy.IO_RELATION,
            max_io_groups=5,
        )

        assert len(result.skills) <= 5
        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert len(all_tools) == len(ops)

        max_tools = max(len(s.allowed_tools) for s in result.skills)
        assert max_tools < len(ops), f"Rebalancing should split dominant anchor, got max={max_tools}/{len(ops)}"

    def test_io_relation_single_type_stays_grouped(self) -> None:
        """Single-type ops stay in their anchor group, not forcibly scattered."""
        ops = []
        for _ in range(100):
            ops.append(
                SourceOperation(
                    name=f"op_str_{_}",
                    description="Str op",
                    parameters=[ParamSpec(name="x", type_hint="str")],
                )
            )

        comp = SourceComponent(
            name="OnlyStr",
            file_path="s.py",
            description="All str ops",
            operations=ops,
        )

        result = group_source_components(
            [comp],
            strategy=GroupStrategy.IO_RELATION,
            max_io_groups=5,
        )

        assert len(result.skills) == 1
        assert len(result.skills[0].allowed_tools) == 100

    def test_io_relation_utility_ops_grouped(self) -> None:
        """Utility (untyped) operations are dissolved into typed buckets."""
        typed_ops = []
        for t in ["str", "int", "bool", "float", "Path"]:
            typed_ops.append(
                SourceOperation(
                    name=f"op_{t}",
                    description=f"Op {t}",
                    parameters=[ParamSpec(name="x", type_hint=t)],
                )
            )

        untyped_ops = []
        for i in range(20):
            untyped_ops.append(
                SourceOperation(
                    name=f"op_none_{i}",
                    description="No type",
                    parameters=[ParamSpec(name="x")],
                )
            )

        comp = SourceComponent(
            name="MixedTypes",
            file_path="m.py",
            description="Mixed typed/untyped",
            operations=typed_ops + untyped_ops,
        )

        result = group_source_components(
            [comp],
            strategy=GroupStrategy.IO_RELATION,
            max_io_groups=5,
        )

        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert len(all_tools) == 25

        # With the dissolution design, untyped ops are distributed
        # round-robin across existing typed buckets — no standalone
        # "utility" group should exist when typed buckets are present.
        utility_group = next((s for s in result.skills if "utility" in s.name), None)
        assert utility_group is None, f"Utility group should be dissolved, got: {[s.name for s in result.skills]}"


# =========================================================================
# MatchType and MappingRule tests
# =========================================================================


class TestMatchType:
    def test_enum_values(self) -> None:
        assert MatchType.SUBSTRING.value == "substring"
        assert MatchType.PREFIX.value == "prefix"
        assert MatchType.SUFFIX.value == "suffix"
        assert MatchType.REGEX.value == "regex"
        assert MatchType.EXACT.value == "exact"


class TestMappingRuleMatches:
    def test_substring_match(self) -> None:
        rule = MappingRule("docker", "docker_ops", "build_image", match_type=MatchType.SUBSTRING)
        assert rule.matches("docker_build")
        assert rule.matches("my_docker_push")
        assert not rule.matches("k8s_apply")

    def test_prefix_match(self) -> None:
        rule = MappingRule("docker_", "docker_ops", "build_image", match_type=MatchType.PREFIX)
        assert rule.matches("docker_build")
        assert rule.matches("docker_push_image")
        assert not rule.matches("my_docker_push")

    def test_suffix_match(self) -> None:
        rule = MappingRule("_check", "check_ops", "health", match_type=MatchType.SUFFIX)
        assert rule.matches("health_check")
        assert rule.matches("status_check")
        assert not rule.matches("check_status")

    def test_regex_match(self) -> None:
        rule = MappingRule("video_encode|video_decode", "video_ops", "video_processing", match_type=MatchType.REGEX)
        assert rule.matches("video_encode")
        assert rule.matches("video_decode")
        assert not rule.matches("audio_encode")

    def test_exact_match(self) -> None:
        rule = MappingRule("rollback", "rollback", "deploy_service", match_type=MatchType.EXACT)
        assert rule.matches("rollback")
        assert not rule.matches("rollback_deployment")
        assert not rule.matches("fast_rollback")

    def test_default_match_type_is_substring(self) -> None:
        rule = MappingRule("docker", "docker_ops", "build_image")
        assert rule.match_type == MatchType.SUBSTRING
        assert rule.matches("docker_build")


# =========================================================================
# KEYWORD_MATCH strategy tests
# =========================================================================


class TestKeywordMatch:
    def test_keyword_match_with_source_components(self) -> None:
        """KEYWORD_MATCH groups SourceComponent operations by MappingRule patterns."""
        ops = [
            SourceOperation(name="docker_build", description="Build image"),
            SourceOperation(name="docker_push", description="Push image"),
            SourceOperation(name="k8s_apply", description="Apply manifest"),
            SourceOperation(name="k8s_get", description="Get resource"),
        ]
        comp = SourceComponent(
            name="CICD",
            file_path="cicd.py",
            description="CI/CD operations",
            operations=ops,
        )
        result = group_source_components(
            [comp],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=[
                MappingRule("docker_", "docker_ops", "build_image", "Docker build", MatchType.PREFIX),
                MappingRule("k8s_", "k8s_ops", "deploy_service", "Kubernetes ops", MatchType.PREFIX),
            ],
        )
        skill_names = [s.name for s in result.skills]
        assert "build_image" in skill_names
        assert "deploy_service" in skill_names
        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert "CICD.docker_build" in all_tools
        assert "CICD.docker_push" in all_tools
        assert "CICD.k8s_apply" in all_tools
        assert "CICD.k8s_get" in all_tools

    def test_keyword_match_prefix_rules(self) -> None:
        """Prefix-type MappingRules match operation name starts."""
        rules = [
            MappingRule("video_", "video_ops", "video_ops", "Video operations", MatchType.PREFIX),
            MappingRule("audio_", "audio_ops", "audio_ops", "Audio operations", MatchType.PREFIX),
        ]
        ops = [
            SourceOperation(name="video_encode", description="Encode video"),
            SourceOperation(name="video_decode", description="Decode video"),
            SourceOperation(name="audio_mix", description="Mix audio"),
            SourceOperation(name="audio_record", description="Record audio"),
        ]
        comp = SourceComponent(name="Media", file_path="media.py", description="Media ops", operations=ops)
        result = group_source_components(
            [comp],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=rules,
        )
        skill_names = [s.name for s in result.skills]
        assert "video_ops" in skill_names
        assert "audio_ops" in skill_names
        video_skill = next(s for s in result.skills if s.name == "video_ops")
        assert "Media.video_encode" in video_skill.allowed_tools
        assert "Media.video_decode" in video_skill.allowed_tools

    def test_keyword_match_auto_prefix_inference(self) -> None:
        """Unmatched operations are auto-grouped by underscore prefix."""
        ops = [
            SourceOperation(name="video_encode", description="Encode"),
            SourceOperation(name="video_decode", description="Decode"),
            SourceOperation(name="audio_mix", description="Mix"),
            SourceOperation(name="audio_record", description="Record"),
        ]
        comp = SourceComponent(name="Media", file_path="media.py", description="Media", operations=ops)
        result = group_source_components(
            [comp],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=[],
        )
        skill_names = [s.name for s in result.skills]
        assert "video_ops" in skill_names
        assert "audio_ops" in skill_names
        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert len(all_tools) == 4

    def test_keyword_match_single_segment_names_in_utility(self) -> None:
        """Single-segment names (no underscore) go to utility bucket."""
        ops = [
            SourceOperation(name="init", description="Initialize"),
            SourceOperation(name="cleanup", description="Clean up"),
        ]
        comp = SourceComponent(name="Core", file_path="core.py", description="Core", operations=ops)
        result = group_source_components(
            [comp],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=[],
        )
        utility_skill = next(s for s in result.skills if s.name == "utility")
        assert "Core.init" in utility_skill.allowed_tools
        assert "Core.cleanup" in utility_skill.allowed_tools

    def test_keyword_match_small_prefix_groups_merged_to_misc(self) -> None:
        """Prefix groups with <2 items are merged into misc."""
        ops = [
            SourceOperation(name="video_encode", description="Encode"),
            SourceOperation(name="video_decode", description="Decode"),
            SourceOperation(name="audio_mix", description="Mix"),
            SourceOperation(name="cache_purge", description="Purge"),
        ]
        comp = SourceComponent(name="Mixed", file_path="mixed.py", description="Mixed", operations=ops)
        result = group_source_components(
            [comp],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=[],
        )
        all_tools = {t for s in result.skills for t in s.allowed_tools}
        assert len(all_tools) == 4
        assert "Mixed.video_encode" in all_tools
        assert "Mixed.cache_purge" in all_tools
        misc_skill = next(s for s in result.skills if s.name == "misc")
        assert "Mixed.audio_mix" in misc_skill.allowed_tools
        assert "Mixed.cache_purge" in misc_skill.allowed_tools

    def test_keyword_match_mixed_explicit_and_auto(self) -> None:
        """Explicit MappingRules + auto prefix inference work together."""
        rules = [
            MappingRule("docker_", "docker_ops", "build_image", "Docker build", MatchType.PREFIX),
        ]
        ops = [
            SourceOperation(name="docker_build", description="Build image"),
            SourceOperation(name="docker_push", description="Push image"),
            SourceOperation(name="video_encode", description="Encode video"),
            SourceOperation(name="video_decode", description="Decode video"),
        ]
        comp = SourceComponent(name="Mixed", file_path="mixed.py", description="Mixed ops", operations=ops)
        result = group_source_components(
            [comp],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=rules,
        )
        skill_names = [s.name for s in result.skills]
        assert "build_image" in skill_names
        assert "video_ops" in skill_names
        docker_skill = next(s for s in result.skills if s.name == "build_image")
        assert "Mixed.docker_build" in docker_skill.allowed_tools
        assert "Mixed.docker_push" in docker_skill.allowed_tools

    def test_keyword_match_regex_rule(self) -> None:
        """Regex-type MappingRule matches pattern via regex search."""
        rules = [
            MappingRule(
                "video_encode|video_decode",
                "video_ops",
                "video_ops",
                "Video codec",
                MatchType.REGEX,
            ),
        ]
        ops = [
            SourceOperation(name="video_encode", description="Encode"),
            SourceOperation(name="video_decode", description="Decode"),
            SourceOperation(name="audio_mix", description="Mix"),
        ]
        comp = SourceComponent(name="Codec", file_path="codec.py", description="Codec", operations=ops)
        result = group_source_components(
            [comp],
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=rules,
        )
        video_skill = next(s for s in result.skills if s.name == "video_ops")
        assert "Codec.video_encode" in video_skill.allowed_tools
        assert "Codec.video_decode" in video_skill.allowed_tools

    def test_keyword_match_with_sdk_methods(self) -> None:
        """KEYWORD_MATCH also works with SdkMethod data (backward compat)."""
        from extensions.sop_converter.sdk_parser import SdkMethod

        methods = [
            SdkMethod(name="docker_build", description="Build image"),
            SdkMethod(name="docker_push", description="Push image"),
            SdkMethod(name="k8s_apply", description="Apply manifest"),
        ]
        grouper = SkillGrouper(
            methods,
            strategy=GroupStrategy.KEYWORD_MATCH,
            mapping_rules=[
                MappingRule("docker_", "docker_ops", "build_image", "Docker build", MatchType.PREFIX),
                MappingRule("k8s_", "k8s_ops", "deploy_service", "Kubernetes ops", MatchType.PREFIX),
            ],
        )
        skills = grouper.group()
        skill_names = [s.name for s in skills]
        assert "build_image" in skill_names
        assert "deploy_service" in skill_names

    def test_keyword_match_empty_input(self) -> None:
        """KEYWORD_MATCH with no methods or components returns empty list."""
        grouper = SkillGrouper([], strategy=GroupStrategy.KEYWORD_MATCH)
        skills = grouper.group()
        assert skills == []

    def test_static_group_respects_match_type(self) -> None:
        """SdkMethod-only path honors match_type via MappingRule.matches().

        Regression for review: ``_static_group`` used a raw ``in`` substring
        check that silently ignored PREFIX / EXACT / REGEX match types.
        """
        from extensions.sop_converter.sdk_parser import SdkMethod

        methods = [
            SdkMethod(name="my_docker_build", description="Docker build (embedded)"),
            SdkMethod(name="docker_build", description="Docker build"),
            SdkMethod(name="rollback_deployment", description="Rollback deployment"),
            SdkMethod(name="rollback", description="Rollback"),
            SdkMethod(name="video_encode", description="Encode video"),
        ]
        grouper = SkillGrouper(
            methods,
            mapping_rules=[
                MappingRule("docker", "docker_ops", "docker_skill", "Docker", match_type=MatchType.PREFIX),
                MappingRule("rollback", "rollback", "rb_skill", "Rollback", match_type=MatchType.EXACT),
                MappingRule("video_encode|video_decode", "video_ops", "v_skill", "Video", match_type=MatchType.REGEX),
            ],
        )
        skills = {s.name: s for s in grouper.group()}
        # PREFIX "docker" must NOT match "my_docker_build" (substring would).
        assert "my_docker_build" not in skills["docker_skill"].allowed_tools
        assert "docker_build" in skills["docker_skill"].allowed_tools
        # EXACT "rollback" must NOT match "rollback_deployment".
        assert "rollback_deployment" not in skills["rb_skill"].allowed_tools
        assert "rollback" in skills["rb_skill"].allowed_tools
        # REGEX pattern still matches via re.search.
        assert "video_encode" in skills["v_skill"].allowed_tools
