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

"""SkillGrouper — groups atomic tools into business-level Skills.

Uses LLM-assisted grouping to cluster related tools by business logic.
Falls back to static rules (from MappingRule config) when LLM is unavailable.
KEYWORD_MATCH strategy: MappingRule pattern matching with catch-all ``sdk_utility``.
IO_RELATION strategy: type-anchor clustering with anchor-set-intersection
    grouping, smallest-first merge, and ClassName.methodName / compName.methodName
    tool naming.
LLM_SEMANTIC strategy: LLM-driven semantic clustering with JSON output parsing,
    validation, and KEYWORD_MATCH fallback.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.sdk_parser import SdkMethod
from enum import Enum
from ..core.source_parser import SourceComponent, SourceOperation

if TYPE_CHECKING:
    from extensions.capabilities.sop_provider_protocol import SOPAssistantProviderProtocol

logger = logging.getLogger(__name__)


@dataclass
class SkillGrouper:
    """Group atomic SDK methods into Skills based on business logic.

    Uses MappingRule config for static grouping when LLM is not available.
    The group method accepts a requirements hint that can be used by LLM
    to determine which tools belong together.
    """

    _DEFAULT_MAX_IO_GROUPS = 15

    def __init__(
        self,
        methods: list[SdkMethod],
        *,
        mapping_rules: list[MappingRule] | None = None,
        strategy: GroupStrategy | list[GroupStrategy] | None = None,
        source_components: list[SourceComponent] | None = None,
        max_io_groups: int | None = None,
        llm_provider: SOPAssistantProviderProtocol | None = None,
    ) -> None:
        self._methods = methods
        self._custom_rules = mapping_rules  # None = user didn't provide custom rules
        self._rules = mapping_rules or []
        self._llm_rules: list[MappingRule] | None = None  # LLM-generated rules (isolated)
        self._strategy = strategy
        self._source_components = source_components or []
        self._max_io_groups = max_io_groups or self._DEFAULT_MAX_IO_GROUPS
        self._llm_provider = llm_provider
        self._grouped: list[SkillSpec] | None = None

    def group(self, requirements: str = "") -> list[SkillSpec]:
        """Group methods into Skills.

        Args:
            requirements: Business requirements hint (e.g., "CI/CD pipeline",
                "data processing"). Passed to LLM when available for smarter
                grouping. Falls back to static MappingRule matching.
        """
        if self._grouped is not None:
            return self._grouped

        if self._strategy is None:
            self._grouped = self._static_group()
        elif isinstance(self._strategy, list):
            self._grouped = self._dispatch_list_strategy(requirements)
        else:
            self._grouped = self._dispatch_single_strategy(requirements)

        return self._grouped

    def _dispatch_list_strategy(self, requirements: str) -> list[SkillSpec]:
        """Dispatch a strategy list by fixed priority.

        Priority: COMPONENT_GROUP > KEYWORD_MATCH > LLM_SEMANTIC > IO_RELATION.
        KEYWORD_MATCH without SourceComponents falls back to the static path
        (component-aware keyword matching, directory auto-rules and
        COMP_NAME / FILE_PATH targets do not apply to SdkMethod-only input).
        """
        if GroupStrategy.COMPONENT_GROUP in self._strategy:
            return self._component_group()
        if GroupStrategy.KEYWORD_MATCH in self._strategy:
            if self._source_components:
                self._maybe_auto_rules()
                return self._keyword_match_group()
            return self._static_group()
        if GroupStrategy.LLM_SEMANTIC in self._strategy:
            logger.warning("LLM semantic grouping is unavailable; falling back to static grouping")
            return self._static_group()
        if GroupStrategy.IO_RELATION in self._strategy:
            return self._io_relation_group()
        return self._static_group()

    def _dispatch_single_strategy(self, requirements: str) -> list[SkillSpec]:
        """Dispatch a single GroupStrategy value."""
        if self._strategy == GroupStrategy.KEYWORD_MATCH:
            if self._source_components:
                self._maybe_auto_rules()
                return self._keyword_match_group()
            return self._static_group()
        if self._strategy == GroupStrategy.COMPONENT_GROUP:
            return self._component_group()
        if self._strategy == GroupStrategy.IO_RELATION:
            return self._io_relation_group()
        if self._strategy == GroupStrategy.LLM_SEMANTIC:
            logger.warning("LLM semantic grouping is unavailable; falling back to static grouping")
            return self._static_group()
        return self._static_group()

    def _static_group(self) -> list[SkillSpec]:
        """Group SdkMethod tools using static MappingRule patterns.

        SdkMethod-only path: matching runs against the method name
        (OP_NAME) via ``MappingRule.matches()``, which honors match_type.

        Component-aware features are NOT supported here BY DESIGN — they
        require SourceComponent metadata and live on the SourceComponent
        path only (``_keyword_match_group`` / ``_group_with_llm``):
          - match_target COMP_NAME / FILE_PATH
          - KEYWORD_MATCH directory auto-rules (``_maybe_auto_rules``)
          - LLM semantic grouping
        """
        skill_map: dict[str, SkillSpec] = {}
        unmatched: list[SdkMethod] = []

        for method in self._methods:
            matched = False
            for rule in self._rules:
                if rule.matches(method.name):
                    self._append_to_skill(skill_map, rule, method)
                    matched = True
                    break
            if not matched:
                unmatched.append(method)

        # Put unmatched tools in a default skill
        if unmatched:
            skill_map["_unmatched"] = SkillSpec(
                name="sdk_utility",
                description="SDK utility methods",
                allowed_tools=[m.name for m in unmatched],
                argument_names=[],
            )

        return list(skill_map.values())

    def _append_to_skill(
        self,
        skill_map: dict[str, SkillSpec],
        rule: MappingRule,
        method: SdkMethod,
    ) -> None:
        """Attach a matched SdkMethod to the rule's SkillSpec."""
        if rule.skill_name not in skill_map:
            skill_map[rule.skill_name] = SkillSpec(
                name=rule.skill_name,
                description=rule.description or f"Skill: {rule.skill_name}",
                allowed_tools=[],
            )
        skill = skill_map[rule.skill_name]
        if method.name not in skill.allowed_tools:
            skill.allowed_tools.append(method.name)
        if method.parameters:
            skill.argument_names.extend(method.parameters)

    def _maybe_auto_rules(self) -> None:
        """If no custom mapping rules were provided, auto-generate from directory structure.

        Only applies to KEYWORD_MATCH + SourceComponent path.
        Does NOT affect _static_group (SdkMethod path) or other strategies.
        """
        if self._custom_rules is not None:
            return  # User provided custom rules — respect them.
        auto = self._auto_generate_rules(self._source_components, self._max_io_groups)
        if auto:
            self._rules = auto

    @staticmethod
    def _auto_generate_rules(
        components: list[SourceComponent],
        max_groups: int = 15,
    ) -> list[MappingRule]:
        """Auto-generate MappingRules from directory structure of SourceComponents.

        Algorithm — path prefix tree cutting:
          1. Collect all file_paths, split into segments by OS separator.
          2. At each depth d, group components by their path prefix of d
             segments.  Pick the depth whose group count is **closest to**
             max_groups (preferring slightly over if equidistant), so that
             we get the finest-grained grouping that still fits the budget.
          3. When a depth yields more groups than max_groups, merge smallest
             groups by path-segment Jaccard similarity until ≤ max_groups.
          4. For each final group, use the **distinguishing path segment**
             (the deepest segment that is unique to this group) as the
             method_pattern for a FILE_PATH / SUBSTRING rule.

        This is only used when strategy=KEYWORD_MATCH, source_components is
        non-empty, and no custom ``--mapping-rules`` were provided.  The
        SdkMethod path (``_static_group``) is completely unaffected.
        """
        if not components:
            return []

        comp_paths = SkillGrouper._collect_path_segments(components)
        if not any(segs for _, segs in comp_paths):
            # No meaningful path structure — cannot auto-generate.
            return []

        best_depth = SkillGrouper._pick_cut_depth(comp_paths, max_groups)
        groups = SkillGrouper._build_groups_at_depth(comp_paths, best_depth)
        groups = SkillGrouper._merge_smallest_groups(groups, max_groups)

        # Step 5: generate MappingRule(s) per group; Step 6: sort by specificity.
        all_group_keys = list(groups.keys())
        rules: list[MappingRule] = []
        for group_key in all_group_keys:
            rules.extend(SkillGrouper._rules_for_path_group(group_key, all_group_keys, rules))
        rules.sort(key=lambda r: _pattern_specificity(r, all_group_keys))
        return rules

    @staticmethod
    def _collect_path_segments(
        components: list[SourceComponent],
    ) -> list[tuple[str, list[str]]]:
        """Step 1: normalize each component's directory path into segments."""
        comp_paths: list[tuple[str, list[str]]] = []  # (comp_name, segments)
        for comp in components:
            fp = comp.file_path.replace("\\", "/")
            # Remove the final .py filename — we only care about directories.
            if fp.endswith(".py"):
                fp = fp.rsplit("/", 1)[0] if "/" in fp else ""
            segments = [s for s in fp.split("/") if s]
            comp_paths.append((comp.name, segments))
        return comp_paths

    @staticmethod
    def _pick_cut_depth(
        comp_paths: list[tuple[str, list[str]]],
        max_groups: int,
    ) -> int:
        """Step 2: pick the shallowest depth whose group count ≥ max_groups.

        This gives the finest-grained grouping that can be merged down to
        exactly max_groups.  If no depth reaches max_groups, use the deepest
        depth available.
        """
        max_depth = max((len(segs) for _, segs in comp_paths), default=0)
        best_depth = 1
        best_count_at_depth = 0
        for d in range(1, max_depth + 1):
            groups_d: dict[str, list[str]] = {}
            for comp_name, segs in comp_paths:
                prefix_key = "/".join(segs[:d]) if len(segs) >= d else "/".join(segs)
                groups_d.setdefault(prefix_key, []).append(comp_name)
            count = len(groups_d)
            if count >= max_groups:
                best_depth = d
                best_count_at_depth = count
                break
            # Track the deepest depth with most groups as fallback.
            if count > best_count_at_depth:
                best_depth = d
                best_count_at_depth = count
            # Early exit: if group count starts declining, stop.
            if d > 2 and count < best_count_at_depth:
                break
        return best_depth

    @staticmethod
    def _build_groups_at_depth(
        comp_paths: list[tuple[str, list[str]]],
        depth: int,
    ) -> dict[str, list[str]]:
        """Step 3: group components by path prefix of the cut depth."""
        groups: dict[str, list[str]] = {}  # prefix_key → [comp_name, ...]
        for comp_name, segs in comp_paths:
            prefix_key = "/".join(segs[:depth]) if len(segs) >= depth else "/".join(segs)
            groups.setdefault(prefix_key, []).append(comp_name)
        return groups

    @staticmethod
    def _merge_smallest_groups(
        groups: dict[str, list[str]],
        max_groups: int,
    ) -> dict[str, list[str]]:
        """Step 4: merge smallest groups until ≤ max_groups (segment Jaccard)."""
        while len(groups) > max_groups:
            keys = sorted(groups.keys(), key=lambda k: len(groups[k]))
            smallest_key = keys[0]
            # Find most similar neighbor by shared path segments.
            smallest_segs = set(smallest_key.split("/"))
            best_neighbor = keys[1]
            best_sim = -1.0
            for other_key in keys[1:]:
                other_segs = set(other_key.split("/"))
                inter = len(smallest_segs & other_segs)
                union = len(smallest_segs | other_segs)
                sim = inter / max(union, 1)
                if sim > best_sim:
                    best_sim = sim
                    best_neighbor = other_key
            # Merge smallest into neighbor.
            merged_key = f"{smallest_key}|{best_neighbor}"
            groups[merged_key] = groups.pop(smallest_key) + groups.pop(best_neighbor)
        return groups

    @staticmethod
    def _rules_for_path_group(
        group_key: str,
        all_group_keys: list[str],
        existing_rules: list[MappingRule],
    ) -> list[MappingRule]:
        """Step 5: generate MappingRule(s) for one (possibly merged) group.

        Single-path groups get one rule with the most distinguishing path
        segment; merged groups get one rule per sub_key, all pointing to a
        shared skill_name derived from the common ancestor segment.
        """
        sub_keys = group_key.split("|")

        # Collect all segments from OTHER groups for uniqueness check.
        other_segments: set[str] = set()
        for ok in all_group_keys:
            if ok != group_key:
                for part in ok.split("|"):
                    other_segments.update(part.split("/"))

        if len(sub_keys) == 1:
            best_pattern = _best_distinguishing_pattern(sub_keys, other_segments)
            skill_name = best_pattern.replace("-", "_").replace(" ", "_").lower()
            return [
                MappingRule(
                    method_pattern=best_pattern,
                    tool_name="",
                    skill_name=skill_name,
                    description=f"Auto-grouped from path: {group_key}",
                    match_type=MatchType.SUBSTRING,
                    match_target=MatchTarget.FILE_PATH,
                )
            ]

        skill_name = SkillGrouper._merged_skill_name(sub_keys, other_segments, existing_rules)
        shared_desc = "Auto-grouped from paths: " + ", ".join(sub_keys)
        return [
            MappingRule(
                method_pattern=_best_distinguishing_pattern([sk], other_segments),
                tool_name="",
                skill_name=skill_name,
                description=shared_desc,
                match_type=MatchType.SUBSTRING,
                match_target=MatchTarget.FILE_PATH,
            )
            for sk in sub_keys
        ]

    @staticmethod
    def _merged_skill_name(
        sub_keys: list[str],
        other_segments: set[str],
        existing_rules: list[MappingRule],
    ) -> str:
        """Naming strategy for merged groups.

        1. Prefer the **common ancestor segment** — the deepest path segment
           shared by ALL sub_keys (e.g. "examples" for examples/xxx paths).
        2. If no common ancestor exists, fall back to the most distinguishing
           segment of the deepest sub_key.
        """
        ancestor = _common_ancestor_segment(sub_keys, other_segments)
        if ancestor:
            skill_name = ancestor.replace("-", "_").replace(" ", "_").lower()
        else:
            main_key = max(sub_keys, key=lambda k: len(k.split("/")))
            fallback_pat = _best_distinguishing_pattern([main_key], other_segments)
            skill_name = fallback_pat.replace("-", "_").replace(" ", "_").lower()

        used_names = {r.skill_name for r in existing_rules}
        if skill_name in used_names:
            segs = [s for s in sub_keys[0].split("/") if s]
            skill_name = (
                "_".join(segs[-2:]).replace("-", "_").replace(" ", "_").lower() if len(segs) >= 2 else f"{skill_name}_2"
            )

        # Tag merged groups so users can distinguish them from
        # single-path groups at a glance.
        return skill_name + "_merged"

    def _keyword_match_group(self, rules: list[MappingRule] | None = None) -> list[SkillSpec]:
        """Group SourceComponent operations by MappingRule + auto prefix inference.

        Phase 1 — Explicit rule matching (first-match-wins, respects match_type
                  and match_target).  Rules with match_target=COMP_NAME or
                  FILE_PATH match against component-level fields, so all
                  operations in a matched component are grouped together.
        Phase 2 — Auto prefix inference for unmatched operations:
            a. Extract first underscore prefix (``video_encode`` → ``"video"``)
            b. Single-segment names (no underscore) → ``"utility"``
            c. Prefix groups with ≥ 2 members → ``"{prefix}_ops"`` Skill
            d. Prefix groups with 1 member → ``"misc"`` Skill
        Phase 3 — Merge smallest groups until count ≤ max_io_groups.

        ``rules`` defaults to ``self._rules``; the LLM path passes its
        generated FILE_PATH rules explicitly (stored in ``self._llm_rules``)
        so the shared ``self._rules`` is never polluted.
        """
        if not self._source_components:
            return []

        active_rules = rules if rules is not None else self._rules

        # Build flat item list from SourceComponents, now including file_path.
        items: list[tuple[str, str, str, str, str | None]] = []
        for comp in self._source_components:
            for op in comp.operations:
                items.append((op.name, op.description, comp.name, comp.file_path, op.class_name))

        skill_map: dict[str, SkillSpec] = {}
        matched_keys: set[str] = set()  # "comp_name.op_name" to avoid duplicates
        self._match_explicit_rules(items, skill_map, matched_keys, active_rules)

        # Phase 2 — auto prefix inference for unmatched operations.
        unmatched = [(n, c, cl) for n, _, c, _, cl in items if self._qualified_tool_name(c, cl, n) not in matched_keys]
        self._infer_prefix_groups(skill_map, unmatched)

        # Phase 3 — merge smallest groups until count ≤ max_io_groups.
        self._merge_to_budget(skill_map, self._max_io_groups)
        return list(skill_map.values())

    @staticmethod
    def _qualified_tool_name(comp_name: str, class_name: str | None, op_name: str) -> str:
        """``compName.className.opName`` (or ``compName.opName`` without a class)."""
        return f"{comp_name}.{class_name}.{op_name}" if class_name else f"{comp_name}.{op_name}"

    def _match_explicit_rules(
        self,
        items: list[tuple[str, str, str, str, str | None]],
        skill_map: dict[str, SkillSpec],
        matched_keys: set[str],
        rules: list[MappingRule],
    ) -> None:
        """Phase 1 — explicit MappingRule matching (MatchType + MatchTarget aware)."""
        for op_name, _, comp_name, file_path, class_name in items:
            for rule in rules:
                if rule.match_against(op_name, comp_name, file_path):
                    if rule.skill_name not in skill_map:
                        skill_map[rule.skill_name] = SkillSpec(
                            name=rule.skill_name,
                            description=rule.description or f"Skill: {rule.skill_name}",
                            allowed_tools=[],
                        )
                    qualified = self._qualified_tool_name(comp_name, class_name, op_name)
                    if qualified not in skill_map[rule.skill_name].allowed_tools:
                        skill_map[rule.skill_name].allowed_tools.append(qualified)
                    matched_keys.add(qualified)
                    break

    def _infer_prefix_groups(
        self,
        skill_map: dict[str, SkillSpec],
        unmatched: list[tuple[str, str, str | None]],
    ) -> None:
        """Phase 2 — auto prefix inference for unmatched operations."""
        prefix_groups: dict[str, list[tuple[str, str, str | None]]] = {}
        single_segment: list[tuple[str, str, str | None]] = []

        for name, comp_name, class_name in unmatched:
            prefix = name.split("_")[0] if "_" in name else ""
            if prefix:
                prefix_groups.setdefault(prefix, []).append((name, comp_name, class_name))
            else:
                # No underscore, or empty prefix (_private / __dunder).
                single_segment.append((name, comp_name, class_name))

        for prefix, members in prefix_groups.items():
            if len(members) >= 2:
                self._assign_prefix_skill(skill_map, prefix, members)
            else:
                # Lone prefix groups → "misc".
                self._assign_misc_skill(skill_map, members)

        if single_segment:
            self._assign_utility_skill(skill_map, single_segment)

    def _assign_prefix_skill(
        self,
        skill_map: dict[str, SkillSpec],
        prefix: str,
        members: list[tuple[str, str, str | None]],
    ) -> None:
        """Prefix groups with ≥ 2 members → "{prefix}_ops" Skill."""
        skill_name = f"{prefix}_ops"
        if skill_name not in skill_map:
            skill_map[skill_name] = SkillSpec(
                name=skill_name,
                description=f"Auto-grouped operations with prefix '{prefix}'",
                allowed_tools=[],
            )
        for name, comp_name, class_name in members:
            qualified = self._qualified_tool_name(comp_name, class_name, name)
            if qualified not in skill_map[skill_name].allowed_tools:
                skill_map[skill_name].allowed_tools.append(qualified)

    def _assign_misc_skill(
        self,
        skill_map: dict[str, SkillSpec],
        members: list[tuple[str, str, str | None]],
    ) -> None:
        """Lone prefix groups → "misc" Skill."""
        if "misc" not in skill_map:
            skill_map["misc"] = SkillSpec(
                name="misc",
                description="Miscellaneous operations (small prefix groups)",
                allowed_tools=[],
            )
        for name, comp_name, class_name in members:
            qualified = self._qualified_tool_name(comp_name, class_name, name)
            if qualified not in skill_map["misc"].allowed_tools:
                skill_map["misc"].allowed_tools.append(qualified)

    def _assign_utility_skill(
        self,
        skill_map: dict[str, SkillSpec],
        single_segment: list[tuple[str, str, str | None]],
    ) -> None:
        """Single-segment names → "utility" Skill."""
        skill_map["utility"] = SkillSpec(
            name="utility",
            description="Utility operations (single-segment names)",
            allowed_tools=[],
        )
        for name, comp_name, class_name in single_segment:
            qualified = self._qualified_tool_name(comp_name, class_name, name)
            if qualified not in skill_map["utility"].allowed_tools:
                skill_map["utility"].allowed_tools.append(qualified)

    @staticmethod
    def _merge_to_budget(skill_map: dict[str, SkillSpec], max_groups: int) -> None:
        """Phase 3 — merge smallest groups until count ≤ max_groups (Jaccard)."""
        while len(skill_map) > max_groups:
            keys = list(skill_map.keys())
            smallest_key = min(keys, key=lambda k: len(skill_map[k].allowed_tools))
            smallest_prefixes = _extract_prefixes(skill_map[smallest_key].allowed_tools)
            best_neighbor = smallest_key
            best_sim = -1.0
            for other_key in keys:
                if other_key == smallest_key:
                    continue
                other_prefixes = _extract_prefixes(skill_map[other_key].allowed_tools)
                inter = len(smallest_prefixes & other_prefixes)
                union = len(smallest_prefixes | other_prefixes)
                sim = inter / max(union, 1)
                if sim > best_sim:
                    best_sim = sim
                    best_neighbor = other_key
            # Fallback: merge into next smallest.
            if best_neighbor == smallest_key:
                best_neighbor = next(k for k in keys if k != smallest_key)
            skill_map[best_neighbor].allowed_tools.extend(skill_map[smallest_key].allowed_tools)
            del skill_map[smallest_key]

    def _component_group(self) -> list[SkillSpec]:
        """Group by SourceComponent. Each component becomes a SkillSpec.
        Operations map to allowed_tools.
        """
        if not self._source_components:
            return self._static_group()

        skills: list[SkillSpec] = []
        for component in self._source_components:
            tools = [
                f"{component.name}.{op.class_name}.{op.name}" if op.class_name else f"{component.name}.{op.name}"
                for op in component.operations
            ]
            skills.append(
                SkillSpec(
                    name=component.name,
                    description=component.description or f"Component: {component.name}",
                    allowed_tools=tools,
                )
            )
        return skills

    @staticmethod
    def _sanitize_type_name(raw: str) -> str:
        """Simplify a complex type annotation into a readable slug.

        Rules:
          - ``Annotated[X, ...]``  → extract the inner type ``X``
          - ``Literal[X]`` → ``literal``
          - ``list[X]`` / ``List[X]`` → ``list`` (generic params stripped)
          - ``dict[K, V]`` / ``Dict[K, V]`` → ``dict`` (generic params stripped)
          - ``Optional[X]`` → ``X``
          - ``Union[A, B]`` / ``A | B`` → ``A_B``
          - Other angle-bracket generics → bare name (params stripped)
          - Remove all characters unsafe for filenames
          - Lowercase the result.
        """
        s = raw.strip()

        m = re.match(r"^Literal\[", s)
        if m:
            return "literal"

        m = re.match(r"^Annotated\[(.+?),\s", s)
        if m:
            s = m.group(1).strip()

        m = re.match(r"^Optional\[(.+)\]$", s)
        if m:
            s = m.group(1).strip()

        if "|" in s:
            parts = [p.strip() for p in s.split("|")]
            parts = [re.sub(r"['\"]", "", p) for p in parts]
            s = "_".join(parts)

        m = re.match(r"^Union\[(.+)\]$", s)
        if m:
            parts = [p.strip() for p in m.group(1).split(",")]
            parts = [re.sub(r"['\"]", "", p) for p in parts]
            s = "_".join(parts)

        generic_set = {"list", "List", "dict", "Dict", "set", "Set", "tuple", "Tuple"}
        is_generic = False
        for prefix in generic_set:
            m = re.match(rf"^{prefix}\[.+]$", s)
            if m:
                # Strip generic parameters: dict[str,any] → dict
                s = prefix.lower()
                is_generic = True
                break

        if not is_generic:
            s = re.sub(r"[<\(\[{].*", "", s)

        s = s.replace(".", "_")
        s = re.sub(r"[|\\/:*?\"<>']", "", s)

        s = s.lower()
        return s

    def _io_relation_group(self) -> list[SkillSpec]:
        """Group operations by shared parameter types using type-anchor clustering.

        Tool naming under IO_RELATION:
            Class methods  → ``ClassName.methodName`` (e.g. ``VideoCodec.encode``)
            Top-level func → ``compName.fileStem.methodName`` when ``file_stem``
                             is present (disambiguates same-name functions across
                             files), otherwise ``compName.methodName`` (e.g.
                             ``MyModule.utils.load_config``).

        Phase 1 — Type frequency analysis:
            Count how many operations use each parameter type, then select the
            top-K most frequent types as "anchor types" (K = max_io_groups).

        Phase 2 — Anchor-set-intersection grouping:
            Each operation is assigned to the bucket defined by the exact set of
            its parameter types that intersect with the anchor set.  Operations
            with identical anchor-type combinations always go to the same bucket.
            Operations with no anchor types or no typed parameters are collected
            into a ``utility`` list, which is then dissolved round-robin across
            existing typed buckets.  If no typed buckets exist at all, utility
            ops remain as a single ``utility`` group.

        Phase 3 — Smallest-first merge:
            If the number of groups still exceeds ``max_io_groups``, the
            **smallest** bucket is merged with its most similar neighbor
            (highest Jaccard similarity between type-sets).  This preserves
            large, usefully-distinct buckets while consolidating tiny ones.
            Merge iterates until the target count is met.

        Phase 4 — Readable naming:
            Each group is named after its top 2-3 dominant types, e.g.
            ``io_group_str_path``.  A numeric suffix is appended when
            duplicate names arise.
        """
        if not self._source_components:
            return self._static_group()

        all_ops = self._collect_typed_ops()
        if not all_ops:
            return []

        anchor_types = self._pick_anchor_types(all_ops, self._max_io_groups)
        anchor_set = set(anchor_types)
        bucket_map, utility_ops = self._bucket_by_anchors(all_ops, anchor_set)
        self._dissolve_utility(bucket_map, utility_ops)
        bucket_type_sets = self._bucket_type_sets(bucket_map)
        self._merge_smallest_buckets(bucket_map, bucket_type_sets, self._max_io_groups)
        return self._name_buckets(bucket_map)

    def _collect_typed_ops(self) -> list[tuple[SourceOperation, set[str], str]]:
        """Phase 1 prep — collect (op, sanitized-type-set, comp_name) tuples."""
        all_ops: list[tuple[SourceOperation, set[str], str]] = []
        for component in self._source_components:
            for op in component.operations:
                type_set = (
                    {self._sanitize_type_name(p.type_hint) for p in op.parameters if p.type_hint}
                    if op.parameters
                    else set()
                )
                all_ops.append((op, type_set, component.name))
        return all_ops

    @staticmethod
    def _pick_anchor_types(
        all_ops: list[tuple[SourceOperation, set[str], str]],
        max_groups: int,
    ) -> list[str]:
        """Phase 1 — top-K most frequent parameter types as anchor types."""
        type_freq = Counter()
        for _, ts, _ in all_ops:
            type_freq.update(ts)
        return [t for t, _ in type_freq.most_common(max_groups)]

    @staticmethod
    def _bucket_by_anchors(
        all_ops: list[tuple[SourceOperation, set[str], str]],
        anchor_set: set[str],
    ) -> tuple[
        dict[tuple[str, ...], list[tuple[SourceOperation, set[str], str]]],
        list[tuple[SourceOperation, set[str], str]],
    ]:
        """Phase 2 — bucket ops by anchor-type-set intersection."""
        bucket_map: dict[tuple[str, ...], list[tuple[SourceOperation, set[str], str]]] = {}
        utility_ops: list[tuple[SourceOperation, set[str], str]] = []
        for op, ts, comp_name in all_ops:
            if not ts:
                utility_ops.append((op, ts, comp_name))
                continue
            # Operations with identical anchor-type combinations always go to
            # the same bucket — no op is scattered to an unrelated bucket.
            present_anchors = tuple(sorted(t for t in ts if t in anchor_set))
            if present_anchors:
                bucket_map.setdefault(present_anchors, []).append((op, ts, comp_name))
            else:
                utility_ops.append((op, ts, comp_name))
        return bucket_map, utility_ops

    @staticmethod
    def _dissolve_utility(
        bucket_map: dict[tuple[str, ...], list[tuple[SourceOperation, set[str], str]]],
        utility_ops: list[tuple[SourceOperation, set[str], str]],
    ) -> None:
        """Phase 2 — distribute untyped ops round-robin across typed buckets."""
        if not utility_ops:
            return
        existing_keys = list(bucket_map.keys())
        if existing_keys:
            for idx, op_tuple in enumerate(utility_ops):
                bucket_map[existing_keys[idx % len(existing_keys)]].append(op_tuple)
        else:
            # No typed buckets exist at all — keep as a single utility group.
            bucket_map[("utility",)] = utility_ops

    @staticmethod
    def _bucket_type_sets(
        bucket_map: dict[tuple[str, ...], list[tuple[SourceOperation, set[str], str]]],
    ) -> dict[tuple[str, ...], set[str]]:
        """Combined type set per bucket, used for Jaccard merge similarity."""
        bucket_type_sets: dict[tuple[str, ...], set[str]] = {}
        for anchor, items in bucket_map.items():
            combined = set()
            for _, ts, _ in items:
                combined |= ts
            bucket_type_sets[anchor] = combined
        return bucket_type_sets

    @staticmethod
    def _merge_smallest_buckets(
        bucket_map: dict[tuple[str, ...], list[tuple[SourceOperation, set[str], str]]],
        bucket_type_sets: dict[tuple[str, ...], set[str]],
        max_groups: int,
    ) -> None:
        """Phase 3 — smallest-first merge until bucket count ≤ max_groups.

        Merging the SMALLEST bucket first preserves large, usefully-distinct
        buckets (e.g. str, any, int) while consolidating tiny ones.
        """
        while len(bucket_map) > max_groups:
            keys = list(bucket_map.keys())
            smallest_key = min(keys, key=lambda k: len(bucket_map[k]))
            best_neighbor = smallest_key
            best_sim = -1.0
            for other_key in keys:
                if other_key == smallest_key:
                    continue
                inter = len(bucket_type_sets[smallest_key] & bucket_type_sets[other_key])
                union = len(bucket_type_sets[smallest_key] | bucket_type_sets[other_key])
                sim = inter / max(union, 1)
                if sim > best_sim:
                    best_neighbor = other_key
                    best_sim = sim
            # Fallback: if no similarity data, merge into first other bucket.
            if best_neighbor == smallest_key:
                best_neighbor = next(k for k in keys if k != smallest_key)
            bucket_map[best_neighbor].extend(bucket_map[smallest_key])
            bucket_type_sets[best_neighbor] = bucket_type_sets[best_neighbor] | bucket_type_sets[smallest_key]
            del bucket_map[smallest_key]
            del bucket_type_sets[smallest_key]

    def _name_buckets(
        self,
        bucket_map: dict[tuple[str, ...], list[tuple[SourceOperation, set[str], str]]],
    ) -> list[SkillSpec]:
        """Phase 4 — readable naming by top 2-3 dominant types."""
        skills: list[SkillSpec] = []
        name_counts: Counter = Counter()
        for _, items in bucket_map.items():
            ops_with_comp = [(op, comp_name) for op, _, comp_name in items]
            freq = Counter()
            for _, ts, _ in items:
                freq.update(ts)

            dom_types = [t for t, _ in freq.most_common(3)]
            base_slug = "_".join(dom_types) if dom_types else "utility"
            base_name = f"io_group_{base_slug[:80]}"

            name_counts[base_name] += 1
            skill_name = f"{base_name}_{name_counts[base_name]}" if name_counts[base_name] > 1 else base_name

            type_desc = ", ".join(dom_types) if dom_types else "untyped"
            skills.append(
                SkillSpec(
                    name=skill_name,
                    description=f"Operations sharing types: {type_desc}",
                    allowed_tools=[
                        (
                            f"{op.class_name}.{op.name}"
                            if op.class_name
                            else f"{comp_name}.{op.file_stem}.{op.name}"
                            if op.file_stem
                            else f"{comp_name}.{op.name}"
                        )
                        for op, comp_name in ops_with_comp
                    ],
                )
            )
        return skills

    _LLM_SYSTEM_PROMPT = (
        "你是 SDK 工具分组专家。根据源码目录路径和方法数量，"
        "将目录聚类为 Skill 组。每组 Skill 对应一个子 Agent。\n\n"
        "约束：\n"
        "- 分组数量不超过 {max_groups} 个\n"
        "- 每个目录必须属于恰好一个 Skill\n"
        "- 同一父目录下的子目录通常应归入同一个 Skill\n"
        "- Skill 名称需简洁、有业务含义（如 core_engine, agent_lifecycle），"
        "不要用 io_group_x 这种机械名称\n"
        "- patterns 是目录路径的匹配模式（SUBSTRING 匹配），"
        "只需列出有区分力的片段即可，不要写完整路径\n\n"
        "输出格式（严格 JSON，不要输出任何其他内容）：\n"
        '{{"skills": [{{"name": "...", "description": "...", '
        '"patterns": ["path_fragment1", "path_fragment2"]}}]}}'
    )
