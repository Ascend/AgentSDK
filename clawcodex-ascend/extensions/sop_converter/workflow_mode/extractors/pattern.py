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

"""
PatternExtractor — config-driven custom Pipeline extractor

====================================================
  The official, reusable workflow extractor for the SOP converter.
====================================================

Design goals
------------
Provide a configurable WorkflowExtractorBase subclass for any SDK/project
that parses the project's pipeline definitions (stage enums, state
transitions, gates, decisions, contracts) into a WorkflowGraph IR.

Key differences from ArcExtractor
--------------------------
The old ArcExtractor hardcoded all of AutoResearchClaw's conventions in the class:
  - Paths: ``researchclaw/pipeline/``
  - Variable names: ``STAGE_SEQUENCE``, ``NEXT_STAGE``, ``DECISION_ROLLBACK``, etc.
  - Enum member names: ``RESEARCH_DECISION``
  - No caching: every extraction method parsed files independently

This implementation parameterizes all SDK conventions via ``PipelineConfig``
and reuses AST parsing results through ``SourceScanContext`` for any project.

Extension points
----------------
1. Replace ``SourceScanContext`` → support non-Python projects (YAML / TOML / JSON)
2. Add new extraction methods → override the abstract methods of ``WorkflowExtractorBase``
3. Customize fallback strategies → modify the ``_fallback_stages_*`` methods
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from extensions.sop_converter.workflow_mode.ast_helpers import (
    extract_docstring_first_para,
    find_dict_mapping_assignments,
    find_enum_classes,
    find_gate_assigns,
    get_enum_members_ordered,
    parse_ast,
    parse_contracts_dict,
    parse_enum_dict_mapping_from_expr,
    parse_enum_to_name_dict,
    parse_frozenset_members,
    parse_stage_sequence_from_expr,
    to_kebab,
    parse_string_to_stage_dict,
)
from extensions.sop_converter.workflow_mode.extractors.base import (
    WorkflowExtractorBase,
)
from extensions.sop_converter.workflow_mode.extractors.models import (
    DecisionSpec,
    ExtractedStage,
    GateSpec,
    OutcomeSpec,
    StageContract,
    Transition,
)
from extensions.sop_converter.workflow_mode.scan_context import SourceScanContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PipelineConfig — parameterizes all SDK-specific conventions
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Describes the configuration of an SDK/project's pipeline conventions.

    Through this configuration the user tells the extractor:
    - where to find pipeline definition files
    - what pattern the variable names follow
    - what the specific enum members are called

    Attribute notes:
        name: config name (for logging/debugging)
        description: config description

        pipeline_marker_files: the list of files that identify a pipeline directory.
            Each element is ``(filename, directory_relative_glob)``.
            For example ``("stages.py", "pipeline")`` means ``{source_dir}/pipeline/stages.py``.
            The extractor checks them in order; the first matching set determines the pipeline directory.

        executor_table_patterns: list of variable names for the executor mapping table.
            For example ``["STAGE_EXECUTORS", "_STAGE_EXECUTORS"]``.

        sequence_var_pattern: name of the stage-sequence variable (exact match).
            For example ``"STAGE_SEQUENCE"``.

        transition_var_pattern: regex pattern for transition variables.
            Dicts whose variable name matches this pattern are treated as inter-stage transitions.
            For example ``"NEXT_STAGE|PREVIOUS_STAGE"``.

        gate_var_pattern: regex pattern for gate variables.
            Frozensets whose variable name matches this pattern are treated as gate definitions.
            For example ``"GATE"``.

        decision_var_pattern: regex pattern for decision variables.
            Dicts whose variable name matches this pattern are treated as decision/rollback definitions.
            For example ``"DECISION_ROLLBACK"``.

        contract_var_pattern: regex pattern for contract variables.
            Dicts whose variable name matches this pattern are treated as contract definitions.
            For example ``"CONTRACT"``.

        decision_stage_names: fallback list of decision-stage names.
            Checked in order; the first matching enum member is treated as the decision stage.
            For example ``["RESEARCH_DECISION", "DECISION"]``.
    """

    name: str = "default"
    description: str = ""

    # Pipeline directory discovery
    pipeline_marker_files: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("stages.py", "pipeline"),
            ("contracts.py", "pipeline"),
        ]
    )

    # Variable name patterns
    executor_table_patterns: list[str] = field(
        default_factory=lambda: [
            "STAGE_EXECUTORS",
            "_STAGE_EXECUTORS",
        ]
    )
    sequence_var_pattern: str = "STAGE_SEQUENCE"
    transition_var_pattern: str = "NEXT_STAGE|PREVIOUS_STAGE"
    gate_var_pattern: str = "GATE"
    decision_var_pattern: str = "DECISION_ROLLBACK"
    contract_var_pattern: str = "CONTRACT"

    # Decision stage discovery
    decision_stage_names: list[str] = field(
        default_factory=lambda: [
            "RESEARCH_DECISION",
            "DECISION",
        ]
    )

    # Stage enum suffixes (used for heuristic discovery)
    stage_enum_suffixes: list[str] = field(
        default_factory=lambda: [
            "Stage",
            "Step",
        ]
    )


# ---------------------------------------------------------------------------
# Preset configs — ready-to-use SDK configs
# ---------------------------------------------------------------------------

# Old AutoResearchClaw-style config (for reference / backward compatibility)
ARC_COMPAT_CONFIG = PipelineConfig(
    name="arc-compat",
    description="AutoResearchClaw 兼容配置（参考用）",
    pipeline_marker_files=[
        ("stages.py", "researchclaw/pipeline"),
        ("contracts.py", "researchclaw/pipeline"),
        ("stages.py", "pipeline"),
        ("contracts.py", "pipeline"),
    ],
    executor_table_patterns=["_STAGE_EXECUTORS", "STAGE_EXECUTORS"],
    sequence_var_pattern="STAGE_SEQUENCE",
    transition_var_pattern="NEXT_STAGE|PREVIOUS_STAGE",
    gate_var_pattern="GATE",
    decision_var_pattern="DECISION_ROLLBACK",
    contract_var_pattern="CONTRACT",
    decision_stage_names=["RESEARCH_DECISION", "DECISION"],
)


# ---------------------------------------------------------------------------
# Helper functions — pipeline directory discovery
# ---------------------------------------------------------------------------


def _resolve_pipeline_dir(source_dir: str | Path, config: PipelineConfig) -> Path | None:
    """Locate the pipeline directory according to *config.pipeline_marker_files*.

    For each ``(filename, relative_glob)`` pair, look for
    ``relative_glob/filename`` under ``source_dir``. ``relative_glob`` may contain
    path separators, e.g. ``"researchclaw/pipeline"``.

    If all the marker files exist, return that directory; otherwise keep checking the next pair.
    """
    root = Path(source_dir).resolve()
    for filename, rel_glob in config.pipeline_marker_files:
        candidate = root / rel_glob
        if (candidate / filename).is_file():
            return candidate
    # Fallback: recursive search (slow, but covers projects with non-standard layouts)
    for stages_py in root.rglob("stages.py"):
        parent = stages_py.parent
        if (parent / "contracts.py").is_file():
            return parent
    return None


def resolve_pipeline_dir(source_dir: str | Path) -> Path | None:
    """Locate the pipeline directory using the ARC-compatible default config.

    Convenience wrapper around :func:`_resolve_pipeline_dir` for callers that
    need the legacy AutoResearchClaw layout (see ``ARC_COMPAT_CONFIG``).
    """
    return _resolve_pipeline_dir(source_dir, ARC_COMPAT_CONFIG)


# ---------------------------------------------------------------------------
# PatternExtractor — the main extractor class
# ---------------------------------------------------------------------------


class PatternExtractor(WorkflowExtractorBase):
    """Config-driven Pipeline extractor.

    Accepts a ``PipelineConfig`` describing the SDK conventions, then extracts
    workflow definitions (stages, transitions, gates, decisions, contracts) from the project directory.

    Usage::

        config = PipelineConfig(
            name="my-sdk",
            pipeline_marker_files=[("stages.py", "pipeline")],
        )
        ext = PatternExtractor(config=config, mode="fwa")
        graph = ext.extract("/path/to/project")
    """

    # Default config (used when the user passes none)
    _DEFAULT_CONFIG: ClassVar[PipelineConfig] = PipelineConfig()

    def __init__(
        self,
        *args,
        config: PipelineConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._config: PipelineConfig = config or self._DEFAULT_CONFIG
        self._pipeline_dir: Path | None = None
        self._enum_class: str | None = None
        self._member_to_value: dict[str, int] = {}
        self._member_order: list[tuple[str, int]] = []
        self._stage_sequence: list[int] = []
        self._executor_rel: str | None = None
        self._executor_by_stage: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Initialization and caching
    # ------------------------------------------------------------------

    def _ensure_scan(self, source_dir: Path) -> None:
        """Ensure the SourceScanContext has been built (lazy; repeated calls do not re-parse)."""
        if self._scan is None:  # pylint: disable=access-member-before-definition
            self._scan = SourceScanContext.build(source_dir)  # pylint: disable=attribute-defined-outside-init
        self._populate_enum_info(source_dir)

    def _populate_enum_info(self, source_dir: Path) -> None:
        """Resolve the stage enum, sequence, and executor mapping from the pipeline directory.

        This is the extractor's core initialization step:
        1. Locate the pipeline directory (via marker files)
        2. Resolve the stage enum class (member name → ID mapping)
        3. Resolve STAGE_SEQUENCE (stage execution order)
        4. Resolve the executor mapping table (stage → executor function)

        This method fails safely: if the pipeline directory is missing or files
        are incomplete, it sets ``_pipeline_dir = None``; later extraction methods return empty values.
        """
        pipeline = _resolve_pipeline_dir(source_dir, self._config)
        if pipeline is None:
            return
        self._pipeline_dir = pipeline

        stages_tree = parse_ast(pipeline / "stages.py")
        if stages_tree is None:
            return

        # Take the first enum class as the stage enum
        for cls in find_enum_classes(stages_tree):
            members = get_enum_members_ordered(cls)
            if not members:
                continue
            self._enum_class = cls.name
            self._member_order = members
            self._member_to_value = dict(members)
            break

        if not self._member_to_value:
            return

        enum_names = {self._enum_class} if self._enum_class else set()
        module_assigns = dict(find_dict_mapping_assignments(stages_tree))

        # Stage sequence
        seq_var = self._config.sequence_var_pattern
        if seq_var in module_assigns:
            self._stage_sequence = parse_stage_sequence_from_expr(
                module_assigns[seq_var],
                enum_class_names=enum_names,
                enum_members_ordered=self._member_order,
                module_assigns=module_assigns,
            )
        if not self._stage_sequence:
            self._stage_sequence = [v for _, v in self._member_order]

        # Executor mapping table
        executor_path = pipeline / "executor.py"
        if executor_path.is_file():
            self._executor_rel = executor_path.relative_to(Path(source_dir).resolve()).as_posix()
            exec_tree = parse_ast(executor_path)
            if exec_tree is not None:
                for var_name, dict_expr in find_dict_mapping_assignments(exec_tree):
                    if var_name in self._config.executor_table_patterns:
                        self._executor_by_stage = parse_enum_to_name_dict(
                            dict_expr,
                            enum_names,
                            self._member_to_value,
                        )

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def extract_stages(self, source_dir: Path) -> list[ExtractedStage]:
        """Extract the list of stages.

        Strategy:
        1. Precise extraction: parse the enum class from ``stages.py`` in the pipeline directory
        2. Directory inference: infer stages from the directory structure (see ``_fallback_stages_from_directory``)
        3. Coarse-grained scan: infer from file contents only when ``allow_coarse=True``

        Extension point: override the ``_fallback_stages_from_*`` methods to support custom fallback strategies.
        """
        self._ensure_scan(source_dir)
        if self._pipeline_dir is None:
            return self._fallback_stages(source_dir)

        pipeline = self._pipeline_dir
        stages_tree = parse_ast(pipeline / "stages.py")
        stage_enum_cls = None
        if stages_tree:
            for cls in find_enum_classes(stages_tree):
                if cls.name == self._enum_class:
                    stage_enum_cls = cls
                    break

        desc = extract_docstring_first_para(stage_enum_cls) if stage_enum_cls else ""
        stages: list[ExtractedStage] = []
        for name, value in self._member_order:
            entry_fn = self._executor_by_stage.get(value)
            stages.append(
                ExtractedStage(
                    id=value,
                    name=to_kebab(name),
                    label=name,
                    source_class=self._enum_class,
                    source_value=value,
                    file_path=self._executor_rel,
                    entry_function=entry_fn,
                    description=desc,
                )
            )
        return stages

    def extract_transitions(self, source_dir: Path) -> list[Transition]:
        """Extract inter-stage transition relations.

        Look up dict mappings matching ``transition_var_pattern`` in ``stages.py``
        and parse them into ``(from_stage, to_stage)`` pairs.

        If no explicit transition definitions are found, derive a linear order from ``STAGE_SEQUENCE``.
        """
        self._ensure_scan(source_dir)
        if self._pipeline_dir is None or not self._member_to_value:
            return []

        pipeline = self._pipeline_dir
        stages_tree = parse_ast(pipeline / "stages.py")
        if stages_tree is None:
            return []

        enum_names = {self._enum_class} if self._enum_class else set()
        transitions: list[Transition] = []
        seen: set[tuple[int, int]] = set()
        trans_pattern = re.compile(self._config.transition_var_pattern, re.IGNORECASE)

        for var_name, dict_expr in find_dict_mapping_assignments(stages_tree):
            if trans_pattern.search(var_name):
                pairs = parse_enum_dict_mapping_from_expr(
                    dict_expr,
                    enum_names,
                    self._member_to_value,
                    stage_sequence=self._stage_sequence,
                )
                for from_id, to_id in pairs:
                    key = (from_id, to_id)
                    if key not in seen:
                        seen.add(key)
                        transitions.append(
                            Transition(
                                from_stage=from_id,
                                to_stage=to_id,
                                condition=var_name,
                                is_default=True,
                            )
                        )

        # Fallback: derive a linear order from STAGE_SEQUENCE
        if not transitions and self._stage_sequence:
            for idx, sid in enumerate(self._stage_sequence):
                if idx + 1 < len(self._stage_sequence):
                    nxt = self._stage_sequence[idx + 1]
                    transitions.append(
                        Transition(
                            from_stage=sid,
                            to_stage=nxt,
                            condition=self._config.sequence_var_pattern,
                            is_default=True,
                        )
                    )
        return transitions

    def extract_gates(self, source_dir: Path) -> dict[int, GateSpec]:
        """Extract gates (stages that require manual approval).

        Look up frozenset assignments matching ``gate_var_pattern`` in ``stages.py``
        and mark their members as gates.
        """
        self._ensure_scan(source_dir)
        if self._pipeline_dir is None or not self._member_to_value:
            return {}

        pipeline = self._pipeline_dir
        stages_tree = parse_ast(pipeline / "stages.py")
        if stages_tree is None:
            return {}

        gates: dict[int, GateSpec] = {}
        enum_names = {self._enum_class} if self._enum_class else set()
        gate_pattern = re.compile(self._config.gate_var_pattern, re.IGNORECASE)

        for var_name, value_expr in find_gate_assigns(stages_tree):
            if not gate_pattern.search(var_name):
                continue
            stage_ids = parse_frozenset_members(
                value_expr,
                enum_names,
                self._member_to_value,
            )
            for sid in stage_ids:
                gates[sid] = GateSpec(
                    stage_id=sid,
                    approval_mode="manual",
                    description=f"Gate from {var_name}",
                    source_name=var_name,
                )
        return gates

    def extract_decisions(self, source_dir: Path) -> dict[int, DecisionSpec]:
        """Extract decision nodes (branch/rollback logic).

        Find dict mappings matching ``decision_var_pattern`` in ``stages.py`` and
        parse them into rollback targets; decision stages come from ``decision_stage_names``.

        Extension point: if the SDK uses function definitions instead of dict
        mappings for decisions, override this method or add ``_fallback_decision_funcs()``.
        """
        self._ensure_scan(source_dir)
        if self._pipeline_dir is None or not self._member_to_value:
            return {}

        pipeline = self._pipeline_dir
        stages_tree = parse_ast(pipeline / "stages.py")
        if stages_tree is None:
            return {}

        enum_names = {self._enum_class} if self._enum_class else set()
        decisions: dict[int, DecisionSpec] = {}
        decision_pattern = re.compile(
            self._config.decision_var_pattern,
            re.IGNORECASE,
        )

        for var_name, dict_expr in find_dict_mapping_assignments(stages_tree):
            if not decision_pattern.search(var_name):
                continue
            rollback = parse_string_to_stage_dict(
                dict_expr,
                enum_names,
                self._member_to_value,
            )
            # Discover the decision stage: look it up in the order of the configured `decision_stage_names` list
            decision_stage = self._resolve_decision_stage()
            if decision_stage is None:
                continue
            outcomes = {name: OutcomeSpec(next_stage=sid, max_times=2) for name, sid in rollback.items()}
            decisions[decision_stage] = DecisionSpec(
                stage_id=decision_stage,
                outcomes=outcomes,
                source_func=var_name,
                inferred=False,
            )
        return decisions

    def extract_contracts(self, source_dir: Path) -> dict[int, StageContract]:
        """Extract stage contracts (input/output file definitions).

        Look up dict mappings matching ``contract_var_pattern`` in ``contracts.py``
        and parse them into ``StageContract``.
        """
        self._ensure_scan(source_dir)
        if self._pipeline_dir is None or not self._member_to_value:
            return {}

        pipeline = self._pipeline_dir
        contracts_path = pipeline / "contracts.py"
        tree = parse_ast(contracts_path)
        if tree is None:
            return {}

        enum_names = {self._enum_class} if self._enum_class else set()
        contracts: dict[int, StageContract] = {}
        contract_pattern = re.compile(
            self._config.contract_var_pattern,
            re.IGNORECASE,
        )

        for var_name, dict_expr in find_dict_mapping_assignments(tree):
            if not contract_pattern.search(var_name):
                continue
            parsed = parse_contracts_dict(
                dict_expr,
                enum_names,
                self._member_to_value,
            )
            for stage_id, (inp, out, call_name, dod) in parsed.items():
                contracts[stage_id] = StageContract(
                    stage_id=stage_id,
                    input_files=inp,
                    output_files=out,
                    dod=dod,
                    source_class=call_name,
                )
        return contracts

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _resolve_decision_stage(self) -> int | None:
        """Look up the decision stage ID in the order configured by ``decision_stage_names``.

        If not found, scan all members for one whose name contains "DECISION".
        """
        for name in self._config.decision_stage_names:
            sid = self._member_to_value.get(name)
            if sid is not None:
                return sid
        for member, value in self._member_to_value.items():
            if "DECISION" in member.upper():
                return value
        return None

    def _fallback_stages(self, source_dir: Path) -> list[ExtractedStage]:
        """Fallback: when precise extraction fails, try other strategies.

        Current implementation:
        1. Try ``_fallback_stages_from_directory`` — infer from the directory structure
        2. If ``allow_coarse=True``, try ``_fallback_stages_coarse`` —
           infer from file contents

        Extension point: add new fallback strategies, e.g. parsing from YAML/TOML config files.
        """
        stages = self._fallback_stages_from_directory(source_dir)
        if stages:
            return stages
        if self._allow_coarse:
            return self._fallback_stages_coarse(source_dir)
        return []

    def _fallback_stages_from_directory(
        self,
        source_dir: Path,
    ) -> list[ExtractedStage]:
        """Infer stages from the directory structure.

        Look for subdirectories whose names start with ``stage_`` or ``step_``;
        each such subdirectory is treated as a stage.
        """
        root = Path(source_dir).resolve()
        stages: list[ExtractedStage] = []
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            name = entry.name
            if not (name.startswith("stage_") or name.startswith("step_")):
                continue
            stages.append(
                ExtractedStage(
                    id=len(stages) + 1,
                    name=name,
                    label=name.replace("_", " ").title(),
                    inferred=True,
                    file_path=entry.relative_to(root).as_posix(),
                )
            )
        return stages

    def _fallback_stages_coarse(self, source_dir: Path) -> list[ExtractedStage]:
        """Coarse-grained inference: scan Python files for class/function definitions.

        Suitable for projects with few clues; results are less precise but never fail outright.
        """
        scan = self._scan or SourceScanContext.build(source_dir)
        stages: list[ExtractedStage] = []
        seen_names: set[str] = set()

        for tree in scan.trees.values():
            for cls in find_enum_classes(tree):
                for suffix in self._config.stage_enum_suffixes:
                    if cls.name.endswith(suffix) and cls.name not in seen_names:
                        seen_names.add(cls.name)
                        stages.append(
                            ExtractedStage(
                                id=len(stages) + 1,
                                name=to_kebab(cls.name),
                                label=cls.name,
                                source_class=cls.name,
                                inferred=True,
                                description=extract_docstring_first_para(cls),
                            )
                        )
        return stages
