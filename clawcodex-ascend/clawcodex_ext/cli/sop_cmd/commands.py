#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=fixme,too-many-lines,too-many-nested-blocks
"""Fast-path ``clawcodex-dev sop`` CLI commands.

Usage::

    clawcodex-dev sop convert <sdk_spec> [--out <output_dir>]
        [--requirements "<requirements>"] [--name <agent_name>]
        [--strategy <strategy>] [--skills <skills_dir>]
        [--max-groups <N>] [--mapping-rules <file>]
        [--llm-provider <provider>] [--llm-model <model>]
        [--preview] [--all] [--no-register-tools]

    clawcodex-dev sop convert docker_build,k8s_apply \\
        --out ./.clawcodex --requirements "CI/CD pipeline" --name cicd-agent

    # Source directory auto-detection (默认筛选外部接口，加 --all 包含全部方法):
    clawcodex-dev sop convert ./src \\
        --out ./.clawcodex --strategy component --skills ./skills

Options:

    --strategy <strategy>   Grouping strategy (keyword|component|io|llm).
                            Only used when <sdk_spec> is a directory.
    --skills <skills_dir>   Output path for generated skill files.
    --all                   Include all public methods (disable external
                            interface filtering).  Default: extern-only.
    --no-register-tools     Skip SDK method Tool registration (on by default for
                            source-directory converts).
    --macros-dir <dir>      Directory of handwritten macro YAML files
                            (globs *.yaml/*.yml). Use when macros live
                            outside <sdk_spec>/sop-macros/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from clawcodex_ext.cli.subcommand_registry import register

if TYPE_CHECKING:
    from extensions.sop_converter.skill_grouper import MappingRule


@dataclass
class ConvertOptions:
    sdk_spec: str
    output_dir: str = ""
    requirements: str = ""
    agent_name: str = ""
    strategy: str = ""
    skills_dir: str = ""
    max_groups: int = 0
    mapping_rules_file: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    preview: bool = False
    all_methods: bool = False
    register_tools: bool = True
    mode: str = "auto"
    extractor: str | None = None
    emit_workflow_yaml: bool = False
    emit_stage_agents: bool = False
    emit_bridge: bool = False
    bridge_mode: str = "python"
    bridge_cli: str = ""
    strict_workflow_yaml: bool = False
    json_output: bool = False
    validate_only: bool = False
    interactive: bool = False  # F-50.11.4: generate TODO templates on empty extraction
    # F-REC: optional .cast output path. When set, the convert call is
    # wrapped in a :class:`SopStageProjector` so the SOP CLI's stdout
    # (and the start/done markers) land in an asciicast file. Default
    # empty string keeps the existing behaviour (no recording).
    record_path: str = ""
    # F-57 Phase 4: explicit handwritten macro manifests (repeatable).
    macro_manifests: list[str] | None = None
    # F-57: directory of handwritten macros (globs *.yaml/*.yml).
    macros_dir: str = ""


@register("sop")
def run_sop_command(args: list[str]) -> int:
    """Dispatch ``sop`` sub-subcommands (currently only ``convert``)."""
    if not args:
        print("usage: clawcodex sop convert|catalog …", file=sys.stderr)
        return 2

    command = args[0]
    rest = args[1:]

    if command == "convert":
        return _handle_convert(rest)

    if command == "catalog":
        from extensions.sop_converter.catalog_cli import run_catalog_command

        return run_catalog_command(rest)

    print(f"Unknown sop command: {command}", file=sys.stderr)
    print("usage: clawcodex sop convert|catalog …", file=sys.stderr)
    return 2


@register("pos")
def run_pos_command(args: list[str]) -> int:
    """Backward-compatible alias for ``sop``."""
    return run_sop_command(args)


def _open_capture(record_path: str):
    """Open an asciicast capture when ``--record <path>`` was supplied.

    Returns ``None`` when ``record_path`` is empty (the recording-
    disabled path). The projector + caller are responsible for
    ``close()``.
    """
    if not record_path:
        return None
    try:
        from extensions.capabilities.recorder import AsciicastHeader
        from extensions.recording.asciicast_writer import AsciicastWriter
    except Exception:  # pragma: no cover - import failure fallback
        return None
    path = Path(record_path)
    header = AsciicastHeader(
        width=80,
        height=24,
        title=f"SOP convert ({path.stem})",
        command="clawcodex-dev sop convert --record",
    )
    writer = AsciicastWriter(path, header)
    writer.open()
    return writer.capture


def _parse_convert_args(args: list[str]) -> ConvertOptions:
    """Parse ``sop convert`` arguments."""
    if not args:
        print("error: missing <sdk_spec> argument", file=sys.stderr)
        print(
            "usage: clawcodex sop convert <sdk_spec> [--out <dir>] [--requirements <req>] "
            "[--name <name>] [--strategy <strategy>] [--skills <skills_dir>] "
            "[--max-groups <N>] [--mapping-rules <file>] [--llm-provider <provider>] "
            "[--llm-model <model>] [--preview] [--all] [--mode auto|sdk|hybrid|fwa] "
            "[--extractor <name>] [--emit-workflow-yaml] [--emit-stage-agents] "
            "[--emit-bridge] [--bridge-mode python|cli] [--bridge-cli <cmd>] [--strict-workflow-yaml] "
            "[--macro-manifest <path>] [--macros-dir <dir>] [--json] [--validate]",
            "[--json] [--validate] [--interactive]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    opts = ConvertOptions(sdk_spec=args[0], macro_manifests=[])

    i = 1
    while i < len(args):
        token = args[i]
        if token == "--out" and i + 1 < len(args):  # nosec B105
            opts.output_dir = args[i + 1]
            i += 2
        elif token == "--requirements" and i + 1 < len(args):  # nosec B105
            opts.requirements = args[i + 1]
            i += 2
        elif token == "--name" and i + 1 < len(args):  # nosec B105
            opts.agent_name = args[i + 1]
            i += 2
        elif token == "--strategy" and i + 1 < len(args):  # nosec B105
            opts.strategy = args[i + 1]
            i += 2
        elif token == "--skills" and i + 1 < len(args):  # nosec B105
            opts.skills_dir = args[i + 1]
            i += 2
        elif token == "--max-groups" and i + 1 < len(args):  # nosec B105
            try:
                opts.max_groups = int(args[i + 1])
            except ValueError:
                print(f"error: --max-groups requires an integer, got: {args[i + 1]}", file=sys.stderr)
                raise SystemExit(2)
            i += 2
        elif token == "--mapping-rules" and i + 1 < len(args):  # nosec B105
            opts.mapping_rules_file = args[i + 1]
            i += 2
        elif token == "--llm-provider" and i + 1 < len(args):  # nosec B105
            opts.llm_provider = args[i + 1]
            i += 2
        elif token == "--llm-model" and i + 1 < len(args):  # nosec B105
            opts.llm_model = args[i + 1]
            i += 2
        elif token == "--mode" and i + 1 < len(args):  # nosec B105
            opts.mode = args[i + 1]
            i += 2
        elif token == "--extractor" and i + 1 < len(args):  # nosec B105
            opts.extractor = args[i + 1]
            i += 2
        elif token == "--emit-workflow-yaml":  # nosec B105
            opts.emit_workflow_yaml = True
            i += 1
        elif token == "--emit-stage-agents":  # nosec B105
            opts.emit_stage_agents = True
            i += 1
        elif token == "--emit-bridge":  # nosec B105
            opts.emit_bridge = True
            i += 1
        elif token == "--bridge-mode" and i + 1 < len(args):  # nosec B105
            opts.bridge_mode = args[i + 1]
            i += 2
        elif token == "--bridge-cli" and i + 1 < len(args):  # nosec B105
            opts.bridge_cli = args[i + 1]
            i += 2
        elif token == "--strict-workflow-yaml":  # nosec B105
            opts.strict_workflow_yaml = True
            i += 1
        elif token == "--json":  # nosec B105
            opts.json_output = True
            i += 1
        elif token == "--validate":  # nosec B105
            opts.validate_only = True
            i += 1
        elif token == "--record" and i + 1 < len(args):  # nosec B105
            opts.record_path = args[i + 1]
            i += 2
        elif token == "--preview":  # nosec B105
            opts.preview = True
            i += 1
        elif token == "--all":  # nosec B105
            opts.all_methods = True
            i += 1
        elif token == "--no-register-tools":  # nosec B105
            opts.register_tools = False
            i += 1
        elif token == "--register-tools":  # nosec B105
            opts.register_tools = True
            i += 1
        elif token == "--interactive":  # nosec B105
            opts.interactive = True
            i += 1
        elif token == "--macro-manifest" and i + 1 < len(args):  # nosec B105
            if opts.macro_manifests is None:
                opts.macro_manifests = []
            opts.macro_manifests.append(args[i + 1])
            i += 2
        elif token == "--macros-dir" and i + 1 < len(args):  # nosec B105
            opts.macros_dir = args[i + 1]
            i += 2
        else:
            print(f"error: unknown argument: {token}", file=sys.stderr)
            raise SystemExit(2)

    return opts


def _handle_convert(args: list[str]) -> int:
    """Handle ``sop convert`` — convert an SOP spec into an Agent."""
    try:
        opts = _parse_convert_args(args)
    except SystemExit:
        return 2

    sdk_path = Path(opts.sdk_spec)

    if sdk_path.is_dir():
        return _handle_convert_from_source(opts)

    from extensions.sop_converter.convert_sop_skill import convert_sop_to_agent

    result = convert_sop_to_agent(
        sdk_spec=opts.sdk_spec,
        requirements=opts.requirements,
        agent_name=opts.agent_name,
        bundle_dir=opts.output_dir,
    )

    if result["status"] == "error":
        print(f"error: {result.get('error', 'conversion failed')}", file=sys.stderr)
        return 2

    # F-REC: when ``--record <path>`` is set, mirror the summary block
    # (and the call above) into an asciicast capture. The projector is
    # a no-op when ``record_path`` is empty so the default CLI path
    # pays no cost.
    capture = _open_capture(opts.record_path)
    from extensions.sop_converter.asciicast_projector import SopStageProjector

    with SopStageProjector(capture, title=opts.agent_name or "SOP convert"):
        # Print summary to stdout
        print(f"✅ Converted SOP: {result['agent_type']}")
        print(f"   Description: {result['agent_description']}")
        print(f"   Model: {result.get('model', 'default')}")
        print(f"   Skills: {len(result['skills'])}")
        for skill in result["skills"]:
            print(f"     - {skill['name']} ({', '.join(skill['tools'])})")
        print(f"   Tools: {len(result.get('tools', []))}")
        print(f"   Persistence: {result['persist_status']}")

        if result.get("warnings"):
            for w in result["warnings"]:
                print(f"   Warning: {w}", file=sys.stderr)

    if capture is not None:
        try:
            capture.close()
        except Exception:  # nosec B110
            pass

    # If --out is specified, write the output files to the target directory
    if opts.output_dir:
        _write_output_files(opts.output_dir, result, opts.sdk_spec, opts.requirements, opts.agent_name)

    return 0


def _collect_macro_manifests(opts: ConvertOptions) -> list[Path]:
    """Resolve --macros-dir + --macro-manifest into a flat deduplicated file list."""
    paths: list[Path] = [Path(p) for p in (opts.macro_manifests or [])]
    if opts.macros_dir:
        d = Path(opts.macros_dir)
        if d.is_dir():
            for p in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
                if p.is_file() and p not in paths:
                    paths.append(p)
    return paths


def _handle_convert_from_source(opts: ConvertOptions) -> int:
    """Convert a source code directory into Agents via SourceCodeParser + grouping strategy."""
    sdk_path = Path(opts.sdk_spec)
    from extensions.sop_converter.type_schema import reset_schema_probe_runtime_state
    from extensions.sop_converter.workflow_mode.pipeline import discriminate_and_extract
    from extensions.sop_converter.workflow_mode.extractors.preview import (
        format_discrimination_summary,
        format_workflow_preview,
    )
    from extensions.sop_converter.workflow_mode.mapping import (
        build_workflow_stages,
    )
    from extensions.sop_converter.source_parser import SourceCodeParser
    from extensions.sop_converter.skill_grouper import (
        GroupStrategy,
        group_source_components,
        SkillSpec,
    )
    from extensions.sop_converter.agent_md_writer import (
        AgentMarkdownWriter,
        AgentComponentInfo,
        WorkflowStage,
    )

    reset_schema_probe_runtime_state()

    force_mode = opts.mode if opts.mode != "auto" else None
    interactive = opts.interactive
    disc, workflow_graph = discriminate_and_extract(
        sdk_path,
        force_mode=force_mode,
        extractor=opts.extractor,
        interactive=interactive,
        out_dir=opts.output_dir or None,
    )

    if not opts.json_output:
        print(format_discrimination_summary(disc))
        if workflow_graph:
            print(
                f"   Workflow graph: {len(workflow_graph.stages)} stages, "
                f"{len(workflow_graph.transitions)} transitions, "
                f"quality={workflow_graph.extraction_quality}"
            )
        elif disc.mode in ("fwa", "hybrid"):
            print("   Warning: workflow mode selected but graph extraction returned empty", file=sys.stderr)
            if interactive:
                print(
                    "   → TODO templates generated. Edit and re-run or fill in manually.",
                    file=sys.stderr,
                )

    if opts.json_output:
        payload = disc.to_dict()
        if workflow_graph:
            payload["workflow"] = {
                "stages": len(workflow_graph.stages),
                "transitions": len(workflow_graph.transitions),
                "quality": workflow_graph.extraction_quality,
            }
        print(json.dumps(payload, indent=2))
        if opts.validate_only:
            return 0

    parser = SourceCodeParser(str(sdk_path), extern_only=not opts.all_methods)
    components = parser.parse()

    if not components:
        print("error: No source components found in directory", file=sys.stderr)
        return 2

    from extensions.sop_converter.bundle_venv import (
        bundle_venv_dir,
        bundle_venv_python,
    )
    from extensions.sop_converter.sdk_dependency_resolver import resolve_sdk_dependencies

    sdk_deps = resolve_sdk_dependencies(sdk_path)
    bundle_dir_for_tools = Path(opts.output_dir).resolve() if opts.output_dir else None
    bundle_venv_python_path: str | None = None
    bundle_venv_dir_path: str | None = None
    if bundle_dir_for_tools is not None and sdk_deps.requirements:
        bundle_venv_python_path = str(bundle_venv_python(bundle_dir_for_tools))
        bundle_venv_dir_path = str(bundle_venv_dir(bundle_dir_for_tools))
    if sdk_deps.requirements and not opts.json_output:
        suffix = (
            "bundle venv will be created before schema probing"
            if bundle_venv_python_path
            else "no --out bundle path; venv isolation disabled"
        )
        print(f"   SDK dependencies: {len(sdk_deps.requirements)} from {sdk_deps.source} ({suffix})")

    parsed_strategy = opts.strategy.lower() if opts.strategy else ""
    if parsed_strategy == "keyword":
        group_strategy = GroupStrategy.KEYWORD_MATCH
    elif parsed_strategy == "io":
        group_strategy = GroupStrategy.IO_RELATION
    elif parsed_strategy == "llm":
        group_strategy = GroupStrategy.LLM_SEMANTIC
    else:
        group_strategy = GroupStrategy.COMPONENT_GROUP

    custom_rules: list[MappingRule] | None = None
    if opts.mapping_rules_file and group_strategy == GroupStrategy.KEYWORD_MATCH:
        custom_rules = _load_mapping_rules(opts.mapping_rules_file)

    llm_provider_obj = None
    if group_strategy == GroupStrategy.LLM_SEMANTIC:
        llm_provider_obj = _create_llm_provider(opts.llm_provider, opts.llm_model)
        if llm_provider_obj is None:
            print(
                "warning: LLM_SEMANTIC requires a configured LLM provider; falling back to keyword match strategy",
                file=sys.stderr,
            )
            group_strategy = GroupStrategy.KEYWORD_MATCH

    group_result = group_source_components(
        components,
        strategy=group_strategy,
        max_io_groups=opts.max_groups if opts.max_groups > 0 else None,
        mapping_rules=custom_rules,
        requirements=opts.requirements,
        llm_provider=llm_provider_obj,
    )
    grouped_skills: list[SkillSpec] = group_result.skills

    skill_agent_map = {s.name: f"{s.name}-agent" for s in grouped_skills}

    agent_map = None
    bridge_script_path: Path | None = None
    if workflow_graph:
        from extensions.sop_converter.workflow_mode.capability import (
            StageCapabilityMapper,
            ensure_stage_skills,
        )

        grouped_skills = ensure_stage_skills(
            workflow_graph,
            components,
            grouped_skills,
            sdk_path,
        )
        skill_agent_map = {s.name: f"{s.name}-agent" for s in grouped_skills}

        agent_map = StageCapabilityMapper().map(workflow_graph, components, grouped_skills)

    workflow_stages: list[WorkflowStage] = []
    if workflow_graph and agent_map:
        from extensions.sop_converter.workflow_mode.generator import AgentDefinitionGenerator

        gen = AgentDefinitionGenerator()
        workflow_stages = gen.enrich_workflow_stages(
            workflow_graph,
            agent_map,
            skill_agent_map=skill_agent_map,
        )
    elif workflow_graph:
        workflow_stages = build_workflow_stages(workflow_graph, skill_agent_map=skill_agent_map)

    tool_deps_index = None
    try:
        from extensions.sop_converter.tool_dependencies import build_tool_dependency_index

        tool_deps_index = build_tool_dependency_index(components, source_dir=str(sdk_path))
    except Exception:
        tool_deps_index = None

    # F-57 Phase 4: preview / validate handwritten macros without writing
    if opts.preview or opts.validate_only:
        try:
            from extensions.sop_converter.runtime.macros import (
                MacroConvertError,
                convert_handwritten_macros,
            )

            macro_preview = convert_handwritten_macros(
                source_dir=sdk_path,
                bundle_dir=None,
                manifest_paths=_collect_macro_manifests(opts),
                tool_index=None,
                preview=True,
                validate_only=opts.validate_only,
                register_tools=False,
                persist=False,
            )
            if macro_preview.diagnostics:
                for item in macro_preview.diagnostics:
                    print(f"   Macro diagnostic: {item}", file=sys.stderr)
                if opts.validate_only:
                    return 2
            elif macro_preview.preview and not opts.json_output:
                print(f"   Handwritten macros OK: {len(macro_preview.preview)}")
        except MacroConvertError as exc:
            print(f"error: {exc.to_dict()}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"   Warning: macro preview skipped: {exc}", file=sys.stderr)

    registered: dict[str, str] = {}
    if opts.register_tools and not opts.preview and not opts.validate_only:
        try:
            from extensions.sop_converter.tool_registry_bridge import (
                register_component_tools,
                _to_kebab_case,
            )

            print("   Registering component tools...")
            registered = register_component_tools(
                components,
                str(sdk_path),
                persist=True,
                bundle_dir=bundle_dir_for_tools,
                bundle_id=bundle_dir_for_tools.name if bundle_dir_for_tools else None,
                bundle_venv_python=bundle_venv_python_path,
                sdk_requirements=sdk_deps.requirements,
            )
            if registered:
                print(f"   Registered tools: {len(set(registered.values()))}")
            composite_registered: dict[str, str] = {}
            try:
                from extensions.sop_converter.runtime.composite_tools import (
                    emit_composite_workflow_yaml,
                    register_composite_tools,
                )
                from extensions.sop_converter.runtime.composite_tools.builtin import (
                    builtin_composite_tools,
                )

                composite_registered = register_composite_tools(
                    persist=True,
                    bundle_dir=bundle_dir_for_tools,
                    sdk_source_dir=str(sdk_path),
                )
                if composite_registered:
                    print(f"   Registered composite tools: {len(composite_registered)}")
                # F-55: auto-promote lifecycle recovery composite tools into
                # skills whose allowed_tools intersect the agent_lifecycle group.
                try:
                    from extensions.sop_converter.runtime.composite_tools.builtin import (
                        lifecycle_tools_for_skill,
                    )
                    from extensions.sop_converter.core.dependency.models import (
                        ToolDependencyGraph,
                    )

                    lifecycle_graph = ToolDependencyGraph.detect_from_components(components)
                    for skill in grouped_skills:
                        extras = lifecycle_tools_for_skill(
                            skill.allowed_tools,
                            lifecycle_graph,
                            composite_registered,
                        )
                        for tool_name in extras:
                            if tool_name not in skill.allowed_tools:
                                skill.allowed_tools.insert(0, tool_name)
                except Exception as exc:
                    print(
                        f"   Warning: lifecycle tool promotion failed: {exc}",
                        file=sys.stderr,
                    )
                for skill in grouped_skills:
                    if skill.name in ("agent_teams-skill", "agent_teams"):
                        for tool_name in composite_registered.values():
                            if tool_name not in skill.allowed_tools:
                                skill.allowed_tools.insert(0, tool_name)
                if opts.output_dir:
                    out_composite = Path(opts.output_dir)
                    composite_project_name = opts.agent_name or sdk_path.name
                    for spec in builtin_composite_tools():
                        wf = emit_composite_workflow_yaml(
                            spec,
                            out_composite,
                            project_name=composite_project_name,
                        )
                        if wf:
                            print(f"   Composite workflow: {wf.name}")
            except Exception as exc:
                print(
                    f"   Warning: composite tool registration failed: {exc}",
                    file=sys.stderr,
                )

            # F-57 Phase 4: handwritten / template macros (sop-macros/ + --macro-manifest)
            try:
                from extensions.sop_converter.runtime.macros import (
                    MacroConvertError,
                    convert_handwritten_macros,
                )

                tool_index = set(registered.values()) if registered else set()
                tool_index.update(str(name) for name in composite_registered.values())
                macro_result = convert_handwritten_macros(
                    source_dir=sdk_path,
                    bundle_dir=bundle_dir_for_tools,
                    manifest_paths=_collect_macro_manifests(opts),
                    tool_index=tool_index,
                    preview=False,
                    validate_only=False,
                    register_tools=True,
                    persist=bool(bundle_dir_for_tools),
                )
                if macro_result.registered_tools:
                    print(f"   Registered handwritten macros: {len(macro_result.registered_tools)}")
                    # One owner skill per macro — never broadcast onto every domain.
                    from extensions.sop_converter.runtime.macros.overview_intent import (
                        assign_macros_to_owner_skills,
                    )

                    ownership = assign_macros_to_owner_skills(
                        grouped_skills,
                        macro_result.macros,
                    )
                    if ownership:
                        for tool_name, skill_name in ownership.items():
                            print(f"      macro {tool_name} → skill {skill_name}")
                    else:
                        # Fallback only when ownership cannot be resolved from
                        # covered_tools: keep tools registered but unattached.
                        print(
                            "   Warning: no owner skill resolved for handwritten macros; "
                            "macros stay registered but are not added to domain skills",
                            file=sys.stderr,
                        )
                elif macro_result.written_paths:
                    print(f"   Persisted handwritten macros: {len(macro_result.written_paths)}")
            except MacroConvertError as exc:
                print(
                    f"error: handwritten macro convert failed: {exc.to_dict()}",
                    file=sys.stderr,
                )
                return 2
            except Exception as exc:
                print(
                    f"   Warning: handwritten macro convert skipped: {exc}",
                    file=sys.stderr,
                )
        except ImportError as exc:
            print(
                f"   Warning: tool registration skipped (missing module: {exc})",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"   Warning: tool registration failed: {exc}",
                file=sys.stderr,
            )

        # Always convert tool names to kebab-case, even if registration partially failed.
        # _to_kebab_case() provides a reliable fallback for tools whose wrappers
        # couldn't be generated (e.g. due to syntax errors).
        try:
            for skill in grouped_skills:
                seen_tools: set[str] = set()
                ordered_tools: list[str] = []
                for t in skill.allowed_tools:
                    mapped = registered.get(t, _to_kebab_case(t)) if registered else _to_kebab_case(t)
                    if mapped not in seen_tools:
                        seen_tools.add(mapped)
                        ordered_tools.append(mapped)
                skill.allowed_tools = ordered_tools
        except NameError:
            # _to_kebab_case not imported (ImportError path taken above) — skip
            pass
        if registered:
            print(f"   Registered tools: {len(set(registered.values()))}")
        # Sync StageAgentMap profiles with rewritten tool names so
        # stage agents reference the actual registered spec names.
        if agent_map:
            skill_tool_map = {s.name: s.allowed_tools for s in grouped_skills}
            for profile in agent_map.by_stage_id.values():
                if profile.mapped_skill and profile.mapped_skill in skill_tool_map:
                    profile.recommended_tools = list(skill_tool_map[profile.mapped_skill])

    # Build per-skill AgentComponentInfo for the overview agent.
    # Macro tools belong in the overview「宏工具意图」table only — do not list
    # them under every (or even the owner) domain agent's capabilities.
    macro_tool_names: set[str] = set()
    try:
        from extensions.sop_converter.bundle_context import load_bundle_macro_routes

        if opts.output_dir:
            for route in load_bundle_macro_routes(Path(opts.output_dir)):
                target = str(getattr(route, "target_tool", "") or "").strip()
                if target:
                    macro_tool_names.add(target)
    except Exception:  # nosec B110
        pass

    overview_info: list[AgentComponentInfo] = []
    for skill in grouped_skills:
        caps = [t for t in skill.allowed_tools if str(t).strip() and str(t).strip() not in macro_tool_names][:5]
        overview_info.append(
            AgentComponentInfo(
                name=f"{skill.name}-agent",
                description=skill.description,
                capabilities=caps,
                invoke_pattern=f"@{skill.name}-agent {{task}}",
            )
        )

    overview_count = 1 if len(overview_info) > 1 else 0
    total_agents = len(grouped_skills) + overview_count

    if opts.preview:
        if workflow_graph:
            print("[Preview] Workflow extraction:")
            print(format_workflow_preview(workflow_graph, disc))
        else:
            print(f"[Preview] Workflow mode: {disc.mode} (score={disc.total_score:.2f}), no graph extracted")
        strategy_labels = {
            "IO_RELATION": "IO_RELATION",
            "KEYWORD_MATCH": "KEYWORD_MATCH",
            "LLM_SEMANTIC": "LLM_SEMANTIC",
            "COMPONENT_GROUP": "COMPONENT_GROUP",
        }
        label = strategy_labels.get(group_strategy.name, group_strategy.name)
        print(f"[Preview] Strategy: {label}")
        if not opts.all_methods:
            print("   Filter: external interfaces only (default)")
        else:
            print("   Filter: --all (all public methods)")
        print(f"   Source Components: {len(components)}")
        print(f"   Grouped Skills: {len(grouped_skills)}")
        print(f"   Agent file count: {total_agents} ({len(grouped_skills)} + {overview_count} overview)")
        if workflow_stages:
            print(f"   Workflow stages: {len(workflow_stages)}")
        if len(components) != len(grouped_skills):
            reduction = 100 - int(len(grouped_skills) / len(components) * 100)
            print(f"   Agent 缩减率: {reduction}% ({len(components)} → {total_agents})")
        print("   Skills:")
        for skill in grouped_skills:
            sample_tools = skill.allowed_tools[:3]
            sample_str = ", ".join(sample_tools)
            if len(skill.allowed_tools) > 3:
                sample_str += ", …"
            print(f"     - {skill.name}-agent ({len(skill.allowed_tools)} tools): {sample_str}")
        if group_result.unmatched_tools:
            for w in group_result.unmatched_tools:
                print(f"   Warning: unmatched tool: {w}", file=sys.stderr)
        return 0

    if opts.validate_only:
        return 0

    project_name = opts.agent_name or sdk_path.name
    overview_agent_name = opts.agent_name or "clawcodex-overview"
    # Hybrid/FWA mode: auto-emit workflow YAML, stage agents, bridge
    if disc.mode in ("fwa", "hybrid"):
        if not opts.emit_workflow_yaml:
            opts.emit_workflow_yaml = True
        if not opts.emit_bridge:
            opts.emit_bridge = True
    emit_stage_agents = opts.emit_stage_agents or (disc.mode in ("fwa", "hybrid"))
    out_path = Path(opts.output_dir) if opts.output_dir else None
    from extensions.sop_converter.bundle_workflow import workflow_artifacts_enabled

    emit_workflow_bundle = (
        workflow_graph is not None
        and agent_map is not None
        and out_path is not None
        and workflow_artifacts_enabled(
            has_mapped_stages=agent_map.has_mapped_stages,
            workflow_mode=disc.mode,
        )
    )
    if workflow_graph and agent_map and out_path and not emit_workflow_bundle:
        print(
            "   Warning: workflow artifacts skipped (no mapped stages; use fwa/hybrid mode or fix F-50-C mapping)",
            file=sys.stderr,
        )
    elif workflow_graph and agent_map and not agent_map.has_mapped_stages and emit_workflow_bundle:
        print(
            "   Warning: emitting workflow scaffold with unmapped stages "
            "(mapping_confidence=0; stage agents use fallback names)",
            file=sys.stderr,
        )

    if workflow_graph and agent_map and out_path and emit_workflow_bundle and opts.emit_bridge:
        from extensions.sop_converter.workflow_mode.bridge import BridgeGenerator

        bridge_script_path = BridgeGenerator().generate(
            workflow_graph,
            agent_map,
            sdk_path,
            out_path,
            mode=opts.bridge_mode,
            project_name=project_name,
            cli_entry=opts.bridge_cli or None,
        )
        if opts.register_tools and bridge_script_path is not None:
            from extensions.sop_converter.workflow_mode.bridge.mcp_adapter import (
                bridge_tool_name,
                register_bridge_tool,
            )

            try:
                register_bridge_tool(
                    bridge_tool_name(project_name),
                    bridge_script_path,
                    persist=True,
                    bundle_dir=out_path,
                )
            except Exception as exc:
                print(
                    f"   Warning: bridge tool registration failed: {exc}",
                    file=sys.stderr,
                )

    # Finalize stage agent names before workflow.yaml / overview reference them.
    if workflow_graph and agent_map and emit_workflow_bundle and emit_stage_agents:
        from extensions.sop_converter.workflow_mode.generator import (
            AgentDefinitionGenerator,
            stage_agent_existing_names,
        )
        from extensions.sop_converter.workflow_mode.mapping import (
            sync_overview_component_agents,
            sync_workflow_stages_agents,
        )

        gen = AgentDefinitionGenerator()
        existing_stage_names = stage_agent_existing_names(
            grouped_skills,
            workflow_graph,
            overview_agent_name=overview_agent_name if len(overview_info) > 1 else None,
        )
        gen.finalize_stage_agent_names(
            workflow_graph,
            agent_map,
            project_name=project_name,
            existing_agent_names=existing_stage_names,
        )
        workflow_stages = sync_workflow_stages_agents(
            workflow_stages,
            workflow_graph,
            agent_map,
        )
        overview_info = sync_overview_component_agents(
            overview_info,
            workflow_graph,
            agent_map,
        )

    if workflow_graph and agent_map and out_path and emit_workflow_bundle and opts.emit_workflow_yaml:
        from extensions.sop_converter.workflow_mode.schema import emit_engine_workflow_yaml

        yaml_path = emit_engine_workflow_yaml(
            workflow_graph,
            agent_map,
            out_path,
            workflow_name=project_name,
            strict=opts.strict_workflow_yaml,
        )
        print(f"   Workflow YAML: {yaml_path}")

    # ── Normal mode: write files + print summary ──
    writer = AgentMarkdownWriter()
    if opts.output_dir:
        from extensions.sop_converter.workflow_mode.generator import (
            AgentDefinitionGenerator,
            coarse_agent_skills,
        )

        out_path = Path(opts.output_dir)
        from extensions.sop_converter.cross_domain_orchestration import (
            write_orchestration_routes,
        )
        from extensions.sop_converter.sdk_overview import write_sdk_overview

        overview_md_path = write_sdk_overview(
            out_path,
            components,
            skills=grouped_skills,
            sdk_source_dir=sdk_path,
            group_strategy=group_strategy,
        )
        print(f"   SDK overview: {overview_md_path.name}")
        orch_path = write_orchestration_routes(
            out_path,
            grouped_skills,
            components=components,
            tool_deps_index=tool_deps_index,
        )
        print(f"   Orchestration routes: {orch_path.name}")

        for skill in coarse_agent_skills(grouped_skills, workflow_graph):
            skill_name = f"{skill.name}-skill"
            agent_def = {
                "name": f"{skill.name}-agent",
                "description": skill.description,
                "tools": skill.allowed_tools,
                "skills": [skill_name],
            }
            writer.write_agent(agent_def, out_path, bundle=out_path)

        # Overview agent — always generated when there are 2+ agents
        if len(overview_info) > 1:
            writer.write_overview_agent(
                name=opts.agent_name or "clawcodex-overview",
                description=f"Overview agent for {opts.agent_name or 'project'}",
                component_agents=overview_info,
                workflow_stages=workflow_stages,
                output_dir=out_path,
                sdk_source_dir=sdk_path,
            )

        # F-50-E: write hybrid/wrapper/native stage agents last so coarse write_agent cannot overwrite them
        if workflow_graph and agent_map and emit_workflow_bundle and emit_stage_agents:
            from extensions.sop_converter.workflow_mode.generator import stage_agent_existing_names

            existing = stage_agent_existing_names(
                grouped_skills,
                workflow_graph,
                overview_agent_name=overview_agent_name if len(overview_info) > 1 else None,
            )
            bridge_rel = None
            if bridge_script_path and out_path:
                try:
                    bridge_rel = str(bridge_script_path.relative_to(out_path))
                except ValueError:
                    bridge_rel = str(bridge_script_path)
            AgentDefinitionGenerator().generate_stage_agents(
                workflow_graph,
                agent_map,
                out_path,
                project_name=project_name,
                bridge_script=bridge_rel,
                existing_agent_names=existing,
                write_skills=True,
            )

    if opts.skills_dir:
        from extensions.sop_converter.task_guide import format_flat_skill_markdown
        from extensions.sop_converter.workflow_mode.generator import coarse_agent_skills

        skills_path = Path(opts.skills_dir)
        skills_path.mkdir(parents=True, exist_ok=True)
        stage_contracts: dict[str, object] = {}
        if workflow_graph:
            for stage in workflow_graph.stages:
                c = workflow_graph.contracts.get(stage.id)
                if c is not None:
                    stage_contracts[stage.name] = c
        for skill in coarse_agent_skills(grouped_skills, workflow_graph):
            skill_file = skills_path / f"{skill.name}-skill.md"
            skill_file.write_text(
                format_flat_skill_markdown(
                    skill,
                    components=components,
                    contract=stage_contracts.get(skill.name),
                    tool_deps_index=tool_deps_index,
                    bundle=out_path,
                ),
                encoding="utf-8",
            )
            print(f"   Skill file: {skill_file}")

    # Print summary — show actual agent count after grouping
    print(f"✅ Converted source directory to Agents: {sdk_path}")
    print(f"   Source components: {len(components)}")
    print(f"   Grouped skills (agents): {len(grouped_skills)}")
    print(f"   Workflow mode: {disc.mode} (score={disc.total_score:.2f})")
    if workflow_stages:
        print(f"   Workflow stages in overview: {len(workflow_stages)}")
    if len(components) != len(grouped_skills):
        reduction = 100 - int(len(grouped_skills) / len(components) * 100)
        print(f"   Agent reduction: {reduction}% ({len(components)} components → {total_agents} agents)")
    for i, skill in enumerate(grouped_skills):
        print(f"     Agent {i + 1}: {skill.name}-agent")
        print(f"       Description: {skill.description}")
        print(f"       Tools: {len(skill.allowed_tools)}")
    if group_result.unmatched_tools:
        for w in group_result.unmatched_tools:
            print(f"   Warning: unmatched tool: {w}", file=sys.stderr)

    print(f"   Strategy: {group_strategy.name}")

    if opts.output_dir:
        from extensions.sop_converter.bundle_manifest import write_bundle_manifest

        bundle_dir = Path(opts.output_dir)
        bridge_rel: str | None = None
        if bridge_script_path is not None:
            try:
                bridge_rel = str(bridge_script_path.relative_to(bundle_dir))
            except ValueError:
                bridge_rel = str(bridge_script_path)
        workflow_rel = "workflow.yaml" if (bundle_dir / "workflow.yaml").is_file() else None
        manifest_path = write_bundle_manifest(
            bundle_dir,
            sdk_source_dir=sdk_path,
            bundle_id=bundle_dir.name,
            workflow_yaml=workflow_rel,
            bridge_script=bridge_rel,
            workflow_mode=disc.mode if workflow_graph else None,
            sdk_requirements=sdk_deps.requirements,
            bundle_venv_dir=bundle_venv_dir_path,
        )
        print(f"   Bundle manifest: {manifest_path}")

    return 0


def _load_mapping_rules(file_path: str) -> list[MappingRule]:
    """Load custom MappingRule definitions from a YAML or JSON file.

    Expected format (YAML):
        rules:
          - method_pattern: "docker_"
            tool_name: "docker_ops"
            skill_name: "build_image"
            description: "Docker build operations"
            match_type: "prefix"
          - method_pattern: "video_encode|video_decode"
            tool_name: "video_ops"
            skill_name: "video_processing"
            description: "Video codec operations"
            match_type: "regex"
          - method_pattern: "core"
            tool_name: "core_ops"
            skill_name: "core"
            description: "Core module operations"
            match_type: "substring"
            match_target: "comp_name"

    Expected format (JSON):
        {
          "rules": [
            {"method_pattern": "docker_", "tool_name": "docker_ops", "skill_name": "build_image", "description": "...", "match_type": "prefix"}
          ]
        }
    """
    from extensions.sop_converter.skill_grouper import MappingRule, MatchType, MatchTarget

    path = Path(file_path)
    if not path.is_file():
        print(f"error: mapping rules file not found: {file_path}", file=sys.stderr)
        raise SystemExit(2)

    raw = path.read_text(encoding="utf-8")

    try:
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(raw)
            except ImportError:
                print(
                    "error: PyYAML not installed, use JSON format for mapping rules",
                    file=sys.stderr,
                )
                raise SystemExit(2)
        else:
            data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        print(f"error: failed to parse mapping rules file: {exc}", file=sys.stderr)
        raise SystemExit(2)

    rules_raw = data.get("rules", [])
    rules: list[MappingRule] = []
    for entry in rules_raw:
        match_type_str = entry.get("match_type", "substring")
        try:
            match_type = MatchType(match_type_str)
        except ValueError:
            match_type = MatchType.SUBSTRING
        match_target_str = entry.get("match_target", "op_name")
        try:
            match_target = MatchTarget(match_target_str)
        except ValueError:
            match_target = MatchTarget.OP_NAME
        method_pattern = entry.get("method_pattern", "")
        skill_name = entry.get("skill_name", "")
        if not method_pattern or not skill_name:
            print(
                f"warning: skipping mapping rule entry missing method_pattern or skill_name: {entry}",
                file=sys.stderr,
            )
            continue
        rules.append(
            MappingRule(
                method_pattern=method_pattern,
                tool_name=entry.get("tool_name", ""),
                skill_name=skill_name,
                description=entry.get("description", ""),
                match_type=match_type,
                match_target=match_target,
            )
        )

    return rules


def _create_llm_provider(provider_name: str, model: str) -> object | None:
    """Create an LLM provider for LLM_SEMANTIC strategy.

    Uses the project's ``build_provider_from_config`` infrastructure.
    Falls back to the default provider from config when ``provider_name``
    is empty.  Returns None when no provider can be created (missing
    API key, unknown provider, etc.).
    """
    try:
        from src.providers.runtime import build_provider_from_config
        from src.config import get_default_provider

        resolved_name = provider_name or get_default_provider()
        resolved_model = model or None
        return build_provider_from_config(resolved_name, model=resolved_model)
    except Exception as exc:
        print(f"warning: failed to create LLM provider '{provider_name}': {exc}", file=sys.stderr)
        return None


def _write_output_files(
    out_dir: str,
    result: dict,
    sdk_spec: str,
    requirements: str,
    agent_name: str,
) -> None:
    """Write agent definition + skill files to ``out_dir`` in loadable format.

    Output layout:
      ``.claude/agents/<name>.md``         — agent definition (loadable by ``@agent-<name>``)
      ``.atomcode/skills/<name>/SKILL.md``   — skill files (loadable by skill system)
      ``workflows/sop-<name>.yaml``          — orchestrator workflow (unchanged)
      ``skills/sop-<name>-<skill>/SKILL.md`` — legacy compat (deprecated)
    """
    from extensions.sop_converter.agent_md_writer import AgentMarkdownWriter

    base = Path(out_dir).resolve()
    workflows_dir = base / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    name = result["agent_type"]
    skill_files = result.get("skill_files", [])
    skill_names = [s["name"] for s in result["skills"]]

    # --- .claude/agents/<name>.md — loadable by @agent-name ---
    writer = AgentMarkdownWriter()
    agent_def = {
        "name": name,
        "description": result["agent_description"],
        "model": result.get("model", "default"),
        "tools": result.get("tools", []),
        "skills": skill_names,
    }
    agent_path = writer.write_agent(agent_def, base, bundle=base)
    print(f"   Agent: {agent_path}")

    # --- .atomcode/skills/<name>/SKILL.md — loadable by skill system ---
    skills_for_writer = []
    for skill in result["skills"]:
        skills_for_writer.append(
            {
                "name": skill["name"],
                "description": skill["description"],
                "allowed_tools": skill["tools"],
                "parameters": [],
                "source_code": "",
            }
        )
    if skills_for_writer:
        skill_paths = writer.write_skills(skills_for_writer, base, bundle=base)
        for sp in skill_paths:
            print(f"   Skill: {sp}")

    # --- Legacy compat: skills/sop-<name>-<skill>/SKILL.md (deprecated) ---
    # TODO: remove in next major version; users should migrate to .atomcode/skills/
    import warnings

    warnings.warn(
        "The `skills/` output directory is deprecated and will be removed "
        "in a future release. Use `.atomcode/skills/` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    skills_dir = base / "skills"
    for skill in result["skills"]:
        skill_name = skill["name"]
        skill_dir = skills_dir / f"sop-{name}-{skill_name}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        lines = [
            "---",
            f"name: {skill_name}",
            f"description: {skill['description']}",
            "user-invocable: true",
            "allowed-tools:",
        ]
        for tool in skill["tools"]:
            lines.append(f"  - {tool}")
        lines.append("---")
        lines.append("")
        lines.append(f"# Skill: {skill_name}")
        lines.append("")
        lines.append(skill["description"])
        lines.append("")
        lines.append("## Included Tools")
        for tool in skill["tools"]:
            lines.append(f"- `{tool}`")
        skill_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"   Skill (legacy): {skill_path}")

    # --- Workflow orchestration graph (unchanged) ---
    workflow_path = workflows_dir / f"sop-{name}.yaml"
    workflow_lines = [
        f"# Workflow: {name}",
        "# Auto-generated by clawcodex sop convert",
        "# This is a SOP execution graph (nodes: id/agent/skill/tools), NOT an",
        "# orchestrator WORKFLOW.md. The `sop-` prefix + `nodes:` schema keep it",
        "# distinct from orchestrator configs (name/tracker/workspace/stages).",
        "",
        "nodes:",
    ]
    for skill in result["skills"]:
        workflow_lines.append(f"  - id: {skill['name']}")
        workflow_lines.append(f"    agent: {name}")
        workflow_lines.append(f"    skill: {skill['name']}")
        workflow_lines.append(f"    tools: [{', '.join(skill['tools'])}]")
    workflow_path.write_text("\n".join(workflow_lines), encoding="utf-8")
    print(f"   Workflow: {workflow_path}")

    # --- Also copy any old-format skill files from converter ---
    for src_path_str in skill_files:
        src_path = Path(src_path_str)
        if src_path.exists():
            skills_dir.mkdir(parents=True, exist_ok=True)
            dest = skills_dir / src_path.name
            if not dest.exists():
                dest.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
