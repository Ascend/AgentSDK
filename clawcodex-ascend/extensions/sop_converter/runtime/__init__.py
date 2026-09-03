"""sop_converter runtime — agent runtime integration layer.

This subpackage depends on ``clawcodex_ext.*`` and ``extensions.capabilities.*``
Protocols.  It is NOT importable without the Claude Code runtime environment.

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.5.
"""

from .agent_builder import (
    AgentBuilder,
    AgentBuildResult,
    AgentPersistenceSpec,
    persist_converted_agent,
    write_agent_markdown,
)
from .agent_md_writer import AgentMarkdownWriter, AgentComponentInfo, WorkflowStage
from .bundle_agents import register_bundle_agents
from .bundle_context import (
    BundleContext,
    filter_tools_for_bundle,
    get_active_bundle,
    ensure_bundle_tools_registered,
    set_active_bundle,
    build_bundle_context,
    is_sop_converter_spec_source,
    collect_tool_names_from_skills,
    collect_tool_names_from_bundle_specs,
    is_pos_converter_tool,
    load_bundle_persisted_tools,
    prune_registry_to_bundle,
    activate_bundle_isolation,
    load_bundle_macro_routes,
    apply_sdk_source_working_directory,
)
from .bundle_skills import (
    register_bundle_skills,
    BundleSkillLoadResult,
    resolve_bundle_skill_workspace,
)
from .bundle_discovery import (
    overview_has_sop_skills,
    list_workspace_bundle_candidates,
    discover_workspace_bundle,
)
from .composite_runtime import (
    CompositeWorkflowSpec,
    CompositeWorkflowStep,
    CompositeWorkflowRunner,
    CompositeWorkflowError,
    StepTrace,
    CompositeResult,
    normalize_workflow_output,
)
from .composite_workflows import invoke_existing_agent_workflow, resume_resource_workflow
from .cross_domain_orchestration import (
    OrchestrationStep,
    OrchestrationRoute,
    skill_name_to_agent,
    build_tool_to_agent_map,
    discover_orchestration_routes,
    generate_orchestration_routes_markdown,
    write_orchestration_routes,
    format_orchestration_routes_block,
)
from .sdk_overview import (
    generate_io_sdk_overview_markdown,
    generate_sdk_overview_markdown,
    write_sdk_overview,
    format_sdk_overview_block,
)
from .convert_sop_skill import convert_sop_to_agent, get_prompt_for_command
from .skill_grouper import (
    SkillGrouper,
    SkillSpec,
    GroupStrategy,
    MatchType,
    MatchTarget,
    MappingRule,
    GroupResult,
    group_into_skills,
    group_source_components,
)
from .sop_exploration_guard import check_bundle_source_exploration, sop_exploration_permission_check
from .sop_routing import (
    looks_like_direct_sdk_execution,
    requested_agent_types_in_prompt,
    list_domain_agent_types,
    check_bundle_agent_delegation,
    refresh_domain_agent_sop_prompts,
)
from .startup_agent import build_bundle_overview_agent_definition
from .task_guide import (
    build_operation_index,
    is_entry_point,
    generate_task_guide_markdown,
    append_task_guide_to_skill_body,
    format_flat_skill_markdown,
)
from .tool_registry_bridge import (
    register_component_tools,
    register_http_tools,
    resolve_catalog_handle_from_args,
    operation_to_spec,
)


__all__ = [
    "AgentBuilder",
    "AgentBuildResult",
    "AgentPersistenceSpec",
    "persist_converted_agent",
    "write_agent_markdown",
    "AgentMarkdownWriter",
    "AgentComponentInfo",
    "WorkflowStage",
    "register_bundle_agents",
    "BundleContext",
    "filter_tools_for_bundle",
    "get_active_bundle",
    "ensure_bundle_tools_registered",
    "set_active_bundle",
    "build_bundle_context",
    "is_sop_converter_spec_source",
    "collect_tool_names_from_skills",
    "collect_tool_names_from_bundle_specs",
    "is_pos_converter_tool",
    "load_bundle_persisted_tools",
    "prune_registry_to_bundle",
    "activate_bundle_isolation",
    "load_bundle_macro_routes",
    "apply_sdk_source_working_directory",
    "register_bundle_skills",
    "BundleSkillLoadResult",
    "resolve_bundle_skill_workspace",
    "overview_has_sop_skills",
    "list_workspace_bundle_candidates",
    "discover_workspace_bundle",
    "CompositeWorkflowSpec",
    "CompositeWorkflowStep",
    "CompositeWorkflowRunner",
    "CompositeWorkflowError",
    "StepTrace",
    "CompositeResult",
    "normalize_workflow_output",
    "invoke_existing_agent_workflow",
    "resume_resource_workflow",
    "OrchestrationStep",
    "OrchestrationRoute",
    "skill_name_to_agent",
    "build_tool_to_agent_map",
    "discover_orchestration_routes",
    "generate_orchestration_routes_markdown",
    "write_orchestration_routes",
    "format_orchestration_routes_block",
    "generate_io_sdk_overview_markdown",
    "generate_sdk_overview_markdown",
    "write_sdk_overview",
    "format_sdk_overview_block",
    "convert_sop_to_agent",
    "get_prompt_for_command",
    "SkillGrouper",
    "SkillSpec",
    "GroupStrategy",
    "MatchType",
    "MatchTarget",
    "MappingRule",
    "GroupResult",
    "group_into_skills",
    "group_source_components",
    "check_bundle_source_exploration",
    "sop_exploration_permission_check",
    "looks_like_direct_sdk_execution",
    "requested_agent_types_in_prompt",
    "list_domain_agent_types",
    "check_bundle_agent_delegation",
    "refresh_domain_agent_sop_prompts",
    "build_bundle_overview_agent_definition",
    "build_operation_index",
    "is_entry_point",
    "generate_task_guide_markdown",
    "append_task_guide_to_skill_body",
    "format_flat_skill_markdown",
    "register_component_tools",
    "register_http_tools",
    "resolve_catalog_handle_from_args",
    "operation_to_spec",
]
