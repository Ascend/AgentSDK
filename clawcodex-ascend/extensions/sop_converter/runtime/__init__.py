"""Agent-runtime integration layer for the SOP converter."""

# This module is a compatibility facade: imported names are its public API and
# ``__all__`` is derived dynamically after the imports.
# ruff: noqa: F401

from .agent_builder import (
    AgentBuildResult,
    AgentBuilder,
    AgentPersistenceSpec,
    persist_converted_agent,
    write_agent_markdown,
)
from .agent_md_writer import AgentComponentInfo, AgentMarkdownWriter, WorkflowStage
from .bundle_agents import register_bundle_agents
from .bundle_context import (
    BundleContext,
    activate_bundle_isolation,
    apply_sdk_source_working_directory,
    build_bundle_context,
    collect_tool_names_from_bundle_specs,
    collect_tool_names_from_skills,
    ensure_bundle_tools_registered,
    filter_tools_for_bundle,
    get_active_bundle,
    is_pos_converter_tool,
    is_sop_converter_spec_source,
    load_bundle_macro_routes,
    load_bundle_persisted_tools,
    prune_registry_to_bundle,
    set_active_bundle,
)
from .bundle_discovery import (
    discover_workspace_bundle,
    list_workspace_bundle_candidates,
    overview_has_sop_skills,
)
from .bundle_skills import BundleSkillLoadResult, register_bundle_skills, resolve_bundle_skill_workspace
from .composite_runtime import (
    CompositeResult,
    CompositeWorkflowError,
    CompositeWorkflowRunner,
    CompositeWorkflowSpec,
    CompositeWorkflowStep,
    StepTrace,
    normalize_workflow_output,
)
from .composite_workflows import invoke_existing_agent_workflow, resume_resource_workflow
from .convert_sop_skill import convert_sop_to_agent, get_prompt_for_command
from .cross_domain_orchestration import (
    OrchestrationRoute,
    OrchestrationStep,
    build_tool_to_agent_map,
    discover_orchestration_routes,
    format_orchestration_routes_block,
    generate_orchestration_routes_markdown,
    skill_name_to_agent,
    write_orchestration_routes,
)
from .sdk_overview import (
    format_sdk_overview_block,
    generate_io_sdk_overview_markdown,
    generate_sdk_overview_markdown,
    write_sdk_overview,
)
from .skill_grouper import (
    GroupResult,
    GroupStrategy,
    MappingRule,
    MatchTarget,
    MatchType,
    SkillGrouper,
    SkillSpec,
    group_into_skills,
    group_source_components,
)
from .sop_exploration_guard import check_bundle_source_exploration, sop_exploration_permission_check
from .sop_routing import (
    check_bundle_agent_delegation,
    list_domain_agent_types,
    looks_like_direct_sdk_execution,
    refresh_domain_agent_sop_prompts,
    requested_agent_types_in_prompt,
)
from .startup_agent import build_bundle_overview_agent_definition
from .task_guide import (
    append_task_guide_to_skill_body,
    build_operation_index,
    format_flat_skill_markdown,
    generate_task_guide_markdown,
    is_entry_point,
)
from .tool_registry_bridge import (
    operation_to_spec,
    register_component_tools,
    register_http_tools,
    resolve_catalog_handle_from_args,
)

__all__ = [name for name in globals() if not name.startswith("_")]
