"""Transform professional workflows into reusable agents, skills, and tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime.composite_tools.models import CompositeToolSpec

from .adapters import DEFAULTS as _DEFAULTS, fill_defaults as _fill_defaults

_fill_defaults(_DEFAULTS)

from .core import (  # noqa: E402
    AGENT_MD_TEMPLATE,
    AGENT_TEMPLATE,
    OVERVIEW_AGENT_TEMPLATE,
    SKILL_MD_TEMPLATE_JINJA,
    SKILL_TEMPLATE,
    MacroCoverage,
    MappingRule,
    ParamSpec,
    ResourceCatalog,
    ResourceHandler,
    ResourceRecord,
    SdkMethod,
    SdkParser,
    SourceCodeParser,
    SourceComponent,
    SourceOperation,
    ToolRetrievalIndex,
    ToolRetrievalProfile,
    get_resource_handler,
    get_resource_record,
    load_tool_retrieval_index,
    register_resource_handler,
    require_resource_handler,
    resolve_agent_by_type,
    resolve_default_agent,
    resolve_resource_catalog_path,
)
from .resource_catalog import build_resource_record_from_create  # noqa: E402
from .runtime import (  # noqa: E402
    AgentBuildResult,
    AgentBuilder,
    AgentComponentInfo,
    AgentMarkdownWriter,
    GroupStrategy,
    MatchTarget,
    MatchType,
    SkillGrouper,
    SkillSpec,
    WorkflowStage,
    group_source_components,
    register_component_tools,
    register_http_tools,
)
from .workflow_mode import (  # noqa: E402
    DiscriminationResult,
    WorkflowDiscriminator,
    discriminate_and_extract,
    extract_workflow,
)


def register_composite_tools(
    *,
    persist: bool = True,
    bundle_dir: Path | None = None,
    sdk_source_dir: str = "",
) -> dict[str, str]:
    """Register composite tools after the runtime package is initialized."""
    from .runtime.composite_tools import register_composite_tools as _register

    return _register(
        persist=persist,
        bundle_dir=bundle_dir,
        sdk_source_dir=sdk_source_dir,
    )


def emit_composite_workflow_yaml(
    spec: CompositeToolSpec,
    output_dir: str | Path,
    *,
    project_name: str = "",
) -> Path | None:
    """Emit composite workflow YAML through the canonical runtime."""
    from .runtime.composite_tools import emit_composite_workflow_yaml as _emit

    return _emit(spec, output_dir, project_name=project_name)


__all__ = [
    "AGENT_MD_TEMPLATE",
    "AGENT_TEMPLATE",
    "AgentBuildResult",
    "AgentBuilder",
    "AgentComponentInfo",
    "AgentMarkdownWriter",
    "DiscriminationResult",
    "GroupStrategy",
    "MacroCoverage",
    "MappingRule",
    "MatchTarget",
    "MatchType",
    "OVERVIEW_AGENT_TEMPLATE",
    "ParamSpec",
    "ResourceCatalog",
    "ResourceHandler",
    "ResourceRecord",
    "SKILL_MD_TEMPLATE_JINJA",
    "SKILL_TEMPLATE",
    "SdkMethod",
    "SdkParser",
    "SkillGrouper",
    "SkillSpec",
    "SourceCodeParser",
    "SourceComponent",
    "SourceOperation",
    "ToolRetrievalIndex",
    "ToolRetrievalProfile",
    "WorkflowDiscriminator",
    "WorkflowStage",
    "build_resource_record_from_create",
    "discriminate_and_extract",
    "emit_composite_workflow_yaml",
    "extract_workflow",
    "get_resource_handler",
    "get_resource_record",
    "group_source_components",
    "load_tool_retrieval_index",
    "register_component_tools",
    "register_composite_tools",
    "register_http_tools",
    "register_resource_handler",
    "require_resource_handler",
    "resolve_agent_by_type",
    "resolve_default_agent",
    "resolve_resource_catalog_path",
]
