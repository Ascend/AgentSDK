"""SOP converter — transforms professional workflows into reusable Agents.

Three-layer mapping:
    SOP (Standard Operating Procedure)     → Agent
    workflow steps                → Skill
    SDK interfaces               → atomic tools

Architecture::

    SDK Spec + Requirements
         │
         ▼
    SdkParser ──────────────────► atomic_tools: list[str]
         │
         ▼
    SkillGrouper ────────────────► skills: list[SkillSpec]
         │
         ▼
    AgentBuilder ────────────────► agent: AgentDefinition
         │
         ▼
    Persistence / Registration
"""

# ── Dependency Injection ────────────────────────────────────────────────────
# Populate the module-level DEFAULTS container with the default adapter
# implementations (Layer 1.5).  This must happen before any SOP converter
# module imports ``DEFAULTS`` so the factories are available at runtime.
# See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.4.
from .adapters import DEFAULTS as _DEFAULTS, fill_defaults as _fill_defaults

_fill_defaults(_DEFAULTS)

# ── Core (pure algorithm, no clawcodex_ext / src dependency) ────────────────
from .core import (  # noqa: E402  (DI fill must run first)
    SdkParser,
    SdkMethod,
    SourceCodeParser,
    SourceComponent,
    SourceOperation,
    ParamSpec,
    AGENT_TEMPLATE,
    SKILL_TEMPLATE,
    MappingRule,
    AGENT_MD_TEMPLATE,
    SKILL_MD_TEMPLATE_JINJA,
    OVERVIEW_AGENT_TEMPLATE,
    resolve_default_agent,
    resolve_agent_by_type,
    ResourceCatalog,
    ResourceRecord,
    get_resource_record,
    resolve_resource_catalog_path,
    ResourceHandler,
    get_resource_handler,
    register_resource_handler,
    require_resource_handler,
    MacroCoverage,
    ToolRetrievalIndex,
    ToolRetrievalProfile,
    load_tool_retrieval_index,
)

# F-56 catalog write facade (canonical implementation lives in core/).
from .resource_catalog import build_resource_record_from_create  # noqa: E402

# ── Runtime (agent runtime integration layer) ────────────────────────────────
from .runtime import (  # noqa: E402
    SkillGrouper,
    SkillSpec,
    GroupStrategy,
    MatchType,
    MatchTarget,
    group_source_components,
    AgentBuilder,
    AgentBuildResult,
    AgentMarkdownWriter,
    AgentComponentInfo,
    WorkflowStage,
    register_component_tools,
    register_http_tools,
)

# ── workflow_mode (kept independent) ────────────────────────────────────────
from .workflow_mode import (  # noqa: E402
    WorkflowDiscriminator,
    DiscriminationResult,
    discriminate_and_extract,
    extract_workflow,
)


def register_composite_tools(*args, **kwargs):
    from .runtime.composite_tools import register_composite_tools as _register

    return _register(*args, **kwargs)


def emit_composite_workflow_yaml(*args, **kwargs):
    from .runtime.composite_tools import emit_composite_workflow_yaml as _emit

    return _emit(*args, **kwargs)


__all__ = [
    "SdkParser",
    "SdkMethod",
    "SkillGrouper",
    "SkillSpec",
    "GroupStrategy",
    "MatchType",
    "MatchTarget",
    "group_source_components",
    "AgentBuilder",
    "AgentBuildResult",
    "AGENT_TEMPLATE",
    "SKILL_TEMPLATE",
    "MappingRule",
    "AGENT_MD_TEMPLATE",
    "SKILL_MD_TEMPLATE_JINJA",
    "OVERVIEW_AGENT_TEMPLATE",
    "SourceCodeParser",
    "SourceComponent",
    "SourceOperation",
    "ParamSpec",
    "AgentMarkdownWriter",
    "AgentComponentInfo",
    "WorkflowStage",
    "resolve_default_agent",
    "resolve_agent_by_type",
    "register_component_tools",
    "register_http_tools",
    "ResourceCatalog",
    "ResourceRecord",
    "build_resource_record_from_create",
    "get_resource_record",
    "resolve_resource_catalog_path",
    "ResourceHandler",
    "get_resource_handler",
    "register_resource_handler",
    "require_resource_handler",
    "MacroCoverage",
    "ToolRetrievalIndex",
    "ToolRetrievalProfile",
    "load_tool_retrieval_index",
    "register_composite_tools",
    "emit_composite_workflow_yaml",
    "WorkflowDiscriminator",
    "DiscriminationResult",
    "discriminate_and_extract",
    "extract_workflow",
]
