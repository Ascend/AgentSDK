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

"""sop_converter core — pure algorithm layer.

This subpackage contains ONLY algorithm/parsing logic with zero
dependencies on ``clawcodex_ext.*`` or ``src.*``.  It can be imported
and tested without any Claude Code runtime.

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.5.
"""

from typing import Any

from .sdk_parser import SdkParser, SdkMethod, SdkParam, SdkParseResult, parse_sdk_spec
from .source_parser import (
    SourceCodeParser,
    SourceComponent,
    SourceOperation,
    ParamSpec,
    detect_interactive_input,
    is_async_generator_operation,
    infer_type_hint_from_description,
)
from .templates import (
    MappingRule,
    AGENT_TEMPLATE,
    SKILL_TEMPLATE,
    AGENT_MD_TEMPLATE,
    SKILL_MD_TEMPLATE_JINJA,
    OVERVIEW_AGENT_TEMPLATE,
    MAPPING_RULES_TEMPLATE,
    SKILL_FRAGMENT_TEMPLATE,
)
from .default_agent import resolve_default_agent, resolve_agent_by_type
from .resource_catalog import (
    ResourceCatalog,
    ResourceRecord,
    ResourceCatalogLocation,
    CatalogExecutionContext,
    ResourceCatalogError,
    get_resource_record,
    resolve_resource_catalog_path,
    get_agent_record,
    resource_error,
    normalize_resource_type,
    agent_entry_to_resource_record,
    build_resource_record_from_create,
    iter_resource_catalog_locations,
)
from .resource_handlers import (
    ResourceHandler,
    ResourceHandlerError,
    get_resource_handler,
    register_resource_handler,
    require_resource_handler,
    registered_resource_types,
    ensure_builtin_handlers,
)
from .tool_retrieval import (
    MacroCoverage,
    ToolRetrievalIndex,
    ToolRetrievalProfile,
    load_tool_retrieval_index,
    write_tool_retrieval_index,
    retrieval_index_path,
    index_from_routes,
    normalize_tool_ref,
    resolve_tool_references,
)
from .asciicast_projector import SopStageProjector
from .search_tags import generate_search_tags
from .intent_tags import (
    collect_intent_phrases,
    format_search_suggestions,
    get_intent_tags,
    enrich_with_intent_tags,
)
from .type_schema import reset_schema_probe_runtime_state
from .sdk_dependency_resolver import SdkDependencySpec, resolve_sdk_dependencies
from .sdk_serialization import (
    coerce_sdk_type,
    to_jsonable,
    dumps_sdk_result,
    normalize_mapping_inputs,
    coerce_mapping_value,
    resolve_env_references,
)
from .tool_dependencies import (
    ToolOperationDeps,
    build_tool_dependency_index,
    dependency_schema_fragment,
    enrich_input_schema_with_dependencies,
    dependency_description_suffix,
    dependency_search_tags,
    sanitize_type_name,
    extract_type_roots,
    raw_tool_name,
    to_kebab_tool_name,
)
from .tool_state import (
    resolve_sessions_dir,
    tool_state_path,
    load_tool_state,
    save_tool_state,
    set_session_secret,
    get_session_secrets,
    is_configure_tool,
    is_cli_secret_consumer,
    enrich_tool_input,
    persist_configure_secrets,
)
from .path_resolver import (
    resolve_source_file,
    infer_extra_sys_path_entries,
    format_extra_sys_path_inserts,
)
from .import_alias_resolver import resolve_module_path, type_identity_key, ModuleImportIndex
from .runtime_paths import (
    is_wsl_runtime,
    wsl_path_to_windows_path,
    windows_path_to_wsl_path,
    normalize_runtime_path,
)
from .sop_prompts import (
    format_sdk_source_dir_block,
    agent_type_to_skill_name,
    pick_pipeline_execute_tool,
    infer_stage_label_from_skill,
    stage_agent_sop_body,
    domain_agent_sop_body,
    format_overview_stage_pipeline_block,
    append_sop_overview_routing,
)
from .workflow_project import (
    read_workflow_project_name,
    read_workflow_first_stage_skill_name,
    read_workflow_stage_for_agent,
    read_workflow_stage_pipeline,
    is_prefixed_stage_agent,
)
from .bundle_manifest import (
    BundleManifest,
    manifest_path_for_bundle,
    write_bundle_manifest,
    read_bundle_manifest,
    resolve_sdk_source_dir,
)
from .bundle_resources import (
    ResourceBinding,
    load_resource_bindings,
)
from .bundle_workflow import (
    bundle_dir_from_workflow_yaml,
    resolve_bundle_workflow_yaml,
    discover_workflow_yaml,
    workflow_artifacts_enabled,
)
from .agent_catalog import AgentCatalog, AgentCatalogEntry
from .agent_runtime import materialize_agent, invoke_agent, AgentRuntimeError


_BUNDLE_VENV_EXPORTS = frozenset(
    {
        "bundle_venv_dir",
        "bundle_venv_python",
        "bundle_venv_site_packages",
        "activate_bundle_venv_imports",
        "in_process_bundle_venv_reexec",
        "is_venv_ready",
        "ensure_bundle_venv",
        "ensure_bundle_venv_and_reexec",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily expose bundle-venv APIs from the top-level canonical module."""

    if name in _BUNDLE_VENV_EXPORTS:
        from .. import bundle_venv

        return getattr(bundle_venv, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The bundle-venv names are provided lazily by ``__getattr__`` above.
# pylint: disable=undefined-all-variable
__all__ = [
    "SdkParser",
    "SdkMethod",
    "SdkParam",
    "SdkParseResult",
    "parse_sdk_spec",
    "SourceCodeParser",
    "SourceComponent",
    "SourceOperation",
    "ParamSpec",
    "detect_interactive_input",
    "is_async_generator_operation",
    "infer_type_hint_from_description",
    "MappingRule",
    "AGENT_TEMPLATE",
    "SKILL_TEMPLATE",
    "AGENT_MD_TEMPLATE",
    "SKILL_MD_TEMPLATE_JINJA",
    "OVERVIEW_AGENT_TEMPLATE",
    "MAPPING_RULES_TEMPLATE",
    "SKILL_FRAGMENT_TEMPLATE",
    "resolve_default_agent",
    "resolve_agent_by_type",
    "ResourceCatalog",
    "ResourceRecord",
    "ResourceCatalogLocation",
    "CatalogExecutionContext",
    "ResourceCatalogError",
    "get_resource_record",
    "resolve_resource_catalog_path",
    "get_agent_record",
    "resource_error",
    "agent_entry_to_resource_record",
    "build_resource_record_from_create",
    "iter_resource_catalog_locations",
    "ResourceHandler",
    "ResourceHandlerError",
    "get_resource_handler",
    "register_resource_handler",
    "require_resource_handler",
    "registered_resource_types",
    "ensure_builtin_handlers",
    "MacroCoverage",
    "ToolRetrievalIndex",
    "ToolRetrievalProfile",
    "load_tool_retrieval_index",
    "write_tool_retrieval_index",
    "retrieval_index_path",
    "index_from_routes",
    "normalize_tool_ref",
    "resolve_tool_references",
    "SopStageProjector",
    "generate_search_tags",
    "collect_intent_phrases",
    "format_search_suggestions",
    "get_intent_tags",
    "enrich_with_intent_tags",
    "reset_schema_probe_runtime_state",
    "SdkDependencySpec",
    "resolve_sdk_dependencies",
    "coerce_sdk_type",
    "to_jsonable",
    "dumps_sdk_result",
    "normalize_mapping_inputs",
    "coerce_mapping_value",
    "resolve_env_references",
    "ToolOperationDeps",
    "build_tool_dependency_index",
    "dependency_schema_fragment",
    "enrich_input_schema_with_dependencies",
    "dependency_description_suffix",
    "dependency_search_tags",
    "sanitize_type_name",
    "extract_type_roots",
    "raw_tool_name",
    "to_kebab_tool_name",
    "resolve_sessions_dir",
    "tool_state_path",
    "load_tool_state",
    "save_tool_state",
    "set_session_secret",
    "get_session_secrets",
    "is_configure_tool",
    "is_cli_secret_consumer",
    "enrich_tool_input",
    "persist_configure_secrets",
    "resolve_source_file",
    "infer_extra_sys_path_entries",
    "format_extra_sys_path_inserts",
    "resolve_module_path",
    "type_identity_key",
    "ModuleImportIndex",
    "is_wsl_runtime",
    "wsl_path_to_windows_path",
    "windows_path_to_wsl_path",
    "normalize_runtime_path",
    "format_sdk_source_dir_block",
    "agent_type_to_skill_name",
    "pick_pipeline_execute_tool",
    "infer_stage_label_from_skill",
    "stage_agent_sop_body",
    "domain_agent_sop_body",
    "format_overview_stage_pipeline_block",
    "append_sop_overview_routing",
    "read_workflow_project_name",
    "read_workflow_first_stage_skill_name",
    "read_workflow_stage_for_agent",
    "read_workflow_stage_pipeline",
    "is_prefixed_stage_agent",
    "BundleManifest",
    "manifest_path_for_bundle",
    "write_bundle_manifest",
    "read_bundle_manifest",
    "resolve_sdk_source_dir",
    "ResourceBinding",
    "load_resource_bindings",
    "bundle_dir_from_workflow_yaml",
    "resolve_bundle_workflow_yaml",
    "discover_workflow_yaml",
    "workflow_artifacts_enabled",
    "bundle_venv_dir",
    "bundle_venv_python",
    "bundle_venv_site_packages",
    "activate_bundle_venv_imports",
    "in_process_bundle_venv_reexec",
    "is_venv_ready",
    "ensure_bundle_venv",
    "ensure_bundle_venv_and_reexec",
    "AgentCatalog",
    "AgentCatalogEntry",
    "materialize_agent",
    "invoke_agent",
    "AgentRuntimeError",
    # Keep legacy aliases for backward compatibility
    "normalize_resource_type",
]
