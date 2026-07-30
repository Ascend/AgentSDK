"""Tool execution services — streaming executor, orchestrator, and tool hooks."""

from __future__ import annotations

from .orchestrator import (
    Batch,
    ToolUseBlock,
    _mark_tool_use_as_complete,
    classify_concurrency_safe,
    partition_tool_calls,
    run_tools,
)
from .orchestrator import (
    MessageUpdate as OrchestratorMessageUpdate,
)
from .streaming_executor import (
    BASH_TOOL_NAME,
    MessageUpdate,
    StreamingToolExecutor,
    TrackedTool,
)
from .tool_execution import (
    ContextModifier,
    MessageUpdateLazy,
    classify_tool_error,
    run_tool_use,
)
from .tool_hooks import (
    PreToolUseResult,
    resolve_hook_permission_decision,
    run_post_tool_use_failure_hooks,
    run_post_tool_use_hooks,
    run_pre_tool_use_hooks,
)
from .tool_result_persistence import (
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    PERSISTED_OUTPUT_CLOSING_TAG,
    PERSISTED_OUTPUT_TAG,
    PREVIEW_SIZE_BYTES,
    PersistedToolResult,
    PersistResult,
    PersistToolResultError,
    build_large_tool_result_message,
    compute_block_chars,
    generate_preview,
    get_persistence_threshold,
    is_persist_error,
    is_tool_result_content_empty,
    maybe_persist_large_tool_result,
    persist_tool_result,
    process_tool_result_block,
    resolve_tool_results_dir,
)

__all__ = [
    "BASH_TOOL_NAME",
    "DEFAULT_MAX_RESULT_SIZE_CHARS",
    "PERSISTED_OUTPUT_CLOSING_TAG",
    "PERSISTED_OUTPUT_TAG",
    "PREVIEW_SIZE_BYTES",
    "Batch",
    "ContextModifier",
    "MessageUpdate",
    "MessageUpdateLazy",
    "OrchestratorMessageUpdate",
    "PersistResult",
    "PersistToolResultError",
    "PersistedToolResult",
    "PreToolUseResult",
    "StreamingToolExecutor",
    "ToolUseBlock",
    "TrackedTool",
    "_mark_tool_use_as_complete",
    "build_large_tool_result_message",
    "classify_concurrency_safe",
    "classify_tool_error",
    "compute_block_chars",
    "generate_preview",
    "get_persistence_threshold",
    "is_persist_error",
    "is_tool_result_content_empty",
    "maybe_persist_large_tool_result",
    "partition_tool_calls",
    "persist_tool_result",
    "process_tool_result_block",
    "resolve_hook_permission_decision",
    "resolve_tool_results_dir",
    "run_post_tool_use_failure_hooks",
    "run_post_tool_use_hooks",
    "run_pre_tool_use_hooks",
    "run_tool_use",
    "run_tools",
]
