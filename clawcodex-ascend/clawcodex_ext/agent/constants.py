"""Agent system constants.

Mirrors typescript/src/constants/tools.ts and typescript/src/tools/AgentTool/constants.ts.
"""

from __future__ import annotations

# --- Tool name constants ---
AGENT_TOOL_NAME = "Agent"
LEGACY_AGENT_TOOL_NAME = "Task"
WORKFLOW_TOOL_NAME = "Workflow"

# --- Built-in agent type identifiers ---
VERIFICATION_AGENT_TYPE = "verification"

# Built-in agents that run once and return a report — the parent never
# sends messages back to continue them. Skip the agentId/SendMessage/usage
# trailer for these to save tokens.
ONE_SHOT_BUILTIN_AGENT_TYPES: frozenset[str] = frozenset(
    [
        "Explore",
        "Plan",
    ]
)

# --- Tool filtering sets ---

# Tools always blocked for ALL agents (built-in and custom).
# Mirrors ALL_AGENT_DISALLOWED_TOOLS from typescript/src/constants/tools.ts.
ALL_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset(
    [
        "TaskOutput",
        "ExitPlanMode",
        "EnterPlanMode",
        AGENT_TOOL_NAME,
        # Prevent recursive workflow execution inside subagents.
        WORKFLOW_TOOL_NAME,
        "AskUserQuestion",
        "TaskStop",
        "Brief",
    ]
)

# Additional tools blocked for custom (non-built-in) agents.
# Mirrors CUSTOM_AGENT_DISALLOWED_TOOLS.
CUSTOM_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset(
    [
        *ALL_AGENT_DISALLOWED_TOOLS,
    ]
)

# Whitelist of tools allowed for async (background) agents.
# Mirrors ASYNC_AGENT_ALLOWED_TOOLS from typescript/src/constants/tools.ts
# (:53-69). "PowerShell" (via TS SHELL_TOOL_NAMES) is deliberately absent —
# this port has no PowerShell tool. This set is BOTH the async-agent runtime
# whitelist (agent_tool_utils.filter_tools_for_agent) and the source of the
# coordinator's rendered worker-tools list (coordinator/mode.py).
ASYNC_AGENT_ALLOWED_TOOLS: frozenset[str] = frozenset(
    [
        "Read",
        "WebSearch",
        "TodoWrite",
        "Grep",
        "WebFetch",
        "Glob",
        "Bash",
        "Edit",
        "Write",
        "NotebookEdit",
        "Skill",
        "StructuredOutput",
        "ToolSearch",
        "EnterWorktree",
        "ExitWorktree",
        # Background agents participating in an LKB Plan need the same
        # public task surface as foreground agents. TaskOutput/TaskStop
        # remain blocked by ALL_AGENT_DISALLOWED_TOOLS.
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskUpdate",
        "Lkb",
    ]
)

# Default agent system prompt when agent definition doesn't provide one.
# Mirrors DEFAULT_AGENT_PROMPT from typescript/src/constants/prompts.ts.
DEFAULT_AGENT_PROMPT = (
    "You are an agent for Claw Codex. Given the user's message, you should use the "
    "tools available to complete the task. Complete the task fully\u2014don't gold-plate, "
    "but don't leave it half-done. When you complete the task, respond with a concise "
    "report covering what was done and any key findings \u2014 the caller will relay this "
    "to the user, so it only needs the essentials."
)

# Fork-specific constants
FORK_SUBAGENT_TYPE = "fork"
FORK_BOILERPLATE_TAG = "fork-boilerplate"
FORK_DIRECTIVE_PREFIX = "Your directive: "

# When an agent's allowlist exceeds this count, inline tool names in Agent-tool
# prompts are replaced with a count summary to avoid token blow-up.
MAX_INLINE_TOOL_DISPLAY: int = 20

# Minimal tool set for SOP-converted proxy agents. Domain tools load on demand
# via Skill (allowed_tools filter) or ToolSearch (deferred schemas).
POS_PROXY_BASE_TOOLS: frozenset[str] = frozenset(
    [
        "Skill",
        "ToolSearch",
        "Agent",
        "Read",
        "TodoWrite",
        "StructuredOutput",
        "resource-catalog",
        # F-57 session-macro register / promote (not in convert bundles).
        "register-macro-workflow",
        "register-macro-from-trace",
        "promote-macro-workflow",
    ]
)

# Domain sub-agents: no codebase exploration — SDK via Skill + ToolSearch only.
# Read + resource-catalog allow F-56 catalog inspect without workspace Grep.
# Macro tools stay visible so subagents hit capability gate (not missing-tool).
POS_SOP_DOMAIN_AGENT_TOOLS: frozenset[str] = frozenset(
    [
        "Skill",
        "ToolSearch",
        "Bash",
        "Read",
        "TodoWrite",
        "StructuredOutput",
        "resource-catalog",
        "register-macro-workflow",
        "register-macro-from-trace",
        "promote-macro-workflow",
    ]
)

# Base tools always kept visible when a skill's allowed_tools filter runs.
SKILL_CONTEXT_BASE_TOOLS: frozenset[str] = frozenset(
    [
        "agent",
        "skill",
        "toolsearch",
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "listdir",
        "webfetch",
        "task",
        "todos",
        "advisor",
    ]
)
