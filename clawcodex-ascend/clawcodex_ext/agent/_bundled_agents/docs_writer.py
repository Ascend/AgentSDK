"""Bundled ``docs-writer`` agent."""

from __future__ import annotations

from clawcodex_ext.agent.policy import (
    IDENTITY_DOCS_WRITER,
    NORM_CODE_AUTHOR,
    TOOL_SET_AUTHOR,
    build_agent_prompt,
)
from clawcodex_ext.agent.registry import AgentRegistry

_SYSTEM_PROMPT = build_agent_prompt(
    identity=IDENTITY_DOCS_WRITER,
    norms=[NORM_CODE_AUTHOR],
    extra=(
        "Only create or edit documentation files when the user has "
        "explicitly asked for documentation. Default to editing existing "
        "files. Do not add sections that the user did not ask for."
    ),
)


@AgentRegistry.register(
    "docs-writer",
    when_to_use=(
        "Documentation specialist. Use when the user explicitly asks for documentation to be written or updated."
    ),
    tools=TOOL_SET_AUTHOR,
    permission_mode="acceptEdits",
)
def _docs_writer_prompt() -> str:
    return _SYSTEM_PROMPT
