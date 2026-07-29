"""Agent registry — programmatic registration of AgentDefinition for
``clawcodex_ext`` and ``extensions/`` paths.

Mirrors the shape of :class:`extensions.capabilities.adapter_protocol.AdapterRegistry`
(``@AdapterRegistry.register(...)`` decorator used by
``clawcodex_ext/agent/_outlines_adapter.py``) so extension authors have a
single familiar pattern to add both adapters and agents.

Usage
-----
Decorator form (system prompt produced by the decorated function)::

    from clawcodex_ext.agent.registry import AgentRegistry
    from clawcodex_ext.agent.policy import IDENTITY_CODE_REVIEWER, build_agent_prompt

    @AgentRegistry.register(
        "code-reviewer",
        when_to_use="Use after writing code to get an independent review.",
        tools=["Read", "Glob", "Grep", "Bash"],
        disallowed_tools=["Edit", "Write"],
        permission_mode="default",
    )
    def _code_reviewer_prompt() -> str:
        return build_agent_prompt(identity=IDENTITY_CODE_REVIEWER, ...)

Explicit form (caller owns the AgentDefinition)::

    AgentRegistry.register_definition(my_agent_def)

Registration is idempotent and last-wins by ``agent_type``. A re-registration
of an existing ``agent_type`` is logged at INFO level and silently replaces
the previous definition; downstream code can introspect the override via
:func:`extensions.capabilities.adapter_protocol.AdapterInfo` semantics or
simply by looking up the new entry in :meth:`AgentRegistry.all`.

This registry is *in-process* state. Call :meth:`AgentRegistry.clear` in
tests, and avoid mutating it from concurrent threads without external
synchronisation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from clawcodex_ext.agent.constants import ALL_AGENT_DISALLOWED_TOOLS

if TYPE_CHECKING:
    from clawcodex_ext.agent.agent_definitions import AgentDefinition, AgentSource

logger = logging.getLogger(__name__)


# Public source-tag constants consumed by the merge-order wiring in
# ``src/agent/load_agents_dir.py``. The string values must stay in sync
# with the ``AgentSource`` Literal in ``src/agent/agent_definitions.py``.
SOURCE_CLAWCODEX_EXT = "clawcodex_ext"
SOURCE_EXTENSIONS = "extensions"


def _normalise_disallowed(disallowed: list[str] | None) -> list[str] | None:
    """Merge ALL_AGENT_DISALLOWED_TOOLS into the user-supplied deny-list.

    Mirrors the runtime behaviour in :mod:`src.agent.agent_tool_utils`
    where the hard-coded deny-list is always enforced on top of the
    agent's own preferences. We do it here at registration time so the
    stored ``AgentDefinition`` already reflects the effective toolset.
    """
    if disallowed is None:
        return list(ALL_AGENT_DISALLOWED_TOOLS)
    seen: set[str] = set(disallowed)
    merged: list[str] = list(disallowed)
    for tool in ALL_AGENT_DISALLOWED_TOOLS:
        if tool not in seen:
            merged.append(tool)
            seen.add(tool)
    return merged


class AgentRegistry:
    """In-process registry of :class:`AgentDefinition` instances.

    Agents are registered at module-import time via the :meth:`register`
    decorator or the :meth:`register_definition` function. Lookups are
    last-wins by ``agent_type``.

    The class-level state is process-global by design — it matches the
    existing ``AdapterRegistry`` precedent in
    ``extensions/capabilities/adapter_protocol.py``.
    """

    # Class-level state. ``_by_type`` mirrors ``_definitions`` for O(1) lookup.
    _definitions: list[AgentDefinition] = []
    _by_type: dict[str, AgentDefinition] = {}

    # ---- Query API ---------------------------------------------------

    @classmethod
    def all(cls) -> list[AgentDefinition]:
        """Return a snapshot of all registered agents in registration order."""
        return list(cls._definitions)

    @classmethod
    def by_source(cls, source: AgentSource) -> list[AgentDefinition]:
        """Return all registered agents whose ``source`` matches."""
        return [a for a in cls._definitions if a.source == source]

    @classmethod
    def find(cls, agent_type: str) -> AgentDefinition | None:
        """Look up an agent by its ``agent_type``."""
        return cls._by_type.get(agent_type)

    @classmethod
    def clear(cls) -> None:
        """Drop every registered agent. Intended for tests and process
        boundaries (e.g., REPL session reset), not for production use.
        """
        cls._definitions.clear()
        cls._by_type.clear()

    # ---- Registration API --------------------------------------------

    @classmethod
    def register_definition(cls, agent: "AgentDefinition") -> "AgentDefinition":
        """Register a pre-built :class:`AgentDefinition` directly.

        Last-wins by ``agent_type``. Returns the stored definition (which
        may differ from the input if a previous entry of the same type
        was replaced).
        """
        return cls._add(agent)

    @classmethod
    def register(
        cls,
        agent_type: str,
        *,
        when_to_use: str,
        tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        model: str | None = None,
        permission_mode: str | None = None,
        max_turns: int | None = None,
        background: bool = False,
        color: str | None = None,
        memory: str | None = None,
        omit_claude_md: bool = False,
        skills: list[str] | None = None,
        isolation: str | None = None,
        source: str = SOURCE_CLAWCODEX_EXT,
        base_dir: str = "clawcodex_ext",
    ) -> Callable[[Callable[..., str]], "AgentDefinition"]:
        """Decorator that registers an agent whose system prompt is
        produced by the decorated callable.

        Parameters
        ----------
        agent_type
            Unique identifier used in ``subagent_type=...`` and
            ``@agent-foo`` mentions.
        when_to_use
            One-line description surfaced to the *parent* agent so it
            knows when to spawn this sub-agent. Required.
        tools
            Explicit tool allow-list. ``None`` or ``["*"]`` means "all".
        disallowed_tools
            Additional tools to deny on top of the always-denied set
            (``ALL_AGENT_DISALLOWED_TOOLS``).
        permission_mode
            See ``src/permissions/types.py`` for accepted values.
        source
            Either ``SOURCE_CLAWCODEX_EXT`` (default) or
            ``SOURCE_EXTENSIONS`` for third-party packages.
        """

        def decorator(prompt_fn: Callable[..., str]) -> "AgentDefinition":
            from clawcodex_ext.agent.agent_definitions import AgentDefinition

            normalised = _normalise_disallowed(disallowed_tools)
            agent = AgentDefinition(
                agent_type=agent_type,
                when_to_use=when_to_use,
                tools=tools,
                source=source,  # type: ignore[arg-type]
                base_dir=base_dir,
                model=model,
                permission_mode=permission_mode,  # type: ignore[arg-type]
                max_turns=max_turns,
                background=background,
                color=color,
                memory=memory,
                omit_claude_md=omit_claude_md,
                disallowed_tools=normalised,
                skills=skills,
                isolation=isolation,  # type: ignore[arg-type]
                required_mcp_servers=None,
                mcp_servers=None,
                effort=None,
                get_system_prompt=prompt_fn,
            )
            return cls._add(agent)

        return decorator

    # ---- Internal ----------------------------------------------------

    @classmethod
    def _add(cls, agent: "AgentDefinition") -> "AgentDefinition":
        existing = cls._by_type.get(agent.agent_type)
        if existing is not None and existing is not agent:
            logger.info(
                "agent registry: %r (source=%s) overridden by %r (source=%s)",
                existing.agent_type,
                existing.source,
                agent.agent_type,
                agent.source,
            )
            try:
                cls._definitions.remove(existing)
            except ValueError:
                # The list can drift if callers mutate it directly; the
                # by_type map is the source of truth, so this is safe.
                pass
        cls._definitions.append(agent)
        cls._by_type[agent.agent_type] = agent
        return agent


register = AgentRegistry.register


__all__ = [
    "AgentRegistry",
    "SOURCE_CLAWCODEX_EXT",
    "SOURCE_EXTENSIONS",
    "register",
]
