"""Built-in default templates (P85-E).

P85-E ships a small, opinionated set of canonical templates that every
install starts with. The set mirrors the built-in agents in
:mod:`src.agent.agent_definitions` so users have a familiar starting
point: an agent that says ``template: explore`` should behave the same
as spawning the built-in :data:`~src.agent.agent_definitions.EXPLORE_AGENT`.

The set is intentionally narrow:

* ``general-purpose`` — full tool access; the default for ad-hoc work.
* ``explore`` — read-only file-search agent (no Edit/Write/Bash).
* ``plan`` — read-only architect agent; inherits the parent model.
* ``fix`` — full tools + a bounded turn budget, suited for bug-fix loops.
* ``review`` — read-only + a larger turn budget for code-review passes.

Storage strategy — **Python data, not YAML**. The five templates are
constants, not user-editable, so they live as :class:`Template`
dataclass instances rather than as ``*.yml`` files shipped alongside
the code. That keeps the import graph clean (no PyYAML required to
materialise defaults) and makes the templates inspectable from
Python without a filesystem round-trip.

Source label — ``SOURCE_BUILT_IN = "built-in"`` matches the
:class:`~src.agent.agent_definitions.AgentSource` literal so the
``/template list --source built-in`` filter works the same way users
expect from ``/agent list``.

Precedence — built-ins register **first** inside
:func:`~src.services.templates.bootstrap.bootstrap_default_templates`,
then user / project / managed layers can overwrite them. That way
"a user template named ``plan``" wins automatically without any code
change; no flag is required to shadow a built-in.
"""

from __future__ import annotations

from .exceptions import TemplateAlreadyExistsError
from .models import Template
from .registry import TemplateRegistry

# Source label for built-in templates. Mirrors the literal in
# :class:`~src.agent.agent_definitions.AgentSource` so the same string
# works on both sides of the agent ↔ template boundary.
SOURCE_BUILT_IN = "built-in"

# Tools that perform file mutations or spawn other agents. The Explore
# and Plan templates forbid these so the resulting agent cannot
# accidentally rewrite the project it is reading.
_READ_ONLY_DISALLOWED: tuple[str, ...] = (
    "Agent",
    "ExitPlanMode",
    "Edit",
    "Write",
    "NotebookEdit",
)

# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------
#
# Each ``fields`` mapping contains ONLY keys that are real
# :class:`~src.agent.agent_definitions.AgentDefinition` field names
# minus the non-overridable set (agent_type, source, base_dir,
# template, get_system_prompt, callback). The resolver silently drops
# any other key, so a typo here is cheap — but we keep the surface
# tight so a typo is obvious at code-review time.
#
# ``metadata`` carries machine-readable tags so future consumers
# (e.g. a TUI template picker) can filter or group the catalogue
# without parsing the prose description.


_GENERAL_PURPOSE = Template(
    id="general-purpose",
    title="General Purpose Agent",
    description=(
        "Versatile agent for ad-hoc work. Full tool access, acceptEdits "
        "permission mode, no turn budget. Use when the task does not fit "
        "a more specialised template."
    ),
    fields={
        "tools": ["*"],
        "permission_mode": "acceptEdits",
    },
    metadata={
        "kind": "agent",
        "category": "general",
        "tags": "general,default,all-tools",
        "schema_version": "1",
        "version": "1",
    },
    source=SOURCE_BUILT_IN,
)


_EXPLORE = Template(
    id="explore",
    title="Explore Agent",
    description=(
        "Read-only file-search agent. Has Glob / Grep / Read / Bash "
        "(read-only commands) but cannot Edit, Write, or spawn "
        "sub-agents. Use when you need to map a codebase without "
        "risking accidental mutations."
    ),
    fields={
        "disallowed_tools": list(_READ_ONLY_DISALLOWED),
        "omit_claude_md": True,
    },
    metadata={
        "kind": "agent",
        "category": "search",
        "tags": "read-only,search,explore",
        "schema_version": "1",
        "version": "1",
    },
    source=SOURCE_BUILT_IN,
)


_PLAN = Template(
    id="plan",
    title="Plan Agent",
    description=(
        "Read-only architect agent. Same restrictions as Explore but "
        "inherits the parent's model so it can run on the same model "
        "the caller is using. Use for design / planning passes where "
        "you want consistent reasoning."
    ),
    fields={
        "disallowed_tools": list(_READ_ONLY_DISALLOWED),
        "model": "inherit",
        "omit_claude_md": True,
    },
    metadata={
        "kind": "agent",
        "category": "design",
        "tags": "read-only,plan,architect,inherit",
        "schema_version": "1",
        "version": "1",
    },
    source=SOURCE_BUILT_IN,
)


_FIX = Template(
    id="fix",
    title="Fix Agent",
    description=(
        "Full-tool agent with a bounded turn budget, suited for "
        "targeted bug-fix loops. Use when the diagnosis is already "
        "clear and you want the agent to make focused edits without "
        "wandering off into a long investigation."
    ),
    fields={
        "tools": ["*"],
        "permission_mode": "acceptEdits",
        "max_turns": 30,
    },
    metadata={
        "kind": "agent",
        "category": "edit",
        "tags": "fix,edit,bounded,all-tools",
        "schema_version": "1",
        "version": "1",
    },
    source=SOURCE_BUILT_IN,
)


_REVIEW = Template(
    id="review",
    title="Review Agent",
    description=(
        "Read-only agent with a generous turn budget, suited for "
        "multi-file code-review passes. Same restrictions as Explore "
        "but no early exit — use when you want a thorough sweep "
        "without risk of edits."
    ),
    fields={
        "disallowed_tools": list(_READ_ONLY_DISALLOWED),
        "max_turns": 50,
        "omit_claude_md": True,
    },
    metadata={
        "kind": "agent",
        "category": "review",
        "tags": "read-only,review,thorough",
        "schema_version": "1",
        "version": "1",
    },
    source=SOURCE_BUILT_IN,
)


# Canonical ordered set. The order is the display order in
# ``/template list`` and the order they are registered during
# bootstrap. Kept alphabetical by id for deterministic output.
_BUILT_IN_TEMPLATES: tuple[Template, ...] = (
    _EXPLORE,
    _FIX,
    _GENERAL_PURPOSE,
    _PLAN,
    _REVIEW,
)


def get_built_in_templates() -> tuple[Template, ...]:
    """Return the canonical built-in :class:`Template` set.

    Returns a fresh tuple each call (the underlying dataclasses are
    frozen so callers cannot mutate them). Tests use this to assert
    the catalogue without re-reading the module-level constants.
    """
    return tuple(_BUILT_IN_TEMPLATES)


def register_built_in_templates(
    registry: TemplateRegistry,
    *,
    overwrite: bool = True,
) -> int:
    """Register every built-in template into ``registry``.

    Returns the number of templates **newly added** (overwrites are
    not counted). When ``overwrite=False``, an existing template with
    the same id is preserved and the call is a no-op for that entry —
    :class:`TemplateAlreadyExistsError` is caught silently so the
    function is always safe to call repeatedly.

    The default ``overwrite=True`` matches :func:`bootstrap_default_templates`'s
    contract: built-ins are written first, then user / project /
    managed layers can re-write them with ``overwrite=True`` again,
    so the last-registered copy wins.
    """
    added = 0
    for tpl in _BUILT_IN_TEMPLATES:
        if tpl.id in registry:
            # Already present — only treat as "added" when the caller
            # asked us to overwrite AND the existing copy is itself
            # not the canonical built-in. Either way the count is
            # the number of NEW entries, so an overwrite is 0.
            try:
                registry.register(tpl, overwrite=overwrite)
            except TemplateAlreadyExistsError:
                continue
            continue
        try:
            registry.register(tpl, overwrite=overwrite)
            added += 1
        except TemplateAlreadyExistsError:
            # Race with another thread that registered the same id
            # between our ``in`` check and ``register``. Leave the
            # existing entry in place.
            continue
    return added


__all__ = [
    "SOURCE_BUILT_IN",
    "get_built_in_templates",
    "register_built_in_templates",
]
