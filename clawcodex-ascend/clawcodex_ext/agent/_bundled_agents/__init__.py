"""Bundled decoupled agents.

Importing this package triggers the ``@AgentRegistry.register`` side
effect of every submodule. Callers should import it only after
``src.agent.agent_definitions`` has finished initialising.
"""

from __future__ import annotations

from clawcodex_ext.agent._bundled_agents import (  # noqa: F401
    code_reviewer,
    docs_writer,
    test_runner,
)
