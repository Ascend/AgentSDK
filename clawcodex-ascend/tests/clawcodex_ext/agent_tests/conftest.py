"""Pytest fixtures for ``clawcodex_ext.agent`` tests.

The :class:`AgentRegistry` is process-global state. We use a
**save/restore** strategy so the bundled agents (which are eagerly
registered on first import of ``clawcodex_ext.agent``) stay visible
to every test, while mutations made by individual tests are
reverted on teardown. Tests that *want* a clean slate can call
:func:`AgentRegistry.clear` themselves.
"""

from __future__ import annotations


# pylint: disable=E0611
import pytest


@pytest.fixture(autouse=True)
def _reset_agent_registry():
    """Snapshot the in-process agent registry around each test.

    The first import of :mod:`clawcodex_ext.agent` (which happens
    transitively when this conftest imports ``AgentRegistry``)
    registers the bundled agents. Save that initial state and
    restore it on teardown so subsequent tests still see the
    bundled agents.
    """
    from clawcodex_ext.agent.registry import AgentRegistry

    saved_defs = list(AgentRegistry._definitions)
    saved_by_type = dict(AgentRegistry._by_type)
    try:
        yield
    finally:
        AgentRegistry._definitions = saved_defs
        AgentRegistry._by_type = saved_by_type
