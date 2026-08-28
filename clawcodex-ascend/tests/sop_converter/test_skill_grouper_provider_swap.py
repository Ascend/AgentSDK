#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Backward-compatibility test: BaseProvider → SOPAssistantProviderProtocol adapter.

Verifies:
- SOPAssistantProviderAdapter.from_provider() wraps a BaseProvider correctly
- chat() returns a string response
- The adapter is runtime_checkable as SOPAssistantProviderProtocol

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §4.2 and §5.
"""

from __future__ import annotations

from typing import Any

import pytest

from extensions.capabilities.sop_provider_protocol import (
    SOPAssistantProviderProtocol,
)


@pytest.fixture
def fake_provider():
    """Create a minimal fake BaseProvider-like object for testing."""

    class FakeProvider:
        """Mimics clawcodex_ext.providers.base.BaseProvider interface."""

        def chat(self, messages: list[dict[str, str]]) -> Any:
            class FakeResponse:
                content = "Hello from the fake provider!"

            return FakeResponse()

    return FakeProvider()


def test_sop_provider_adapter_import():
    """SOPAssistantProviderAdapter is importable."""
    from extensions.sop_converter.adapters.sop_provider_adapter import (
        SOPAssistantProviderAdapter,
    )

    assert SOPAssistantProviderAdapter is not None


def test_sop_provider_adapter_from_provider(fake_provider):
    """from_provider() wraps a BaseProvider and returns a SOPAssistantProviderProtocol."""
    from extensions.sop_converter.adapters.sop_provider_adapter import (
        SOPAssistantProviderAdapter,
    )

    adapter = SOPAssistantProviderAdapter.from_provider(fake_provider)
    assert isinstance(adapter, SOPAssistantProviderProtocol)


def test_sop_provider_adapter_chat(fake_provider):
    """chat() returns a string response."""
    from extensions.sop_converter.adapters.sop_provider_adapter import (
        SOPAssistantProviderAdapter,
    )

    adapter = SOPAssistantProviderAdapter.from_provider(fake_provider)
    response = adapter.chat([{"role": "user", "content": "Hello"}])
    assert isinstance(response, str)
    assert "Hello" in response


def test_sop_provider_adapter_standalone():
    """SOPAssistantProviderAdapter can be used standalone (without a BaseProvider)
    when wrapped around a simple callable object.
    """
    from extensions.sop_converter.adapters.sop_provider_adapter import (
        SOPAssistantProviderAdapter,
    )

    class StaticProvider:
        def chat(self, messages):
            class FakeResponse:
                content = "Static reply"

            return FakeResponse()

    adapter = SOPAssistantProviderAdapter(StaticProvider())
    assert isinstance(adapter, SOPAssistantProviderProtocol)
    response = adapter.chat([{"role": "user", "content": "Hi"}])
    assert response == "Static reply"


def test_sop_provider_defaults_is_none():
    """DEFAULTS.sop_provider is None by default (no LLM dependency)."""
    from extensions.sop_converter.adapters import DEFAULTS, fill_defaults

    fill_defaults(DEFAULTS)
    assert DEFAULTS.sop_provider is None


def test_skill_grouper_accepts_protocol():
    """skill_grouper's group_source_components accepts SOPAssistantProviderProtocol
    (or None) without importing BaseProvider.
    """
    # This is a structural test — we verify the function signature accepts
    # the Protocol type.  The actual runtime behaviour is tested in the
    # skill_grouper unit tests.
    import inspect

    from extensions.sop_converter.runtime.skill_grouper import (
        group_source_components,
    )

    sig = inspect.signature(group_source_components)
    # The function should have a 'sop_provider' param (or 'llm_provider' param)
    params = sig.parameters
    has_sop_provider = "sop_provider" in params or "llm_provider" in params
    assert has_sop_provider, (
        f"group_source_components should accept sop_provider or llm_provider; got params: {list(params.keys())}"
    )
