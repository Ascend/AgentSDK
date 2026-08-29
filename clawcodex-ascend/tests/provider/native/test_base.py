#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for the ``NativeProvider`` abstract base class ."""

from __future__ import annotations

import pytest

from src.providers.native.base import NativeProvider
from src.providers.native.capabilities import (
    CAP_REASONING,
    CAP_STREAMING_TOOLS,
    CAP_STRUCTURED_OUTPUT,
    CAP_VISION,
)


class _NativeAdapter(NativeProvider):
    """Concrete subclass used for the abstract-method contract tests."""

    capabilities = {CAP_STRUCTURED_OUTPUT, CAP_VISION}

    def get_provider_name(self) -> str:
        return "test"

    def chat(self, messages, tools=None, **kwargs):  # pragma: no cover - abstract filler
        raise NotImplementedError

    def chat_stream(self, messages, tools=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def get_available_models(self) -> list[str]:  # pragma: no cover
        return []


def test_check_capabilities_empty_required_is_true() -> None:
    """An empty required-set is vacuously satisfied (subset of any set)."""
    assert _NativeAdapter.check_capabilities(set()) is True


def test_check_capabilities_subset() -> None:
    """Asking for a strict subset of supported caps returns True."""
    assert _NativeAdapter.check_capabilities({CAP_STRUCTURED_OUTPUT}) is True
    assert _NativeAdapter.check_capabilities({CAP_VISION}) is True
    assert _NativeAdapter.check_capabilities({CAP_STRUCTURED_OUTPUT, CAP_VISION}) is True


def test_check_capabilities_unsupported() -> None:
    """Asking for a capability the provider doesn't have returns False."""
    assert _NativeAdapter.check_capabilities({CAP_STREAMING_TOOLS}) is False
    assert _NativeAdapter.check_capabilities({CAP_REASONING}) is False
    assert _NativeAdapter.check_capabilities({CAP_STRUCTURED_OUTPUT, CAP_REASONING}) is False


def test_has_capability_single() -> None:
    assert _NativeAdapter.has_capability(CAP_VISION) is True
    assert _NativeAdapter.has_capability(CAP_REASONING) is False


def test_default_capabilities_is_empty_set() -> None:
    """A subclass that forgets to override ``capabilities`` exposes
    an empty set, not a parent's value.
    """

    class _Forgot(NativeProvider):
        def get_provider_name(self) -> str:
            return "forgot"

        def chat(self, messages, tools=None, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def chat_stream(self, messages, tools=None, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def get_available_models(self) -> list[str]:  # pragma: no cover
            return []

    assert _Forgot.capabilities == set()


def test_get_provider_name_is_abstract() -> None:
    """Subclasses must implement ``get_provider_name``; instantiating
    a class that doesn't should fail at construction time.
    """

    class _NoName(NativeProvider):
        def chat(self, messages, tools=None, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def chat_stream(self, messages, tools=None, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def get_available_models(self) -> list[str]:  # pragma: no cover
            return []

    with pytest.raises(TypeError):
        _NoName(api_key="k")  # type: ignore[abstract]


def test_get_sdk_client_falls_back_to_attribute() -> None:
    """``get_sdk_client`` returns the held client if one exists."""
    sentinel = object()

    class _WithClient(_NativeAdapter):
        def __init__(self):
            self.client = sentinel

    instance = _WithClient()
    assert instance.get_sdk_client() is sentinel


def test_get_sdk_client_returns_none_when_no_client() -> None:
    """Providers that don't hold a client (composition wrappers)
    get a clean ``None`` rather than ``AttributeError``.
    """

    class _NoClient(_NativeAdapter):
        def __init__(self) -> None:
            # Skip ``BaseProvider.__init__``'s required ``api_key``
            # — we're testing the attribute lookup, not the
            # constructor contract.
            pass

    instance = _NoClient()
    # ``_NativeAdapter`` doesn't set ``client`` in ``__init__`` so
    # ``getattr(..., "client", None)`` returns ``None``.
    assert instance.get_sdk_client() is None
