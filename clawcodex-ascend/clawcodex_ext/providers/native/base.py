#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Base class for native SDK adapters.

Native providers sit alongside the existing LiteLLM-based
``BaseProvider`` implementations. The motivation is the same as CCB's
multi-adapter design: the LiteLLM universal facade flattens over a
hundred backends, but the cost is that platform-specific features
(Gemini's ``SafetySetting``, OpenAI's structured-output ``response_format``
JSON-Schema mode, Grok's tool-calling variant) get lost in translation.

Native adapters keep the same ``BaseProvider`` contract (``chat`` /
``chat_stream`` / ``chat_stream_response`` / ``get_available_models``)
so they drop into the existing query loop unchanged, but expose a
``capabilities`` registry that lets the rest of the system reason about
which provider can satisfy a given request without round-tripping the
network.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from clawcodex_ext.providers.base import BaseProvider


class NativeProvider(BaseProvider):
    """Abstract base for native SDK adapters.

    Subclasses must:

    * declare a ``capabilities`` class attribute (a ``set[str]`` of
      :data:`~clawcodex_ext.providers.native.capabilities.CAP_*` identifiers);
    * implement :meth:`get_provider_name` returning a stable lowercase
      identifier (``"openai"``, ``"gemini"``, ``"grok"`` ...).

    The default :meth:`check_capabilities` is a pure subset check —
    callers can pass a request's required capabilities and ask "is this
    provider sufficient?". The check is intentionally permissive
    (subset, not equality) so that adding a new capability to a
    provider doesn't break callers that only asked for a smaller set.
    """

    # Class-level capability set. Subclasses must override. Empty by
    # default so a forgotten override surfaces immediately as "no
    # capabilities" rather than silently inheriting from a sibling.
    capabilities: ClassVar[set[str]] = set()

    @classmethod
    def check_capabilities(cls, required: set[str]) -> bool:
        """Return whether *all* required capabilities are supported.

        Args:
            required: A set of ``CAP_*`` identifiers the caller needs.

        Returns:
            ``True`` if ``required`` is a subset of
            ``cls.capabilities``. An empty ``required`` set returns
            ``True`` (vacuous truth).
        """
        if not required:
            return True
        return required.issubset(cls.capabilities)

    @classmethod
    def has_capability(cls, capability: str) -> bool:
        """Convenience single-capability check."""
        return capability in cls.capabilities

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a stable, lowercase provider identifier.

        The value is used as a key for the native-provider registry
        and as the ``provider`` field on the model-routing telemetry.
        Examples: ``"openai"``, ``"gemini"``, ``"grok"``.
        """

    # Native adapters typically hold a single SDK client. The helper
    # below documents the convention; subclasses are free to ignore it
    # if they need a different lifecycle (e.g. a pool).
    def get_sdk_client(self):  # pragma: no cover - thin pass-through
        """Return the underlying SDK client, if one is held.

        Most native adapters cache an SDK client in ``__init__`` and
        expose it as ``self.client``; this helper just normalises the
        access so callers (e.g. tests, telemetry) don't have to know
        the attribute name.
        """
        return getattr(self, "client", None)


__all__ = ["NativeProvider"]
