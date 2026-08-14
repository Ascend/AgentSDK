# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Adapter Protocol — unified interface for optional dependency adapters.

Shared helpers for the optional-dependency adapters (outlines, gitpython,
pluggy, treesitter, pydantic, frontmatter, litellm), replacing the
duplicated ``os.getenv`` + ``try/except ImportError`` pattern:

* ``env_switch()`` — environment-variable gate.
* ``dependency_available()`` — optional-dependency detection.
* ``AdapterRegistry`` — central registry for adapter discovery.
* ``AdapterProtocol`` — structural typing base for adapters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "AdapterInfo",
    "AdapterProtocol",
    "AdapterRegistry",
    "dependency_available",
    "env_switch",
]


def env_switch(var_name: str, default: str = "true") -> bool:
    """Return ``True`` when *var_name* is ``'true'``, ``'1'``, or absent.

    Example: ``_USE_GITPYTHON = env_switch("CLAW_USE_GITPYTHON")``.
    """
    return os.getenv(var_name, default).lower() in ("true", "1")


def dependency_available(module_name: str) -> bool:
    """Return ``True`` if *module_name* can be imported.

    Example: ``_GITPYTHON_AVAILABLE = dependency_available("git")``.
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def is_provider_adapter(adapter: type) -> bool:
    """Return ``True`` if *adapter* is a provider adapter (has chat/stream)."""
    return hasattr(adapter, "chat") or hasattr(adapter, "chat_stream")


@dataclass
class AdapterInfo:
    """Metadata about a registered adapter."""

    name: str
    env_var: str | None = None
    dependency: str | None = None
    description: str = ""
    is_enabled_by_default: bool = True


class AdapterRegistry:
    """Central registry for optional-dependency adapters.

    Adapters self-register via the :meth:`register` classmethod decorator
    (e.g. ``@AdapterRegistry.register("gitpython", env_var="CLAW_USE_GITPYTHON")``).
    Discovery: :meth:`list` / :meth:`get`.
    """

    _adapters: dict[str, AdapterInfo] = {}
    _loaded: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        *,
        env_var: str | None = None,
        dependency: str | None = None,
        description: str = "",
        is_enabled_by_default: bool = True,
    ) -> Any:
        """Decorator that registers an adapter class (returned unchanged)."""
        info = AdapterInfo(
            name=name,
            env_var=env_var,
            dependency=dependency,
            description=description or name,
            is_enabled_by_default=is_enabled_by_default,
        )

        def decorator(adapter_cls: type) -> type:
            cls._adapters[name] = info
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> AdapterInfo | None:
        """Return metadata for *name*, or ``None``."""
        return cls._adapters.get(name)

    @classmethod
    def list(cls) -> dict[str, AdapterInfo]:
        """Return all registered adapters keyed by name."""
        return dict(cls._adapters)

    @classmethod
    def is_enabled(cls, name: str) -> bool:
        """Check whether adapter *name* is enabled (env-var gate)."""
        info = cls._adapters.get(name)
        if info is None:
            return False
        if info.env_var is None:
            return True
        default = "true" if info.is_enabled_by_default else "false"
        return env_switch(info.env_var, default=default)

    @classmethod
    def is_dependency_available(cls, name: str) -> bool:
        """Check whether adapter *name*'s dependency is installed."""
        info = cls._adapters.get(name)
        if info is None or info.dependency is None:
            return False
        return dependency_available(info.dependency)


class AdapterProtocol(Protocol):
    """Structural typing for optional-dependency adapters.

    An adapter module should expose ``is_available() -> bool``.
    """

    name: str

    def is_available(self) -> bool:
        """Return ``True`` when the dependency is installed and the
        env var doesn't disable it.
        """
        ...
